"""
FEM-augmented evaluation pipeline for outrunner BLDC motor design.

Replaces (or supplements) the analytical EM/thermal/mechanical evaluations
with true 2-D finite element solvers from scikit-fem.  Falls back to the
analytical pipeline automatically if FEM solve raises an exception.

Workflow:
    1. Build geometry + winding from design vector  (same as optimizer.py)
    2. Run FEM EM (magnetostatic) → torque, Bg, losses
    3. Run FEM Thermal (steady-state heat conduction) → temperature field
    4. Run FEM Mechanical (plane-stress centrifugal + modal) → stress, modes
    5. Analytical CFD (unchanged — no FEM equivalent)
    6. Package objectives + constraints (same interface as optimizer.py)

If any FEM step fails, that domain falls back to the analytical solver,
so the optimizer always returns a result.

Key References:
  [1] T. Gustafsson, G. McBain, "scikit-fem," JOSS, 2020.
  [2] K. Deb et al., "NSGA-II," IEEE Trans. Evol. Comp., 2002.
  [3] J. Blank, K. Deb, "pymoo," IEEE Access, 2020.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import warnings
import numpy as np

from .materials import MaterialDatabase
from .electromagnetic import (
    MotorSpecs, ElectromagneticModel, EMResults,
    select_slot_pole, GeometryParams, WindingConfig, compute_winding_factor,
)
from .thermal import ThermalModel, ThermalResults
from .mechanical import MechanicalModel, MechanicalResults
from .cfd import CFDModel, CFDResults

from .optimizer import (
    DESIGN_VARS, N_VARS, _bounds, EvalResult,
)


MU_0 = 4 * np.pi * 1e-7


def _build_geometry_winding(x: np.ndarray,
                            slot_pole: Tuple[int, int],
                            materials: MaterialDatabase,
                            specs: MotorSpecs,
                            ) -> Tuple[GeometryParams, WindingConfig, ElectromagneticModel]:
    """Build geometry, winding, and analytical EM model from design vector."""
    n_slots, n_poles = slot_pole

    R_so = x[0]
    L_stack = x[1]
    hm = x[2]
    g_air = x[3]
    sd_frac = x[4]
    tw_frac = x[5]
    tpc = max(1, int(round(x[6])))
    d_wire = x[7] * 1e-3
    mag_arc = x[8]
    t_yoke_r = x[9]

    geo = GeometryParams()
    geo.stator_outer_radius = R_so
    geo.stack_length = L_stack
    geo.magnet_thickness = hm
    geo.airgap = g_air

    tau_s = 2 * np.pi * R_so / n_slots
    geo.tooth_width = tw_frac * tau_s
    geo.slot_opening = max(tau_s * 0.25, 0.001)
    geo.stator_yoke_thickness = max(0.002, (R_so - geo.shaft_radius - 0.002) * 0.25)
    avail = R_so - geo.stator_yoke_thickness - geo.shaft_radius - 0.002
    geo.slot_depth = max(sd_frac * avail, 0.003)
    geo.stator_inner_radius = geo.shaft_radius + 0.001

    geo.rotor_inner_radius = R_so + g_air
    geo.rotor_yoke_thickness = t_yoke_r
    geo.rotor_outer_radius = geo.rotor_inner_radius + hm + t_yoke_r

    materials.magnet.thickness = hm
    materials.magnet.arc_fraction = mag_arc
    materials.magnet.length = L_stack

    wdg = WindingConfig()
    wdg.num_poles = n_poles
    wdg.num_slots = n_slots
    wdg.turns_per_coil = tpc
    wdg.wire_diameter = d_wire
    wdg.num_layers = 2
    wdg.coil_span = 1
    wdg.winding_factor = compute_winding_factor(n_slots, n_poles)
    wdg.end_turn_length = 1.5 * geo.tooth_width + 0.010

    # Phase resistance
    coils_per_phase = n_slots // 3
    N_total = tpc * coils_per_phase * wdg.num_layers
    l_turn = 2 * L_stack + 2 * wdg.end_turn_length
    wire_area = wdg.wire_area * wdg.num_strands
    rho_cu = materials.copper.resistivity(80.0)
    wdg.phase_resistance = rho_cu * N_total * l_turn / max(wire_area, 1e-12)

    # Inductance (simplified)
    p = n_poles // 2
    tau_p = np.pi * R_so / max(p, 1)
    g_total = g_air + hm / materials.magnet.mu_rec
    L_gap = MU_0 * N_total**2 * L_stack * tau_p / (np.pi * max(p, 1) * g_total) * 1.5
    L_slot = MU_0 * N_total**2 * L_stack * coils_per_phase * \
             geo.slot_depth / (3 * max(geo.slot_opening, 0.001))
    wdg.phase_inductance = L_gap + L_slot

    em_model = ElectromagneticModel(specs, materials)
    em_model.geometry = geo
    em_model.winding = wdg

    return geo, wdg, em_model


def evaluate_design_fem(x: np.ndarray,
                        specs: MotorSpecs,
                        materials: MaterialDatabase,
                        slot_pole: Tuple[int, int] = (12, 14),
                        T_ambient: float = 25.0,
                        max_thermal_iters: int = 5,
                        mesh_density: int = 80,
                        ) -> EvalResult:
    """
    Evaluate a single design vector using FEM solvers where available.

    Falls back to analytical solvers for any domain that fails.

    Same interface and objective/constraint layout as optimizer.evaluate_design.
    """
    res = EvalResult()
    res.design_vector = x.copy()
    res.slot_pole = slot_pole
    n_slots, n_poles = slot_pole

    INFEASIBLE_OBJ = np.array([0, 1e6, 0])
    INFEASIBLE_CON = np.array([100, 100, 100, 100, 100, 1])

    try:
        geo, wdg, em_model = _build_geometry_winding(x, slot_pole, materials, specs)
    except Exception:
        res.objectives = INFEASIBLE_OBJ
        res.constraints = INFEASIBLE_CON
        return res

    # --- FEM EM (with analytical fallback) ---
    em_res = None
    fem_em_used = False
    T_mag = 80.0
    T_wind = 100.0

    try:
        from .fem_electromagnetic import FEMElectromagneticModel
        fem_em = FEMElectromagneticModel(
            specs, geo, wdg, materials, mesh_density=mesh_density
        )
        fem_result = fem_em.solve_magnetostatic(
            rotor_angle_deg=0.0, T_magnet=T_mag, T_winding=T_wind
        )
        # Map FEM EM results to analytical EMResults for compatibility
        em_res = em_model.compute_performance(T_magnet=T_mag, T_winding=T_wind)
        # Override with FEM values where available and physically plausible
        # Sanity bounds: Bg < 3T, torque > 0, back_emf reasonable
        if 0.01 < fem_result.Bg1_fundamental < 3.0:
            em_res.airgap_flux_density = fem_result.Bg1_fundamental
        if 0 < fem_result.torque_maxwell < 1000:
            em_res.torque_avg = fem_result.torque_maxwell
        if 0 < fem_result.back_emf_peak < 1e4:
            em_res.back_emf_peak = fem_result.back_emf_peak
            em_res.back_emf_rms = fem_result.back_emf_peak / np.sqrt(2)
        if 0 < fem_result.iron_loss_stator < 1e6:
            em_res.iron_loss_stator = fem_result.iron_loss_stator
        if 0 < fem_result.magnet_eddy_loss < 1e6:
            em_res.magnet_loss = fem_result.magnet_eddy_loss
        # Recalculate totals with FEM-updated losses
        em_res.total_loss = (em_res.copper_loss + em_res.iron_loss_stator +
                             em_res.iron_loss_rotor + em_res.magnet_loss)
        P_in = em_res.power_output + em_res.total_loss
        em_res.efficiency = em_res.power_output / P_in if P_in > 0 else 0
        fem_em_used = True
    except Exception:
        pass

    if em_res is None:
        # Pure analytical fallback
        try:
            em_res = em_model.compute_performance(T_magnet=T_mag, T_winding=T_wind)
        except Exception:
            res.objectives = INFEASIBLE_OBJ
            res.constraints = INFEASIBLE_CON
            return res

    # --- Coupled EM-thermal iteration (using FEM thermal when possible) ---
    th_res = None
    temperature_field = None

    for _iter in range(max_thermal_iters):
        try:
            em_res_iter = em_model.compute_performance(T_magnet=T_mag, T_winding=T_wind)
            if fem_em_used:
                # Keep FEM overrides
                em_res_iter.airgap_flux_density = em_res.airgap_flux_density
        except Exception:
            break

        # Try FEM thermal
        try:
            from .fem_thermal import FEMThermalModel
            fem_th = FEMThermalModel(geo, wdg, materials, T_ambient,
                                     mesh_density=mesh_density)
            fem_th_res = fem_th.solve_steady_state(em_res_iter, specs.target_rpm)
            # Convert to analytical-compatible ThermalResults
            th_res = fem_th.to_thermal_results()
            temperature_field = fem_th_res.temperature_field
        except Exception:
            # Analytical fallback
            th_model = ThermalModel(geo, wdg, materials, T_ambient)
            th_res = th_model.build_and_solve(em_res_iter, specs.target_rpm)
            temperature_field = None

        dT_mag = abs(th_res.T_magnets - T_mag)
        dT_wind = abs(th_res.T_winding_slot - T_wind)
        T_mag = th_res.T_magnets
        T_wind = th_res.T_winding_slot
        if dT_mag < 1.0 and dT_wind < 1.0:
            break

    if th_res is None:
        # If thermal completely failed, use analytical
        th_model = ThermalModel(geo, wdg, materials, T_ambient)
        th_res = th_model.build_and_solve(em_res, specs.target_rpm)

    # --- CFD (always analytical — no FEM equivalent) ---
    cfd_model = CFDModel(geo, wdg)
    cfd_res = cfd_model.compute(specs.target_rpm, T_air=(T_ambient + T_mag) / 2)

    # --- FEM Mechanical (with analytical fallback) ---
    mech_res = None
    fem_mech_results = None

    try:
        from .fem_mechanical import FEMMechanicalModel
        fem_mech = FEMMechanicalModel(geo, wdg, materials,
                                      mesh_density=mesh_density)
        fem_mech_results = fem_mech.solve_static(
            rpm=specs.target_rpm,
            temperature_field=temperature_field,
        )
        fem_mech.solve_modal(n_modes=6)
    except Exception:
        fem_mech_results = None

    # Always run analytical mechanical for bearing/shaft/mounting calcs
    mech_model = MechanicalModel(geo, wdg, materials)
    mech_res = mech_model.run_full_analysis(
        specs.target_rpm, em_res.torque_avg, em_res
    )

    # If FEM mechanical succeeded, overlay the stress results
    if fem_mech_results is not None:
        # FEM provides more accurate stress distribution
        mech_res.rotor_stress.max_hoop_stress = fem_mech_results.rotor_hoop_stress_max
        if fem_mech_results.rotor_safety_factor > 0:
            pass  # Keep analytical retention margin for now (FEM gives region stress)
        if fem_mech_results.natural_frequencies_hz:
            mech_res.vibration.natural_frequencies_hz = \
                fem_mech_results.natural_frequencies_hz[:3]

    # --- Assemble objectives / constraints ---
    total_loss = (em_res.total_loss + cfd_res.windage_loss +
                  mech_res.bearings.friction_loss_total)
    P_out = em_res.power_output
    P_in = P_out + total_loss
    efficiency = P_out / P_in if P_in > 0 else 0

    weights = em_model.compute_weight_breakdown()
    total_mass = weights["total_kg"]
    torque_density = em_res.torque_avg / max(total_mass, 0.001)

    res.objectives = np.array([
        -efficiency,
        total_mass,
        -torque_density,
    ])

    res.constraints = np.array([
        th_res.T_winding_slot - materials.copper.max_temp,
        th_res.T_magnets - materials.magnet.max_temp,
        1.5 - mech_res.shaft.shaft_safety_factor,
        1.5 - mech_res.rotor_stress.magnet_retention_margin,
        em_res.current_density - 10.0,
        0.0 if mech_res.bearings.speed_ok else 1.0,
    ])

    # Guard against NaN/Inf
    if np.any(np.isnan(res.objectives)) or np.any(np.isinf(res.objectives)):
        res.objectives = INFEASIBLE_OBJ
        res.constraints = INFEASIBLE_CON
        res.feasible = False
        return res
    if np.any(np.isnan(res.constraints)) or np.any(np.isinf(res.constraints)):
        res.constraints = np.where(
            np.isnan(res.constraints) | np.isinf(res.constraints),
            100.0, res.constraints
        )

    res.feasible = np.all(res.constraints <= 0)
    res.em = em_res
    res.thermal = th_res
    res.mech = mech_res
    res.cfd = cfd_res
    return res


# ---------------------------------------------------------------------------
# NSGA-II with FEM evaluations
# ---------------------------------------------------------------------------
def run_optimisation_fem(specs: MotorSpecs,
                         materials: MaterialDatabase,
                         slot_pole: Optional[Tuple[int, int]] = None,
                         T_ambient: float = 25.0,
                         pop_size: int = 20,
                         n_gen: int = 15,
                         mesh_density: int = 60,
                         ) -> Dict:
    """
    Run multi-objective optimisation with FEM solvers.

    Lower default pop_size and n_gen vs analytical because FEM is ~100×
    slower per evaluation.  Accepts mesh_density to trade accuracy vs speed.

    Returns same dict format as optimizer.run_optimisation.
    """
    from .optimizer import _bounds, N_VARS

    if slot_pole is None:
        candidates = select_slot_pole(specs.target_rpm, specs.target_torque,
                                      specs.voltage)
        slot_pole = (candidates[0][0], candidates[0][1]) if candidates else (12, 14)

    lb, ub = _bounds()

    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.termination import get_termination

        class MotorFEMProblem(Problem):
            def __init__(self):
                super().__init__(n_var=N_VARS, n_obj=3, n_ieq_constr=6,
                                 xl=lb, xu=ub)

            def _evaluate(self, X, out, *args, **kwargs):
                F = np.zeros((X.shape[0], 3))
                G = np.zeros((X.shape[0], 6))
                for i in range(X.shape[0]):
                    r = evaluate_design_fem(
                        X[i], specs, materials, slot_pole, T_ambient,
                        mesh_density=mesh_density,
                    )
                    F[i] = r.objectives
                    G[i] = r.constraints
                out["F"] = F
                out["G"] = G

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        result = pymoo_minimize(
            MotorFEMProblem(), algorithm,
            get_termination("n_gen", n_gen),
            seed=42, verbose=False,
        )

        if result.X is not None and len(result.X.shape) == 2:
            X_pareto = result.X
            F_pareto = result.F
        elif result.X is not None:
            X_pareto = result.X.reshape(1, -1)
            F_pareto = result.F.reshape(1, -1)
        else:
            X_pareto = np.empty((0, N_VARS))
            F_pareto = np.empty((0, 3))
        algo = "NSGA-II (FEM)"

    except ImportError:
        # Fallback DE with FEM
        rng = np.random.default_rng(42)
        pop = lb + (ub - lb) * rng.random((pop_size, N_VARS))
        fitness = np.full(pop_size, np.inf)

        def penalised(xx):
            r = evaluate_design_fem(
                xx, specs, materials, slot_pole, T_ambient,
                mesh_density=mesh_density,
            )
            obj = -0.5 * r.objectives[0] + 0.3 * r.objectives[1] - 0.2 * r.objectives[2]
            penalty = 1000 * np.sum(np.maximum(r.constraints, 0) ** 2)
            return obj + penalty, r

        for gen in range(n_gen):
            for i in range(pop_size):
                idxs = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                a, b, c = pop[idxs]
                mutant = np.clip(a + 0.8 * (b - c), lb, ub)
                mask = rng.random(N_VARS) < 0.9
                if not np.any(mask):
                    mask[rng.integers(N_VARS)] = True
                trial = np.where(mask, mutant, pop[i])
                f_trial, _ = penalised(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

        order = np.argsort(fitness)
        X_pareto = pop[order[:min(10, pop_size)]]
        F_pareto = np.array([
            evaluate_design_fem(xx, specs, materials, slot_pole, T_ambient,
                                mesh_density=mesh_density).objectives
            for xx in X_pareto
        ])
        algo = "DE-fallback (FEM)"

    # Re-evaluate Pareto set for full results
    full_results = [
        evaluate_design_fem(xx, specs, materials, slot_pole, T_ambient,
                            mesh_density=mesh_density)
        for xx in X_pareto
    ]
    F_pareto = np.array([r.objectives for r in full_results])

    # Knee point
    if F_pareto.shape[0] > 0:
        F_min = F_pareto.min(axis=0)
        F_max = F_pareto.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1
        F_norm = (F_pareto - F_min) / F_range
        dist = np.linalg.norm(F_norm, axis=1)
        best_idx = int(np.argmin(dist))
    else:
        best_idx = 0

    return {
        "pareto_X": X_pareto,
        "pareto_F": F_pareto,
        "pareto_results": full_results,
        "best_idx": best_idx,
        "algorithm": algo,
        "slot_pole": slot_pole,
    }