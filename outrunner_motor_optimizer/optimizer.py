"""
Multi-objective optimisation engine for outrunner BLDC motor design.

Uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) via the pymoo
library for Pareto-optimal trade-off between efficiency, mass, torque
density, thermal safety, and mechanical margins.

Falls back to a pure-NumPy differential-evolution implementation if
pymoo is unavailable.

Key References:
  [1] K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist
      Multiobjective Genetic Algorithm: NSGA-II," IEEE Trans. Evol. Comp.,
      vol. 6, no. 2, pp. 182-197, 2002.
  [2] J. Blank, K. Deb, "pymoo: Multi-Objective Optimization in Python,"
      IEEE Access, vol. 8, pp. 89497-89509, 2020.
  [3] R. Storn, K. Price, "Differential Evolution — A Simple and Efficient
      Heuristic for Global Optimization over Continuous Spaces," J. Global
      Optim., vol. 11, pp. 341-359, 1997.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import warnings

from .materials import MaterialDatabase
from .electromagnetic import (
    MotorSpecs, ElectromagneticModel, EMResults,
    select_slot_pole, GeometryParams, WindingConfig,
)
from .thermal import ThermalModel, ThermalResults
from .mechanical import MechanicalModel, MechanicalResults
from .cfd import CFDModel, CFDResults


# ---------------------------------------------------------------------------
# Design vector & bounds
# ---------------------------------------------------------------------------
DESIGN_VARS = [
    # name,                   lower,   upper,  description
    ("stator_outer_radius",   0.012,   0.150,  "Stator outer radius [m]"),
    ("stack_length",          0.008,   0.200,  "Active stack length [m]"),
    ("magnet_thickness",      0.001,   0.010,  "Magnet radial thickness [m]"),
    ("airgap",                0.0004,  0.002,  "Mechanical airgap [m]"),
    ("slot_depth_frac",       0.3,     0.7,    "Slot depth / available radial space"),
    ("tooth_width_frac",      0.3,     0.65,   "Tooth width / slot pitch"),
    ("turns_per_coil",        1,       80,     "Turns per coil (integer)"),
    ("wire_diameter_mm",      0.2,     3.0,    "Wire diameter [mm]"),
    ("magnet_arc_frac",       0.6,     0.95,   "Magnet arc / pole arc"),
    ("rotor_yoke_thickness",  0.001,   0.008,  "Rotor back-iron thickness [m]"),
]

N_VARS = len(DESIGN_VARS)


def _bounds():
    lb = np.array([v[1] for v in DESIGN_VARS])
    ub = np.array([v[2] for v in DESIGN_VARS])
    return lb, ub


# ---------------------------------------------------------------------------
# Objective evaluator
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Single design-point evaluation."""
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(3))
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(6))
    em: Optional[EMResults] = None
    thermal: Optional[ThermalResults] = None
    mech: Optional[MechanicalResults] = None
    cfd: Optional[CFDResults] = None
    feasible: bool = False
    design_vector: Optional[np.ndarray] = None
    slot_pole: Tuple[int, int] = (12, 14)


def evaluate_design(x: np.ndarray,
                    specs: MotorSpecs,
                    materials: MaterialDatabase,
                    slot_pole: Tuple[int, int] = (12, 14),
                    T_ambient: float = 25.0,
                    max_thermal_iters: int = 5,
                    ) -> EvalResult:
    """
    Evaluate a single design vector through the coupled multiphysics pipeline.

    Objectives (minimise):
      f0 = -efficiency        (maximise efficiency)
      f1 = total_mass         (minimise mass)
      f2 = -torque_density    (maximise T/kg)

    Constraints (g ≤ 0):
      g0 = T_winding - T_limit_winding
      g1 = T_magnet  - T_limit_magnet
      g2 = 1.5 - shaft_safety_factor
      g3 = 1.5 - magnet_retention_margin
      g4 = J_current - 10  (A/mm², keep current density reasonable)
      g5 = 1.0 - bearing_speed_ok  (0 if OK, 1 if violated)
    """
    res = EvalResult()
    res.design_vector = x.copy()
    res.slot_pole = slot_pole
    n_slots, n_poles = slot_pole

    # Unpack design vector
    R_so = x[0]
    L_stack = x[1]
    hm = x[2]
    g_air = x[3]
    sd_frac = x[4]
    tw_frac = x[5]
    tpc = max(1, int(round(x[6])))
    d_wire = x[7] * 1e-3  # mm → m
    mag_arc = x[8]
    t_yoke_r = x[9]

    # Build geometry
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

    # Update magnet geometry
    materials.magnet.thickness = hm
    materials.magnet.arc_fraction = mag_arc
    materials.magnet.length = L_stack

    # Build winding
    wdg = WindingConfig()
    wdg.num_poles = n_poles
    wdg.num_slots = n_slots
    wdg.turns_per_coil = tpc
    wdg.wire_diameter = d_wire
    wdg.num_layers = 2
    wdg.coil_span = 1

    # Create EM model & compute winding factor
    em_model = ElectromagneticModel(specs, materials)
    em_model.geometry = geo

    from .electromagnetic import compute_winding_factor
    wdg.winding_factor = compute_winding_factor(n_slots, n_poles)

    # End turn length
    wdg.end_turn_length = 1.5 * geo.tooth_width + 0.010

    # Phase resistance
    coils_per_phase = n_slots // 3
    N_total = tpc * coils_per_phase * wdg.num_layers
    l_turn = 2 * L_stack + 2 * wdg.end_turn_length
    wire_area = wdg.wire_area * wdg.num_strands

    rho_cu = materials.copper.resistivity(80.0)
    wdg.phase_resistance = rho_cu * N_total * l_turn / max(wire_area, 1e-12)

    # Inductance (simplified)
    MU_0 = 4 * np.pi * 1e-7
    p = n_poles // 2
    tau_p = np.pi * R_so / max(p, 1)
    g_total = g_air + hm / materials.magnet.mu_rec
    L_gap = MU_0 * N_total**2 * L_stack * tau_p / (np.pi * max(p, 1) * g_total) * 1.5
    L_slot = MU_0 * N_total**2 * L_stack * coils_per_phase * \
             geo.slot_depth / (3 * max(geo.slot_opening, 0.001))
    wdg.phase_inductance = L_gap + L_slot

    em_model.winding = wdg

    # --- Coupled EM-Thermal iteration ---
    T_mag = 80.0
    T_wind = 100.0

    for _iter in range(max_thermal_iters):
        try:
            em_res = em_model.compute_performance(T_magnet=T_mag, T_winding=T_wind)
        except Exception:
            # Infeasible geometry
            res.objectives = np.array([0, 1e6, 0])
            res.constraints = np.array([100, 100, 100, 100, 100, 1])
            return res

        # Thermal
        th_model = ThermalModel(geo, wdg, materials, T_ambient)
        th_res = th_model.build_and_solve(em_res, specs.target_rpm)

        # Check convergence
        dT_mag = abs(th_res.T_magnets - T_mag)
        dT_wind = abs(th_res.T_winding_slot - T_wind)
        T_mag = th_res.T_magnets
        T_wind = th_res.T_winding_slot
        if dT_mag < 1.0 and dT_wind < 1.0:
            break

    # CFD (windage)
    cfd_model = CFDModel(geo, wdg)
    cfd_res = cfd_model.compute(specs.target_rpm, T_air=(T_ambient + T_mag) / 2)

    # Mechanical
    mech_model = MechanicalModel(geo, wdg, materials)
    mech_res = mech_model.run_full_analysis(
        specs.target_rpm, em_res.torque_avg, em_res
    )

    # Total losses including windage and bearing friction
    total_loss = em_res.total_loss + cfd_res.windage_loss + mech_res.bearings.friction_loss_total
    P_out = em_res.power_output
    P_in = P_out + total_loss
    efficiency = P_out / P_in if P_in > 0 else 0

    # Mass
    weights = em_model.compute_weight_breakdown()
    total_mass = weights["total_kg"]

    # Torque density
    torque_density = em_res.torque_avg / max(total_mass, 0.001)

    # --- Objectives (minimise) ---
    res.objectives = np.array([
        -efficiency,          # maximise → negate
        total_mass,
        -torque_density,      # maximise → negate
    ])

    # --- Constraints (g ≤ 0 means feasible) ---
    res.constraints = np.array([
        th_res.T_winding_slot - materials.copper.max_temp,
        th_res.T_magnets - materials.magnet.max_temp,
        1.5 - mech_res.shaft.shaft_safety_factor,
        1.5 - mech_res.rotor_stress.magnet_retention_margin,
        em_res.current_density - 10.0,
        0.0 if mech_res.bearings.speed_ok else 1.0,
    ])

    # Guard against NaN/Inf propagation (causes pymoo comparison failures)
    if np.any(np.isnan(res.objectives)) or np.any(np.isinf(res.objectives)):
        res.objectives = np.array([0, 1e6, 0])
        res.constraints = np.array([100, 100, 100, 100, 100, 1])
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
# NSGA-II wrapper (pymoo)
# ---------------------------------------------------------------------------
def _try_pymoo_nsga2(specs, materials, slot_pole, T_ambient, pop_size, n_gen):
    """Attempt pymoo-based NSGA-II. Returns (X_pareto, F_pareto, results)."""
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    lb, ub = _bounds()

    class MotorProblem(Problem):
        def __init__(self):
            super().__init__(n_var=N_VARS, n_obj=3, n_ieq_constr=6,
                             xl=lb, xu=ub)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 3))
            G = np.zeros((X.shape[0], 6))
            for i in range(X.shape[0]):
                r = evaluate_design(X[i], specs, materials, slot_pole, T_ambient)
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

    problem = MotorProblem()
    termination = get_termination("n_gen", n_gen)

    result = pymoo_minimize(problem, algorithm, termination, seed=42, verbose=False)

    if result.X is not None and len(result.X.shape) == 2:
        X_pareto = result.X
        F_pareto = result.F
    elif result.X is not None:
        X_pareto = result.X.reshape(1, -1)
        F_pareto = result.F.reshape(1, -1)
    else:
        X_pareto = np.empty((0, N_VARS))
        F_pareto = np.empty((0, 3))

    # Re-evaluate Pareto set to get full result objects
    full_results = []
    for x in X_pareto:
        full_results.append(evaluate_design(x, specs, materials, slot_pole, T_ambient))

    return X_pareto, F_pareto, full_results


# ---------------------------------------------------------------------------
# Fallback: simple differential evolution (NumPy only)
# ---------------------------------------------------------------------------
def _fallback_de(specs, materials, slot_pole, T_ambient, pop_size, n_gen):
    """Differential evolution with penalty-based constraint handling."""
    lb, ub = _bounds()
    rng = np.random.default_rng(42)

    # Initialise population
    pop = lb + (ub - lb) * rng.random((pop_size, N_VARS))
    fitness = np.full(pop_size, np.inf)

    def penalised_scalar(x):
        r = evaluate_design(x, specs, materials, slot_pole, T_ambient)
        # Weighted sum + penalty
        obj = -0.5 * r.objectives[0] + 0.3 * r.objectives[1] + -0.2 * r.objectives[2]
        penalty = 1000 * np.sum(np.maximum(r.constraints, 0) ** 2)
        return obj + penalty, r

    best_results = []

    for gen in range(n_gen):
        for i in range(pop_size):
            # Mutation (DE/rand/1)
            idxs = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            a, b, c = pop[idxs]
            F_de = 0.8
            mutant = a + F_de * (b - c)
            mutant = np.clip(mutant, lb, ub)

            # Crossover
            CR = 0.9
            mask = rng.random(N_VARS) < CR
            if not np.any(mask):
                mask[rng.integers(N_VARS)] = True
            trial = np.where(mask, mutant, pop[i])

            f_trial, r_trial = penalised_scalar(trial)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

    # Collect top results
    order = np.argsort(fitness)
    X_pareto = pop[order[:min(10, pop_size)]]
    full_results = []
    for x in X_pareto:
        full_results.append(evaluate_design(x, specs, materials, slot_pole, T_ambient))
    F_pareto = np.array([r.objectives for r in full_results])

    return X_pareto, F_pareto, full_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_optimisation(specs: MotorSpecs,
                     materials: MaterialDatabase,
                     slot_pole: Optional[Tuple[int, int]] = None,
                     T_ambient: float = 25.0,
                     pop_size: int = 40,
                     n_gen: int = 30,
                     ) -> Dict:
    """
    Run multi-objective optimisation of the motor design.

    Returns dict with keys:
      - pareto_X       : array of Pareto-optimal design vectors
      - pareto_F       : corresponding objective values
      - pareto_results : list[EvalResult] with full physics results
      - best_idx       : index of the "knee" design (balanced)
      - algorithm      : "NSGA-II" or "DE-fallback"
    """
    # Auto-select slot-pole if not provided
    if slot_pole is None:
        candidates = select_slot_pole(specs.target_rpm, specs.target_torque,
                                      specs.voltage)
        if candidates:
            slot_pole = (candidates[0][0], candidates[0][1])
        else:
            slot_pole = (12, 14)

    # Try pymoo first
    try:
        X, F, results = _try_pymoo_nsga2(
            specs, materials, slot_pole, T_ambient, pop_size, n_gen
        )
        algo = "NSGA-II"
    except ImportError:
        warnings.warn("pymoo not installed — falling back to differential evolution.")
        X, F, results = _fallback_de(
            specs, materials, slot_pole, T_ambient, pop_size, n_gen
        )
        algo = "DE-fallback"

    # Pick the "knee" point: closest to ideal (utopia) point in normalised space
    if F.shape[0] > 0:
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1
        F_norm = (F - F_min) / F_range
        dist = np.linalg.norm(F_norm, axis=1)
        best_idx = int(np.argmin(dist))
    else:
        best_idx = 0

    return {
        "pareto_X": X,
        "pareto_F": F,
        "pareto_results": results,
        "best_idx": best_idx,
        "algorithm": algo,
        "slot_pole": slot_pole,
    }
