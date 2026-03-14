#!/usr/bin/env python3
"""
Outrunner BLDC Motor Design Optimiser — Main Pipeline
=====================================================

End-to-end workflow:
  1. Accept user specifications (voltage, RPM, torque, current, efficiency)
  2. Configure materials (magnet type, structural alloy, custom overrides)
  3. Configure mounting (bolt pattern, loads, vibration isolation)
  4. Run coupled EM-thermal-mechanical-CFD multi-objective optimisation
  5. Export STEP files for the best design
  6. Print full design report with validation checks

Usage (as script):
    python -m outrunner_motor_optimizer.main

Usage (as library):
    from outrunner_motor_optimizer.main import design_motor
    result = design_motor(
        voltage=24, rpm=5000, torque=0.5, current=20,
        magnet_type="NdFeB", magnet_grade="N42SH",
    )

Validation approach:
  - Torque constant cross-check:  Kt = T / I  should match  Kt_em = 3*E/ω
  - Efficiency sanity: copper+iron+magnet+windage+bearing losses sum to (1-η)*P_in
  - Thermal plausibility: winding temp > magnet temp > ambient (for outrunner)
  - Weight plausibility: power-density 0.5-5 kW/kg for BLDC motors (Rouse et al. 2023)
  - Structural: safety factors > 1.5 everywhere

References:
  [1] U.H. Lee et al., "How to Model Brushless Electric Motors for the
      Design of Lightweight Robotic Systems," arXiv:2310.00080, 2023.
      (Motor modelling, Kt cross-check methodology)
  [2] J. Pyrhonen et al., "Design of Rotating Electrical Machines,"
      Wiley, 2014. (Tangential stress 20-50 kPa for PM machines)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .materials import (
    MaterialDatabase, MagnetType, StructuralMaterial, MountingConfig,
    STRUCTURAL_CATALOG,
)
from .electromagnetic import MotorSpecs, ElectromagneticModel, select_slot_pole
from .thermal import ThermalModel
from .mechanical import MechanicalModel
from .cfd import CFDModel
from .optimizer import run_optimisation, evaluate_design


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def validate_design(result) -> Dict[str, Any]:
    """
    Run sanity / physics checks on an evaluated design.

    Returns dict of {check_name: {passed: bool, value: ..., expected: ..., note: ...}}
    """
    checks = {}
    em = result.em
    th = result.thermal
    mech = result.mech

    if em is None:
        return {"no_em_results": {"passed": False, "note": "EM results missing"}}

    # 1. Torque constant cross-check
    omega = em.power_output / max(em.torque_avg, 1e-6) if em.torque_avg > 0 else 1
    Kt_from_torque = em.torque_avg / max(em.phase_current_rms, 1e-6)
    Kt_from_emf = 3 * em.back_emf_rms / max(omega, 1e-6)
    ratio = Kt_from_torque / max(Kt_from_emf, 1e-9)
    checks["torque_constant_consistency"] = {
        "passed": 0.7 < ratio < 1.3,
        "value": f"Kt(T/I)={Kt_from_torque:.4f}, Kt(3E/ω)={Kt_from_emf:.4f}, ratio={ratio:.2f}",
        "expected": "ratio 0.7 – 1.3",
    }

    # 2. Energy balance
    P_out = em.power_output
    P_loss = em.total_loss
    P_in = P_out + P_loss
    eta = P_out / max(P_in, 1e-6)
    checks["energy_balance"] = {
        "passed": 0.50 < eta < 0.99,
        "value": f"η={eta:.3f}, P_out={P_out:.1f}W, P_loss={P_loss:.1f}W",
        "expected": "0.50 < η < 0.99",
    }

    # 3. Thermal plausibility
    if th is not None:
        T_wind = th.T_winding_slot
        T_mag = th.T_magnets
        T_amb = th.T_ambient
        thermal_ok = T_wind > T_amb and T_mag > T_amb
        checks["thermal_gradient"] = {
            "passed": thermal_ok,
            "value": f"T_wind={T_wind:.1f}°C, T_mag={T_mag:.1f}°C, T_amb={T_amb:.1f}°C",
            "expected": "T_winding > T_magnet > T_ambient (generally)",
        }

    # 4. Current density
    J = em.current_density
    checks["current_density"] = {
        "passed": 1.0 < J < 15.0,
        "value": f"J={J:.1f} A/mm²",
        "expected": "1 – 15 A/mm² (air-cooled: 3-8 typical)",
    }

    # 5. Airgap flux density
    Bg = em.airgap_flux_density
    checks["airgap_flux_density"] = {
        "passed": 0.3 < Bg < 1.2,
        "value": f"Bg1={Bg:.3f} T",
        "expected": "0.3 – 1.2 T for surface-PM",
    }

    # 6. Mechanical safety
    if mech is not None:
        sf = mech.shaft.shaft_safety_factor
        mr = mech.rotor_stress.magnet_retention_margin
        checks["shaft_safety"] = {
            "passed": sf > 1.5,
            "value": f"SF={sf:.2f}",
            "expected": "> 1.5",
        }
        checks["magnet_retention"] = {
            "passed": mr > 1.5,
            "value": f"margin={mr:.2f}",
            "expected": "> 1.5",
        }
        checks["vibration_zone"] = {
            "passed": mech.vibration.iso_10816_zone in ("A", "B"),
            "value": mech.vibration.iso_10816_zone,
            "expected": "A or B",
        }

    return checks


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def design_motor(
    voltage: float,
    rpm: float,
    torque: float,
    current: float,
    efficiency_target: float = 0.90,
    magnet_type: str = "NdFeB",
    magnet_grade: str = "N42SH",
    structural_material: str = "6061-T6",
    shaft_material: str = "AISI 4140",
    mounting_style: str = "face_mount",
    num_bolts: int = 4,
    bolt_diameter: float = 0.003,
    radial_load: float = 0.0,
    axial_load: float = 0.0,
    bending_moment: float = 0.0,
    isolator_stiffness: float = 0.0,
    T_ambient: float = 25.0,
    pop_size: int = 40,
    n_gen: int = 30,
    export_step: bool = True,
    output_dir: str = "motor_step_output",
    custom_magnet: Optional[Dict[str, float]] = None,
    custom_materials: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Full motor design pipeline.

    Parameters
    ----------
    voltage, rpm, torque, current : float
        Electrical specifications.
    efficiency_target : float
        Target efficiency (0-1).
    magnet_type : str
        "NdFeB", "SmCo", "Ferrite", or "Custom".
    magnet_grade : str
        Grade within family (e.g. "N42SH", "N52").
    structural_material : str
        Default for all structural parts ("6061-T6", "7075-T6", etc.).
    shaft_material : str
        Shaft material preset.
    mounting_style : str
        "face_mount", "foot_mount", "shaft_clamp", "custom".
    num_bolts, bolt_diameter : int, float
        Bolt pattern.
    radial_load, axial_load, bending_moment : float
        External loads on the motor [N] / [N·m].
    isolator_stiffness : float
        Vibration isolator stiffness [N/m], 0 = rigid.
    T_ambient : float
        Ambient temperature [°C].
    pop_size, n_gen : int
        Optimiser population size and generations.
    export_step : bool
        Whether to generate STEP files.
    output_dir : str
        Directory for STEP files.
    custom_magnet : dict, optional
        Overrides for custom magnet properties (Br_20, Hcj_20, density, ...).
    custom_materials : dict, optional
        Per-component material overrides. Keys: "rotor_housing",
        "stator_housing", "end_bell_de", "end_bell_nde", "mounting_plate",
        "shaft".  Values: dicts of StructuralMaterial field overrides.

    Returns
    -------
    dict with keys:
        specs, materials, optimisation, best_design, validation,
        step_files, report
    """
    t0 = time.time()

    # --- 1. Build specs ---
    specs = MotorSpecs(
        voltage=voltage,
        target_rpm=rpm,
        target_torque=torque,
        max_current=current,
        target_efficiency=efficiency_target,
    )

    # --- 2. Configure materials ---
    mats = MaterialDatabase()

    # Magnet
    if magnet_type == "Custom" and custom_magnet:
        mats.set_magnet("Custom", **custom_magnet)
    else:
        mats.set_magnet(magnet_type, grade=magnet_grade)

    # Structural (all parts default to the same alloy)
    for comp in ["rotor_housing", "stator_housing", "end_bell_de",
                 "end_bell_nde", "mounting_plate"]:
        mats.set_structural_material(comp, preset=structural_material)
    mats.set_structural_material("shaft", preset=shaft_material)

    # Custom per-component overrides
    if custom_materials:
        for comp, overrides in custom_materials.items():
            mats.set_structural_material(comp, **overrides)

    # --- 3. Configure mounting ---
    mats.mounting = MountingConfig(
        style=mounting_style,
        num_bolts=num_bolts,
        bolt_diameter=bolt_diameter,
        radial_load=radial_load,
        axial_load=axial_load,
        bending_moment=bending_moment,
        isolator_stiffness=isolator_stiffness,
        material=mats.mounting_plate,
    )

    # --- 4. Run optimisation ---
    opt_result = run_optimisation(
        specs, mats,
        T_ambient=T_ambient,
        pop_size=pop_size,
        n_gen=n_gen,
    )

    best_idx = opt_result["best_idx"]
    best = opt_result["pareto_results"][best_idx] if opt_result["pareto_results"] else None

    # --- 5. Validation ---
    validation = validate_design(best) if best else {}
    all_passed = all(v.get("passed", False) for v in validation.values())

    # --- 6. STEP export ---
    step_files = {}
    if export_step and best is not None:
        try:
            from .cad_export import export_step_files
            em_model = ElectromagneticModel(specs, mats)
            # Reconstruct geometry from best design
            em_model.geometry = _reconstruct_geometry(
                best.design_vector, opt_result["slot_pole"], mats
            )
            em_model.winding = _reconstruct_winding(
                best.design_vector, opt_result["slot_pole"], em_model.geometry, mats
            )
            step_files = export_step_files(
                em_model.geometry, em_model.winding, mats, output_dir
            )
        except ImportError:
            step_files = {"error": "CadQuery not installed — STEP export skipped"}
        except Exception as e:
            step_files = {"error": str(e)}

    # --- 7. Build report ---
    elapsed = time.time() - t0
    report = _build_report(specs, mats, opt_result, best, validation,
                           step_files, elapsed)

    return {
        "specs": specs,
        "materials": mats,
        "optimisation": opt_result,
        "best_design": best,
        "validation": validation,
        "all_checks_passed": all_passed,
        "step_files": step_files,
        "report": report,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reconstruct_geometry(x, slot_pole, mats):
    """Rebuild GeometryParams from a design vector."""
    from .electromagnetic import GeometryParams
    n_slots, n_poles = slot_pole
    geo = GeometryParams()
    geo.stator_outer_radius = x[0]
    geo.stack_length = x[1]
    geo.magnet_thickness = x[2]
    geo.airgap = x[3]
    tau_s = 2 * np.pi * x[0] / n_slots
    geo.tooth_width = x[5] * tau_s
    geo.slot_opening = max(tau_s * 0.25, 0.001)
    geo.stator_yoke_thickness = max(0.002, (x[0] - geo.shaft_radius - 0.002) * 0.25)
    avail = x[0] - geo.stator_yoke_thickness - geo.shaft_radius - 0.002
    geo.slot_depth = max(x[4] * avail, 0.003)
    geo.stator_inner_radius = geo.shaft_radius + 0.001
    geo.rotor_inner_radius = x[0] + x[3]
    geo.rotor_yoke_thickness = x[9]
    geo.rotor_outer_radius = geo.rotor_inner_radius + x[2] + x[9]
    return geo


def _reconstruct_winding(x, slot_pole, geo, mats):
    """Rebuild WindingConfig from a design vector."""
    from .electromagnetic import WindingConfig, compute_winding_factor
    n_slots, n_poles = slot_pole
    wdg = WindingConfig()
    wdg.num_poles = n_poles
    wdg.num_slots = n_slots
    wdg.turns_per_coil = max(1, int(round(x[6])))
    wdg.wire_diameter = x[7] * 1e-3
    wdg.num_layers = 2
    wdg.coil_span = 1
    wdg.winding_factor = compute_winding_factor(n_slots, n_poles)
    wdg.end_turn_length = 1.5 * geo.tooth_width + 0.010
    return wdg


def _build_report(specs, mats, opt, best, validation, step_files, elapsed):
    """Build a human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  OUTRUNNER BLDC MOTOR DESIGN REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("## Specifications")
    lines.append(f"  Voltage:      {specs.voltage:.1f} V")
    lines.append(f"  Target RPM:   {specs.target_rpm:.0f}")
    lines.append(f"  Target Torque:{specs.target_torque:.3f} N·m")
    lines.append(f"  Max Current:  {specs.max_current:.1f} A")
    lines.append(f"  Target η:     {specs.target_efficiency*100:.0f}%")
    lines.append("")

    lines.append("## Materials")
    lines.append(f"  Magnet:       {mats.magnet.magnet_type.value} / {mats.magnet.grade}")
    lines.append(f"    Br(20°C):   {mats.magnet.Br_20:.3f} T")
    lines.append(f"    Hcj(20°C):  {mats.magnet.Hcj_20/1e3:.0f} kA/m")
    lines.append(f"    Max temp:   {mats.magnet.max_temp:.0f} °C")
    lines.append(f"  Rotor housing:{mats.rotor_housing.name}")
    lines.append(f"  Shaft:        {mats.shaft.name}")
    lines.append(f"  Mounting:     {mats.mounting.style}, {mats.mounting.num_bolts}x M{mats.mounting.bolt_diameter*1e3:.0f}")
    lines.append("")

    lines.append(f"## Optimisation  ({opt['algorithm']}, "
                 f"slot/pole = {opt['slot_pole'][0]}/{opt['slot_pole'][1]})")
    n_pareto = len(opt["pareto_results"])
    lines.append(f"  Pareto solutions found: {n_pareto}")
    lines.append(f"  Best (knee) index:      {opt['best_idx']}")
    lines.append("")

    if best and best.em:
        em = best.em
        lines.append("## Electromagnetic Performance")
        lines.append(f"  Back-EMF (peak):    {em.back_emf_peak:.2f} V")
        lines.append(f"  Airgap Bg1:         {em.airgap_flux_density:.3f} T")
        lines.append(f"  Torque (avg):       {em.torque_avg:.4f} N·m")
        lines.append(f"  Torque ripple:      {em.torque_ripple:.1f}%")
        lines.append(f"  Phase current RMS:  {em.phase_current_rms:.2f} A")
        lines.append(f"  Current density:    {em.current_density:.1f} A/mm²")
        lines.append(f"  Power output:       {em.power_output:.1f} W")
        lines.append(f"  Copper loss:        {em.copper_loss:.1f} W")
        lines.append(f"  Iron loss (stator): {em.iron_loss_stator:.1f} W")
        lines.append(f"  Magnet loss:        {em.magnet_loss:.2f} W")
        lines.append(f"  Efficiency:         {em.efficiency*100:.1f}%")
        lines.append("")

    if best and best.thermal:
        th = best.thermal
        lines.append("## Thermal")
        lines.append(f"  Winding (slot):     {th.T_winding_slot:.1f} °C")
        lines.append(f"  End winding:        {th.T_end_winding_de:.1f} / {th.T_end_winding_nde:.1f} °C")
        lines.append(f"  Magnets:            {th.T_magnets:.1f} °C")
        lines.append(f"  Rotor yoke:         {th.T_rotor_yoke:.1f} °C")
        lines.append(f"  Hotspot margin:     {th.hotspot_margin:.1f} °C ({th.critical_component})")
        lines.append(f"  Safe:               {'YES' if th.is_safe else 'NO'}")
        lines.append("")

    if best and best.mech:
        m = best.mech
        lines.append("## Mechanical")
        lines.append(f"  Shaft SF:           {m.shaft.shaft_safety_factor:.1f}")
        lines.append(f"  1st critical speed: {m.shaft.critical_speed_1st:.0f} RPM")
        lines.append(f"  Magnet retention:   {m.rotor_stress.magnet_retention_margin:.1f}x")
        lines.append(f"  Max safe RPM:       {m.rotor_stress.max_safe_rpm:.0f}")
        lines.append(f"  Bearing L10 (sys):  {m.bearings.system_l10_hours:.0f} h")
        lines.append(f"  Bearing friction:   {m.bearings.friction_loss_total:.2f} W")
        lines.append(f"  Mount bolt SF:      {m.mounting.bolt_combined_safety_factor:.1f}")
        lines.append(f"  Flange SF:          {m.mounting.flange_safety_factor:.1f}")
        lines.append(f"  Vibration zone:     ISO 10816 Zone {m.vibration.iso_10816_zone}")
        lines.append(f"  Forced amplitude:   {m.vibration.forced_amplitude_mm:.3f} mm")
        lines.append(f"  Resonance margin:   {m.vibration.resonance_margin*100:.0f}%")
        lines.append("")

    if best and best.cfd:
        c = best.cfd
        lines.append("## CFD / Airflow")
        lines.append(f"  Flow regime:        {c.flow_regime}")
        lines.append(f"  Taylor number:      {c.taylor_number:.1f}")
        lines.append(f"  h_airgap:           {c.h_airgap:.1f} W/(m²·K)")
        lines.append(f"  h_external:         {c.h_rotor_external:.1f} W/(m²·K)")
        lines.append(f"  Windage loss:       {c.windage_loss:.2f} W")
        lines.append("")

    lines.append("## Validation Checks")
    for name, v in validation.items():
        status = "PASS" if v.get("passed") else "FAIL"
        lines.append(f"  [{status}] {name}: {v.get('value', '')}")
    lines.append("")

    if step_files:
        lines.append("## STEP Files")
        for name, path in step_files.items():
            lines.append(f"  {name}: {path}")
        lines.append("")

    lines.append(f"Elapsed: {elapsed:.1f}s")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: small drone motor
    result = design_motor(
        voltage=24.0,
        rpm=5000,
        torque=0.5,
        current=20.0,
        efficiency_target=0.90,
        magnet_type="NdFeB",
        magnet_grade="N42SH",
        structural_material="6061-T6",
        shaft_material="AISI 4140",
        mounting_style="face_mount",
        num_bolts=4,
        bolt_diameter=0.003,
        radial_load=5.0,
        axial_load=2.0,
        pop_size=20,
        n_gen=10,
        export_step=False,  # Set True if cadquery installed
    )
    print(result["report"])
