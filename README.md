# Outrunner BLDC Motor Design Optimizer

A coupled electromagnetic-thermal-mechanical-CFD simulation and multi-objective
optimization framework for designing outrunner (external-rotor) brushless DC
permanent magnet motors, with parametric STEP file export.

## Features

- **Electromagnetic model** — Analytical subdomain method (Zhu & Howe 1993), Bertotti
  iron loss, Carter coefficient, winding factor computation, cogging torque estimation
- **Thermal model** — 10-node Lumped Parameter Thermal Network (LPTN) with
  Taylor-Couette airgap convection (Becker & Kaye 1962), temperature-dependent
  material properties
- **Mechanical model** — Rotor hoop stress (Timoshenko), shaft critical speed
  (Dunkerley), bearing L10 life (ISO 281), VDI 2230 bolt analysis, Roark flange
  bending, 3-DOF vibration/modal analysis with ISO 10816 classification
- **CFD model** — Correlation-based convective heat transfer (Kreith 1968,
  Owen & Rogers 1989), windage loss, self-pumping airflow estimate
- **Multi-objective optimization** — NSGA-II via pymoo (Deb et al. 2002) with
  DE fallback; optimizes efficiency, mass, and torque density simultaneously
- **CAD export** — CadQuery-based parametric STEP file generation for stator,
  rotor, magnets, shaft, mounting flange, and full assembly

## Material System

- **Magnets**: NdFeB (7 grades: N35 through N52), SmCo 2:17, Ferrite, or fully
  custom user-defined magnets with temperature-dependent Br/Hcj
- **Structural**: 6061-T6 aluminium (default for all structural components),
  7075-T6, AISI 4140 steel, Ti-6Al-4V titanium, or fully custom
- **Mounting**: Face mount, foot mount, shaft clamp, or custom bolt patterns
  with full shear/tensile/vibration analysis

## Quick Start

```python
from outrunner_motor_optimizer import design_motor

result = design_motor(
    voltage=24.0,
    rpm=5000,
    torque=0.5,
    current=20.0,
    efficiency_target=0.90,
    magnet_type="NdFeB",        # or "SmCo", "Ferrite", "Custom"
    magnet_grade="N42SH",
    structural_material="6061-T6",
    shaft_material="AISI 4140",
    mounting_style="face_mount",
    num_bolts=4,
    bolt_diameter=0.003,        # M3
    radial_load=5.0,            # N
    axial_load=2.0,             # N
    pop_size=40,                # NSGA-II population
    n_gen=30,                   # generations
    export_step=True,           # requires cadquery
    output_dir="motor_step_output",
)

print(result["report"])
print("All validation checks passed:", result["all_checks_passed"])
```

### Custom magnet example

```python
result = design_motor(
    voltage=24.0, rpm=5000, torque=0.5, current=20.0,
    magnet_type="Custom",
    custom_magnet={
        "Br_20": 1.1,
        "Hcj_20": 1200e3,
        "density": 7600,
        "max_temp": 200,
        "alpha_Br": -0.001,
    },
)
```

### Custom structural material per component

```python
result = design_motor(
    voltage=24.0, rpm=5000, torque=0.5, current=20.0,
    custom_materials={
        "rotor_housing": {"preset": "7075-T6"},
        "shaft": {"preset": "Ti-6Al-4V"},
    },
)
```

## CLI Usage

```bash
python -m outrunner_motor_optimizer.main
```

## Installation

```bash
pip install numpy pymoo
# For STEP export (optional):
# conda install -c conda-forge -c cadquery cadquery
```

## Architecture

```
User Specs (V, RPM, T, I, η)
  → Material selection (magnet + structural + mounting)
  → Slot-pole selection (scored candidate list)
  → NSGA-II multi-objective optimization loop:
      ├── Electromagnetic design (analytical subdomain)
      ├── Coupled EM-Thermal iteration (LPTN)
      ├── CFD/Cooling (Nusselt correlations)
      └── Mechanical analysis (stress, shaft, bearings, mounting, vibration)
  → Pareto front → knee-point selection
  → Validation checks (Kt cross-check, energy balance, thermal, mechanical)
  → STEP export (CadQuery parametric models)
```

## Validation Checks

The pipeline automatically runs these physics sanity checks:

| Check | Criterion |
|-------|-----------|
| Torque constant consistency | Kt(T/I) vs Kt(3E/ω) ratio within 0.7–1.3 |
| Energy balance | 0.50 < η < 0.99 |
| Thermal gradient | T_winding > T_magnet > T_ambient |
| Current density | 1–15 A/mm² |
| Airgap flux density | 0.3–1.2 T for surface-PM |
| Shaft safety factor | > 1.5 |
| Magnet retention | > 1.5× |
| Vibration | ISO 10816 Zone A or B |

## References

- Z.Q. Zhu, D. Howe, IEEE Trans. Magnetics, 1993 (subdomain method)
- G. Bertotti, IEEE Trans. Magnetics, 1988 (iron loss separation)
- A. Boglietti et al., IEEE Trans. Ind. Electron., 2009 (LPTN)
- D. Staton, A. Cavagnino, IEEE Trans. Ind. Electron., 2008 (convection)
- S.P. Timoshenko, J.N. Goodier, 1970 (rotating cylinder stress)
- ISO 281:2007 (bearing life)
- VDI 2230 (bolted joint analysis)
- S.S. Rao, Mechanical Vibrations, 6th ed., 2017
- K. Deb et al., IEEE Trans. Evol. Comp., 2002 (NSGA-II)
- K.M. Becker, J. Kaye, ASME J. Heat Transfer, 1962 (Taylor-Couette)
- F. Kreith, Advances in Heat Transfer, 1968 (rotating cylinder)
- J.M. Owen, R.H. Rogers, 1989 (rotating disc)
