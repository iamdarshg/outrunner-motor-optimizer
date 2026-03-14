# Outrunner BLDC Motor Design Optimizer

A coupled electromagnetic-thermal-mechanical-CFD simulation and multi-objective
optimization framework for designing outrunner (external-rotor) brushless DC
permanent magnet motors, with parametric STEP file export.

Supports both fast analytical solvers and full 2-D finite element method (FEM)
solvers via scikit-fem, with automatic fallback.

## Features

### Analytical Solvers (default)
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

### FEM Solvers (use_fem=True)
- **FEM Electromagnetic** — 2-D magnetostatic A_z formulation, element-wise
  permeability assignment (stator iron µ_r=800, PM remanence source term),
  Maxwell stress tensor torque extraction on a single airgap contour
- **FEM Thermal** — Steady-state heat conduction (-∇·(k∇T) = q) with
  Dirichlet boundary conditions, effective air conductivity, volumetric
  heat sources from EM losses
- **FEM Mechanical** — Plane-stress linear elasticity with centrifugal body
  forces (rotor-only rotation), thermal pre-stress coupling, eigenvalue
  modal analysis for natural frequencies

### Common
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

### Analytical mode (fast, ~30s)

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
```

### FEM mode (higher fidelity, ~10min)

```python
result = design_motor(
    voltage=24.0,
    rpm=5000,
    torque=0.5,
    current=20.0,
    use_fem=True,               # Enable 2-D FEM solvers
    fem_mesh_density=60,        # Circumferential mesh points
    pop_size=20,                # Smaller pop for FEM (slower)
    n_gen=15,                   # Fewer generations
    export_step=False,
)
```

### Using FEM solvers directly

```python
from outrunner_motor_optimizer import (
    FEMElectromagneticModel, FEMThermalModel, FEMMechanicalModel,
    MotorSpecs, GeometryParams, WindingConfig, MaterialDatabase,
)

specs = MotorSpecs(voltage=24, target_rpm=5000, target_torque=0.5, max_current=20)
geo = GeometryParams()   # use defaults or configure
wdg = WindingConfig()
mats = MaterialDatabase()

# EM solve
fem_em = FEMElectromagneticModel(specs, geo, wdg, mats)
em_results = fem_em.solve_magnetostatic()

# Thermal solve (needs analytical EM results for loss inputs)
from outrunner_motor_optimizer.electromagnetic import ElectromagneticModel
em_model = ElectromagneticModel(specs, mats)
em_model.geometry = geo
em_model.winding = wdg
em_res = em_model.compute_performance()

fem_th = FEMThermalModel(geo, wdg, mats)
th_results = fem_th.solve_steady_state(em_res, rpm=5000)

# Mechanical solve
fem_mech = FEMMechanicalModel(geo, wdg, mats)
mech_results = fem_mech.solve_static(rpm=5000)
fem_mech.solve_modal(n_modes=6)
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
pip install numpy pymoo scipy

# For FEM solvers (optional but recommended):
pip install scikit-fem[all] meshio

# For STEP export (optional):
# conda install -c conda-forge -c cadquery cadquery
```

## Architecture

```
User Specs (V, RPM, T, I, η)
  → Material selection (magnet + structural + mounting)
  → Slot-pole selection (scored candidate list)
  → NSGA-II multi-objective optimization loop:
      ├── Electromagnetic design
      │     ├── Analytical: subdomain method
      │     └── FEM: 2-D magnetostatic A_z (scikit-fem)
      ├── Coupled EM-Thermal iteration
      │     ├── Analytical: LPTN
      │     └── FEM: steady-state heat conduction
      ├── CFD/Cooling (Nusselt correlations — always analytical)
      └── Mechanical analysis
            ├── Analytical: stress, shaft, bearings, mounting, vibration
            └── FEM: plane-stress elasticity + eigenvalue modal
  → Pareto front → knee-point selection
  → Validation checks (Kt cross-check, energy balance, thermal, mechanical)
  → STEP export (CadQuery parametric models)
```

## Modules

| Module | Description |
|--------|-------------|
| `materials.py` | Material database with temperature-dependent properties |
| `electromagnetic.py` | Analytical EM design (subdomain method) |
| `thermal.py` | LPTN thermal model |
| `mechanical.py` | Analytical stress, shaft, bearing, mounting, vibration |
| `cfd.py` | Correlation-based convection and windage |
| `optimizer.py` | NSGA-II / DE multi-objective optimizer (analytical) |
| `cad_export.py` | CadQuery STEP file generation |
| `main.py` | High-level pipeline API |
| `fem_electromagnetic.py` | 2-D FEM magnetostatic solver |
| `fem_thermal.py` | 2-D FEM steady-state thermal solver |
| `fem_mechanical.py` | 2-D FEM elasticity + modal solver |
| `fem_optimizer.py` | FEM-augmented optimizer with analytical fallback |

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
- O.C. Zienkiewicz, R.L. Taylor, 2013 (FEM for solid mechanics)
- ISO 281:2007 (bearing life)
- VDI 2230 (bolted joint analysis)
- S.S. Rao, Mechanical Vibrations, 6th ed., 2017
- K. Deb et al., IEEE Trans. Evol. Comp., 2002 (NSGA-II)
- T. Gustafsson, G. McBain, JOSS, 2020 (scikit-fem)
- K.M. Becker, J. Kaye, ASME J. Heat Transfer, 1962 (Taylor-Couette)
- F. Kreith, Advances in Heat Transfer, 1968 (rotating cylinder)
- J.M. Owen, R.H. Rogers, 1989 (rotating disc)

## License

MIT