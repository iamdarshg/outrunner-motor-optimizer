"""
Outrunner BLDC Motor Design Optimizer
=====================================

A coupled electromagnetic-thermal-mechanical-CFD simulation and optimization
framework for designing outrunner (external-rotor) brushless DC permanent
magnet motors, with STEP file export of all components.

Architecture:
    User Specs (V, RPM, Torque, I, η)
        → Material selection (NdFeB / SmCo / Ferrite / Custom magnets;
          6061-T6 default structural; user-injectable custom materials)
        → Mounting definition (bolt pattern, loads, vibration isolation)
        → Electromagnetic Design (analytical subdomain + Carter coeff)
        → Thermal Analysis (LPTN with temp-dependent materials)
        → Mechanical Analysis (stress, bearing life, shaft, mounting shear,
          vibration/modal)
        → CFD/Cooling (Nusselt correlations for rotating bodies)
        → Multi-objective Optimization (NSGA-II via pymoo / DE fallback)
        → STEP Export (CadQuery parametric 3D models)

References & Algorithms Used:
    Electromagnetic:
      - Subdomain analytical method for surface-PM machines
        Ref: Z.Q. Zhu et al., IEEE Trans. Magnetics, 1993 & 2003.
      - Winding factor: El-Refaie (2010), Pyrhonen et al. (2014).
      - Iron loss: Bertotti (1988) loss separation model.

    Thermal:
      - Lumped Parameter Thermal Network (LPTN)
        Ref: Boglietti et al., IEEE Trans. Ind. Electron., 2009.
        Ref: Staton & Cavagnino, IEEE Trans. Ind. Electron., 2008.

    Mechanical:
      - Rotor stress: Timoshenko thick-walled rotating cylinder (1970).
      - Shaft critical speed: Dunkerley / Rayleigh method.
      - Bearing life: ISO 281:2007, L10.
      - Mounting: VDI 2230 (simplified bolt analysis), Roark plate bending.
      - Vibration: lumped-mass 3-DOF modal, ISO 10816 classification.
        Ref: Rao, "Mechanical Vibrations," 6th ed., 2017.
        Ref: Genta, "Dynamics of Rotating Systems," 2005.

    CFD/Cooling:
      - Taylor-Couette: Becker & Kaye (1962).
      - Rotating cylinder: Kreith (1968).
      - Rotating disc: Owen & Rogers (1989).

    Optimization:
      - NSGA-II: Deb et al., IEEE Trans. Evol. Comp., 2002.
      - pymoo: Blank & Deb, IEEE Access, 2020.
      - Fallback: Differential Evolution (Storn & Price, 1997).

    CAD Export:
      - CadQuery (OCCT kernel) → ISO 10303 STEP files.

License: MIT
"""

__version__ = "3.0.0"

from .materials import (
    MaterialDatabase, MagnetProperties, MagnetType, StructuralMaterial,
    MountingConfig, BearingProperties, CopperProperties,
    SteelLaminationProperties, STRUCTURAL_CATALOG, MAGNET_CATALOG,
)
from .electromagnetic import MotorSpecs, ElectromagneticModel, GeometryParams, WindingConfig
from .thermal import ThermalModel
from .mechanical import MechanicalModel
from .cfd import CFDModel
from .optimizer import run_optimisation, evaluate_design
from .main import design_motor, validate_design

# FEM modules (optional — require scikit-fem)
try:
    from .fem_electromagnetic import FEMElectromagneticModel, FEMEMResults
    from .fem_thermal import FEMThermalModel, FEMThermalResults
    from .fem_mechanical import FEMMechanicalModel, FEMMechanicalResults
    from .fem_optimizer import run_optimisation_fem, evaluate_design_fem
    HAS_FEM = True
except ImportError:
    HAS_FEM = False

__all__ = [
    "design_motor",
    "validate_design",
    "MaterialDatabase",
    "MagnetProperties",
    "MagnetType",
    "StructuralMaterial",
    "MountingConfig",
    "MotorSpecs",
    "ElectromagneticModel",
    "ThermalModel",
    "MechanicalModel",
    "CFDModel",
    "run_optimisation",
    "evaluate_design",
    "STRUCTURAL_CATALOG",
    "MAGNET_CATALOG",
    # FEM exports
    "FEMElectromagneticModel",
    "FEMThermalModel",
    "FEMMechanicalModel",
    "FEMEMResults",
    "FEMThermalResults",
    "FEMMechanicalResults",
    "run_optimisation_fem",
    "evaluate_design_fem",
    "HAS_FEM",
]