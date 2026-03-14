"""
Temperature-dependent material property database for electric motor design.

Supports user-selectable magnet types (NdFeB, SmCo, Ferrite, custom) and
fully custom material injection for all components.  Default structural
material is 6061-T6 aluminum throughout.

Material models based on:
- Bertotti's iron loss separation model (G. Bertotti, "General Properties
  of Power Losses in Soft Ferromagnetic Materials," IEEE Transactions on
  Magnetics, vol. 24, no. 1, 1988.)
- NdFeB demagnetization curves from Arnold Magnetic Technologies datasheets
  and K&J Magnetics BH curve data.
- SmCo data from Electron Energy Corporation datasheets (EEC 2-17 series).
- Ferrite data from TDK FB Series datasheets.
- Temperature coefficients from IEC 60404 and manufacturer datasheets.
- Copper resistivity model: IEC 60228 standard.
- 6061-T6 aluminium properties: ASM Aerospace Specification Metals Inc.
- AISI 4140 shaft steel: MatWeb / ASM data.

References:
  [1] G. Bertotti, "Hysteresis in Magnetism," Academic Press, 1998.
  [2] J. Pyrhonen, T. Jokinen, V. Hrabovcova, "Design of Rotating
      Electrical Machines," Wiley, 2nd ed., 2014. (Ch. 3)
  [3] Arnold Magnetic Technologies, "N42SH Datasheet," 2023.
  [4] S. Ruoho et al., "Temperature Dependence of Resistivity of Sintered
      Rare-Earth Permanent Magnets," IEEE Trans. Magn., 2010.
  [5] Electron Energy Corp., "2:17 SmCo Magnets Datasheet," 2022.
  [6] TDK Corporation, "Ferrite Magnets FB Series Datasheet," 2021.
  [7] ASM Aerospace Specification Metals, "Aluminum 6061-T6," 2023.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from enum import Enum
import numpy as np
import copy


# ---------------------------------------------------------------------------
# Generic structural material
# ---------------------------------------------------------------------------
@dataclass
class StructuralMaterial:
    """
    Generic isotropic structural material with mechanical + thermal props.
    Default: 6061-T6 aluminium (ASM Aerospace Specification Metals).
    """
    name: str = "6061-T6 Aluminium"
    density: float = 2710.0              # [kg/m³]
    thermal_conductivity: float = 167.0  # [W/(m·K)]
    specific_heat: float = 896.0         # [J/(kg·K)]
    resistivity: float = 3.99e-8         # [Ohm·m]
    yield_strength: float = 276e6        # [Pa]
    ultimate_strength: float = 310e6     # [Pa]
    youngs_modulus: float = 68.9e9       # [Pa]
    poissons_ratio: float = 0.33
    thermal_expansion: float = 23.6e-6   # [1/°C]
    fatigue_strength: float = 96.5e6     # [Pa] at 5e8 cycles (Aluminum)
    shear_modulus: float = 26.0e9        # [Pa]
    shear_strength: float = 207e6        # [Pa]
    max_service_temp: float = 170.0      # [°C]

    def yield_at_temp(self, T: float) -> float:
        """Yield strength derated with temperature (empirical for 6061-T6)."""
        if T <= 100.0:
            return self.yield_strength
        # Linear de-rate: ~20 % drop from 100 °C to 200 °C
        factor = max(1.0 - 0.002 * (T - 100.0), 0.3)
        return self.yield_strength * factor


# Pre-built structural material catalogue
STRUCTURAL_CATALOG: Dict[str, StructuralMaterial] = {
    "6061-T6": StructuralMaterial(),  # default
    "7075-T6": StructuralMaterial(
        name="7075-T6 Aluminium", density=2810.0,
        thermal_conductivity=130.0, specific_heat=960.0,
        resistivity=5.15e-8, yield_strength=503e6,
        ultimate_strength=572e6, youngs_modulus=71.7e9,
        poissons_ratio=0.33, thermal_expansion=23.4e-6,
        fatigue_strength=159e6, shear_modulus=26.9e9,
        shear_strength=331e6, max_service_temp=130.0,
    ),
    "AISI 4140": StructuralMaterial(
        name="AISI 4140 Steel", density=7850.0,
        thermal_conductivity=42.0, specific_heat=473.0,
        resistivity=2.2e-7, yield_strength=655e6,
        ultimate_strength=1020e6, youngs_modulus=205e9,
        poissons_ratio=0.29, thermal_expansion=12.3e-6,
        fatigue_strength=420e6, shear_modulus=80.0e9,
        shear_strength=655e6, max_service_temp=400.0,
    ),
    "Ti-6Al-4V": StructuralMaterial(
        name="Ti-6Al-4V Titanium", density=4430.0,
        thermal_conductivity=6.7, specific_heat=526.0,
        resistivity=1.78e-6, yield_strength=880e6,
        ultimate_strength=950e6, youngs_modulus=113.8e9,
        poissons_ratio=0.342, thermal_expansion=8.6e-6,
        fatigue_strength=510e6, shear_modulus=44.0e9,
        shear_strength=550e6, max_service_temp=300.0,
    ),
}


# ---------------------------------------------------------------------------
# Copper winding
# ---------------------------------------------------------------------------
@dataclass
class CopperProperties:
    """
    Copper winding material with temperature-dependent resistivity.
    Model: rho(T) = rho_20 * (1 + alpha*(T - 20))
    Per IEC 60228: alpha_Cu ~ 0.00393 /°C for annealed copper.
    """
    rho_20: float = 1.724e-8
    alpha: float = 0.00393
    density: float = 8960.0
    thermal_conductivity: float = 401.0
    specific_heat: float = 385.0
    max_temp: float = 180.0        # Class H insulation limit
    fill_factor: float = 0.45

    def resistivity(self, T: float) -> float:
        return self.rho_20 * (1.0 + self.alpha * (T - 20.0))

    def conductivity(self, T: float) -> float:
        return 1.0 / self.resistivity(T)

    def thermal_cond_at_temp(self, T: float) -> float:
        return self.thermal_conductivity * (1.0 - 0.0003 * (T - 20.0))


# ---------------------------------------------------------------------------
# Steel lamination
# ---------------------------------------------------------------------------
@dataclass
class SteelLaminationProperties:
    """
    Silicon steel lamination (e.g. M250-35A / M19-29Ga).
    Iron loss via Bertotti's separation model (1988).
    """
    name: str = "M250-35A"
    density: float = 7650.0
    saturation_flux: float = 1.8
    relative_permeability: float = 5000.0
    lamination_thickness: float = 0.35e-3
    stacking_factor: float = 0.97
    k_h: float = 0.02
    alpha_steinmetz: float = 1.8
    k_e: float = 5.0e-5
    k_ex: float = 1.0e-4
    thermal_conductivity_radial: float = 28.0
    thermal_conductivity_axial: float = 1.5
    specific_heat: float = 460.0
    max_temp: float = 200.0

    def bh_curve(self, H: np.ndarray) -> np.ndarray:
        mu_0 = 4 * np.pi * 1e-7
        a = mu_0 * self.relative_permeability / (np.pi / 2 * self.saturation_flux)
        return (2 * self.saturation_flux / np.pi) * np.arctan(a * H)

    def iron_loss_density(self, f: float, B_peak: float, T: float = 20.0) -> float:
        B_peak = abs(B_peak)
        f = abs(f)
        if B_peak < 1e-12 or f < 1e-6:
            return 0.0
        tf_h = 1.0 - 0.0003 * (T - 20.0)
        tf_e = 1.0 + 0.0001 * (T - 20.0)
        P_h = self.k_h * f * B_peak ** self.alpha_steinmetz * tf_h
        P_e = self.k_e * (f * B_peak) ** 2 * tf_e
        P_ex = self.k_ex * (f * B_peak) ** 1.5
        result = P_h + P_e + P_ex
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return max(result, 0.0)


# ---------------------------------------------------------------------------
# Magnet type enumeration & generic magnet dataclass
# ---------------------------------------------------------------------------
class MagnetType(Enum):
    NdFeB = "NdFeB"
    SmCo = "SmCo"
    Ferrite = "Ferrite"
    Custom = "Custom"


@dataclass
class MagnetProperties:
    """
    Generic permanent-magnet material.

    Works for NdFeB, SmCo, Ferrite, or any user-supplied magnet.
    Temperature-dependent Br and Hcj models:
        Br(T) = Br_20 * (1 + alpha_Br * (T - 20))
        Hcj(T) = Hcj_20 * (1 + alpha_Hcj * (T - 20))

    References (by magnet family):
      NdFeB  — Arnold Magnetic Technologies datasheets; K&J Magnetics BH data.
      SmCo   — Electron Energy Corp. EEC 2-17 series.
      Ferrite — TDK FB-series datasheets.
    """
    magnet_type: MagnetType = MagnetType.NdFeB
    grade: str = "N42SH"

    # Magnetic properties at 20 °C
    Br_20: float = 1.30            # Remanence [T]
    Hcj_20: float = 1590e3         # Intrinsic coercivity [A/m]
    Hcb_20: float = 995e3          # Normal coercivity [A/m]
    BHmax_20: float = 334e3        # Max energy product [J/m³]
    mu_rec: float = 1.05           # Recoil permeability

    # Temperature coefficients
    alpha_Br: float = -0.0011      # [1/°C]
    alpha_Hcj: float = -0.006      # [1/°C]
    alpha_Hcb: float = -0.005      # [1/°C]

    # Physical / thermal
    density: float = 7500.0
    resistivity: float = 1.6e-6
    thermal_conductivity: float = 8.0
    specific_heat: float = 440.0
    max_temp: float = 150.0
    curie_temp: float = 340.0

    # Geometry (set during design)
    thickness: float = 0.003
    arc_fraction: float = 0.85
    length: float = 0.030

    # --- derived helpers ---
    def Br(self, T: float) -> float:
        return self.Br_20 * (1.0 + self.alpha_Br * (T - 20.0))

    def Hcj(self, T: float) -> float:
        return self.Hcj_20 * (1.0 + self.alpha_Hcj * (T - 20.0))

    def Hcb(self, T: float) -> float:
        return self.Hcb_20 * (1.0 + self.alpha_Hcb * (T - 20.0))

    def check_demagnetization(self, H_demag: float, T: float) -> bool:
        """True if SAFE (operating point above the knee)."""
        return abs(H_demag) < 0.8 * abs(self.Hcj(T))

    def eddy_current_loss_density(self, f: float, B_peak: float, T: float = 20.0) -> float:
        rho_T = self.resistivity * (1.0 + 0.001 * (T - 20.0))
        if rho_T <= 0 or abs(f) < 1e-6:
            return 0.0
        result = (np.pi * self.thickness * f * B_peak) ** 2 / (6.0 * rho_T)
        return 0.0 if (np.isnan(result) or np.isinf(result)) else max(result, 0.0)


# Pre-built magnet catalogue --------------------------------------------------
def _ndfeb_grade(grade: str = "N42SH") -> MagnetProperties:
    """Return NdFeB magnet properties for common grades."""
    base = MagnetProperties(magnet_type=MagnetType.NdFeB)
    grades = {
        "N35":   dict(Br_20=1.17, Hcj_20=955e3,  BHmax_20=263e3, max_temp=80,  grade="N35"),
        "N42":   dict(Br_20=1.30, Hcj_20=955e3,  BHmax_20=334e3, max_temp=80,  grade="N42"),
        "N42H":  dict(Br_20=1.30, Hcj_20=1353e3, BHmax_20=334e3, max_temp=120, grade="N42H"),
        "N42SH": dict(Br_20=1.30, Hcj_20=1592e3, BHmax_20=334e3, max_temp=150, grade="N42SH"),
        "N40UH": dict(Br_20=1.26, Hcj_20=1990e3, BHmax_20=318e3, max_temp=180, grade="N40UH"),
        "N38EH": dict(Br_20=1.22, Hcj_20=2388e3, BHmax_20=294e3, max_temp=200, grade="N38EH"),
        "N52":   dict(Br_20=1.44, Hcj_20=876e3,  BHmax_20=422e3, max_temp=60,  grade="N52"),
    }
    if grade in grades:
        for k, v in grades[grade].items():
            setattr(base, k, v)
    else:
        # fall back to N42SH
        for k, v in grades["N42SH"].items():
            setattr(base, k, v)
    return base


def _smco_grade(grade: str = "SmCo 2:17") -> MagnetProperties:
    """Samarium Cobalt (Sm2Co17) — high-temperature, lower Br than NdFeB."""
    return MagnetProperties(
        magnet_type=MagnetType.SmCo, grade=grade,
        Br_20=1.08, Hcj_20=2000e3, Hcb_20=820e3, BHmax_20=230e3,
        mu_rec=1.05,
        alpha_Br=-0.0003, alpha_Hcj=-0.002, alpha_Hcb=-0.002,
        density=8400.0, resistivity=8.6e-7,
        thermal_conductivity=12.0, specific_heat=370.0,
        max_temp=300.0, curie_temp=800.0,
    )


def _ferrite_grade(grade: str = "Y30BH") -> MagnetProperties:
    """Hard ferrite (ceramic) — cheap, low Br, good temperature stability."""
    return MagnetProperties(
        magnet_type=MagnetType.Ferrite, grade=grade,
        Br_20=0.39, Hcj_20=240e3, Hcb_20=230e3, BHmax_20=30e3,
        mu_rec=1.1,
        alpha_Br=-0.002, alpha_Hcj=0.004,  # Note: positive for ferrite
        alpha_Hcb=0.003,
        density=4900.0, resistivity=1e4,  # effectively insulator
        thermal_conductivity=4.0, specific_heat=800.0,
        max_temp=250.0, curie_temp=450.0,
    )


MAGNET_CATALOG: Dict[str, Callable[..., MagnetProperties]] = {
    "NdFeB": _ndfeb_grade,
    "SmCo":  _smco_grade,
    "Ferrite": _ferrite_grade,
}


# ---------------------------------------------------------------------------
# Bearing
# ---------------------------------------------------------------------------
@dataclass
class BearingProperties:
    """
    Deep-groove ball bearing — ISO 281:2007 L10 life model.
    """
    type: str = "deep_groove_ball"
    bore_diameter: float = 0.010
    outer_diameter: float = 0.026
    width: float = 0.008
    dynamic_load_rating: float = 4620.0   # C [N]
    static_load_rating: float = 1960.0    # C0 [N]
    max_speed: float = 40000.0            # [RPM]
    life_exponent: float = 3.0
    friction_coefficient: float = 0.0015
    grease_temp_limit: float = 120.0
    # Stiffness for vibration analysis (radial, N/m)
    radial_stiffness: float = 5.0e7

    def l10_life_hours(self, load_N: float, rpm: float) -> float:
        if load_N <= 0 or rpm <= 0:
            return float("inf")
        L10_rev = (self.dynamic_load_rating / load_N) ** self.life_exponent
        return L10_rev * 1e6 / (60.0 * rpm)

    def friction_torque(self, load_N: float) -> float:
        return self.friction_coefficient * load_N * self.bore_diameter / 2.0

    def friction_loss(self, load_N: float, rpm: float) -> float:
        return self.friction_torque(load_N) * rpm * 2 * np.pi / 60.0


# ---------------------------------------------------------------------------
# Mounting geometry
# ---------------------------------------------------------------------------
@dataclass
class MountingConfig:
    """
    Motor mounting / interface geometry.

    Supports four common arrangements:
      - face_mount   : Bolted flange on drive-end face (e.g. drone arm clamp)
      - foot_mount   : Feet under stator housing (IEC B3-style)
      - shaft_clamp  : Shaft clamped into external structure
      - custom       : User-defined bolt circle, thickness, etc.

    All structural parts default to 6061-T6.
    """
    style: str = "face_mount"

    # Bolt circle
    num_bolts: int = 4
    bolt_circle_diameter: float = 0.040   # [m]
    bolt_diameter: float = 0.003          # M3  [m]
    bolt_grade: float = 8.8              # ISO grade → tensile ~ 800 MPa

    # Flange / foot plate
    flange_thickness: float = 0.003       # [m]
    flange_outer_diameter: float = 0.050  # [m]

    # Material for mounting hardware (default 6061-T6)
    material: StructuralMaterial = field(default_factory=StructuralMaterial)

    # Loaded directions -------------------------------------------------------
    axial_load: float = 0.0               # [N] along shaft axis
    radial_load: float = 0.0              # [N] perpendicular to shaft
    bending_moment: float = 0.0           # [N·m]

    # For vibration isolation
    isolator_stiffness: float = 0.0       # [N/m], 0 = rigid mount
    isolator_damping: float = 0.0         # [N·s/m]


# ---------------------------------------------------------------------------
# Central material database
# ---------------------------------------------------------------------------
@dataclass
class MaterialDatabase:
    """
    Central material registry for the motor optimiser.

    Every structural component defaults to 6061-T6.  Users may override any
    field, inject fully custom materials, or select from the built-in
    catalogue.  Magnet type is selectable (NdFeB / SmCo / Ferrite / Custom).
    """
    copper: CopperProperties = field(default_factory=CopperProperties)
    steel: SteelLaminationProperties = field(default_factory=SteelLaminationProperties)
    magnet: MagnetProperties = field(default_factory=lambda: _ndfeb_grade("N42SH"))

    # Structural — all default to 6061-T6
    rotor_housing: StructuralMaterial = field(default_factory=StructuralMaterial)
    stator_housing: StructuralMaterial = field(default_factory=StructuralMaterial)
    end_bell_de: StructuralMaterial = field(default_factory=StructuralMaterial)
    end_bell_nde: StructuralMaterial = field(default_factory=StructuralMaterial)
    mounting_plate: StructuralMaterial = field(default_factory=StructuralMaterial)

    # Shaft — defaults to AISI 4140 steel (overridable)
    shaft: StructuralMaterial = field(
        default_factory=lambda: copy.deepcopy(STRUCTURAL_CATALOG["AISI 4140"])
    )

    # Bearings
    bearing_drive: BearingProperties = field(default_factory=BearingProperties)
    bearing_non_drive: BearingProperties = field(default_factory=BearingProperties)

    # Mounting
    mounting: MountingConfig = field(default_factory=MountingConfig)

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------
    def set_magnet(self, magnet_type: str, grade: str = "", **overrides) -> None:
        """
        Select a magnet family and (optional) grade from the catalogue,
        or pass magnet_type="Custom" with keyword overrides.

        Examples
        --------
        >>> db = MaterialDatabase()
        >>> db.set_magnet("NdFeB", grade="N52")
        >>> db.set_magnet("SmCo")
        >>> db.set_magnet("Ferrite", grade="Y30BH")
        >>> db.set_magnet("Custom", Br_20=1.1, Hcj_20=1200e3, density=7600,
        ...               max_temp=200, alpha_Br=-0.001)
        """
        if magnet_type in MAGNET_CATALOG:
            self.magnet = MAGNET_CATALOG[magnet_type](grade or magnet_type)
        elif magnet_type == "Custom":
            self.magnet = MagnetProperties(magnet_type=MagnetType.Custom,
                                           grade=grade or "Custom")
        else:
            raise ValueError(
                f"Unknown magnet type '{magnet_type}'. "
                f"Choose from {list(MAGNET_CATALOG.keys())} or 'Custom'."
            )
        # Apply any user overrides on top
        for k, v in overrides.items():
            if hasattr(self.magnet, k):
                setattr(self.magnet, k, v)

    def set_structural_material(self, component: str,
                                material: Optional[StructuralMaterial] = None,
                                preset: Optional[str] = None,
                                **overrides) -> None:
        """
        Set or override the structural material for a named component.

        Parameters
        ----------
        component : str
            One of: rotor_housing, stator_housing, end_bell_de, end_bell_nde,
            mounting_plate, shaft
        material : StructuralMaterial, optional
            Fully custom material instance.
        preset : str, optional
            Name from STRUCTURAL_CATALOG ("6061-T6", "7075-T6", "AISI 4140",
            "Ti-6Al-4V").
        **overrides
            Patch individual fields on top of preset / current material.

        Examples
        --------
        >>> db.set_structural_material("shaft", preset="AISI 4140")
        >>> db.set_structural_material("rotor_housing", preset="7075-T6")
        >>> db.set_structural_material("mounting_plate",
        ...     material=StructuralMaterial(name="Custom CF plate",
        ...         density=1600, yield_strength=600e6, youngs_modulus=70e9,
        ...         thermal_conductivity=5.0, specific_heat=900, ...))
        """
        valid = {"rotor_housing", "stator_housing", "end_bell_de",
                 "end_bell_nde", "mounting_plate", "shaft"}
        if component not in valid:
            raise ValueError(f"component must be one of {valid}")

        if material is not None:
            mat = copy.deepcopy(material)
        elif preset is not None:
            if preset not in STRUCTURAL_CATALOG:
                raise ValueError(
                    f"Unknown preset '{preset}'. "
                    f"Available: {list(STRUCTURAL_CATALOG.keys())}"
                )
            mat = copy.deepcopy(STRUCTURAL_CATALOG[preset])
        else:
            mat = copy.deepcopy(getattr(self, component))

        for k, v in overrides.items():
            if hasattr(mat, k):
                setattr(mat, k, v)

        setattr(self, component, mat)

    def auto_select_magnet_grade(self, max_operating_temp: float) -> None:
        """Pick the cheapest NdFeB grade that survives the temperature."""
        ordered = [
            (80,  "N42"),
            (120, "N42H"),
            (150, "N42SH"),
            (180, "N40UH"),
            (200, "N38EH"),
        ]
        for t_lim, grade in ordered:
            if max_operating_temp <= t_lim:
                self.set_magnet("NdFeB", grade=grade)
                return
        self.set_magnet("NdFeB", grade="N38EH")

    def select_bearing(self, shaft_diameter: float,
                       target_speed: float) -> None:
        """Auto-select bearing from simplified catalogue."""
        catalog = {
            0.005: (0.016, 0.005, 1380,  540,  60000, 2.0e7),
            0.008: (0.022, 0.007, 3450,  1370, 48000, 3.5e7),
            0.010: (0.026, 0.008, 4620,  1960, 40000, 5.0e7),
            0.012: (0.028, 0.008, 5070,  2360, 38000, 5.5e7),
            0.015: (0.032, 0.009, 5590,  2850, 34000, 7.0e7),
            0.017: (0.035, 0.010, 6050,  3250, 32000, 8.0e7),
            0.020: (0.042, 0.012, 9360,  4500, 28000, 1.2e8),
            0.025: (0.047, 0.012, 11200, 5850, 24000, 1.5e8),
            0.030: (0.055, 0.013, 13300, 7350, 20000, 2.0e8),
        }
        bores = sorted(catalog.keys())
        sel = min(bores, key=lambda x: abs(x - shaft_diameter))
        od, w, C, C0, max_rpm, k_radial = catalog[sel]
        for brg in [self.bearing_drive, self.bearing_non_drive]:
            brg.bore_diameter = sel
            brg.outer_diameter = od
            brg.width = w
            brg.dynamic_load_rating = C
            brg.static_load_rating = C0
            brg.max_speed = max_rpm
            brg.radial_stiffness = k_radial

    # Legacy compatibility aliases
    @property
    def aluminum(self) -> StructuralMaterial:
        return self.rotor_housing

    @property
    def shaft_steel(self) -> StructuralMaterial:
        return self.shaft
