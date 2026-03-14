"""
Simplified CFD / airflow cooling model for outrunner BLDC motors.

Uses validated Nusselt-number correlations rather than full Navier-Stokes
to estimate convective heat transfer at each motor surface, and windage
(aerodynamic drag) losses.

Key References:
  [1] K.M. Becker, J. Kaye, "Measurements of Diabatic Flow in an Annulus
      With an Inner Rotating Cylinder," ASME J. Heat Transfer, vol. 84,
      pp. 97-105, 1962.
  [2] F. Kreith, "Convection Heat Transfer in Rotating Systems," Advances
      in Heat Transfer, vol. 5, Academic Press, 1968.
  [3] D. Staton, A. Cavagnino, "Convection Heat Transfer and Flow
      Computations in Electric Machines Thermal Models," IEEE Trans. Ind.
      Electron., vol. 55, no. 10, 2008.
  [4] J.M. Owen, R.H. Rogers, "Flow and Heat Transfer in Rotating-Disc
      Systems," Research Studies Press, 1989.
  [5] G.I. Taylor, "Stability of a Viscous Liquid Contained between Two
      Rotating Cylinders," Phil. Trans. Royal Soc. A, vol. 223, 1923.
      (Taylor-Couette instability criterion)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .electromagnetic import GeometryParams, WindingConfig


@dataclass
class AirProperties:
    """Temperature-dependent dry air at 1 atm."""
    def density(self, T: float) -> float:
        """[kg/m³] ideal-gas approximation."""
        return 1.225 * 293.15 / (T + 273.15)

    def kinematic_viscosity(self, T: float) -> float:
        """[m²/s] Sutherland's law approximation."""
        return 1.46e-5 * ((T + 273.15) / 293.15) ** 1.5 * (293.15 + 110.4) / (T + 273.15 + 110.4)

    def thermal_conductivity(self, T: float) -> float:
        """[W/(m·K)]."""
        return 0.0257 * ((T + 273.15) / 293.15) ** 0.8

    def prandtl(self, T: float) -> float:
        return 0.71

    def dynamic_viscosity(self, T: float) -> float:
        return self.density(T) * self.kinematic_viscosity(T)


AIR = AirProperties()


@dataclass
class CFDResults:
    """Airflow / cooling results."""
    h_airgap: float = 0.0             # [W/(m²·K)]
    h_rotor_external: float = 0.0
    h_endcap_de: float = 0.0
    h_endcap_nde: float = 0.0
    h_stator_bore: float = 0.0

    taylor_number: float = 0.0
    flow_regime: str = "laminar"       # laminar / Taylor-vortex / turbulent

    windage_loss: float = 0.0         # [W] aerodynamic drag on rotor
    total_convective_loss: float = 0.0 # [W] (not thermal loss — windage only)

    air_mass_flow_kg_s: float = 0.0   # Estimated self-pumping airflow [kg/s]


class CFDModel:
    """
    Correlation-based convective heat transfer and windage model.
    """

    def __init__(self, geometry: GeometryParams, winding: WindingConfig):
        self.geo = geometry
        self.wdg = winding
        self.results = CFDResults()

    def compute(self, rpm: float, T_air: float = 40.0) -> CFDResults:
        """
        Evaluate all convection coefficients and windage loss.

        Parameters
        ----------
        rpm : float   Operating speed [RPM].
        T_air : float Bulk air temperature estimate [°C].
        """
        res = CFDResults()
        geo = self.geo
        omega = rpm * 2 * np.pi / 60.0

        nu = AIR.kinematic_viscosity(T_air)
        k = AIR.thermal_conductivity(T_air)
        rho = AIR.density(T_air)
        Pr = AIR.prandtl(T_air)
        mu = AIR.dynamic_viscosity(T_air)

        delta = geo.airgap
        R_rotor = geo.rotor_inner_radius
        R_outer = geo.rotor_outer_radius
        L = geo.stack_length

        # =================================================================
        # 1. Airgap (Taylor-Couette) — Becker & Kaye (1962) [1]
        # =================================================================
        v_gap = omega * R_rotor
        Re_gap = v_gap * delta / nu if nu > 0 else 0

        Ta = Re_gap * np.sqrt(delta / R_rotor) if R_rotor > 0 else 0
        res.taylor_number = Ta

        Ta_crit = 41.3   # Critical Taylor number (Taylor, 1923 [5])

        if Ta < Ta_crit:
            Nu_gap = 2.0
            res.flow_regime = "laminar"
        elif Ta < 100:
            Nu_gap = 0.202 * Ta ** 0.63 * Pr ** 0.27
            res.flow_regime = "Taylor-vortex"
        else:
            Nu_gap = 0.386 * Ta ** 0.5 * Pr ** 0.27
            res.flow_regime = "turbulent"

        res.h_airgap = max(Nu_gap * k / delta, 5.0) if delta > 0 else 10.0
        res.h_stator_bore = res.h_airgap  # same gap

        # =================================================================
        # 2. External rotor surface — rotating cylinder  (Kreith 1968 [2])
        # =================================================================
        v_surface = omega * R_outer
        Re_ext = v_surface * 2 * R_outer / nu if nu > 0 else 0

        if Re_ext > 2.5e4:
            Nu_ext = 0.133 * Re_ext ** (2 / 3) * Pr ** (1 / 3)
        elif Re_ext > 100:
            Nu_ext = 0.683 * Re_ext ** 0.466 * Pr ** (1 / 3)
        else:
            Nu_ext = 2.0

        D_ext = 2 * R_outer
        res.h_rotor_external = max(Nu_ext * k / D_ext, 5.0) if D_ext > 0 else 10.0

        # =================================================================
        # 3. End-caps — rotating disc in enclosure (Owen & Rogers 1989 [4])
        # =================================================================
        Re_disc = omega * R_outer ** 2 / nu if nu > 0 else 0

        if Re_disc > 1e5:
            # Turbulent disc
            Nu_disc = 0.0151 * Re_disc ** 0.8 * Pr ** 0.33
        elif Re_disc > 100:
            Nu_disc = 0.456 * Re_disc ** 0.6 * Pr ** 0.33
        else:
            Nu_disc = 2.0

        h_disc = max(Nu_disc * k / (2 * R_outer), 8.0)
        res.h_endcap_de = h_disc
        res.h_endcap_nde = h_disc * 0.9  # NDE slightly less air movement

        # =================================================================
        # 4. Windage (aerodynamic drag) loss
        # =================================================================
        # Cylinder windage: P_w = C_f * π * ρ * ω³ * R⁴ * L
        # C_f depends on Reynolds number:
        Re_cyl = rho * omega * R_outer ** 2 / mu if mu > 0 else 0

        if Re_cyl > 1e4:
            C_f = 0.0725 / Re_cyl ** 0.2  # Turbulent (Kreith)
        elif Re_cyl > 0:
            C_f = 8.0 / Re_cyl              # Laminar (Stokes)
        else:
            C_f = 0

        P_windage_cyl = C_f * np.pi * rho * omega ** 3 * R_outer ** 4 * L

        # Disc windage (two end faces): P_disc = 0.5 * C_m * ρ * ω³ * R⁵
        if Re_disc > 1e5:
            C_m = 0.073 / Re_disc ** 0.2
        elif Re_disc > 0:
            C_m = 3.87 / np.sqrt(Re_disc)
        else:
            C_m = 0

        P_windage_disc = 2 * 0.5 * C_m * rho * omega ** 3 * R_outer ** 5

        res.windage_loss = max(P_windage_cyl + P_windage_disc, 0)
        res.total_convective_loss = res.windage_loss

        # =================================================================
        # 5. Estimated self-pumping airflow (outrunner acts as centrifugal fan)
        # =================================================================
        # Simplified: Δp ≈ 0.5 * ρ * (ω * R_outer)² * efficiency_fan
        eta_fan = 0.1  # very rough
        delta_p = 0.5 * rho * (omega * R_outer) ** 2 * eta_fan
        # Assume flow through annular gap: A = 2 * π * delta * L
        A_flow = 2 * np.pi * delta * L if delta > 0 else 0
        v_flow = np.sqrt(2 * delta_p / rho) if rho > 0 and delta_p > 0 else 0
        res.air_mass_flow_kg_s = rho * A_flow * v_flow

        self.results = res
        return res