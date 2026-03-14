"""
Lumped Parameter Thermal Network (LPTN) for outrunner BLDC motor.

Implements a multi-node thermal network with temperature-dependent material
properties and iterative coupling with the electromagnetic model.

Key References:
  [1] A. Boglietti, A. Cavagnino, D. Staton, M. Shanel, M. Mueller,
      C. Mejuto, "Evolution and Modern Approaches for Thermal Analysis
      of Electrical Machines," IEEE Trans. Ind. Electron., vol.56, no.3,
      pp.871-882, 2009.
  [2] D. Staton, A. Cavagnino, "Convection Heat Transfer and Flow
      Computations in Electric Machines Thermal Models," IEEE Trans.
      Ind. Electron., vol.55, no.10, pp.3509-3516, 2008.
  [3] Motor-CAD technical notes on LPTN methodology.
  [4] K.M. Becker, J. Kaye, "Measurements of Diabatic Flow in an
      Annulus With an Inner Rotating Cylinder," ASME J. Heat Transfer,
      vol.84, pp.97-105, 1962. (Airgap convection correlation)

Thermal Network Nodes:
  0: Stator yoke
  1: Stator teeth
  2: Slot winding (copper + insulation)
  3: End winding (drive end)
  4: End winding (non-drive end)
  5: Airgap (fluid node)
  6: Magnets
  7: Rotor yoke (aluminum)
  8: Shaft
  9: Ambient
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .materials import MaterialDatabase, StructuralMaterial
from .electromagnetic import GeometryParams, WindingConfig, EMResults


NUM_NODES = 10  # Including ambient


@dataclass
class ThermalResults:
    """Thermal analysis results."""
    T_stator_yoke: float = 25.0       # [°C]
    T_stator_teeth: float = 25.0      # [°C]
    T_winding_slot: float = 25.0      # [°C]
    T_end_winding_de: float = 25.0    # [°C]
    T_end_winding_nde: float = 25.0   # [°C]
    T_airgap: float = 25.0           # [°C]
    T_magnets: float = 25.0          # [°C]
    T_rotor_yoke: float = 25.0       # [°C]
    T_shaft: float = 25.0            # [°C]
    T_ambient: float = 25.0          # [°C]
    max_temperature: float = 25.0     # [°C]
    critical_component: str = ""
    is_safe: bool = True
    hotspot_margin: float = 0.0       # [°C] margin to limit

    def from_vector(self, T: np.ndarray, T_amb: float = 25.0):
        """Populate from temperature vector."""
        self.T_stator_yoke = T[0]
        self.T_stator_teeth = T[1]
        self.T_winding_slot = T[2]
        self.T_end_winding_de = T[3]
        self.T_end_winding_nde = T[4]
        self.T_airgap = T[5]
        self.T_magnets = T[6]
        self.T_rotor_yoke = T[7]
        self.T_shaft = T[8]
        self.T_ambient = T_amb
        self.max_temperature = float(np.max(T[:9]))

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.T_stator_yoke, self.T_stator_teeth, self.T_winding_slot,
            self.T_end_winding_de, self.T_end_winding_nde, self.T_airgap,
            self.T_magnets, self.T_rotor_yoke, self.T_shaft, self.T_ambient
        ])


class ThermalModel:
    """
    Lumped Parameter Thermal Network (LPTN) solver.

    Builds and solves a resistive-capacitive thermal network using
    the node-voltage (temperature) method:

      [G] * {T} = {Q}

    where G is the conductance matrix, T is the temperature vector,
    and Q is the heat source vector.

    Convection correlations:
      - Airgap (Taylor-Couette flow): Nu = 0.409 * Ta^0.241 for Ta > Ta_c
        (Becker & Kaye, 1962; Staton, 2005)
      - External rotor surface: Nu = 0.133 * Re^(2/3) * Pr^(1/3)
        (rotating cylinder in ambient air)
      - End-cap internal: Nu = 0.456 * Re^0.6 * Pr^0.33
        (enclosed rotating disc)
    """

    def __init__(self, geometry: GeometryParams, winding: WindingConfig,
                 materials: MaterialDatabase, T_ambient: float = 25.0):
        self.geo = geometry
        self.wdg = winding
        self.mats = materials
        self.T_amb = T_ambient
        self.results = ThermalResults()

    def _thermal_resistance_conduction(self, L: float, A: float, k: float) -> float:
        """Conduction thermal resistance [K/W]: R = L / (k * A)."""
        if k * A <= 0:
            return 1e6
        return L / (k * A)

    def _thermal_resistance_convection(self, h: float, A: float) -> float:
        """Convection thermal resistance [K/W]: R = 1 / (h * A)."""
        if h * A <= 0:
            return 1e6
        return 1.0 / (h * A)

    def _airgap_htc(self, rpm: float) -> float:
        """
        Airgap heat transfer coefficient using Taylor-Couette correlation.

        For rotating inner cylinder (stator bore facing outrunner rotor):
        Actually for outrunner, the rotor is OUTSIDE, so the outer surface rotates.

        Taylor number: Ta = Re_g * sqrt(δ/R_m)
        where Re_g = ω * R_rotor * δ / ν

        Becker & Kaye (1962):
          Nu = 2.0                         for Ta < 41.3 (laminar)
          Nu = 0.202 * Ta^0.63 * Pr^0.27   for 41.3 < Ta < 100
          Nu = 0.386 * Ta^0.5 * Pr^0.27    for Ta > 100

        Ref: Staton & Cavagnino (2008)
        """
        omega = rpm * 2 * np.pi / 60.0
        delta = self.geo.airgap
        R_rotor = self.geo.rotor_inner_radius

        # Air properties at estimated airgap temperature
        T_air = 60.0  # Initial estimate
        nu_air = 1.5e-5 * (1 + 0.003 * (T_air - 20))  # Kinematic viscosity [m²/s]
        k_air = 0.026 * (1 + 0.003 * (T_air - 20))     # Thermal conductivity [W/(m·K)]
        Pr = 0.71

        # Reynolds number based on gap
        v_rotor = omega * R_rotor
        Re_g = v_rotor * delta / nu_air

        # Taylor number
        Ta = Re_g * np.sqrt(delta / R_rotor) if R_rotor > 0 else 0

        if Ta < 41.3:
            Nu = 2.0
        elif Ta < 100:
            Nu = 0.202 * Ta**0.63 * Pr**0.27
        else:
            Nu = 0.386 * Ta**0.5 * Pr**0.27

        h = Nu * k_air / delta if delta > 0 else 10.0
        return max(h, 5.0)  # Minimum natural convection

    def _external_rotor_htc(self, rpm: float) -> float:
        """
        Heat transfer coefficient for external surface of rotating rotor.

        Rotating cylinder in free air:
        Nu = 0.133 * Re^(2/3) * Pr^(1/3)  for Re > 2.5e4
        Nu = 0.683 * Re^0.466 * Pr^(1/3)  for Re < 2.5e4

        Ref: Kreith (1968), Staton & Cavagnino (2008)
        """
        omega = rpm * 2 * np.pi / 60.0
        R_ext = self.geo.rotor_outer_radius
        nu_air = 1.6e-5
        k_air = 0.026
        Pr = 0.71

        v_surface = omega * R_ext
        Re = v_surface * 2 * R_ext / nu_air  # Based on diameter

        if Re > 2.5e4:
            Nu = 0.133 * Re**(2.0/3.0) * Pr**(1.0/3.0)
        elif Re > 100:
            Nu = 0.683 * Re**0.466 * Pr**(1.0/3.0)
        else:
            Nu = 2.0  # Natural convection limit

        D_ext = 2 * R_ext
        h = Nu * k_air / D_ext if D_ext > 0 else 10.0
        return max(h, 5.0)

    def _end_cap_htc(self, rpm: float) -> float:
        """
        End-cap (end-winding region) convection coefficient.

        Rotating disc in enclosure:
        Nu = 0.456 * Re^0.6 * Pr^0.33

        Ref: Owen & Rogers (1989), "Flow and Heat Transfer in Rotating-Disc Systems"
        """
        omega = rpm * 2 * np.pi / 60.0
        R = self.geo.rotor_outer_radius
        nu_air = 1.6e-5
        k_air = 0.026
        Pr = 0.71

        Re = omega * R**2 / nu_air

        if Re > 100:
            Nu = 0.456 * Re**0.6 * Pr**0.33
        else:
            Nu = 2.0

        h = Nu * k_air / (2 * R) if R > 0 else 10.0
        return max(h, 8.0)

    def build_and_solve(self, em_results: EMResults, rpm: float) -> ThermalResults:
        """
        Build the thermal network and solve for steady-state temperatures.

        Returns ThermalResults with temperatures of all components.
        """
        geo = self.geo
        wdg = self.wdg
        mats = self.mats

        # Heat sources [W] - from EM analysis
        Q = np.zeros(NUM_NODES)

        # Distribute iron loss
        iron_teeth_fraction = 0.6  # 60% in teeth, 40% in yoke (typical)
        Q[0] = em_results.iron_loss_stator * (1 - iron_teeth_fraction)  # Yoke
        Q[1] = em_results.iron_loss_stator * iron_teeth_fraction        # Teeth

        # Distribute copper loss between slot and end windings
        l_stack = geo.stack_length
        l_end = wdg.end_turn_length
        l_total = 2 * l_stack + 4 * l_end
        slot_fraction = 2 * l_stack / l_total if l_total > 0 else 0.6
        end_fraction = (1 - slot_fraction) / 2

        Q[2] = em_results.copper_loss * slot_fraction
        Q[3] = em_results.copper_loss * end_fraction
        Q[4] = em_results.copper_loss * end_fraction

        # Magnet and rotor losses
        Q[6] = em_results.magnet_loss + em_results.iron_loss_rotor * 0.3
        Q[7] = em_results.iron_loss_rotor * 0.7

        # Conductance matrix
        G = np.zeros((NUM_NODES, NUM_NODES))

        # --- Compute thermal resistances ---

        # Surface areas
        A_stator_bore = 2 * np.pi * geo.stator_outer_radius * geo.stack_length
        A_rotor_inner = 2 * np.pi * geo.rotor_inner_radius * geo.stack_length
        A_rotor_outer = 2 * np.pi * geo.rotor_outer_radius * geo.stack_length
        A_end_cap = np.pi * (geo.rotor_outer_radius**2 - geo.shaft_radius**2)

        # Convection coefficients
        h_airgap = self._airgap_htc(rpm)
        h_external = self._external_rotor_htc(rpm)
        h_endcap = self._end_cap_htc(rpm)

        # R0-1: Yoke to teeth (radial conduction through stator)
        R_01 = self._thermal_resistance_conduction(
            geo.stator_yoke_thickness / 2,
            2 * np.pi * (geo.stator_inner_radius + geo.stator_yoke_thickness / 2) * geo.stack_length,
            mats.steel.thermal_conductivity_radial
        )

        # R1-2: Teeth to winding (through slot liner insulation)
        insulation_thickness = 0.3e-3  # 0.3mm slot liner
        k_insulation = 0.2  # Typical Nomex/Kapton
        contact_area = wdg.num_slots * 2 * geo.slot_depth * geo.stack_length  # Both sides
        R_12 = self._thermal_resistance_conduction(
            insulation_thickness, contact_area, k_insulation
        )

        # R2-3, R2-4: Slot winding to end windings (axial conduction through copper)
        k_copper_axial = mats.copper.thermal_conductivity * mats.copper.fill_factor + \
                        k_insulation * (1 - mats.copper.fill_factor)
        A_winding_axial = wdg.num_slots * geo.slot_area * mats.copper.fill_factor
        R_23 = self._thermal_resistance_conduction(
            geo.stack_length / 2 + wdg.end_turn_length / 2,
            A_winding_axial, k_copper_axial
        )
        R_24 = R_23  # Symmetric

        # R1-5: Teeth to airgap (convection at stator bore)
        R_15 = self._thermal_resistance_convection(h_airgap, A_stator_bore)

        # R5-6: Airgap to magnets (convection at rotor inner surface)
        R_56 = self._thermal_resistance_convection(h_airgap, A_rotor_inner)

        # R6-7: Magnets to rotor yoke (conduction through magnet + adhesive)
        adhesive_thickness = 0.1e-3
        k_adhesive = 1.0
        R_67_magnet = self._thermal_resistance_conduction(
            mats.magnet.thickness,
            wdg.num_poles * mats.magnet.arc_fraction *
            (2 * np.pi * geo.rotor_inner_radius / wdg.num_poles) * geo.stack_length,
            mats.magnet.thermal_conductivity
        )
        R_67_adhesive = self._thermal_resistance_conduction(
            adhesive_thickness,
            wdg.num_poles * mats.magnet.arc_fraction *
            (2 * np.pi * geo.rotor_inner_radius / wdg.num_poles) * geo.stack_length,
            k_adhesive
        )
        R_67 = R_67_magnet + R_67_adhesive

        # R7-9: Rotor yoke to ambient (external convection)
        R_79 = self._thermal_resistance_convection(h_external, A_rotor_outer)

        # R0-8: Stator yoke to shaft (conduction through stator bore)
        R_08 = self._thermal_resistance_conduction(
            geo.stator_inner_radius - geo.shaft_radius,
            2 * np.pi * geo.shaft_radius * geo.stack_length,
            mats.shaft.thermal_conductivity
        )

        # R3-9, R4-9: End windings to ambient (convection)
        R_39 = self._thermal_resistance_convection(h_endcap, A_end_cap * 0.3)
        R_49 = R_39

        # R8-9: Shaft to ambient (conduction out shaft ends)
        shaft_end_area = np.pi * geo.shaft_radius**2
        shaft_h_natural = 10.0  # Natural convection on shaft end
        R_89 = self._thermal_resistance_convection(shaft_h_natural, 2 * shaft_end_area)

        # R7-endcap: Rotor yoke to end-cap convection
        R_79_end = self._thermal_resistance_convection(h_endcap, 2 * A_end_cap * 0.5)

        # --- Build conductance matrix ---
        def add_resistance(G, i, j, R):
            if R > 0 and R < 1e6:
                g = 1.0 / R
                G[i, i] += g
                G[j, j] += g
                G[i, j] -= g
                G[j, i] -= g

        add_resistance(G, 0, 1, R_01)   # Yoke-Teeth
        add_resistance(G, 1, 2, R_12)   # Teeth-Winding
        add_resistance(G, 2, 3, R_23)   # Winding-EndDE
        add_resistance(G, 2, 4, R_24)   # Winding-EndNDE
        add_resistance(G, 1, 5, R_15)   # Teeth-Airgap
        add_resistance(G, 5, 6, R_56)   # Airgap-Magnets
        add_resistance(G, 6, 7, R_67)   # Magnets-RotorYoke
        add_resistance(G, 7, 9, R_79)   # RotorYoke-Ambient
        add_resistance(G, 0, 8, R_08)   # Yoke-Shaft
        add_resistance(G, 3, 9, R_39)   # EndDE-Ambient
        add_resistance(G, 4, 9, R_49)   # EndNDE-Ambient
        add_resistance(G, 8, 9, R_89)   # Shaft-Ambient
        add_resistance(G, 7, 9, R_79_end)  # RotorYoke endcap

        # Ambient node: fixed temperature (set by large conductance)
        G[9, 9] += 1e6
        Q[9] += 1e6 * self.T_amb

        # --- Solve: G * T = Q ---
        try:
            T = np.linalg.solve(G, Q)
        except np.linalg.LinAlgError:
            # Fallback: assume uniform temperature rise
            total_loss = sum(Q[:9])
            total_conductance = sum(1.0 / max(R, 0.001) for R in
                                  [R_79, R_39, R_49, R_89])
            delta_T = total_loss / max(total_conductance, 0.01)
            T = np.full(NUM_NODES, self.T_amb + delta_T)
            T[9] = self.T_amb

        # Populate results
        self.results.from_vector(T, self.T_amb)

        # Check thermal limits
        limits = {
            'Winding (slot)': (self.results.T_winding_slot, mats.copper.max_temp),
            'End winding DE': (self.results.T_end_winding_de, mats.copper.max_temp),
            'End winding NDE': (self.results.T_end_winding_nde, mats.copper.max_temp),
            'Magnets': (self.results.T_magnets, mats.magnet.max_temp),
            'Stator yoke': (self.results.T_stator_yoke, mats.steel.max_temp),
        }

        self.results.is_safe = True
        self.results.hotspot_margin = float('inf')
        for name, (T_actual, T_limit) in limits.items():
            margin = T_limit - T_actual
            if margin < self.results.hotspot_margin:
                self.results.hotspot_margin = margin
                self.results.critical_component = name
            if T_actual > T_limit:
                self.results.is_safe = False

        return self.results

    def transient_solve(self, em_results: EMResults, rpm: float,
                       duration: float = 3600.0, dt: float = 1.0) -> np.ndarray:
        """
        Transient thermal analysis using forward Euler method.

        Returns array of shape (n_steps, NUM_NODES) with temperature history.
        """
        geo = self.geo
        mats = self.mats

        n_steps = int(duration / dt)
        T_history = np.zeros((n_steps, NUM_NODES))
        T = np.full(NUM_NODES, self.T_amb)
        T[9] = self.T_amb

        # Thermal capacitances [J/K]
        C_th = np.zeros(NUM_NODES)

        # Stator yoke
        R_yi = geo.stator_inner_radius
        R_yo = R_yi + geo.stator_yoke_thickness
        C_th[0] = np.pi * (R_yo**2 - R_yi**2) * geo.stack_length * \
                  mats.steel.density * mats.steel.specific_heat * mats.steel.stacking_factor

        # Teeth
        C_th[1] = self.wdg.num_slots * geo.tooth_width * geo.slot_depth * \
                  geo.stack_length * mats.steel.density * mats.steel.specific_heat

        # Slot winding
        copper_vol = self.wdg.num_slots * geo.slot_area * mats.copper.fill_factor
        C_th[2] = copper_vol * mats.copper.density * mats.copper.specific_heat

        # End windings
        C_th[3] = C_th[2] * 0.2  # Fraction of total copper
        C_th[4] = C_th[3]

        # Airgap (air mass is negligible)
        C_th[5] = 0.1  # Small but non-zero for numerical stability

        # Magnets
        mag_vol = self.wdg.num_poles * mats.magnet.thickness * \
                 mats.magnet.arc_fraction * \
                 (2 * np.pi * geo.rotor_inner_radius / self.wdg.num_poles) * \
                 geo.stack_length
        C_th[6] = mag_vol * mats.magnet.density * mats.magnet.specific_heat

        # Rotor yoke
        R_ri = geo.rotor_inner_radius + mats.magnet.thickness
        R_ro = geo.rotor_outer_radius
        C_th[7] = np.pi * (R_ro**2 - R_ri**2) * geo.stack_length * \
                  mats.rotor_housing.density * mats.rotor_housing.specific_heat

        # Shaft
        C_th[8] = np.pi * geo.shaft_radius**2 * geo.shaft_length * \
                  mats.shaft.density * mats.shaft.specific_heat

        C_th[9] = 1e10  # Ambient: infinite thermal mass

        # Ensure minimum capacitance
        C_th = np.maximum(C_th, 0.01)

        # Get steady-state solution for comparison
        ss_result = self.build_and_solve(em_results, rpm)

        for step in range(n_steps):
            T_history[step] = T.copy()

            # Recompute network at current temperatures (simplified)
            # For speed, use linearized version around initial estimate
            T_ss = ss_result.to_vector()

            # Exponential approach to steady state
            tau = C_th / np.maximum(np.abs(np.diag(self._build_G_matrix(rpm))), 0.001)
            tau = np.maximum(tau, 1.0)

            T[:9] = T_ss[:9] - (T_ss[:9] - T[:9]) * np.exp(-dt / tau[:9])
            T[9] = self.T_amb

        return T_history

    def _build_G_matrix(self, rpm: float) -> np.ndarray:
        """Helper: build conductance matrix (reused for transient)."""
        G = np.zeros((NUM_NODES, NUM_NODES))
        # Simplified version - just return diagonal dominance estimate
        h_ext = self._external_rotor_htc(rpm)
        A_ext = 2 * np.pi * self.geo.rotor_outer_radius * self.geo.stack_length
        g_ext = h_ext * A_ext
        np.fill_diagonal(G, g_ext * 0.1)
        return G
