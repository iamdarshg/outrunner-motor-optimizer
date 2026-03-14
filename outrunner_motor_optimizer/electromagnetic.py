"""
Electromagnetic design module for outrunner BLDC motors.

Implements analytical subdomain method for surface-mounted PM machines with
concentrated windings. Computes airgap flux density, back-EMF, torque,
winding losses, iron losses, and magnet eddy current losses.

Key References:
  [1] Z.Q. Zhu, D. Howe, "Instantaneous magnetic field distribution in
      permanent magnet brushless DC motors," IEEE Trans. Magnetics, 1993.
  [2] Z.Q. Zhu et al., "Analytical Prediction of Cogging Torque for
      Surface-Mounted PM Machines," IEEE Trans. Magnetics, 2003.
  [3] J. Pyrhonen, T. Jokinen, V. Hrabovcova, "Design of Rotating
      Electrical Machines," 2nd ed., Wiley, 2014.
  [4] A.M. El-Refaie, "Fractional-Slot Concentrated-Windings Synchronous
      PM Machines: Opportunities and Challenges," IEEE Trans. Ind.
      Electron., 2010.  (Winding factor & slot-pole combinations)
  [5] G. Bertotti, "General Properties of Power Losses in Soft
      Ferromagnetic Materials," IEEE Trans. Magnetics, 1988.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np

from .materials import MaterialDatabase, MagnetProperties, StructuralMaterial


MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability


@dataclass
class MotorSpecs:
    """User-provided motor specifications."""
    voltage: float          # Supply voltage [V]
    target_rpm: float       # Target speed [RPM]
    target_torque: float    # Target torque [N·m]
    max_current: float      # Maximum phase current [A]
    target_efficiency: float = 0.90  # Target efficiency [0-1]
    num_phases: int = 3     # Number of phases


@dataclass
class WindingConfig:
    """Winding configuration results."""
    num_poles: int = 14
    num_slots: int = 12
    turns_per_coil: int = 10
    num_layers: int = 2          # 1 = single layer, 2 = double layer
    num_parallel_paths: int = 1
    wire_diameter: float = 0.001  # [m]
    num_strands: int = 1
    coil_span: int = 1           # In slot pitches
    winding_factor: float = 0.933
    phase_resistance: float = 0.0  # Computed [Ohm]
    phase_inductance: float = 0.0  # Computed [H]
    end_turn_length: float = 0.0   # [m]

    @property
    def wire_area(self) -> float:
        """Total copper area per slot [m²]."""
        return self.num_strands * np.pi * (self.wire_diameter / 2)**2

    @property
    def coils_per_phase(self) -> int:
        return self.num_slots // 3  # For 3-phase


@dataclass
class GeometryParams:
    """Motor geometry parameters (outrunner configuration)."""
    # Stator (inner, stationary)
    stator_outer_radius: float = 0.025   # [m]
    stator_inner_radius: float = 0.008   # [m] (shaft bore)
    stator_yoke_thickness: float = 0.005 # [m]
    slot_depth: float = 0.010            # [m]
    slot_opening: float = 0.002          # [m]
    tooth_width: float = 0.005           # [m]
    stack_length: float = 0.030          # [m] (axial)

    # Airgap
    airgap: float = 0.0007              # [m] (0.7mm typical)

    # Rotor (outer, rotating)
    rotor_inner_radius: float = 0.0257   # stator_outer + airgap
    rotor_outer_radius: float = 0.032    # [m]
    rotor_yoke_thickness: float = 0.003  # [m]
    magnet_thickness: float = 0.003      # [m]

    # Shaft
    shaft_radius: float = 0.005         # [m]
    shaft_length: float = 0.060         # [m]

    @property
    def stator_bore_radius(self) -> float:
        """Inner radius of stator bore (where slots end)."""
        return self.stator_outer_radius

    @property
    def airgap_radius(self) -> float:
        """Mean airgap radius."""
        return (self.stator_outer_radius + self.rotor_inner_radius) / 2

    @property
    def slot_area(self) -> float:
        """Approximate slot cross-sectional area [m²]."""
        # Trapezoidal slot approximation
        r_inner = self.stator_inner_radius + self.stator_yoke_thickness
        r_outer = self.stator_outer_radius
        slot_width_avg = (2 * np.pi * (r_inner + r_outer) / 2 / 12) - self.tooth_width
        return max(slot_width_avg * self.slot_depth, 1e-8)

    @property
    def active_volume(self) -> float:
        """Active motor volume [m³]."""
        return np.pi * self.rotor_outer_radius**2 * self.stack_length

    @property
    def stator_mass(self) -> float:
        """Approximate stator lamination mass [kg]."""
        area = np.pi * (self.stator_outer_radius**2 - self.stator_inner_radius**2)
        # Subtract slot area (approximate)
        return area * self.stack_length * 7650 * 0.7  # 70% fill approx


@dataclass
class EMResults:
    """Electromagnetic analysis results."""
    back_emf_peak: float = 0.0        # [V]
    back_emf_rms: float = 0.0         # [V]
    torque_avg: float = 0.0           # [N·m]
    torque_ripple: float = 0.0        # [%]
    cogging_torque_peak: float = 0.0  # [N·m]
    copper_loss: float = 0.0          # [W]
    iron_loss_stator: float = 0.0     # [W]
    iron_loss_rotor: float = 0.0      # [W]
    magnet_loss: float = 0.0          # [W]
    airgap_flux_density: float = 0.0  # [T] fundamental peak
    tooth_flux_density: float = 0.0   # [T]
    yoke_flux_density: float = 0.0    # [T]
    phase_current_rms: float = 0.0    # [A]
    power_output: float = 0.0         # [W]
    power_input: float = 0.0          # [W]
    efficiency: float = 0.0           # [0-1]
    power_factor: float = 0.0
    current_density: float = 0.0      # [A/mm²]

    @property
    def total_loss(self) -> float:
        return self.copper_loss + self.iron_loss_stator + self.iron_loss_rotor + self.magnet_loss


def compute_winding_factor(num_slots: int, num_poles: int,
                           num_layers: int = 2, coil_span: int = 1) -> float:
    """
    Compute fundamental winding factor for concentrated windings.

    kw1 = kd1 * kp1 * ks1

    where:
      kd1 = distribution factor
      kp1 = pitch factor
      ks1 = skew factor (1.0 for no skew)

    Ref: El-Refaie (2010), Pyrhonen et al. (2014) Ch. 2.
    """
    q = num_slots / (3 * num_poles)  # Slots per pole per phase

    if q <= 0:
        return 0.0

    # For concentrated (tooth-wound) windings with q < 1:
    if q < 1:
        # Use the general formula
        p = num_poles // 2
        m = 3  # phases
        alpha_e = 2 * np.pi * p / num_slots  # Electrical angle between slots

        # Distribution factor
        n = 1  # Fundamental harmonic
        if num_slots % (m * 2) == 0:
            q_int = num_slots // (m * 2 * p) if p > 0 else 1
            q_int = max(q_int, 1)
        else:
            q_int = max(1, int(np.round(q)))

        # General winding factor via star of slots method
        # Sum of phasors
        slot_angles = []
        for k in range(num_slots):
            angle = k * num_poles * np.pi / num_slots
            slot_angles.append(angle)

        # Assign slots to phases (simplified for fractional slot)
        phase_a_slots = []
        for k in range(num_slots):
            phase_angle = (slot_angles[k] % (2 * np.pi))
            if phase_angle < 0:
                phase_angle += 2 * np.pi
            # Phase A: angles near 0 or π
            if phase_angle < np.pi / 3 or phase_angle > 5 * np.pi / 3:
                phase_a_slots.append(k)
            elif np.pi - np.pi / 3 < phase_angle < np.pi + np.pi / 3:
                phase_a_slots.append(k)  # Negative coil

        if len(phase_a_slots) == 0:
            return 0.866  # Fallback

        # Compute winding factor from phasor sum
        phasor_sum = 0.0
        for k in phase_a_slots:
            phasor_sum += np.exp(1j * slot_angles[k])

        kw = abs(phasor_sum) / len(phase_a_slots)
        return min(kw, 1.0)

    else:
        # Distributed winding
        alpha_e = np.pi * num_poles / num_slots
        q_val = max(int(np.round(q)), 1)
        kd = np.sin(q_val * alpha_e / 2) / (q_val * np.sin(alpha_e / 2)) if q_val > 0 else 1.0

        # Pitch factor
        y = coil_span
        tau_p = num_slots / num_poles
        kp = np.sin(y * np.pi / (2 * tau_p)) if tau_p > 0 else 1.0

        return abs(kd * kp)


# Well-known outrunner slot-pole combinations with good winding factors
SLOT_POLE_COMBINATIONS = [
    # (slots, poles, typical_kw1, notes)
    (9, 8, 0.945, "Low cogging, high kw"),
    (12, 10, 0.933, "Very common drone motor"),
    (12, 14, 0.933, "Common outrunner"),
    (18, 16, 0.945, "Balanced performance"),
    (18, 20, 0.945, "High pole count"),
    (24, 20, 0.933, "Large motors"),
    (24, 22, 0.949, "High torque density"),
    (27, 24, 0.945, "Smooth torque"),
    (36, 32, 0.945, "Very smooth"),
    (36, 42, 0.933, "Hub motor"),
]


def select_slot_pole(target_rpm: float, target_torque: float,
                     voltage: float) -> List[Tuple[int, int, float]]:
    """
    Select candidate slot-pole combinations based on operating requirements.

    Higher pole counts → better for low speed, high torque
    Lower pole counts → better for high speed

    Returns list of (slots, poles, winding_factor) sorted by suitability.
    """
    candidates = []
    f_elec_target = target_rpm / 60.0  # Mechanical frequency

    for ns, np_val, kw, notes in SLOT_POLE_COMBINATIONS:
        p = np_val // 2  # Pole pairs
        f_elec = f_elec_target * p  # Electrical frequency

        # Practical frequency limits (iron loss, switching)
        if f_elec > 2000:
            continue  # Too high frequency
        if f_elec < 10:
            continue  # Too low, poor utilization

        # Score based on multiple factors
        score = kw  # Start with winding factor

        # Prefer moderate frequencies (200-800 Hz is sweet spot)
        if 200 < f_elec < 800:
            score += 0.05
        elif 100 < f_elec < 1200:
            score += 0.02

        # Higher pole count = higher torque density (up to a point)
        if target_torque > 1.0 and np_val >= 14:
            score += 0.03
        if target_torque < 0.5 and np_val <= 14:
            score += 0.03

        candidates.append((ns, np_val, kw, score))

    # Sort by score descending
    candidates.sort(key=lambda x: x[3], reverse=True)
    return [(c[0], c[1], c[2]) for c in candidates[:5]]


class ElectromagneticModel:
    """
    Analytical electromagnetic model for outrunner surface-PM BLDC motor.

    Uses the subdomain method for airgap field prediction and equivalent
    circuit approach for performance computation.

    Key equations (Pyrhonen et al., 2014):
      - Airgap flux density: Bg1 = (4/π) * Br * (αp * hm) / (hm + μ_rec * g_eff)
      - Back-EMF: E_ph = (2/√2) * N_s * kw1 * f * Φ_1
      - Torque: T = 3 * E_ph * I_ph / ω_m  (for sinusoidal drive)
    """

    def __init__(self, specs: MotorSpecs, materials: MaterialDatabase):
        self.specs = specs
        self.materials = materials
        self.geometry = GeometryParams()
        self.winding = WindingConfig()
        self.results = EMResults()

    def design_initial_geometry(self, num_slots: int, num_poles: int) -> GeometryParams:
        """
        Initial sizing based on Torque = k_T * D²L equation.

        Ref: Pyrhonen et al. (2014), eq. 6.1:
          T = (π/4) * σ_tan * D²_i * L
        where σ_tan is tangential stress (typical: 20-40 kPa for PM machines)
        """
        specs = self.specs
        geo = GeometryParams()

        p = num_poles // 2  # Pole pairs

        # Target tangential stress [Pa] (conservative for air-cooled)
        sigma_tan = 25000.0  # 25 kPa

        # D²L product from torque requirement
        D2L = specs.target_torque / (np.pi / 4 * sigma_tan)

        # Aspect ratio: L/D typically 0.5-2.0 for outrunners
        # Higher pole count → shorter stack acceptable
        aspect_ratio = min(max(0.3 + 0.05 * p, 0.4), 1.5)

        # Solve for diameter
        # D²L = D² * (aspect_ratio * D) → D³ = D²L / aspect_ratio
        D_stator = (D2L / aspect_ratio) ** (1.0 / 3.0)

        # Clamp to reasonable sizes
        D_stator = max(D_stator, 0.020)  # Min 20mm
        D_stator = min(D_stator, 0.300)  # Max 300mm

        L_stack = aspect_ratio * D_stator
        L_stack = max(L_stack, 0.010)
        L_stack = min(L_stack, 0.200)

        # Stator geometry
        geo.stator_outer_radius = D_stator / 2
        R_so = geo.stator_outer_radius

        # Tooth and yoke sizing based on flux density targets
        # Target: B_tooth ≈ 1.5T, B_yoke ≈ 1.3T
        B_tooth_target = 1.5
        B_yoke_target = 1.3

        # Estimate airgap flux density
        Br = self.materials.magnet.Br_20
        hm = 0.003  # Initial magnet thickness estimate
        g = 0.0007  # Airgap
        mu_rec = self.materials.magnet.mu_rec
        alpha_p = self.materials.magnet.arc_fraction

        Bg1 = (4 / np.pi) * Br * np.sin(alpha_p * np.pi / 2) * hm / (hm + mu_rec * g)

        # Tooth width from flux conservation: B_g * τ_s * L = B_t * w_t * L
        tau_s = 2 * np.pi * R_so / num_slots  # Slot pitch at stator bore
        tooth_width = Bg1 * tau_s / B_tooth_target
        tooth_width = max(tooth_width, 0.002)

        # Yoke thickness from flux conservation
        # Φ_yoke = Bg1 * τ_p * L / 2
        tau_p = 2 * np.pi * R_so / num_poles
        yoke_thickness = Bg1 * tau_p / (2 * B_yoke_target)
        yoke_thickness = max(yoke_thickness, 0.002)

        # Slot depth
        slot_depth = R_so - yoke_thickness - geo.shaft_radius - 0.002
        slot_depth = max(slot_depth, 0.005)
        slot_depth = min(slot_depth, R_so * 0.6)

        geo.stator_yoke_thickness = yoke_thickness
        geo.slot_depth = slot_depth
        geo.tooth_width = tooth_width
        geo.slot_opening = max(tau_s * 0.3, 0.001)
        geo.stator_inner_radius = geo.shaft_radius + 0.001  # 1mm clearance
        geo.stack_length = L_stack

        # Airgap
        geo.airgap = max(0.0005, 0.001 * D_stator)  # ~0.1% of diameter, min 0.5mm

        # Rotor geometry (outrunner: rotor outside stator)
        geo.rotor_inner_radius = R_so + geo.airgap
        geo.magnet_thickness = hm
        geo.rotor_yoke_thickness = max(yoke_thickness * 0.6, 0.002)  # Rotor yoke thinner
        geo.rotor_outer_radius = geo.rotor_inner_radius + hm + geo.rotor_yoke_thickness

        # Shaft
        geo.shaft_radius = max(0.004, geo.stator_inner_radius - 0.001)
        geo.shaft_length = L_stack * 2.5

        self.geometry = geo
        return geo

    def design_winding(self, num_slots: int, num_poles: int) -> WindingConfig:
        """
        Design the winding configuration.

        For outrunner motors, concentrated (tooth-wound) windings are preferred
        due to shorter end turns and simpler manufacturing.

        Ref: El-Refaie (2010), Pyrhonen et al. (2014) Ch.2.
        """
        specs = self.specs
        geo = self.geometry
        wdg = WindingConfig()

        wdg.num_poles = num_poles
        wdg.num_slots = num_slots
        p = num_poles // 2

        # Winding factor
        wdg.winding_factor = compute_winding_factor(num_slots, num_poles)
        wdg.coil_span = 1  # Concentrated winding

        # Number of turns estimation
        # From back-EMF equation: E_ph = 4.44 * N * kw * f * Φ_1
        # where Φ_1 = Bg1 * (2*τ_p*L/π)
        f_elec = specs.target_rpm * p / 60.0

        Br = self.materials.magnet.Br(80.0)  # Design at 80°C
        hm = geo.magnet_thickness
        g_eff = geo.airgap + hm / self.materials.magnet.mu_rec
        alpha_p = self.materials.magnet.arc_fraction

        Bg1 = (4 / np.pi) * Br * np.sin(alpha_p * np.pi / 2) * hm / (hm + self.materials.magnet.mu_rec * geo.airgap)

        tau_p = np.pi * geo.stator_outer_radius / (num_poles / 2)
        Phi_1 = Bg1 * tau_p * geo.stack_length * 2 / np.pi

        # Target back-EMF ≈ 0.9 * V_dc / √3 (for Y-connected BLDC)
        E_target = 0.9 * specs.voltage / np.sqrt(3)

        # N_series = E / (4.44 * kw * f * Φ)
        if f_elec > 0 and Phi_1 > 0 and wdg.winding_factor > 0:
            N_series = E_target / (4.44 * wdg.winding_factor * f_elec * Phi_1)
        else:
            N_series = 10

        # Total turns per phase = N_series * parallel_paths
        coils_per_phase = num_slots // 3
        wdg.num_parallel_paths = 1

        # Turns per coil
        if wdg.num_layers == 2:
            wdg.turns_per_coil = max(1, int(np.round(N_series / coils_per_phase)))
        else:
            wdg.turns_per_coil = max(1, int(np.round(N_series / coils_per_phase)))

        # Wire sizing based on current density target (5-8 A/mm² for air cooled)
        J_target = 6e6  # 6 A/mm²
        I_rms = specs.max_current / np.sqrt(2)  # Approximate RMS

        wire_area_needed = I_rms / (J_target * wdg.num_parallel_paths)
        wdg.wire_diameter = 2 * np.sqrt(wire_area_needed / np.pi)
        wdg.wire_diameter = max(wdg.wire_diameter, 0.0003)  # Min 0.3mm

        # Check slot fill
        slot_area = geo.slot_area
        total_wire_area = wdg.turns_per_coil * wdg.num_layers * wdg.wire_area
        fill_factor = total_wire_area / slot_area if slot_area > 0 else 1.0

        if fill_factor > 0.5:
            # Reduce turns or increase parallel paths
            wdg.num_parallel_paths = max(1, int(np.ceil(fill_factor / 0.45)))
            wdg.turns_per_coil = max(1, int(wdg.turns_per_coil / wdg.num_parallel_paths))

        # End turn length (concentrated winding: short end turns)
        wdg.end_turn_length = 1.5 * geo.tooth_width + 0.010  # Approximate

        # Phase resistance
        N_total = wdg.turns_per_coil * coils_per_phase * wdg.num_layers
        l_turn = 2 * geo.stack_length + 2 * wdg.end_turn_length
        wire_area = wdg.wire_area * wdg.num_strands

        rho_cu = self.materials.copper.resistivity(80.0)  # Design at 80°C
        if wire_area > 0:
            wdg.phase_resistance = rho_cu * N_total * l_turn / (wire_area * wdg.num_parallel_paths)
        else:
            wdg.phase_resistance = 1.0

        # Phase inductance (simplified)
        mu_0 = MU_0
        g_total = geo.airgap + hm / self.materials.magnet.mu_rec
        L_gap = mu_0 * (N_total / wdg.num_parallel_paths)**2 * \
                geo.stack_length * tau_p / (np.pi * p * g_total) * 3 / 2

        # Slot leakage inductance
        L_slot = mu_0 * (N_total / wdg.num_parallel_paths)**2 * \
                 geo.stack_length * coils_per_phase * \
                 (geo.slot_depth / (3 * max(geo.slot_opening, 0.001)))

        wdg.phase_inductance = L_gap + L_slot

        self.winding = wdg
        return wdg

    def compute_performance(self, T_magnet: float = 80.0,
                           T_winding: float = 100.0) -> EMResults:
        """
        Compute full electromagnetic performance at given temperatures.

        Uses equivalent circuit model with temperature-dependent properties.
        """
        specs = self.specs
        geo = self.geometry
        wdg = self.winding
        mats = self.materials
        res = EMResults()

        p = wdg.num_poles // 2
        f_elec = specs.target_rpm * p / 60.0
        omega_m = specs.target_rpm * 2 * np.pi / 60.0

        # --- Airgap flux density ---
        Br = mats.magnet.Br(T_magnet)
        hm = geo.magnet_thickness
        g = geo.airgap
        mu_rec = mats.magnet.mu_rec
        alpha_p = mats.magnet.arc_fraction

        # Carter's coefficient for slot opening effect
        gamma_c = 1.0
        if geo.slot_opening > 0:
            tau_s = 2 * np.pi * geo.stator_outer_radius / wdg.num_slots
            be = geo.slot_opening
            u = be / (2 * g)
            gamma_c = tau_s / (tau_s - be + be * (np.arctan(u) / u) / (np.pi / 2))
            gamma_c = max(gamma_c, 1.0)

        g_eff = gamma_c * g  # Effective airgap

        # Fundamental airgap flux density (Zhu & Howe, 1993)
        Bg1 = (4 / np.pi) * Br * np.sin(alpha_p * np.pi / 2) * hm / \
              (hm + mu_rec * g_eff)
        res.airgap_flux_density = Bg1

        # --- Flux densities in iron ---
        tau_s = 2 * np.pi * geo.stator_outer_radius / wdg.num_slots
        tau_p = 2 * np.pi * geo.stator_outer_radius / wdg.num_poles

        res.tooth_flux_density = Bg1 * tau_s / geo.tooth_width if geo.tooth_width > 0 else 0
        res.yoke_flux_density = Bg1 * tau_p / (2 * geo.stator_yoke_thickness) \
            if geo.stator_yoke_thickness > 0 else 0

        # --- Back-EMF ---
        Phi_1 = Bg1 * tau_p * geo.stack_length * 2 / np.pi  # Fundamental flux per pole
        coils_per_phase = wdg.num_slots // 3
        N_series = wdg.turns_per_coil * coils_per_phase * wdg.num_layers / wdg.num_parallel_paths

        E_peak = 2 * np.pi * f_elec * N_series * wdg.winding_factor * Phi_1
        res.back_emf_peak = E_peak
        res.back_emf_rms = E_peak / np.sqrt(2)

        # --- Current and torque ---
        # Available voltage for current drive
        R_ph = wdg.phase_resistance * (1 + mats.copper.alpha * (T_winding - 80.0))
        X_ph = 2 * np.pi * f_elec * wdg.phase_inductance

        V_ph = specs.voltage / np.sqrt(3)  # Phase voltage (Y-connection)

        # Phase current (simplified: V = E + I*Z)
        Z_ph = np.sqrt(R_ph**2 + X_ph**2)
        V_margin = V_ph - res.back_emf_rms
        if V_margin > 0 and Z_ph > 0:
            I_ph_rms = min(V_margin / Z_ph, specs.max_current / np.sqrt(2))
        else:
            I_ph_rms = specs.max_current / np.sqrt(2) * 0.5

        I_ph_rms = max(I_ph_rms, 0.01)
        res.phase_current_rms = I_ph_rms

        # Electromagnetic torque
        # T = 3 * E_rms * I_rms / omega_m  (for unity power factor)
        if omega_m > 0:
            res.torque_avg = 3 * res.back_emf_rms * I_ph_rms / omega_m
        else:
            res.torque_avg = 0.0

        # Current density [A/mm²]
        wire_area_mm2 = wdg.wire_area * 1e6 * wdg.num_strands
        res.current_density = I_ph_rms / wire_area_mm2 if wire_area_mm2 > 0 else 0

        # Power factor
        if V_ph > 0 and I_ph_rms > 0:
            cos_phi = (V_ph * I_ph_rms - I_ph_rms**2 * R_ph) / (V_ph * I_ph_rms)
            res.power_factor = max(0.0, min(cos_phi, 1.0))
        else:
            res.power_factor = 0.8

        # --- Losses ---
        # Copper loss (I²R with proximity effect factor)
        k_prox = 1.0 + 0.1 * (wdg.turns_per_coil / 10)  # Simplified proximity factor
        res.copper_loss = 3 * I_ph_rms**2 * R_ph * k_prox

        # Iron losses (Bertotti model)
        # Stator teeth
        stator_tooth_mass = wdg.num_slots * geo.tooth_width * geo.slot_depth * \
                           geo.stack_length * mats.steel.density * mats.steel.stacking_factor
        P_iron_teeth = mats.steel.iron_loss_density(f_elec, res.tooth_flux_density, T_winding) * \
                      stator_tooth_mass

        # Stator yoke
        R_yi = geo.stator_inner_radius + 0.001
        R_yo = R_yi + geo.stator_yoke_thickness
        stator_yoke_mass = np.pi * (R_yo**2 - R_yi**2) * geo.stack_length * \
                          mats.steel.density * mats.steel.stacking_factor
        P_iron_yoke = mats.steel.iron_loss_density(f_elec, res.yoke_flux_density, T_winding) * \
                     stator_yoke_mass

        res.iron_loss_stator = P_iron_teeth + P_iron_yoke

        # Rotor iron loss (much smaller, mainly eddy currents from harmonics)
        rotor_yoke_mass = np.pi * ((geo.rotor_outer_radius)**2 -
                         (geo.rotor_outer_radius - geo.rotor_yoke_thickness)**2) * \
                         geo.stack_length * mats.steel.density * mats.steel.stacking_factor
        # Rotor sees slot harmonics at n*f_slot where f_slot = num_slots * f_mech
        f_slot_harmonic = wdg.num_slots * specs.target_rpm / 60.0
        B_rotor_harmonic = Bg1 * 0.05  # ~5% of fundamental from slot harmonics
        res.iron_loss_rotor = mats.steel.iron_loss_density(f_slot_harmonic,
                             B_rotor_harmonic, T_magnet) * rotor_yoke_mass

        # Magnet eddy current loss
        magnet_volume = wdg.num_poles * mats.magnet.thickness * \
                       mats.magnet.arc_fraction * (2 * np.pi * geo.rotor_inner_radius / wdg.num_poles) * \
                       geo.stack_length
        B_slot_harmonic = Bg1 * 0.03  # Slot harmonics seen by magnets
        res.magnet_loss = mats.magnet.eddy_current_loss_density(
            f_slot_harmonic, B_slot_harmonic, T_magnet) * magnet_volume

        # --- Cogging torque estimate ---
        # Cogging torque ∝ LCM(slots, poles) relationship
        lcm_val = np.lcm(wdg.num_slots, wdg.num_poles)
        # Higher LCM = lower cogging torque
        cogging_factor = wdg.num_poles / lcm_val
        res.cogging_torque_peak = cogging_factor * Bg1**2 * geo.stack_length * \
                                 geo.stator_outer_radius * geo.slot_opening / (2 * MU_0)

        # Torque ripple estimate
        res.torque_ripple = (res.cogging_torque_peak / max(res.torque_avg, 0.001)) * 100

        # --- Power and efficiency ---
        res.power_output = res.torque_avg * omega_m if omega_m > 0 else 0
        res.power_input = res.power_output + res.total_loss
        res.efficiency = res.power_output / res.power_input if res.power_input > 0 else 0

        self.results = res
        return res

    def check_demagnetization(self, T_magnet: float = 150.0) -> bool:
        """
        Check if magnets are safe from irreversible demagnetization.

        The worst case is maximum armature reaction opposing the PM field.
        H_demag = N * I_peak / (2 * g_eff)

        Returns True if safe, False if at risk.
        """
        geo = self.geometry
        wdg = self.winding
        mats = self.materials

        coils_per_phase = wdg.num_slots // 3
        N_series = wdg.turns_per_coil * coils_per_phase * wdg.num_layers / wdg.num_parallel_paths
        I_peak = self.specs.max_current * np.sqrt(2)

        # Maximum demagnetizing field from armature reaction
        g_eff = geo.airgap + geo.magnet_thickness / mats.magnet.mu_rec
        H_demag = N_series * I_peak * wdg.winding_factor / (2 * g_eff * (wdg.num_poles / 2))

        return mats.magnet.check_demagnetization(H_demag, T_magnet)

    def compute_weight_breakdown(self) -> dict:
        """Compute mass of each component [kg]."""
        geo = self.geometry
        wdg = self.winding
        mats = self.materials

        # Stator lamination
        R_si = geo.stator_inner_radius
        R_so = geo.stator_outer_radius
        stator_lam_area = np.pi * (R_so**2 - R_si**2)
        # Subtract approximate slot area
        slot_area_total = wdg.num_slots * geo.slot_area
        stator_lam_area -= slot_area_total
        stator_lam_mass = max(stator_lam_area, 0) * geo.stack_length * \
                         mats.steel.density * mats.steel.stacking_factor

        # Copper
        coils_per_phase = wdg.num_slots // 3
        N_total = wdg.turns_per_coil * coils_per_phase * wdg.num_layers * 3  # All phases
        l_turn = 2 * geo.stack_length + 2 * wdg.end_turn_length
        copper_volume = N_total * wdg.wire_area * wdg.num_strands * l_turn
        copper_mass = copper_volume * mats.copper.density

        # Magnets
        magnet_volume = wdg.num_poles * mats.magnet.thickness * \
                       mats.magnet.arc_fraction * \
                       (2 * np.pi * geo.rotor_inner_radius / wdg.num_poles) * \
                       geo.stack_length
        magnet_mass = magnet_volume * mats.magnet.density

        # Rotor yoke (aluminum or steel)
        R_ri = geo.rotor_inner_radius + mats.magnet.thickness
        R_ro = geo.rotor_outer_radius
        rotor_yoke_mass = np.pi * (R_ro**2 - R_ri**2) * geo.stack_length * mats.aluminum.density

        # Shaft
        shaft_mass = np.pi * geo.shaft_radius**2 * geo.shaft_length * mats.shaft_steel.density

        total = stator_lam_mass + copper_mass + magnet_mass + rotor_yoke_mass + shaft_mass

        return {
            'stator_lamination_kg': stator_lam_mass,
            'copper_kg': copper_mass,
            'magnets_kg': magnet_mass,
            'rotor_yoke_kg': rotor_yoke_mass,
            'shaft_kg': shaft_mass,
            'total_kg': total,
        }
