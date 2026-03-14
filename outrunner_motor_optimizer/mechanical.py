"""
Mechanical / structural analysis module for outrunner BLDC motors.

Covers:
  1. Rotor hoop stress (Timoshenko thick-walled rotating cylinder)
  2. Shaft deflection and critical speed (Dunkerley / Rayleigh)
  3. Bearing load distribution and L10 life
  4. Mounting bolt shear / tensile analysis (VDI 2230 simplified)
  5. Mounting flange bending stress
  6. Vibration & modal analysis (lumped-mass rotordynamic model)
  7. Forced-response from cogging torque and mass imbalance

Key References:
  [1] S.P. Timoshenko, J.N. Goodier, "Theory of Elasticity," 3rd ed.,
      McGraw-Hill, 1970.  (Thick-walled cylinder under centrifugal loading)
  [2] ISO 281:2007, "Rolling bearings — Dynamic load ratings and rating
      life."
  [3] W.C. Young, R.G. Budynas, "Roark's Formulas for Stress and Strain,"
      8th ed., McGraw-Hill, 2012.  (Shaft deflection, plate bending)
  [4] VDI 2230 Part 1, "Systematic calculation of highly stressed bolted
      joints," 2015.  (Bolt preload / shear analysis)
  [5] G. Genta, "Dynamics of Rotating Systems," Springer, 2005.
      (Rotordynamics, critical speed, Campbell diagram)
  [6] S.S. Rao, "Mechanical Vibrations," 6th ed., Pearson, 2017.
      (Lumped-mass vibration, modal analysis)
  [7] D. Staton et al., "Solving the More Difficult Aspects of Electric
      Motor Thermal Analysis," IEEE Trans. Energy Conv., 2005.
      (Combined EM-thermal-mechanical approach)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from .materials import (
    MaterialDatabase, StructuralMaterial, MountingConfig, BearingProperties,
)
from .electromagnetic import GeometryParams, WindingConfig, EMResults


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class RotorStressResults:
    """Centrifugal stress in the rotating rotor shell / magnet ring."""
    hoop_stress_inner: float = 0.0     # [Pa]  at magnet-to-yoke interface
    hoop_stress_outer: float = 0.0     # [Pa]  at rotor OD
    radial_stress_max: float = 0.0     # [Pa]
    magnet_retention_margin: float = 0.0  # safety factor (>1 is OK)
    max_safe_rpm: float = 0.0         # RPM at which SF = 1.0


@dataclass
class ShaftResults:
    """Shaft bending / deflection / critical speed."""
    max_deflection: float = 0.0       # [m]
    critical_speed_1st: float = 0.0   # [RPM]
    critical_speed_ratio: float = 0.0 # operating RPM / 1st critical
    is_subcritical: bool = True
    bending_stress: float = 0.0       # [Pa]
    torsional_stress: float = 0.0     # [Pa]
    combined_stress: float = 0.0      # [Pa] von Mises
    shaft_safety_factor: float = 0.0


@dataclass
class BearingResults:
    """Per-bearing loads and life."""
    load_drive_end: float = 0.0       # [N]
    load_non_drive_end: float = 0.0   # [N]
    l10_drive_hours: float = 0.0
    l10_non_drive_hours: float = 0.0
    system_l10_hours: float = 0.0
    friction_loss_total: float = 0.0  # [W]
    speed_ok: bool = True


@dataclass
class MountingResults:
    """Mounting bolt / flange analysis."""
    bolt_shear_stress: float = 0.0       # [Pa]
    bolt_tensile_stress: float = 0.0     # [Pa]
    bolt_combined_safety_factor: float = 0.0
    flange_bending_stress: float = 0.0   # [Pa]
    flange_safety_factor: float = 0.0
    min_bolt_preload: float = 0.0        # [N]
    pull_out_force: float = 0.0          # [N]
    is_safe: bool = True


@dataclass
class VibrationResults:
    """Vibration / modal analysis."""
    natural_frequencies: List[float] = field(default_factory=list)  # [Hz]
    damping_ratios: List[float] = field(default_factory=list)
    cogging_excitation_freq: float = 0.0   # [Hz]
    imbalance_excitation_freq: float = 0.0 # [Hz]  = 1x RPM
    resonance_margin: float = 0.0          # smallest |f_nat - f_exc| / f_nat
    forced_amplitude_mm: float = 0.0       # [mm] peak displacement at mount
    forced_velocity_mm_s: float = 0.0      # [mm/s] RMS velocity (ISO 10816)
    iso_10816_zone: str = "A"              # A/B/C/D per ISO 10816


@dataclass
class MechanicalResults:
    """Aggregate mechanical analysis."""
    rotor_stress: RotorStressResults = field(default_factory=RotorStressResults)
    shaft: ShaftResults = field(default_factory=ShaftResults)
    bearings: BearingResults = field(default_factory=BearingResults)
    mounting: MountingResults = field(default_factory=MountingResults)
    vibration: VibrationResults = field(default_factory=VibrationResults)
    is_mechanically_safe: bool = True


# ---------------------------------------------------------------------------
# Mechanical model
# ---------------------------------------------------------------------------
class MechanicalModel:
    """
    Coupled structural / rotordynamic analysis for outrunner motors.
    """

    def __init__(self, geometry: GeometryParams, winding: WindingConfig,
                 materials: MaterialDatabase):
        self.geo = geometry
        self.wdg = winding
        self.mats = materials
        self.results = MechanicalResults()

    # ===================================================================
    # 1.  Rotor centrifugal stress  (Timoshenko rotating cylinder)
    # ===================================================================
    def analyse_rotor_stress(self, rpm: float) -> RotorStressResults:
        """
        Hoop and radial stress in the outrunner rotor (magnet ring + yoke).

        Thick-walled cylinder under centrifugal loading (Timoshenko [1]):
          σ_θ = ρ ω² [(3+ν)/8 (a²+b²+a²b²/r²) - (1+3ν)/8 r²]
          σ_r = ρ ω² [(3+ν)/8 (a²+b² - a²b²/r² - r²)]

        where a = inner radius, b = outer radius.
        """
        res = RotorStressResults()
        omega = rpm * 2 * np.pi / 60.0

        # Rotor shell (magnets bonded to aluminium yoke)
        a = self.geo.rotor_inner_radius       # magnet inner surface
        b = self.geo.rotor_outer_radius       # yoke outer surface
        rho = self.mats.rotor_housing.density  # 6061-T6
        nu = self.mats.rotor_housing.poissons_ratio

        # Hoop stress at inner surface (r = a) — worst case
        r = a
        sig_theta_inner = rho * omega**2 * (
            (3 + nu) / 8 * (a**2 + b**2 + a**2 * b**2 / r**2)
            - (1 + 3 * nu) / 8 * r**2
        )
        # Hoop stress at outer surface (r = b)
        r = b
        sig_theta_outer = rho * omega**2 * (
            (3 + nu) / 8 * (a**2 + b**2 + a**2 * b**2 / r**2)
            - (1 + 3 * nu) / 8 * r**2
        )

        # Max radial stress at r = sqrt(a*b)
        r_max_sig_r = np.sqrt(a * b)
        sig_r_max = rho * omega**2 * (3 + nu) / 8 * (
            a**2 + b**2 - a**2 * b**2 / r_max_sig_r**2 - r_max_sig_r**2
        )

        res.hoop_stress_inner = abs(sig_theta_inner)
        res.hoop_stress_outer = abs(sig_theta_outer)
        res.radial_stress_max = abs(sig_r_max)

        # Magnet retention: adhesive must withstand the magnet's centrifugal force
        # F_cent_per_magnet = m_mag * omega² * r_mag_cg
        mag_arc = self.mats.magnet.arc_fraction * 2 * np.pi / self.wdg.num_poles
        mag_area = mag_arc * (a + self.mats.magnet.thickness / 2) * self.geo.stack_length
        mag_vol = mag_area * self.mats.magnet.thickness
        mag_mass = mag_vol * self.mats.magnet.density
        r_cg = a + self.mats.magnet.thickness / 2
        F_cent = mag_mass * omega**2 * r_cg

        # Allowable = adhesive shear strength * bond area (~5 MPa for good epoxy)
        adhesive_shear = 5.0e6  # [Pa] conservative structural epoxy
        bond_area = mag_area  # inner + outer faces ≈ inner face only (conservative)
        F_allowable = adhesive_shear * bond_area

        res.magnet_retention_margin = F_allowable / max(F_cent, 1e-6)

        # Max safe RPM: solve for omega where margin = 1.0
        if mag_mass * r_cg > 0:
            omega_safe = np.sqrt(F_allowable / (mag_mass * r_cg))
            res.max_safe_rpm = omega_safe * 60 / (2 * np.pi)
        else:
            res.max_safe_rpm = 1e6

        self.results.rotor_stress = res
        return res

    # ===================================================================
    # 2.  Shaft deflection & critical speed
    # ===================================================================
    def analyse_shaft(self, rpm: float, torque: float) -> ShaftResults:
        """
        Shaft bending, torsion, and first critical speed.

        Uses Dunkerley's method for 1st critical:
          1/ω²_c = Σ (1/ω²_i)

        with each lumped mass on a simply-supported shaft.
        Ref: Genta (2005) [5], Rao (2017) [6].
        """
        res = ShaftResults()
        geo = self.geo
        mat = self.mats.shaft

        R = geo.shaft_radius
        L = geo.shaft_length
        E = mat.youngs_modulus
        I_shaft = np.pi * R**4 / 4          # Second moment of area
        J_shaft = np.pi * R**4 / 2          # Polar moment of area

        # Rotor mass (magnets + yoke + housing)
        rotor_mass = self._rotor_mass()
        # Stator mass (on shaft, acts at midspan)
        stator_mass = self._stator_mass()

        # Bearing span: assume bearings at 0 and L
        a_rotor = L / 2  # Rotor CG at midspan
        b_rotor = L - a_rotor

        # --- Bending deflection at midspan (simply supported, central load) ---
        # δ = W * a² * b² / (3 * E * I * L)
        if L > 0 and E * I_shaft > 0:
            W = (rotor_mass + stator_mass) * 9.81 + self.mats.mounting.radial_load
            delta = W * (L / 2)**2 * (L / 2)**2 / (3 * E * I_shaft * L)
        else:
            delta = 0
            W = 0
        res.max_deflection = abs(delta)

        # --- Critical speed (Dunkerley) ---
        # ω_i² = 48 * E * I / (m * L³)  for central mass on simply-supported beam
        if L > 0 and (rotor_mass + stator_mass) > 0:
            omega_1_sq = 48 * E * I_shaft / ((rotor_mass + stator_mass) * L**3)
            omega_1 = np.sqrt(max(omega_1_sq, 0))
            res.critical_speed_1st = omega_1 * 60 / (2 * np.pi)
        else:
            res.critical_speed_1st = 1e6

        omega_op = rpm * 2 * np.pi / 60.0
        res.critical_speed_ratio = rpm / max(res.critical_speed_1st, 1)
        res.is_subcritical = res.critical_speed_ratio < 0.7

        # --- Bending stress ---
        M_bend = W * L / 4  # Max bending moment at midspan
        M_bend += self.mats.mounting.bending_moment
        res.bending_stress = M_bend * R / I_shaft if I_shaft > 0 else 0

        # --- Torsional stress ---
        res.torsional_stress = torque * R / J_shaft if J_shaft > 0 else 0

        # --- Von Mises combined ---
        res.combined_stress = np.sqrt(
            res.bending_stress**2 + 3 * res.torsional_stress**2
        )
        res.shaft_safety_factor = mat.yield_strength / max(res.combined_stress, 1)

        self.results.shaft = res
        return res

    # ===================================================================
    # 3.  Bearing loads
    # ===================================================================
    def analyse_bearings(self, rpm: float) -> BearingResults:
        """
        Two-bearing simply-supported load distribution.
        System L10 per ISO 281:  L10_sys = (Σ L10_i^-w)^(-1/w), w ≈ 10/9.
        """
        res = BearingResults()
        geo = self.geo
        L = geo.shaft_length
        mats = self.mats

        total_weight = (self._rotor_mass() + self._stator_mass()) * 9.81
        F_ext = np.sqrt(mats.mounting.radial_load**2 +
                        mats.mounting.axial_load**2)

        # Reaction forces (simply supported)
        res.load_drive_end = total_weight / 2 + F_ext * 0.6
        res.load_non_drive_end = total_weight / 2 + F_ext * 0.4

        res.l10_drive_hours = mats.bearing_drive.l10_life_hours(
            res.load_drive_end, rpm)
        res.l10_non_drive_hours = mats.bearing_non_drive.l10_life_hours(
            res.load_non_drive_end, rpm)

        # System life (Lundberg-Palmgren system reliability)
        w = 10.0 / 9.0
        if res.l10_drive_hours > 0 and res.l10_non_drive_hours > 0:
            res.system_l10_hours = (
                res.l10_drive_hours**(-w) + res.l10_non_drive_hours**(-w)
            ) ** (-1.0 / w)
        else:
            res.system_l10_hours = min(res.l10_drive_hours,
                                       res.l10_non_drive_hours)

        res.friction_loss_total = (
            mats.bearing_drive.friction_loss(res.load_drive_end, rpm) +
            mats.bearing_non_drive.friction_loss(res.load_non_drive_end, rpm)
        )

        res.speed_ok = (rpm <= mats.bearing_drive.max_speed and
                        rpm <= mats.bearing_non_drive.max_speed)

        self.results.bearings = res
        return res

    # ===================================================================
    # 4.  Mounting bolt / flange analysis
    # ===================================================================
    def analyse_mounting(self, rpm: float, torque: float) -> MountingResults:
        """
        Bolt shear, tensile, and flange bending per VDI 2230 (simplified).

        Shear on each bolt:
          τ = F_shear / (n * A_bolt)
        where F_shear comes from torque reaction + radial load.

        Tensile from axial load + bending moment:
          σ_t = F_axial / (n * A_bolt) + M * r / (n * A_bolt * r²_bolt_circle)

        Flange bending: annular plate with central hole, uniform load.
        Ref: Roark [3], Table 11.2, Case 1a.
        """
        res = MountingResults()
        mt = self.mats.mounting
        mat = mt.material
        n = mt.num_bolts
        d_bolt = mt.bolt_diameter
        A_bolt = np.pi * (d_bolt / 2)**2
        r_bc = mt.bolt_circle_diameter / 2

        # --- Bolt shear ---
        # Shear from torque reaction
        if r_bc > 0 and n > 0:
            F_torque_per_bolt = torque / (n * r_bc)
        else:
            F_torque_per_bolt = 0

        # Shear from radial load
        F_radial_per_bolt = mt.radial_load / max(n, 1)

        F_shear_total = np.sqrt(F_torque_per_bolt**2 + F_radial_per_bolt**2)
        res.bolt_shear_stress = F_shear_total / A_bolt if A_bolt > 0 else 0

        # --- Bolt tensile ---
        F_axial_per_bolt = mt.axial_load / max(n, 1)

        # Bending contribution: bolt furthest from neutral axis
        if r_bc > 0 and n > 0:
            F_bending_per_bolt = mt.bending_moment / (n * r_bc)
        else:
            F_bending_per_bolt = 0

        F_tensile_total = F_axial_per_bolt + F_bending_per_bolt

        # Add dynamic factor (centrifugal pull on outrunner rotor)
        omega = rpm * 2 * np.pi / 60.0
        rotor_mass = self._rotor_mass()
        # Worst-case imbalance force: ISO G2.5 quality
        e_imbalance = 2.5 / (rpm / 60.0) * 1e-3 if rpm > 0 else 0  # [m]
        F_imbalance = rotor_mass * omega**2 * e_imbalance
        F_tensile_total += F_imbalance / max(n, 1)

        res.bolt_tensile_stress = F_tensile_total / A_bolt if A_bolt > 0 else 0

        # Bolt strength: 8.8 grade → σ_y ≈ 640 MPa, σ_u ≈ 800 MPa
        bolt_yield = mt.bolt_grade * 1e8 * 0.8  # simplified
        bolt_shear_allow = bolt_yield * 0.577  # von Mises

        # Combined check (von Mises on the bolt)
        sig_vm = np.sqrt(res.bolt_tensile_stress**2 +
                         3 * res.bolt_shear_stress**2)
        res.bolt_combined_safety_factor = bolt_yield / max(sig_vm, 1)

        # Minimum preload (to prevent joint separation under dynamic load)
        res.min_bolt_preload = 1.5 * F_tensile_total
        # Pull-out (thread strip) — simplified
        thread_engagement = d_bolt * 1.5  # standard rule of thumb
        thread_area = np.pi * d_bolt * thread_engagement * 0.6  # 60% shear area
        res.pull_out_force = mat.shear_strength * thread_area

        # --- Flange bending ---
        # Annular plate, uniformly loaded, clamped at bolt circle
        # Max stress at inner edge: σ = k * q * (R/t)²
        # where k ≈ 0.75 for clamped annular plate (Roark Case 1a)
        R_flange = mt.flange_outer_diameter / 2
        t = mt.flange_thickness
        total_flange_load = (mt.axial_load + rotor_mass * 9.81 +
                             F_imbalance)
        if R_flange > 0 and t > 0:
            q = total_flange_load / (np.pi * R_flange**2)
            k_plate = 0.75
            res.flange_bending_stress = k_plate * q * (R_flange / t)**2
        else:
            res.flange_bending_stress = 0

        res.flange_safety_factor = (mat.yield_strength /
                                    max(res.flange_bending_stress, 1))

        res.is_safe = (res.bolt_combined_safety_factor > 1.5 and
                       res.flange_safety_factor > 1.5)

        self.results.mounting = res
        return res

    # ===================================================================
    # 5.  Vibration & modal analysis
    # ===================================================================
    def analyse_vibration(self, rpm: float,
                          em_results: Optional[EMResults] = None
                          ) -> VibrationResults:
        """
        Lumped-mass vibration model (3-DOF: axial, radial, torsional).

        Natural frequencies from eigenvalue analysis of:
          [M]{ẍ} + [C]{ẋ} + [K]{x} = {F(t)}

        Excitation sources:
          - Cogging torque ripple at f_cog = LCM(slots, poles) * f_mech
          - Mass imbalance at f_imb = f_mech  (1× per revolution)
          - Slot-passing frequency: f_slot = n_slots * f_mech

        Ref: Rao (2017) [6], Genta (2005) [5].
        """
        res = VibrationResults()
        geo = self.geo
        mats = self.mats
        mt = mats.mounting

        omega_mech = rpm * 2 * np.pi / 60.0
        f_mech = rpm / 60.0

        # --- Lumped masses ---
        m_rotor = self._rotor_mass()
        m_stator = self._stator_mass()
        m_total = m_rotor + m_stator

        # --- Stiffness matrix (simplified 3-DOF) ---
        # DOF 0: radial (y)
        # DOF 1: axial (z)
        # DOF 2: torsional (θ)

        # Radial stiffness: bearings in parallel + mount isolator in series
        k_bearing_radial = (mats.bearing_drive.radial_stiffness +
                            mats.bearing_non_drive.radial_stiffness)
        if mt.isolator_stiffness > 0:
            k_radial = 1.0 / (1.0 / k_bearing_radial +
                               1.0 / mt.isolator_stiffness)
        else:
            k_radial = k_bearing_radial

        # Axial stiffness: bearing axial stiffness ≈ 0.5 * radial
        k_axial = 0.5 * k_bearing_radial
        if mt.isolator_stiffness > 0:
            k_axial = 1.0 / (1.0 / k_axial + 1.0 / mt.isolator_stiffness)

        # Torsional stiffness: shaft torsional + coupling
        G = mats.shaft.shear_modulus
        J = np.pi * geo.shaft_radius**4 / 2
        k_torsional = G * J / max(geo.shaft_length, 0.01)

        # Polar MOI of rotor about shaft axis
        R_ri = geo.rotor_inner_radius
        R_ro = geo.rotor_outer_radius
        I_polar_rotor = 0.5 * m_rotor * (R_ri**2 + R_ro**2)

        M = np.diag([m_total, m_total, I_polar_rotor])
        K = np.diag([k_radial, k_axial, k_torsional])

        # Damping (proportional: C = α*M + β*K)
        zeta_struct = 0.02  # 2% structural damping
        if mt.isolator_damping > 0:
            c_isolator = mt.isolator_damping
        else:
            c_isolator = 0

        # Eigenvalue analysis: det(K - ω²M) = 0
        # For diagonal system this is trivial:
        nat_freqs_rad = np.sqrt(np.diag(K) / np.diag(M))
        nat_freqs_hz = nat_freqs_rad / (2 * np.pi)
        res.natural_frequencies = sorted(nat_freqs_hz.tolist())

        # Damping ratios
        for i, wn in enumerate(nat_freqs_rad):
            c_crit = 2 * np.sqrt(K[i, i] * M[i, i])
            c_total = 2 * zeta_struct * c_crit + c_isolator
            zeta = c_total / c_crit if c_crit > 0 else zeta_struct
            res.damping_ratios.append(min(zeta, 1.0))

        # --- Excitation frequencies ---
        lcm_val = int(np.lcm(self.wdg.num_slots, self.wdg.num_poles))
        res.cogging_excitation_freq = lcm_val * f_mech
        res.imbalance_excitation_freq = f_mech

        # Excitation list
        excitations = [
            res.imbalance_excitation_freq,      # 1× RPM
            2 * res.imbalance_excitation_freq,   # 2× RPM (misalignment)
            res.cogging_excitation_freq,          # cogging
            self.wdg.num_slots * f_mech,          # slot passing
        ]

        # Resonance margin: smallest normalised gap
        res.resonance_margin = float("inf")
        for fn in res.natural_frequencies:
            for fe in excitations:
                if fn > 0 and fe > 0:
                    margin = abs(fn - fe) / fn
                    if margin < res.resonance_margin:
                        res.resonance_margin = margin

        # --- Forced response (imbalance) ---
        # ISO G2.5 residual imbalance
        if f_mech > 0:
            e_imb = 2.5 / f_mech * 1e-3  # [m]
        else:
            e_imb = 0

        F0 = m_rotor * omega_mech**2 * e_imb
        # Radial response at 1× RPM
        wn_rad = nat_freqs_rad[0] if len(nat_freqs_rad) > 0 else 1e6
        zeta_rad = res.damping_ratios[0] if res.damping_ratios else 0.02

        r_ratio = omega_mech / max(wn_rad, 1e-6)
        H = 1.0 / np.sqrt((1 - r_ratio**2)**2 + (2 * zeta_rad * r_ratio)**2)
        X0 = F0 * H / max(k_radial, 1)

        res.forced_amplitude_mm = abs(X0) * 1e3
        res.forced_velocity_mm_s = abs(X0) * omega_mech * 1e3 / np.sqrt(2)

        # ISO 10816 classification (small machines, Class I)
        v = res.forced_velocity_mm_s
        if v <= 0.71:
            res.iso_10816_zone = "A"
        elif v <= 1.8:
            res.iso_10816_zone = "B"
        elif v <= 4.5:
            res.iso_10816_zone = "C"
        else:
            res.iso_10816_zone = "D"

        # --- Cogging-induced torsional vibration ---
        if em_results is not None and len(nat_freqs_rad) >= 3:
            T_cog = em_results.cogging_torque_peak
            wn_tors = nat_freqs_rad[2]
            zeta_tors = res.damping_ratios[2] if len(res.damping_ratios) > 2 else 0.02
            omega_cog = 2 * np.pi * res.cogging_excitation_freq
            r_t = omega_cog / max(wn_tors, 1e-6)
            H_t = 1.0 / np.sqrt((1 - r_t**2)**2 + (2 * zeta_tors * r_t)**2)
            theta_peak = T_cog * H_t / max(k_torsional, 1)
            # Convert angular to linear at rotor OD
            linear_cog_mm = theta_peak * geo.rotor_outer_radius * 1e3
            # Add to total
            res.forced_amplitude_mm = np.sqrt(
                res.forced_amplitude_mm**2 + linear_cog_mm**2
            )

        self.results.vibration = res
        return res

    # ===================================================================
    # Full analysis
    # ===================================================================
    def run_full_analysis(self, rpm: float, torque: float,
                          em_results: Optional[EMResults] = None
                          ) -> MechanicalResults:
        """Run all mechanical sub-analyses and aggregate."""
        self.analyse_rotor_stress(rpm)
        self.analyse_shaft(rpm, torque)
        self.analyse_bearings(rpm)
        self.analyse_mounting(rpm, torque)
        self.analyse_vibration(rpm, em_results)

        r = self.results
        r.is_mechanically_safe = (
            r.rotor_stress.magnet_retention_margin > 1.5 and
            r.shaft.shaft_safety_factor > 2.0 and
            r.bearings.speed_ok and
            r.mounting.is_safe and
            r.vibration.iso_10816_zone in ("A", "B")
        )
        return r

    # ===================================================================
    # Helpers
    # ===================================================================
    def _rotor_mass(self) -> float:
        geo = self.geo
        mats = self.mats
        # Magnets
        n_poles = self.wdg.num_poles
        mag_arc = mats.magnet.arc_fraction * 2 * np.pi * geo.rotor_inner_radius / n_poles
        mag_vol = n_poles * mag_arc * mats.magnet.thickness * geo.stack_length
        m_mag = mag_vol * mats.magnet.density

        # Rotor yoke (structural aluminium)
        R_ri = geo.rotor_inner_radius + mats.magnet.thickness
        R_ro = geo.rotor_outer_radius
        m_yoke = np.pi * (R_ro**2 - R_ri**2) * geo.stack_length * mats.rotor_housing.density
        return m_mag + m_yoke

    def _stator_mass(self) -> float:
        geo = self.geo
        mats = self.mats
        R_si = geo.stator_inner_radius
        R_so = geo.stator_outer_radius
        area = np.pi * (R_so**2 - R_si**2) * 0.7  # 70 % fill
        m_lam = area * geo.stack_length * mats.steel.density * mats.steel.stacking_factor

        # Copper
        wdg = self.wdg
        cpp = wdg.num_slots // 3
        N_tot = wdg.turns_per_coil * cpp * wdg.num_layers * 3
        l_turn = 2 * geo.stack_length + 2 * wdg.end_turn_length
        cu_vol = N_tot * wdg.wire_area * wdg.num_strands * l_turn
        m_cu = cu_vol * mats.copper.density
        return m_lam + m_cu
