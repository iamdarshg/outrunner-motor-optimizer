"""
2-D Finite Element Electromagnetic Solver for Outrunner BLDC Motors.

Solves the magnetostatic vector-potential equation on a 2-D cross-section:

    -∇·(ν ∇A_z) = J_z  + ∇×(ν M)          in Ω
                A_z = 0                       on ∂Ω

where:
  A_z  — z-component of the magnetic vector potential [Wb/m]
  ν    — magnetic reluctivity (1/μ) [m/H]
  J_z  — impressed current density [A/m²]
  M    — permanent magnet magnetisation [A/m]

The mesh is a polar cross-section of the motor, generated parametrically from
GeometryParams.  Region tags:
  0 — air / airgap
  1 — stator iron (lamination)
  2 — rotor iron (yoke)
  3 — slot copper (winding, +J or −J per phase)
  4 — permanent magnet (remanent Br → equivalent surface current)
  5 — shaft (steel or aluminium)

Post-processing extracts:
  • Airgap flux density (fundamental & harmonics via spatial DFT)
  • Tooth and yoke flux densities (element-average)
  • Back-EMF (from flux linkage vs rotor position)
  • Torque via Maxwell stress tensor integrated on an airgap contour
  • Iron loss from element-level Bertotti model
  • Cogging torque (multi-position sweep)

Key References:
  [1] Z.Q. Zhu, D. Howe, "Instantaneous magnetic field distribution in
      brushless DC motors," IEEE Trans. Magnetics, 1993.
  [2] G. McBain, T. Gustafsson, "scikit-fem: A Python package for finite
      element assembly," JOSS, 2020.
  [3] G. Bertotti, "General Properties of Power Losses in Soft
      Ferromagnetic Materials," IEEE Trans. Magnetics, 1988.
  [4] J. Pyrhonen, T. Jokinen, V. Hrabovcova, "Design of Rotating
      Electrical Machines," Wiley, 2014.  (Ch. 1-4)
  [5] D. Meeker, "Finite Element Method Magnetics (FEMM) Reference Manual,"
      2019.  (Maxwell stress tensor on airgap contour)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import numpy as np
from scipy.sparse.linalg import spsolve

try:
    from skfem import (
        MeshTri, Basis, ElementTriP1, ElementTriP2,
        BilinearForm, LinearForm, asm, enforce, solve,
        InteriorBasis, FacetBasis,
    )
    from skfem.helpers import dot, grad
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False

from .materials import MaterialDatabase
from .electromagnetic import GeometryParams, WindingConfig, EMResults, MotorSpecs

MU_0 = 4.0 * np.pi * 1e-7


# ---------------------------------------------------------------------------
# Mesh generator  (parametric polar cross-section)
# ---------------------------------------------------------------------------
def _generate_motor_mesh(
    geo: GeometryParams,
    wdg: WindingConfig,
    mats: MaterialDatabase,
    n_circum: int = 120,
    n_radial_gap: int = 4,
    rotor_angle_deg: float = 0.0,
) -> Tuple["MeshTri", np.ndarray]:
    """
    Generate a 2-D triangular mesh of the motor cross-section.

    Returns (mesh, region_tags) where region_tags[i] ∈ {0..5} per element.

    Strategy: build concentric annular rings, then triangulate with
    structured quad→split approach for quality control.
    """
    if not HAS_SKFEM:
        raise ImportError("scikit-fem required for FEM EM solver")

    # Radial layers (inside → outside for outrunner)
    r_shaft_inner = 0.0
    r_shaft_outer = geo.shaft_radius
    r_stator_inner = geo.stator_inner_radius
    r_stator_outer = geo.stator_outer_radius
    r_gap_mid = (r_stator_outer + geo.rotor_inner_radius) / 2.0
    r_rotor_inner = geo.rotor_inner_radius
    r_magnet_outer = r_rotor_inner + geo.magnet_thickness
    r_rotor_outer = geo.rotor_outer_radius
    r_domain = r_rotor_outer * 1.3  # some air outside

    # Build structured nodes in (r, theta) then convert
    # Radial breakpoints
    radii = np.array([
        r_shaft_inner + 1e-6,  # avoid r=0 singularity
        r_shaft_outer,
        r_stator_inner,
        r_stator_inner + geo.stator_yoke_thickness,
        r_stator_outer - 0.0001,  # inner edge of airgap
        r_stator_outer,
        r_gap_mid,
        r_rotor_inner,
        r_magnet_outer,
        r_rotor_outer,
        r_domain,
    ])
    radii = np.sort(np.unique(np.clip(radii, 1e-6, None)))

    # Refine each annular layer
    all_r = []
    for i in range(len(radii) - 1):
        nr = max(3, int((radii[i + 1] - radii[i]) / (0.0005)))
        nr = min(nr, 15)
        all_r.append(np.linspace(radii[i], radii[i + 1], nr, endpoint=(i == len(radii) - 2)))
    all_r = np.unique(np.concatenate(all_r))

    n_r = len(all_r)
    n_th = n_circum
    theta = np.linspace(0, 2 * np.pi, n_th, endpoint=False)

    # Node coordinates
    nodes_x = []
    nodes_y = []
    for r in all_r:
        for t in theta:
            nodes_x.append(r * np.cos(t))
            nodes_y.append(r * np.sin(t))

    nodes = np.array([nodes_x, nodes_y])

    # Build quad connectivity → split each quad into 2 triangles
    triangles = []
    for i in range(n_r - 1):
        for j in range(n_th):
            j_next = (j + 1) % n_th
            n0 = i * n_th + j
            n1 = i * n_th + j_next
            n2 = (i + 1) * n_th + j_next
            n3 = (i + 1) * n_th + j
            triangles.append([n0, n1, n3])
            triangles.append([n1, n2, n3])

    elements = np.array(triangles).T

    mesh = MeshTri(nodes, elements)

    # --- Assign region tags ---
    centroids = mesh.p[:, mesh.t].mean(axis=1)  # (2, n_elem)  axis=1 averages over 3 vertices
    cx, cy = centroids[0], centroids[1]
    rc = np.sqrt(cx**2 + cy**2)
    tc = np.arctan2(cy, cx) % (2 * np.pi)

    n_elem = mesh.t.shape[1]
    tags = np.zeros(n_elem, dtype=int)  # default: air (0)

    # Shaft
    tags[rc <= r_shaft_outer] = 5

    # Stator iron (yoke + teeth)
    stator_mask = (rc > r_stator_inner) & (rc < r_stator_outer)

    # Determine slot regions within stator
    n_slots = wdg.num_slots
    slot_angular_width = (2 * np.pi / n_slots) * 0.4  # ~40% of slot pitch is opening
    slot_r_inner = r_stator_inner + geo.stator_yoke_thickness
    slot_r_outer = r_stator_outer - 0.001

    is_slot = np.zeros(n_elem, dtype=bool)
    for s in range(n_slots):
        slot_center = s * 2 * np.pi / n_slots
        # Angular distance considering wrap-around
        dtheta = np.abs((tc - slot_center + np.pi) % (2 * np.pi) - np.pi)
        radial_ok = (rc > slot_r_inner) & (rc < slot_r_outer)
        is_slot |= (dtheta < slot_angular_width / 2) & radial_ok

    tags[stator_mask & ~is_slot] = 1  # stator iron
    tags[stator_mask & is_slot] = 3   # slot copper

    # Rotor iron yoke
    rotor_yoke_mask = (rc > r_magnet_outer) & (rc < r_rotor_outer)
    tags[rotor_yoke_mask] = 2

    # Permanent magnets
    n_poles = wdg.num_poles
    arc_frac = mats.magnet.arc_fraction
    pole_pitch = 2 * np.pi / n_poles
    mag_arc = pole_pitch * arc_frac
    rotor_offset = np.radians(rotor_angle_deg)

    magnet_mask = (rc >= r_rotor_inner) & (rc <= r_magnet_outer)
    for p_idx in range(n_poles):
        center = p_idx * pole_pitch + pole_pitch / 2 + rotor_offset
        dtheta = np.abs((tc - center + np.pi) % (2 * np.pi) - np.pi)
        pole_elements = magnet_mask & (dtheta < mag_arc / 2)
        tags[pole_elements] = 4

    return mesh, tags


def _assign_current_density(
    mesh: "MeshTri",
    tags: np.ndarray,
    wdg: WindingConfig,
    specs: MotorSpecs,
    mats: MaterialDatabase,
    geo: GeometryParams,
) -> np.ndarray:
    """
    Assign current density J_z [A/m²] to each element.
    Concentrated winding: coils around individual teeth, alternating polarity.
    """
    n_elem = mesh.t.shape[1]
    J = np.zeros(n_elem)

    centroids = mesh.p[:, mesh.t].mean(axis=1)
    tc = np.arctan2(centroids[1], centroids[0]) % (2 * np.pi)

    slot_mask = tags == 3
    if not np.any(slot_mask):
        return J

    # Phase assignment (3-phase, concentrated)
    n_slots = wdg.num_slots
    coils_per_phase = n_slots // 3
    I_rms = min(specs.max_current / np.sqrt(2), specs.max_current * 0.7)

    # Current density = N * I / A_slot
    slot_area = geo.slot_area
    if slot_area < 1e-10:
        slot_area = 1e-6
    J_magnitude = wdg.turns_per_coil * I_rms * mats.copper.fill_factor / slot_area

    # Assign ±J to slots in 3-phase pattern
    for s in range(n_slots):
        slot_center = s * 2 * np.pi / n_slots
        phase = s % 3  # phase A=0, B=1, C=2
        coil_idx = s // 3
        # Direction alternates for concentrated windings
        direction = 1 if (s % 2 == 0) else -1

        # Phase current at t=0 (peak of phase A)
        phase_angle = phase * 2 * np.pi / 3
        I_phase = np.cos(phase_angle) * direction

        # Find elements in this slot
        dtheta = np.abs((tc - slot_center + np.pi) % (2 * np.pi) - np.pi)
        slot_width = (2 * np.pi / n_slots) * 0.4 / 2
        in_slot = slot_mask & (dtheta < slot_width)
        J[in_slot] = J_magnitude * I_phase

    return J


def _assign_magnetisation(
    mesh: "MeshTri",
    tags: np.ndarray,
    wdg: WindingConfig,
    mats: MaterialDatabase,
    geo: GeometryParams,
    rotor_angle_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Mx, My) per element for radially-magnetised surface PMs.
    M = Br/μ₀ in the radial direction, alternating N/S poles.
    """
    n_elem = mesh.t.shape[1]
    Mx = np.zeros(n_elem)
    My = np.zeros(n_elem)

    centroids = mesh.p[:, mesh.t].mean(axis=1)
    cx, cy = centroids[0], centroids[1]
    rc = np.sqrt(cx**2 + cy**2)
    tc = np.arctan2(cy, cx) % (2 * np.pi)

    mag_mask = tags == 4
    if not np.any(mag_mask):
        return Mx, My

    Br = mats.magnet.Br(80.0)  # at operating temp
    M_mag = Br / MU_0

    n_poles = wdg.num_poles
    pole_pitch = 2 * np.pi / n_poles
    rotor_offset = np.radians(rotor_angle_deg)

    for p_idx in range(n_poles):
        center = p_idx * pole_pitch + pole_pitch / 2 + rotor_offset
        polarity = 1 if (p_idx % 2 == 0) else -1

        dtheta = np.abs((tc - center + np.pi) % (2 * np.pi) - np.pi)
        arc_half = pole_pitch * mats.magnet.arc_fraction / 2
        pole_elems = mag_mask & (dtheta < arc_half)

        # Radial direction at each element
        cos_t = cx[pole_elems] / np.maximum(rc[pole_elems], 1e-10)
        sin_t = cy[pole_elems] / np.maximum(rc[pole_elems], 1e-10)
        Mx[pole_elems] = polarity * M_mag * cos_t
        My[pole_elems] = polarity * M_mag * sin_t

    return Mx, My


# ---------------------------------------------------------------------------
# FEM solver
# ---------------------------------------------------------------------------
@dataclass
class FEMEMResults:
    """Results from FEM electromagnetic analysis."""
    airgap_Bn: np.ndarray = field(default_factory=lambda: np.array([]))
    airgap_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    Bg1_fundamental: float = 0.0
    Bg_harmonics: np.ndarray = field(default_factory=lambda: np.array([]))

    flux_density_field: np.ndarray = field(default_factory=lambda: np.array([]))
    potential_field: np.ndarray = field(default_factory=lambda: np.array([]))

    tooth_Bmax: float = 0.0
    yoke_Bmax: float = 0.0

    torque_maxwell: float = 0.0
    flux_linkage_a: float = 0.0

    back_emf_peak: float = 0.0
    iron_loss_stator: float = 0.0
    iron_loss_rotor: float = 0.0
    magnet_eddy_loss: float = 0.0
    copper_loss: float = 0.0
    total_loss: float = 0.0
    efficiency: float = 0.0

    mesh_nodes: int = 0
    mesh_elements: int = 0


class FEMElectromagneticModel:
    """
    2-D finite element electromagnetic solver for outrunner PM motors.

    Uses scikit-fem to assemble and solve the magnetostatic A_z equation
    on a triangulated cross-section, then post-processes for motor
    performance parameters.
    """

    def __init__(self, specs: MotorSpecs, geometry: GeometryParams,
                 winding: WindingConfig, materials: MaterialDatabase,
                 mesh_density: int = 120):
        if not HAS_SKFEM:
            raise ImportError(
                "scikit-fem is required for FEM EM analysis. "
                "Install with: pip install scikit-fem[all]"
            )
        self.specs = specs
        self.geo = geometry
        self.wdg = winding
        self.mats = materials
        self.mesh_density = mesh_density
        self.results = FEMEMResults()

    def solve_magnetostatic(self, rotor_angle_deg: float = 0.0,
                            T_magnet: float = 80.0,
                            T_winding: float = 100.0) -> FEMEMResults:
        """
        Solve 2-D magnetostatic problem for a single rotor position.

        Returns FEMEMResults with field solution, airgap flux density,
        torque, and loss breakdown.
        """
        geo = self.geo
        wdg = self.wdg
        mats = self.mats
        specs = self.specs

        # --- 1. Generate mesh ---
        mesh, tags = _generate_motor_mesh(
            geo, wdg, mats,
            n_circum=self.mesh_density,
            rotor_angle_deg=rotor_angle_deg,
        )

        self.results.mesh_nodes = mesh.p.shape[1]
        self.results.mesh_elements = mesh.t.shape[1]

        # --- 2. Material properties per element ---
        n_elem = mesh.t.shape[1]

        # Reluctivity ν = 1/μ for each element
        nu = np.ones(n_elem) / MU_0  # default: air

        # Stator iron — cap relative permeability for linear FEM to
        # approximate saturated operating point (no nonlinear iteration).
        # Unsaturated μᵣ may be 3000-5000, but at typical operating Bₜₒₒₜₕ
        # ≈ 1.4-1.8 T the effective μᵣ ≈ 200-800 for silicon steel.
        mu_r_eff = min(mats.steel.relative_permeability, 800.0)
        mu_iron = MU_0 * mu_r_eff
        nu[tags == 1] = 1.0 / mu_iron

        # Rotor iron (yoke) — same effective permeability
        nu[tags == 2] = 1.0 / mu_iron

        # Magnets — relative permeability ≈ μ_rec
        mu_mag = MU_0 * mats.magnet.mu_rec
        nu[tags == 4] = 1.0 / mu_mag

        # Shaft
        nu[tags == 5] = 1.0 / (MU_0 * 100)  # steel shaft, moderate permeability

        # Copper slots — air-like permeability
        nu[tags == 3] = 1.0 / MU_0

        # --- 3. Source terms ---
        # Current density
        J_z = _assign_current_density(mesh, tags, wdg, specs, mats, geo)

        # Magnetisation (permanent magnets)
        Mx, My = _assign_magnetisation(mesh, tags, wdg, mats, geo, rotor_angle_deg)

        # --- 4. Assemble FEM system ---
        basis = Basis(mesh, ElementTriP1())

        # Use element-wise DiscreteField to preserve sharp material boundaries
        from skfem.element.discrete_field import DiscreteField
        n_qp = basis.X.shape[-1]  # quadrature points per element

        def elem_to_qp(data):
            """Tile element-constant data to (n_elem, n_qp) for DiscreteField."""
            return DiscreteField(np.tile(data[:, None], (1, n_qp)))

        @BilinearForm
        def stiffness_form(u, v, w):
            return w.nu * dot(grad(u), grad(v))

        @LinearForm
        def rhs_form(v, w):
            # Current source + PM source (curl of remanent magnetisation)
            # Weak form: ∫ Jz·v dΩ + ∫ (Mx ∂v/∂y − My ∂v/∂x) dΩ
            # where M = Br/μ₀ in radial direction.  No ν factor here:
            # the ν is already on the LHS bilinear form.
            j_term = w.J * v
            mag_term = w.Mx * grad(v)[1] - w.My * grad(v)[0]
            return j_term + mag_term

        K = stiffness_form.assemble(basis, nu=elem_to_qp(nu))
        f = rhs_form.assemble(
            basis,
            J=elem_to_qp(J_z),
            Mx=elem_to_qp(Mx),
            My=elem_to_qp(My),
        )

        # Boundary condition: A_z = 0 on outer boundary
        boundary = mesh.boundary_nodes()
        K, f = enforce(K, f, D=boundary)

        # --- 5. Solve ---
        A_z = solve(K, f)
        self.results.potential_field = A_z

        # --- 6. Post-process: B = curl(A_z) = (∂A_z/∂y, -∂A_z/∂x) ---
        # Compute gradient at element centroids
        Bx = np.zeros(n_elem)
        By = np.zeros(n_elem)

        for i in range(n_elem):
            n0, n1, n2 = mesh.t[:, i]
            x0, y0 = mesh.p[0, n0], mesh.p[1, n0]
            x1, y1 = mesh.p[0, n1], mesh.p[1, n1]
            x2, y2 = mesh.p[0, n2], mesh.p[1, n2]

            # Area of triangle
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            if area < 1e-20:
                continue

            # Gradient of A_z (constant per element for P1)
            A0, A1, A2 = A_z[n0], A_z[n1], A_z[n2]
            dAdx = ((A0 * (y1 - y2) + A1 * (y2 - y0) + A2 * (y0 - y1))
                    / (2 * area))
            dAdy = ((A0 * (x2 - x1) + A1 * (x0 - x2) + A2 * (x1 - x0))
                    / (2 * area))

            # B = curl(A) for 2D: Bx = ∂Az/∂y, By = -∂Az/∂x
            Bx[i] = dAdy
            By[i] = -dAdx

        B_mag = np.sqrt(Bx**2 + By**2)
        self.results.flux_density_field = B_mag

        # --- 7. Extract airgap flux density ---
        r_gap = (geo.stator_outer_radius + geo.rotor_inner_radius) / 2.0
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        rc = np.sqrt(centroids[0]**2 + centroids[1]**2)
        tc_ang = np.arctan2(centroids[1], centroids[0])

        # Select elements nearest to mid-gap radius (single contour band).
        # Use a narrow band and then pick one element per angular bin.
        gap_tol = geo.airgap * 0.6
        gap_candidates = np.where(np.abs(rc - r_gap) < gap_tol)[0]
        contour_idx = np.array([], dtype=int)

        if len(gap_candidates) > 0:
            # Bin by angle to get one element per angular position
            n_bins = min(self.mesh_density, 360)
            bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            gap_angles = tc_ang[gap_candidates]
            # Wrap to [-pi, pi]
            gap_angles_wrap = np.arctan2(np.sin(gap_angles), np.cos(gap_angles))

            contour_idx = []  # one index per bin
            contour_theta = []
            for b in range(n_bins):
                in_bin = (gap_angles_wrap >= bin_edges[b]) & (gap_angles_wrap < bin_edges[b + 1])
                bin_elem_ids = gap_candidates[in_bin]
                if len(bin_elem_ids) > 0:
                    # pick element closest to r_gap
                    best = bin_elem_ids[np.argmin(np.abs(rc[bin_elem_ids] - r_gap))]
                    contour_idx.append(best)
                    contour_theta.append((bin_edges[b] + bin_edges[b + 1]) / 2)

            contour_idx = np.array(contour_idx)
            contour_theta = np.array(contour_theta)
            sort_order = np.argsort(contour_theta)
            contour_idx = contour_idx[sort_order]
            contour_theta = contour_theta[sort_order]

            # Radial B: B_r = Bx cosθ + By sinθ
            Br_gap = (Bx[contour_idx] * np.cos(contour_theta) +
                      By[contour_idx] * np.sin(contour_theta))

            self.results.airgap_theta = contour_theta
            self.results.airgap_Bn = Br_gap

            # FFT for fundamental
            if len(Br_gap) > 10:
                n_pts = len(Br_gap)
                fft_vals = np.fft.fft(Br_gap)
                harmonics = 2.0 * np.abs(fft_vals[:n_pts // 2]) / n_pts
                p = wdg.num_poles // 2
                if p < len(harmonics):
                    self.results.Bg1_fundamental = harmonics[p]
                else:
                    self.results.Bg1_fundamental = (
                        np.max(harmonics[1:]) if len(harmonics) > 1 else 0
                    )
                self.results.Bg_harmonics = harmonics[:min(50, len(harmonics))]

        # --- 8. Tooth and yoke flux densities ---
        r_tooth_inner = geo.stator_inner_radius + geo.stator_yoke_thickness
        tooth_elements = (tags == 1) & (rc > r_tooth_inner)
        yoke_elements = (tags == 1) & (rc <= r_tooth_inner + geo.stator_yoke_thickness)

        if np.any(tooth_elements):
            # Use 80th percentile to avoid tooth-tip singularities
            self.results.tooth_Bmax = np.percentile(B_mag[tooth_elements], 80)
        if np.any(yoke_elements):
            self.results.yoke_Bmax = np.percentile(B_mag[yoke_elements], 80)

        # --- 9. Torque via Maxwell stress tensor ---
        # T = (L_stack * r² / μ₀) ∮ B_r B_θ dθ  on a single mid-gap contour
        if len(contour_idx) > 3:
            Bt_gap = (-Bx[contour_idx] * np.sin(contour_theta) +
                       By[contour_idx] * np.cos(contour_theta))
            dtheta = 2 * np.pi / len(contour_idx)  # uniform angular spacing
            torque_per_length = (r_gap**2 / MU_0 *
                                 np.sum(Br_gap * Bt_gap) * dtheta)
            self.results.torque_maxwell = torque_per_length * geo.stack_length

        # --- 10. Losses ---
        p = wdg.num_poles // 2
        f_elec = specs.target_rpm * p / 60.0
        omega_m = specs.target_rpm * 2 * np.pi / 60.0

        # Iron loss (Bertotti, element-wise)
        P_iron_stator = 0.0
        P_iron_rotor = 0.0
        for i in range(n_elem):
            area_elem = _triangle_area(mesh, i)
            vol_elem = area_elem * geo.stack_length

            if tags[i] == 1:  # stator iron
                B_elem = B_mag[i]
                p_loss = mats.steel.iron_loss_density(f_elec, B_elem)
                mass_elem = vol_elem * mats.steel.density * mats.steel.stacking_factor
                P_iron_stator += p_loss * mass_elem
            elif tags[i] == 2:  # rotor iron
                B_elem = B_mag[i] * 0.1  # rotor sees slot harmonics
                f_slot = wdg.num_slots * specs.target_rpm / 60.0
                p_loss = mats.steel.iron_loss_density(f_slot, B_elem)
                mass_elem = vol_elem * mats.steel.density * mats.steel.stacking_factor
                P_iron_rotor += p_loss * mass_elem

        self.results.iron_loss_stator = P_iron_stator
        self.results.iron_loss_rotor = P_iron_rotor

        # Copper loss
        I_rms = specs.max_current / np.sqrt(2)
        rho_cu = mats.copper.resistivity(T_winding)
        coils_per_phase = wdg.num_slots // 3
        N_total = wdg.turns_per_coil * coils_per_phase * wdg.num_layers
        l_turn = 2 * geo.stack_length + 2 * wdg.end_turn_length
        wire_area = wdg.wire_area * wdg.num_strands
        if wire_area > 1e-12:
            R_phase = rho_cu * N_total * l_turn / wire_area
        else:
            R_phase = 1.0
        self.results.copper_loss = 3 * I_rms**2 * R_phase

        # Magnet eddy loss
        mag_vol = (wdg.num_poles * mats.magnet.thickness *
                   mats.magnet.arc_fraction *
                   (2 * np.pi * geo.rotor_inner_radius / wdg.num_poles) *
                   geo.stack_length)
        f_slot = wdg.num_slots * specs.target_rpm / 60.0
        B_slot_harmonic = self.results.Bg1_fundamental * 0.03
        self.results.magnet_eddy_loss = (mats.magnet.eddy_current_loss_density(
            f_slot, B_slot_harmonic, T_magnet) * mag_vol)

        self.results.total_loss = (self.results.copper_loss +
                                   self.results.iron_loss_stator +
                                   self.results.iron_loss_rotor +
                                   self.results.magnet_eddy_loss)

        P_out = abs(self.results.torque_maxwell * omega_m)
        P_in = P_out + self.results.total_loss
        self.results.efficiency = P_out / P_in if P_in > 0 else 0

        # Back-EMF estimate from flux linkage
        if self.results.Bg1_fundamental > 0:
            tau_p = np.pi * geo.stator_outer_radius / max(p, 1)
            Phi_1 = self.results.Bg1_fundamental * tau_p * geo.stack_length * 2 / np.pi
            N_s = wdg.turns_per_coil * coils_per_phase * wdg.num_layers / wdg.num_parallel_paths
            self.results.back_emf_peak = 2 * np.pi * f_elec * N_s * wdg.winding_factor * Phi_1
            self.results.flux_linkage_a = N_s * wdg.winding_factor * Phi_1

        return self.results

    def compute_cogging_torque(self, n_positions: int = 24) -> np.ndarray:
        """
        Compute cogging torque by solving at multiple rotor positions
        with zero current (no-load).

        Returns array of (angle_deg, torque_Nm) pairs.
        """
        # Save original current
        orig_current = self.specs.max_current
        self.specs.max_current = 0.0  # no-load

        angles = np.linspace(0, 360.0 / self.wdg.num_poles, n_positions, endpoint=False)
        torques = np.zeros(n_positions)

        for i, angle in enumerate(angles):
            res = self.solve_magnetostatic(rotor_angle_deg=angle)
            torques[i] = res.torque_maxwell

        self.specs.max_current = orig_current
        return np.column_stack([angles, torques])

    def to_em_results(self) -> EMResults:
        """Convert FEM results to the standard EMResults format for compatibility."""
        fem = self.results
        specs = self.specs
        wdg = self.wdg
        geo = self.geo

        p = wdg.num_poles // 2
        omega_m = specs.target_rpm * 2 * np.pi / 60.0

        em = EMResults()
        em.airgap_flux_density = fem.Bg1_fundamental
        em.tooth_flux_density = fem.tooth_Bmax
        em.yoke_flux_density = fem.yoke_Bmax
        em.back_emf_peak = fem.back_emf_peak
        em.back_emf_rms = fem.back_emf_peak / np.sqrt(2)
        em.torque_avg = abs(fem.torque_maxwell)
        em.copper_loss = fem.copper_loss
        em.iron_loss_stator = fem.iron_loss_stator
        em.iron_loss_rotor = fem.iron_loss_rotor
        em.magnet_loss = fem.magnet_eddy_loss
        em.power_output = abs(fem.torque_maxwell * omega_m) if omega_m > 0 else 0
        em.power_input = em.power_output + fem.total_loss
        em.efficiency = em.power_output / em.power_input if em.power_input > 0 else 0

        # Current density
        I_rms = specs.max_current / np.sqrt(2)
        wire_area_mm2 = wdg.wire_area * 1e6 * wdg.num_strands
        em.current_density = I_rms / wire_area_mm2 if wire_area_mm2 > 0 else 0
        em.phase_current_rms = I_rms

        return em


def _triangle_area(mesh, elem_idx):
    """Compute area of a triangle element."""
    n0, n1, n2 = mesh.t[:, elem_idx]
    x0, y0 = mesh.p[0, n0], mesh.p[1, n0]
    x1, y1 = mesh.p[0, n1], mesh.p[1, n1]
    x2, y2 = mesh.p[0, n2], mesh.p[1, n2]
    return 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))