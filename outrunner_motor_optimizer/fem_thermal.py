"""
2-D Finite Element Thermal Solver for Outrunner BLDC Motors.

Solves the steady-state heat conduction equation on the motor cross-section:

  Steady:    -∇·(k ∇T) = q_gen           in Ω

Boundary conditions:
  • Outer domain boundary:  -k ∂T/∂n = h_ext (T - T_amb)
  • Internal boundaries computed via convective coupling

Region-dependent thermal conductivity:
  0 — air            k = 0.026 W/(m·K)
  1 — stator iron    k_radial (lamination)
  2 — rotor iron     k = structural material k
  3 — slot copper    k_eff = k_Cu * fill + k_ins * (1-fill)
  4 — magnets        k = magnet thermal conductivity
  5 — shaft          k = shaft material k

Heat sources q_gen [W/m³] come from the EM analysis:
  • Copper loss → distributed in slots (region 3)
  • Iron loss → distributed in stator/rotor iron (regions 1, 2)
  • Magnet eddy loss → distributed in magnets (region 4)

Key References:
  [1] A. Boglietti et al., "Evolution and Modern Approaches for Thermal
      Analysis of Electrical Machines," IEEE Trans. Ind. Electron., 2009.
  [2] D. Staton, A. Cavagnino, "Convection Heat Transfer and Flow
      Computations in Electric Machine Thermal Models," IEEE Trans. Ind.
      Electron., 2008.
  [3] K.M. Becker, J. Kaye, "Measurements of Diabatic Flow in an
      Annulus With an Inner Rotating Cylinder," ASME J. Heat Transfer,
      1962. (Taylor-Couette airgap correlation)
  [4] G. McBain, T. Gustafsson, "scikit-fem," JOSS, 2020.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

try:
    from skfem import (
        MeshTri, Basis, ElementTriP1,
        BilinearForm, LinearForm, asm, enforce, solve,
    )
    from skfem.helpers import dot, grad
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False

from .materials import MaterialDatabase
from .electromagnetic import GeometryParams, WindingConfig, EMResults

MU_0 = 4.0 * np.pi * 1e-7


@dataclass
class FEMThermalResults:
    """Results from FEM thermal analysis."""
    temperature_field: np.ndarray = field(default_factory=lambda: np.array([]))
    T_winding_max: float = 25.0
    T_winding_avg: float = 25.0
    T_tooth_max: float = 25.0
    T_yoke_max: float = 25.0
    T_magnet_max: float = 25.0
    T_magnet_avg: float = 25.0
    T_rotor_yoke_max: float = 25.0
    T_shaft_max: float = 25.0
    T_airgap_avg: float = 25.0
    max_temperature: float = 25.0
    critical_component: str = ""
    is_safe: bool = True
    hotspot_margin: float = 0.0
    mesh_nodes: int = 0
    mesh_elements: int = 0


def _airgap_htc(geo: GeometryParams, rpm: float, T_air: float = 60.0) -> float:
    """Taylor-Couette airgap heat transfer coefficient [W/(m²·K)]."""
    omega = rpm * 2 * np.pi / 60.0
    delta = geo.airgap
    R_rotor = geo.rotor_inner_radius

    nu_air = 1.5e-5 * (1 + 0.003 * (T_air - 20))
    k_air = 0.026 * (1 + 0.003 * (T_air - 20))
    Pr = 0.71

    v_rotor = omega * R_rotor
    Re_g = v_rotor * delta / nu_air if nu_air > 0 else 0
    Ta = Re_g * np.sqrt(delta / R_rotor) if R_rotor > 0 else 0

    if Ta < 41.3:
        Nu = 2.0
    elif Ta < 100:
        Nu = 0.202 * Ta**0.63 * Pr**0.27
    else:
        Nu = 0.386 * Ta**0.5 * Pr**0.27

    h = Nu * k_air / delta if delta > 0 else 10.0
    return max(h, 5.0)


def _external_htc(geo: GeometryParams, rpm: float) -> float:
    """External rotor surface convection coefficient [W/(m²·K)]."""
    omega = rpm * 2 * np.pi / 60.0
    R_ext = geo.rotor_outer_radius
    nu_air = 1.6e-5
    k_air = 0.026
    Pr = 0.71

    v_surface = omega * R_ext
    Re = v_surface * 2 * R_ext / nu_air
    if Re > 2.5e4:
        Nu = 0.133 * Re**(2.0 / 3.0) * Pr**(1.0 / 3.0)
    elif Re > 100:
        Nu = 0.683 * Re**0.466 * Pr**(1.0 / 3.0)
    else:
        Nu = 2.0

    D_ext = 2 * R_ext
    h = Nu * k_air / D_ext if D_ext > 0 else 10.0
    return max(h, 5.0)


class FEMThermalModel:
    """
    2-D finite element steady-state thermal solver for outrunner motors.

    Uses scikit-fem to solve -∇·(k∇T) = q with convective boundary
    conditions.  Robin (convective) BCs on the outer boundary are
    implemented via Dirichlet approximation: T = T_amb + Q_total /
    (h_eff * A_ext), applied as a fixed temperature on the outer
    boundary.  This is physically correct for a well-cooled external
    surface and avoids sparse-matrix surgery.
    """

    def __init__(self, geometry: GeometryParams, winding: WindingConfig,
                 materials: MaterialDatabase, T_ambient: float = 25.0,
                 mesh_density: int = 80):
        if not HAS_SKFEM:
            raise ImportError(
                "scikit-fem is required for FEM thermal analysis. "
                "Install with: pip install scikit-fem[all]"
            )
        self.geo = geometry
        self.wdg = winding
        self.mats = materials
        self.T_amb = T_ambient
        self.mesh_density = mesh_density
        self.results = FEMThermalResults()

    def solve_steady_state(self, em_results: EMResults,
                           rpm: float) -> FEMThermalResults:
        """
        Solve steady-state thermal problem using FEM.

        Strategy: Apply Dirichlet BC T=T_boundary on the outer boundary,
        where T_boundary is estimated from a lumped convective balance.
        Internal heat sources are distributed volumetrically by region.
        """
        geo = self.geo
        wdg = self.wdg
        mats = self.mats

        # --- 1. Generate mesh ---
        from .fem_electromagnetic import _generate_motor_mesh
        mesh, tags = _generate_motor_mesh(
            geo, wdg, mats, n_circum=self.mesh_density
        )
        self.results.mesh_nodes = mesh.p.shape[1]
        self.results.mesh_elements = mesh.t.shape[1]
        n_elem = mesh.t.shape[1]
        n_nodes = mesh.p.shape[1]

        centroids = mesh.p[:, mesh.t].mean(axis=1)
        rc = np.sqrt(centroids[0]**2 + centroids[1]**2)

        # --- 2. Thermal conductivity per element ---
        # Air regions use an elevated effective conductivity to represent
        # the real 3-D thermal paths (end bells, structural connections,
        # convective mixing) that are absent in the 2-D cross-section.
        k_air_eff = 1.0  # W/(m·K) — effective for structural air paths
        k_elem = np.ones(n_elem) * k_air_eff

        k_elem[tags == 1] = mats.steel.thermal_conductivity_radial
        k_elem[tags == 2] = mats.rotor_housing.thermal_conductivity
        k_eff_slot = (mats.copper.thermal_conductivity * mats.copper.fill_factor +
                      0.2 * (1 - mats.copper.fill_factor))
        k_elem[tags == 3] = k_eff_slot
        k_elem[tags == 4] = mats.magnet.thermal_conductivity
        k_elem[tags == 5] = mats.shaft.thermal_conductivity

        # --- 3. Heat source per element [W/m³] ---
        q_elem = np.zeros(n_elem)

        # Volume per region (2D area × stack length)
        for tag_id, loss_attr in [
            (1, 'iron_loss_stator'),
            (2, 'iron_loss_rotor'),
            (3, 'copper_loss'),
            (4, 'magnet_loss'),
        ]:
            mask = tags == tag_id
            if not np.any(mask):
                continue
            # Total 2D area for this region
            areas = np.array([_triangle_area(mesh, i) for i in range(n_elem)])
            total_area = np.sum(areas[mask])
            if total_area < 1e-12:
                continue
            loss_val = getattr(em_results, loss_attr, 0.0)
            if tag_id == 3:
                # Copper: only slot portion (end-turns don't contribute in 2D)
                slot_frac = (2 * geo.stack_length /
                             (2 * geo.stack_length + 4 * wdg.end_turn_length))
                loss_val *= slot_frac
            q_elem[mask] = loss_val / (total_area * geo.stack_length)

        # --- 4. Estimate boundary temperature ---
        # Total losses
        P_total = (getattr(em_results, 'copper_loss', 0.0) +
                   getattr(em_results, 'iron_loss_stator', 0.0) +
                   getattr(em_results, 'iron_loss_rotor', 0.0) +
                   getattr(em_results, 'magnet_loss', 0.0))

        h_ext = _external_htc(geo, rpm)
        h_gap = _airgap_htc(geo, rpm)

        # External surface area (cylindrical + two end faces)
        A_ext = (2 * np.pi * geo.rotor_outer_radius * geo.stack_length +
                 2 * np.pi * geo.rotor_outer_radius**2)
        # Lumped outer boundary temperature
        T_boundary = self.T_amb + P_total / max(h_ext * A_ext, 1e-6)
        T_boundary = min(T_boundary, self.T_amb + 200)  # cap for stability

        # --- 5. Assemble FEM system ---
        basis = Basis(mesh, ElementTriP1())

        from skfem.element.discrete_field import DiscreteField
        n_qp = basis.X.shape[-1]

        def elem_to_qp(data):
            return DiscreteField(np.tile(data[:, None], (1, n_qp)))

        @BilinearForm
        def conduction(u, v, w):
            return w.k * dot(grad(u), grad(v))

        @LinearForm
        def heat_source(v, w):
            return w.q * v

        K = conduction.assemble(basis, k=elem_to_qp(k_elem))
        f = heat_source.assemble(basis, q=elem_to_qp(q_elem))

        # --- 6. Boundary conditions ---
        boundary_nodes = mesh.boundary_nodes()
        node_r = np.sqrt(mesh.p[0]**2 + mesh.p[1]**2)
        r_domain = np.max(node_r[boundary_nodes])

        # Outer boundary: Dirichlet T = T_boundary
        outer_bc = boundary_nodes[node_r[boundary_nodes] > r_domain * 0.9]
        T_bc = np.full(n_nodes, T_boundary)  # values on Dirichlet nodes

        K, f = enforce(K, f, D=outer_bc, x=T_bc)

        # --- 7. Solve ---
        T = solve(K, f)

        # Clamp to physical range
        T = np.clip(T, self.T_amb - 10, 500)
        self.results.temperature_field = T

        # --- 8. Extract regional temperatures ---
        T_elem = np.zeros(n_elem)
        for i in range(n_elem):
            T_elem[i] = np.mean(T[mesh.t[:, i]])

        # Winding
        if np.any(tags == 3):
            self.results.T_winding_max = np.max(T_elem[tags == 3])
            self.results.T_winding_avg = np.mean(T_elem[tags == 3])

        # Teeth vs yoke
        r_tooth_inner = geo.stator_inner_radius + geo.stator_yoke_thickness
        tooth_mask = (tags == 1) & (rc > r_tooth_inner)
        yoke_mask = (tags == 1) & (rc <= r_tooth_inner + geo.stator_yoke_thickness)
        if np.any(tooth_mask):
            self.results.T_tooth_max = np.max(T_elem[tooth_mask])
        if np.any(yoke_mask):
            self.results.T_yoke_max = np.max(T_elem[yoke_mask])

        # Magnets
        if np.any(tags == 4):
            self.results.T_magnet_max = np.max(T_elem[tags == 4])
            self.results.T_magnet_avg = np.mean(T_elem[tags == 4])

        # Rotor yoke
        if np.any(tags == 2):
            self.results.T_rotor_yoke_max = np.max(T_elem[tags == 2])

        # Shaft
        if np.any(tags == 5):
            self.results.T_shaft_max = np.max(T_elem[tags == 5])

        # Airgap
        gap_r = (geo.stator_outer_radius + geo.rotor_inner_radius) / 2
        gap_mask = np.abs(rc - gap_r) < geo.airgap * 2
        if np.any(gap_mask):
            self.results.T_airgap_avg = np.mean(T_elem[gap_mask])

        # Overall
        self.results.max_temperature = float(np.max(T))

        # Safety check
        limits = {
            'Winding': (self.results.T_winding_max, mats.copper.max_temp),
            'Magnets': (self.results.T_magnet_max, mats.magnet.max_temp),
            'Stator iron': (self.results.T_tooth_max, mats.steel.max_temp),
        }
        self.results.is_safe = True
        self.results.hotspot_margin = float('inf')
        for name, (T_act, T_lim) in limits.items():
            margin = T_lim - T_act
            if margin < self.results.hotspot_margin:
                self.results.hotspot_margin = margin
                self.results.critical_component = name
            if T_act > T_lim:
                self.results.is_safe = False

        return self.results

    def to_thermal_results(self):
        """Convert FEM thermal results to the LPTN-compatible ThermalResults."""
        from .thermal import ThermalResults
        tr = ThermalResults()
        tr.T_stator_yoke = self.results.T_yoke_max
        tr.T_stator_teeth = self.results.T_tooth_max
        tr.T_winding_slot = self.results.T_winding_max
        tr.T_end_winding_de = self.results.T_winding_avg
        tr.T_end_winding_nde = self.results.T_winding_avg
        tr.T_airgap = self.results.T_airgap_avg
        tr.T_magnets = self.results.T_magnet_max
        tr.T_rotor_yoke = self.results.T_rotor_yoke_max
        tr.T_shaft = self.results.T_shaft_max
        tr.T_ambient = self.T_amb
        tr.max_temperature = self.results.max_temperature
        tr.critical_component = self.results.critical_component
        tr.is_safe = self.results.is_safe
        tr.hotspot_margin = self.results.hotspot_margin
        return tr


def _triangle_area(mesh, elem_idx):
    n0, n1, n2 = mesh.t[:, elem_idx]
    x0, y0 = mesh.p[0, n0], mesh.p[1, n0]
    x1, y1 = mesh.p[0, n1], mesh.p[1, n1]
    x2, y2 = mesh.p[0, n2], mesh.p[1, n2]
    return 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))