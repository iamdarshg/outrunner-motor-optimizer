"""
2-D Finite Element Mechanical / Structural Solver for Outrunner BLDC Motors.

Solves:
  1. Linear elasticity (plane stress) for centrifugal + thermal stress
  2. Eigenvalue problem for natural frequencies (modal analysis)
  3. Element-level stress extraction for safety factors

Governing equation (static):
    -∇·σ = f_body     in Ω
    σ = C : (ε - ε_th)
    ε = ½(∇u + (∇u)ᵀ)
    ε_th = α ΔT I

where:
  u   — displacement vector
  σ   — Cauchy stress tensor
  ε   — total strain tensor
  C   — elasticity tensor (plane stress)
  f_body — body force (centrifugal: ρ ω² r ê_r)
  α   — thermal expansion coefficient
  ΔT  — temperature rise above reference

Modal analysis:
    K φ = ω² M φ

Region-dependent material properties (E, ν, ρ, α) follow the same
region tags as the EM/thermal meshes.

Key References:
  [1] S.P. Timoshenko, J.N. Goodier, "Theory of Elasticity," 3rd ed.,
      McGraw-Hill, 1970.
  [2] S.S. Rao, "Mechanical Vibrations," 6th ed., Pearson, 2017.
  [3] G. McBain, T. Gustafsson, "scikit-fem," JOSS, 2020.
  [4] O.C. Zienkiewicz, R.L. Taylor, "The Finite Element Method for
      Solid and Structural Mechanics," 7th ed., Elsevier, 2013.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from scipy.sparse.linalg import eigsh

try:
    from skfem import (
        MeshTri, Basis, ElementTriP1, ElementVector,
        BilinearForm, LinearForm, asm, enforce, solve, condense,
    )
    from skfem.helpers import dot, grad, sym_grad, ddot, transpose, eye
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False

from .materials import MaterialDatabase, StructuralMaterial
from .electromagnetic import GeometryParams, WindingConfig


@dataclass
class FEMMechanicalResults:
    """Results from FEM mechanical analysis."""
    displacement_field: np.ndarray = field(default_factory=lambda: np.array([]))
    von_mises_stress: np.ndarray = field(default_factory=lambda: np.array([]))
    max_displacement: float = 0.0
    max_von_mises: float = 0.0

    # Per-region stress results
    rotor_hoop_stress_max: float = 0.0
    stator_stress_max: float = 0.0
    magnet_stress_max: float = 0.0
    shaft_stress_max: float = 0.0

    # Safety factors
    rotor_safety_factor: float = 0.0
    stator_safety_factor: float = 0.0
    shaft_safety_factor: float = 0.0
    magnet_retention_factor: float = 0.0

    # Modal analysis
    natural_frequencies_hz: List[float] = field(default_factory=list)
    mode_shapes: np.ndarray = field(default_factory=lambda: np.array([]))

    mesh_nodes: int = 0
    mesh_elements: int = 0
    is_safe: bool = True


class FEMMechanicalModel:
    """
    2-D finite element mechanical solver for outrunner motors.

    Solves plane-stress linear elasticity with centrifugal body forces
    and optional thermal pre-stress, plus eigenvalue (modal) analysis.
    """

    def __init__(self, geometry: GeometryParams, winding: WindingConfig,
                 materials: MaterialDatabase, mesh_density: int = 80):
        if not HAS_SKFEM:
            raise ImportError(
                "scikit-fem is required for FEM mechanical analysis. "
                "Install with: pip install scikit-fem[all]"
            )
        self.geo = geometry
        self.wdg = winding
        self.mats = materials
        self.mesh_density = mesh_density
        self.results = FEMMechanicalResults()

    def solve_static(self, rpm: float,
                     temperature_field: Optional[np.ndarray] = None,
                     T_ref: float = 25.0) -> FEMMechanicalResults:
        """
        Solve static stress analysis under centrifugal loading.

        Parameters
        ----------
        rpm : float
            Rotational speed [RPM].
        temperature_field : array, optional
            Nodal temperature field from thermal FEM.
            If provided, thermal stresses are included.
        T_ref : float
            Reference temperature for thermal expansion.
        """
        geo = self.geo
        mats = self.mats

        # --- 1. Generate mesh ---
        from .fem_electromagnetic import _generate_motor_mesh
        mesh, tags = _generate_motor_mesh(geo, self.wdg, mats,
                                          n_circum=self.mesh_density)
        self.results.mesh_nodes = mesh.p.shape[1]
        self.results.mesh_elements = mesh.t.shape[1]
        n_elem = mesh.t.shape[1]
        n_nodes = mesh.p.shape[1]

        centroids = mesh.p[:, mesh.t].mean(axis=1)
        rc = np.sqrt(centroids[0]**2 + centroids[1]**2)

        omega = rpm * 2 * np.pi / 60.0

        # --- 2. Material properties per element ---
        E_elem = np.ones(n_elem) * 1e6      # air (dummy stiffness)
        nu_elem = np.ones(n_elem) * 0.3
        rho_elem = np.ones(n_elem) * 1.225   # air density
        alpha_elem = np.zeros(n_elem)         # thermal expansion

        # Stator iron (lamination)
        mask_1 = tags == 1
        E_elem[mask_1] = 200e9     # approximate for silicon steel
        nu_elem[mask_1] = 0.29
        rho_elem[mask_1] = mats.steel.density
        alpha_elem[mask_1] = 12e-6

        # Rotor yoke (structural material)
        mask_2 = tags == 2
        E_elem[mask_2] = mats.rotor_housing.youngs_modulus
        nu_elem[mask_2] = mats.rotor_housing.poissons_ratio
        rho_elem[mask_2] = mats.rotor_housing.density
        alpha_elem[mask_2] = mats.rotor_housing.thermal_expansion

        # Slot copper (composite: copper + insulation)
        mask_3 = tags == 3
        E_elem[mask_3] = 120e9 * mats.copper.fill_factor + 3e9 * (1 - mats.copper.fill_factor)
        nu_elem[mask_3] = 0.33
        rho_elem[mask_3] = (mats.copper.density * mats.copper.fill_factor +
                            1200 * (1 - mats.copper.fill_factor))
        alpha_elem[mask_3] = 17e-6

        # Magnets
        mask_4 = tags == 4
        E_elem[mask_4] = 160e9   # NdFeB typical
        nu_elem[mask_4] = 0.24
        rho_elem[mask_4] = mats.magnet.density
        alpha_elem[mask_4] = 5e-6

        # Shaft
        mask_5 = tags == 5
        E_elem[mask_5] = mats.shaft.youngs_modulus
        nu_elem[mask_5] = mats.shaft.poissons_ratio
        rho_elem[mask_5] = mats.shaft.density
        alpha_elem[mask_5] = mats.shaft.thermal_expansion

        # Air elements: moderately soft (prevent singularity;
        # too-soft air causes huge displacements that pollute results)
        mask_0 = tags == 0
        E_elem[mask_0] = 1e6  # 1 MPa
        rho_elem[mask_0] = 1.225

        # --- 3. Assemble stiffness matrix ---
        e = ElementVector(ElementTriP1())
        basis = Basis(mesh, e)

        # Use element-wise DiscreteField for material data (sharp boundaries)
        from skfem.element.discrete_field import DiscreteField
        n_qp = basis.X.shape[-1]

        def elem_to_qp(data):
            return DiscreteField(np.tile(data[:, None], (1, n_qp)))

        # Pre-compute Lamé parameters per element (plane stress)
        mu_elem = E_elem / (2 * (1 + nu_elem))
        lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - nu_elem))

        _I2 = np.array([[1, 0], [0, 1]], dtype=float).reshape(2, 2, 1, 1)

        @BilinearForm
        def stiffness(u, v, w):
            def C_eps(eps):
                tr_eps = eps[0, 0] + eps[1, 1]
                return w.lam * tr_eps * _I2 + 2 * w.mu * eps
            return ddot(C_eps(sym_grad(u)), sym_grad(v))

        K = stiffness.assemble(basis,
                               lam=elem_to_qp(lam_elem),
                               mu=elem_to_qp(mu_elem))

        # --- 5. Body force: centrifugal (rotor only) ---
        # Outrunner: only rotor (tag 2), magnets (tag 4), and outer air
        # rotate.  Stator (1), copper (3), shaft (5) are stationary.
        rho_rotating = rho_elem.copy()
        rho_rotating[tags == 1] = 0.0  # stator iron
        rho_rotating[tags == 3] = 0.0  # slot copper
        rho_rotating[tags == 5] = 0.0  # shaft
        # Air inside stator bore also stationary
        centroids_local = mesh.p[:, mesh.t].mean(axis=1)
        r_centroids = np.sqrt(centroids_local[0]**2 + centroids_local[1]**2)
        stator_air = (tags == 0) & (r_centroids < geo.stator_outer_radius)
        rho_rotating[stator_air] = 0.0

        @LinearForm
        def centrifugal_force(v, w):
            rho = w.rho
            x = w.x[0]
            y = w.x[1]
            fx = rho * omega**2 * x
            fy = rho * omega**2 * y
            return fx * v[0] + fy * v[1]

        f = centrifugal_force.assemble(basis, rho=elem_to_qp(rho_rotating))

        # --- 6. Thermal pre-stress (if temperature field provided) ---
        if temperature_field is not None and len(temperature_field) == n_nodes:
            dT = temperature_field - T_ref

            @LinearForm
            def thermal_force(v, w):
                E_val = w.E
                nu_val = w.nu
                alpha_val = w.alpha
                dT_val = w.dT
                # Thermal strain contribution to force vector
                # f_th = ∫ C:ε_th · ∇v dΩ
                # For plane stress: σ_th = E*α*ΔT/(1-ν) * I
                coeff = E_val * alpha_val * dT_val / (1 - nu_val)
                # Divergence form: contributes to both x and y components
                return coeff * (grad(v)[0, 0] + grad(v)[1, 1])

            # Map dT (nodal) to element-averages then to qp
            dT_elem = np.zeros(n_elem)
            for ii in range(n_elem):
                dT_elem[ii] = np.mean(dT[mesh.t[:, ii]])

            f_th = thermal_force.assemble(
                basis,
                E=elem_to_qp(E_elem),
                nu=elem_to_qp(nu_elem),
                alpha=elem_to_qp(alpha_elem),
                dT=elem_to_qp(dT_elem),
            )
            f += f_th

        # --- 7. Boundary conditions ---
        # Fix shaft bore (inner surface) in both x and y
        inner_nodes = np.where(
            np.sqrt(mesh.p[0]**2 + mesh.p[1]**2) < geo.shaft_radius * 1.1
        )[0]

        # For vector elements: DOFs are [u_x_0, u_x_1, ..., u_y_0, u_y_1, ...]
        # or interleaved depending on ElementVector convention
        # In scikit-fem, ElementVector(ElementTriP1()) gives 2*n_nodes DOFs
        dof_x = inner_nodes
        dof_y = inner_nodes + n_nodes
        fixed_dofs = np.concatenate([dof_x, dof_y])

        # --- 8. Solve ---
        K_bc, f_bc = enforce(K, f, D=fixed_dofs)
        u = solve(K_bc, f_bc)

        self.results.displacement_field = u

        # --- 9. Post-process: stress extraction ---
        ux = u[:n_nodes]
        uy = u[n_nodes:]
        disp_mag = np.sqrt(ux**2 + uy**2)
        self.results.max_displacement = np.max(disp_mag)

        # Compute element-level stresses
        von_mises = np.zeros(n_elem)
        for i in range(n_elem):
            n0, n1, n2 = mesh.t[:, i]
            x0, y0 = mesh.p[0, n0], mesh.p[1, n0]
            x1, y1 = mesh.p[0, n1], mesh.p[1, n1]
            x2, y2 = mesh.p[0, n2], mesh.p[1, n2]

            area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
            if area < 1e-20:
                continue

            # Displacement gradients (constant per P1 element)
            # B matrix approach
            dN = np.array([
                [y1-y2, y2-y0, y0-y1],
                [x2-x1, x0-x2, x1-x0],
            ]) / (2 * area)

            u_e = np.array([ux[n0], ux[n1], ux[n2]])
            v_e = np.array([uy[n0], uy[n1], uy[n2]])

            eps_xx = dN[0] @ u_e
            eps_yy = dN[1] @ v_e
            eps_xy = 0.5 * (dN[1] @ u_e + dN[0] @ v_e)

            E = E_elem[i]
            nu_val = nu_elem[i]

            # Plane stress σ
            factor = E / (1 - nu_val**2)
            sig_xx = factor * (eps_xx + nu_val * eps_yy)
            sig_yy = factor * (eps_yy + nu_val * eps_xx)
            sig_xy = E / (2 * (1 + nu_val)) * 2 * eps_xy

            # Von Mises (plane stress)
            von_mises[i] = np.sqrt(sig_xx**2 - sig_xx*sig_yy + sig_yy**2 + 3*sig_xy**2)

        self.results.von_mises_stress = von_mises
        self.results.max_von_mises = np.max(von_mises)

        # Per-region max stress and safety factors
        self._compute_regional_stresses(tags, von_mises)

        return self.results

    def solve_modal(self, n_modes: int = 6) -> FEMMechanicalResults:
        """
        Solve eigenvalue problem for natural frequencies.

        K φ = ω² M φ

        Returns the first n_modes natural frequencies in Hz.
        """
        geo = self.geo
        mats = self.mats

        from .fem_electromagnetic import _generate_motor_mesh
        mesh, tags = _generate_motor_mesh(geo, self.wdg, mats,
                                          n_circum=self.mesh_density)
        n_elem = mesh.t.shape[1]
        n_nodes = mesh.p.shape[1]

        # Material properties (same as static)
        E_elem = np.ones(n_elem) * 1e3
        nu_elem = np.ones(n_elem) * 0.3
        rho_elem = np.ones(n_elem) * 1.225

        mask_1 = tags == 1
        E_elem[mask_1] = 200e9
        nu_elem[mask_1] = 0.29
        rho_elem[mask_1] = mats.steel.density

        mask_2 = tags == 2
        E_elem[mask_2] = mats.rotor_housing.youngs_modulus
        nu_elem[mask_2] = mats.rotor_housing.poissons_ratio
        rho_elem[mask_2] = mats.rotor_housing.density

        mask_3 = tags == 3
        E_elem[mask_3] = 120e9 * mats.copper.fill_factor
        nu_elem[mask_3] = 0.33
        rho_elem[mask_3] = mats.copper.density * mats.copper.fill_factor

        mask_4 = tags == 4
        E_elem[mask_4] = 160e9
        nu_elem[mask_4] = 0.24
        rho_elem[mask_4] = mats.magnet.density

        mask_5 = tags == 5
        E_elem[mask_5] = mats.shaft.youngs_modulus
        nu_elem[mask_5] = mats.shaft.poissons_ratio
        rho_elem[mask_5] = mats.shaft.density

        e = ElementVector(ElementTriP1())
        basis = Basis(mesh, e)

        from skfem.element.discrete_field import DiscreteField
        n_qp = basis.X.shape[-1]

        def elem_to_qp(data):
            return DiscreteField(np.tile(data[:, None], (1, n_qp)))

        mu_elem_m = E_elem / (2 * (1 + nu_elem))
        lam_elem_m = E_elem * nu_elem / ((1 + nu_elem) * (1 - nu_elem))
        _I2 = np.array([[1, 0], [0, 1]], dtype=float).reshape(2, 2, 1, 1)

        @BilinearForm
        def stiffness_form(u, v, w):
            def C_eps(eps):
                tr_eps = eps[0, 0] + eps[1, 1]
                return w.lam * tr_eps * _I2 + 2 * w.mu * eps
            return ddot(C_eps(sym_grad(u)), sym_grad(v))

        @BilinearForm
        def mass_form(u, v, w):
            return w.rho * (u[0] * v[0] + u[1] * v[1])

        K = stiffness_form.assemble(basis,
                                    lam=elem_to_qp(lam_elem_m),
                                    mu=elem_to_qp(mu_elem_m))
        M = mass_form.assemble(basis,
                               rho=elem_to_qp(rho_elem))

        # Fix shaft bore
        inner_nodes = np.where(
            np.sqrt(mesh.p[0]**2 + mesh.p[1]**2) < self.geo.shaft_radius * 1.1
        )[0]
        dof_x = inner_nodes
        dof_y = inner_nodes + n_nodes
        fixed_dofs = np.concatenate([dof_x, dof_y])

        # Condense (remove fixed DOFs for eigenvalue solve)
        all_dofs = np.arange(K.shape[0])
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        K_free = K[np.ix_(free_dofs, free_dofs)]
        M_free = M[np.ix_(free_dofs, free_dofs)]

        # Solve generalised eigenvalue problem
        try:
            n_solve = min(n_modes, K_free.shape[0] - 2)
            if n_solve < 1:
                n_solve = 1
            eigenvalues, eigenvectors = eigsh(
                K_free.tocsc(), k=n_solve, M=M_free.tocsc(),
                sigma=0, which='LM'  # shift-invert for smallest
            )

            # Convert to Hz
            omega_sq = np.sort(np.real(eigenvalues))
            omega_sq = omega_sq[omega_sq > 0]
            freqs_hz = np.sqrt(omega_sq) / (2 * np.pi)

            self.results.natural_frequencies_hz = freqs_hz.tolist()
        except Exception:
            # Fallback: approximate frequencies
            self.results.natural_frequencies_hz = []

        return self.results

    def _compute_regional_stresses(self, tags: np.ndarray,
                                    von_mises: np.ndarray):
        """Extract per-region stress and safety factors.

        Uses 90th percentile instead of max to avoid mesh-related
        stress singularities at material interfaces.
        """
        mats = self.mats
        res = self.results

        # Rotor yoke
        rotor_mask = tags == 2
        if np.any(rotor_mask) and np.sum(rotor_mask) > 3:
            res.rotor_hoop_stress_max = np.percentile(von_mises[rotor_mask], 90)
            res.rotor_safety_factor = (mats.rotor_housing.yield_strength /
                                       max(res.rotor_hoop_stress_max, 1))
        else:
            res.rotor_safety_factor = 100.0

        # Stator
        stator_mask = tags == 1
        if np.any(stator_mask) and np.sum(stator_mask) > 3:
            res.stator_stress_max = np.percentile(von_mises[stator_mask], 90)
            res.stator_safety_factor = 350e6 / max(res.stator_stress_max, 1)
        else:
            res.stator_safety_factor = 100.0

        # Magnets
        mag_mask = tags == 4
        if np.any(mag_mask) and np.sum(mag_mask) > 3:
            res.magnet_stress_max = np.percentile(von_mises[mag_mask], 90)
            res.magnet_retention_factor = 30e6 / max(res.magnet_stress_max, 1)
        else:
            res.magnet_retention_factor = 100.0

        # Shaft
        shaft_mask = tags == 5
        if np.any(shaft_mask) and np.sum(shaft_mask) > 3:
            res.shaft_stress_max = np.percentile(von_mises[shaft_mask], 90)
            res.shaft_safety_factor = (mats.shaft.yield_strength /
                                       max(res.shaft_stress_max, 1))
        else:
            res.shaft_safety_factor = 100.0

        # Overall max uses 95th percentile for robustness
        res.max_von_mises = np.percentile(von_mises[von_mises > 0], 95) if np.any(von_mises > 0) else 0

        res.is_safe = (res.rotor_safety_factor > 1.5 and
                       res.stator_safety_factor > 1.5 and
                       res.shaft_safety_factor > 2.0 and
                       res.magnet_retention_factor > 1.5)