"""
Parametric 3D CAD model generation and STEP export for outrunner motors.

Uses CadQuery (OCCT kernel) to build solid models of every component:
  - Stator lamination stack (with slots and teeth)
  - Rotor bell / yoke (aluminium housing)
  - Permanent magnets (individual arcs)
  - Shaft
  - Bearing seats (simplified cylinders)
  - Mounting flange / bolt pattern
  - Full assembly

All models are parametric — dimensions come directly from the optimised
GeometryParams, WindingConfig, and MountingConfig objects.

Output: ISO 10303 STEP files (.step) per component + full assembly.

References:
  [1] CadQuery documentation — https://cadquery.readthedocs.io
  [2] ISO 10303 (STEP) file format standard.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict

from .electromagnetic import GeometryParams, WindingConfig
from .materials import MaterialDatabase, MountingConfig

# CadQuery import with graceful fallback
try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False


def _require_cadquery():
    if not HAS_CADQUERY:
        raise ImportError(
            "CadQuery is required for STEP export.  Install with:\n"
            "  conda install -c conda-forge -c cadquery cadquery\n"
            "or\n"
            "  pip install cadquery"
        )


# ---------------------------------------------------------------------------
# Individual component builders
# ---------------------------------------------------------------------------

def build_stator(geo: GeometryParams, wdg: WindingConfig) -> "cq.Workplane":
    """
    Stator lamination stack: annular ring with radial slots.
    """
    _require_cadquery()
    import cadquery as cq

    R_outer = geo.stator_outer_radius * 1e3   # mm
    R_inner = geo.stator_inner_radius * 1e3
    L = geo.stack_length * 1e3
    n_slots = wdg.num_slots
    slot_w = geo.slot_opening * 1e3
    slot_d = geo.slot_depth * 1e3

    # Base annular cylinder
    stator = (
        cq.Workplane("XY")
        .circle(R_outer)
        .circle(R_inner)
        .extrude(L)
    )

    # Cut slots (rectangular approximation)
    for i in range(n_slots):
        angle = i * 360.0 / n_slots
        # Slot as a box positioned at the bore
        slot_r = R_inner + geo.stator_yoke_thickness * 1e3 + slot_d / 2
        slot = (
            cq.Workplane("XY")
            .center(0, 0)
            .transformed(rotate=(0, 0, angle))
            .center(slot_r, 0)
            .rect(slot_d, slot_w)
            .extrude(L)
        )
        stator = stator.cut(slot)

    return stator


def build_rotor_yoke(geo: GeometryParams, mats: MaterialDatabase) -> "cq.Workplane":
    """
    Rotor bell / yoke (outer rotating shell) — aluminium.
    Includes end-cap on one side (drive end).
    """
    _require_cadquery()
    import cadquery as cq

    R_inner = (geo.rotor_inner_radius + mats.magnet.thickness) * 1e3
    R_outer = geo.rotor_outer_radius * 1e3
    L = geo.stack_length * 1e3
    end_cap_t = 2.0  # mm

    # Main cylinder
    rotor = (
        cq.Workplane("XY")
        .circle(R_outer)
        .circle(R_inner)
        .extrude(L + end_cap_t)
    )

    # End cap (solid disc closing one end)
    cap = (
        cq.Workplane("XY")
        .workplane(offset=L)
        .circle(R_outer)
        .circle(geo.shaft_radius * 1e3 + 1.0)  # bearing seat hole
        .extrude(end_cap_t)
    )
    rotor = rotor.union(cap)

    return rotor


def build_magnets(geo: GeometryParams, wdg: WindingConfig,
                  mats: MaterialDatabase) -> "cq.Workplane":
    """
    Individual magnet arcs bonded to rotor inner surface.
    Returns a single compound of all magnets.
    """
    _require_cadquery()
    import cadquery as cq

    n_poles = wdg.num_poles
    R_inner = geo.rotor_inner_radius * 1e3
    t_mag = mats.magnet.thickness * 1e3
    L = geo.stack_length * 1e3
    arc_frac = mats.magnet.arc_fraction
    pole_arc_deg = 360.0 / n_poles
    mag_arc_deg = pole_arc_deg * arc_frac

    magnets = None
    for i in range(n_poles):
        angle_start = i * pole_arc_deg + (pole_arc_deg - mag_arc_deg) / 2
        # Build arc segment as revolution of rectangle
        mag = (
            cq.Workplane("XZ")
            .center(R_inner + t_mag / 2, L / 2)
            .rect(t_mag, L)
            .revolve(mag_arc_deg, (0, 0, 0), (0, 0, 1),
                     combine=False)
            .rotate((0, 0, 0), (0, 0, 1), angle_start)
        )
        if magnets is None:
            magnets = mag
        else:
            magnets = magnets.union(mag)

    return magnets


def build_shaft(geo: GeometryParams) -> "cq.Workplane":
    """Motor shaft — simple stepped cylinder."""
    _require_cadquery()
    import cadquery as cq

    R = geo.shaft_radius * 1e3
    L = geo.shaft_length * 1e3

    shaft = (
        cq.Workplane("XY")
        .circle(R)
        .extrude(L)
    )

    return shaft


def build_mounting_flange(geo: GeometryParams,
                          mounting: MountingConfig) -> "cq.Workplane":
    """
    Mounting flange with bolt holes.
    """
    _require_cadquery()
    import cadquery as cq
    import math

    R_flange = mounting.flange_outer_diameter / 2 * 1e3
    t = mounting.flange_thickness * 1e3
    R_center_hole = geo.shaft_radius * 1e3 + 1.0  # clearance

    flange = (
        cq.Workplane("XY")
        .circle(R_flange)
        .circle(R_center_hole)
        .extrude(t)
    )

    # Bolt holes
    r_bc = mounting.bolt_circle_diameter / 2 * 1e3
    d_bolt = mounting.bolt_diameter * 1e3

    for i in range(mounting.num_bolts):
        angle = i * 360.0 / mounting.num_bolts
        rad = math.radians(angle)
        cx = r_bc * math.cos(rad)
        cy = r_bc * math.sin(rad)
        hole = (
            cq.Workplane("XY")
            .center(cx, cy)
            .circle(d_bolt / 2 + 0.15)  # clearance hole
            .extrude(t)
        )
        flange = flange.cut(hole)

    return flange


# ---------------------------------------------------------------------------
# Export orchestrator
# ---------------------------------------------------------------------------

def export_step_files(geo: GeometryParams,
                      wdg: WindingConfig,
                      mats: MaterialDatabase,
                      output_dir: str = "motor_step_output",
                      ) -> Dict[str, str]:
    """
    Build all components and export as individual STEP files + assembly.

    Returns dict mapping component name → file path.
    """
    _require_cadquery()
    import cadquery as cq

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    components = {}

    # 1. Stator
    try:
        stator = build_stator(geo, wdg)
        p = str(out / "stator.step")
        cq.exporters.export(stator, p)
        paths["stator"] = p
        components["stator"] = stator
    except Exception as e:
        paths["stator"] = f"FAILED: {e}"

    # 2. Rotor yoke
    try:
        rotor = build_rotor_yoke(geo, mats)
        p = str(out / "rotor_yoke.step")
        cq.exporters.export(rotor, p)
        paths["rotor_yoke"] = p
        components["rotor_yoke"] = rotor
    except Exception as e:
        paths["rotor_yoke"] = f"FAILED: {e}"

    # 3. Magnets
    try:
        magnets = build_magnets(geo, wdg, mats)
        p = str(out / "magnets.step")
        cq.exporters.export(magnets, p)
        paths["magnets"] = p
        components["magnets"] = magnets
    except Exception as e:
        paths["magnets"] = f"FAILED: {e}"

    # 4. Shaft
    try:
        shaft = build_shaft(geo)
        p = str(out / "shaft.step")
        cq.exporters.export(shaft, p)
        paths["shaft"] = p
        components["shaft"] = shaft
    except Exception as e:
        paths["shaft"] = f"FAILED: {e}"

    # 5. Mounting flange
    try:
        flange = build_mounting_flange(geo, mats.mounting)
        p = str(out / "mounting_flange.step")
        cq.exporters.export(flange, p)
        paths["mounting_flange"] = p
        components["mounting_flange"] = flange
    except Exception as e:
        paths["mounting_flange"] = f"FAILED: {e}"

    # 6. Assembly (union of all successful components)
    try:
        assy = cq.Assembly()
        for name, solid in components.items():
            assy.add(solid, name=name)
        p = str(out / "full_assembly.step")
        assy.save(p)
        paths["full_assembly"] = p
    except Exception as e:
        paths["full_assembly"] = f"FAILED: {e}"

    return paths
