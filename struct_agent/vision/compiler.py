"""
struct_agent/vision/compiler.py
══════════════════════════════
Deterministic structural grid compiler that converts intermediate semantic
FloorPlanDraft representations into mathematically exact struct_core.Project models.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from struct_core import (
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    LoadCase,
    LoadCaseFactor,
    LoadCombination,
    Metadata,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    parse_units,
)

from .schemas import BeamDraft, ColumnDraft, FloorPlanDraft, GridLineDraft, StoryDraft


def _resolve_grid_axes(
    grids_x: List[GridLineDraft], grids_y: List[GridLineDraft]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Accumulate grid line spacings to calculate exact (X, Y) coordinates.
    """
    coords_x: Dict[str, float] = {}
    curr_x = 0.0
    for idx, gx in enumerate(grids_x):
        label = str(gx.label).strip()
        if gx.coordinate is not None:
            curr_x = float(gx.coordinate)
        coords_x[label] = round(curr_x, 4)
        curr_x += float(gx.spacing_to_next)

    coords_y: Dict[str, float] = {}
    curr_y = 0.0
    for idx, gy in enumerate(grids_y):
        label = str(gy.label).strip()
        if gy.coordinate is not None:
            curr_y = float(gy.coordinate)
        coords_y[label] = round(curr_y, 4)
        curr_y += float(gy.spacing_to_next)

    return coords_x, coords_y


def _find_grid_pair(
    pair: Tuple[str, str], coords_x: Dict[str, float], coords_y: Dict[str, float]
) -> Tuple[str, str]:
    """
    Determine which token in a (g1, g2) pair corresponds to the X-axis and which to Y-axis.
    """
    p0, p1 = str(pair[0]).strip(), str(pair[1]).strip()
    if p0 in coords_x and p1 in coords_y:
        return p0, p1
    elif p1 in coords_x and p0 in coords_y:
        return p1, p0
    # Fallback heuristic: check case-insensitively
    for kx in coords_x:
        if kx.lower() == p0.lower():
            for ky in coords_y:
                if ky.lower() == p1.lower():
                    return kx, ky
        elif kx.lower() == p1.lower():
            for ky in coords_y:
                if ky.lower() == p0.lower():
                    return kx, ky
    return p0, p1


def build_project_from_draft(
    draft: FloorPlanDraft,
    existing_project: Optional[Project] = None,
) -> Project:
    """
    Compile a FloorPlanDraft into an exact struct_core.Project.

    Features:
    - Exact nodal coordinates snapped along orthogonal or designated grid axes
    - Continuous column stacking across stories
    - Perimeter and interior beam framing along grid lines
    - Ground restraints at Level 0
    - Concrete material & rectangular section definitions
    - Factored load combinations (1.4D and 1.2D+1.6L)
    """
    if not draft.grids_x or not draft.grids_y:
        raise ValueError("FloorPlanDraft requires at least one X grid and one Y grid.")
    if not draft.stories:
        raise ValueError("FloorPlanDraft requires at least one story level.")

    # 1. Resolve Grid System Coordinates
    coords_x, coords_y = _resolve_grid_axes(draft.grids_x, draft.grids_y)
    x_labels = list(coords_x.keys())
    y_labels = list(coords_y.keys())

    # 2. Resolve Vertical Story Elevations
    story_elevations: List[Tuple[int, str, float]] = [(0, "Ground", 0.0)]
    curr_z = 0.0
    for s_idx, story in enumerate(draft.stories):
        if story.elevation is not None:
            curr_z = float(story.elevation)
        else:
            curr_z += float(story.height)
        story_elevations.append((s_idx + 1, story.name, round(curr_z, 4)))

    num_levels = len(story_elevations)

    # 3. Resolve Column Grid Intersections
    target_columns: List[Tuple[str, str, float, float, str]] = []
    if draft.columns:
        for c in draft.columns:
            gx, gy = _find_grid_pair(c.grid, coords_x, coords_y)
            sec_name = c.section or f"COL_{int((c.b or 0.45)*1000)}x{int((c.h or 0.45)*1000)}"
            target_columns.append((gx, gy, float(c.b or 0.45), float(c.h or 0.45), sec_name))
    else:
        # Default: columns at all grid intersections
        for gx in x_labels:
            for gy in y_labels:
                target_columns.append((gx, gy, 0.45, 0.45, "COL_450x450"))

    # 4. Initialize Project and Materials
    if existing_project is not None:
        proj = existing_project
        proj.metadata.title = draft.title
        proj.metadata.units = parse_units(draft.units)
        proj.model = StructuralModel()
    else:
        proj = Project(
            metadata=Metadata(title=draft.title, units=parse_units(draft.units)),
            model=StructuralModel(),
        )

    fc = float(draft.concrete_fc or 30.0)
    E_conc = round(4700.0 * math.sqrt(fc) * 1000.0, 1)  # kPa (kN/m²)
    mat_conc = ElasticMaterial(
        id="CONC",
        name=f"Concrete f'c={fc:.0f}MPa",
        E=E_conc,
        nu=0.20,
        rho=2400.0,
    )
    proj.model.materials.append(mat_conc)

    # Sections map
    created_sections: Dict[str, RectangularSection] = {}

    def _get_or_create_section(sec_id: str, b_val: float, h_val: float) -> str:
        if sec_id not in created_sections:
            sec = RectangularSection(id=sec_id, name=sec_id, b=round(b_val, 4), h=round(h_val, 4))
            created_sections[sec_id] = sec
            proj.model.sections.append(sec)
        return sec_id

    # 5. Generate Nodes and Base Supports
    node_id_map: Dict[Tuple[str, str, int], str] = {}
    node_counter = 1

    col_grid_set = {(c[0], c[1]) for c in target_columns}

    for lvl_idx, lvl_name, z_val in story_elevations:
        for gx, gy in col_grid_set:
            if gx in coords_x and gy in coords_y:
                x_val = coords_x[gx]
                y_val = coords_y[gy]
                nid = f"N_{gx}_{gy}_L{lvl_idx}"
                n_obj = Node(id=nid, x=x_val, y=y_val, z=z_val)
                proj.model.nodes.append(n_obj)
                node_id_map[(gx, gy, lvl_idx)] = nid
                node_counter += 1

                # Ground level boundary restraint
                if lvl_idx == 0:
                    if draft.support_type.lower() == "pinned":
                        proj.model.supports.append(Support.pinned_3d(nid))
                    else:
                        proj.model.supports.append(Support.fixed_3d(nid))

    # 6. Generate Columns
    elem_counter = 1
    for col_info in target_columns:
        gx, gy, b_col, h_col, sec_name = col_info
        sec_id = _get_or_create_section(sec_name, b_col, h_col)

        for lvl_idx in range(1, num_levels):
            bot_key = (gx, gy, lvl_idx - 1)
            top_key = (gx, gy, lvl_idx)
            if bot_key in node_id_map and top_key in node_id_map:
                cid = f"COL_{gx}_{gy}_S{lvl_idx}"
                elem = BeamElement(
                    id=cid,
                    start_node=node_id_map[bot_key],
                    end_node=node_id_map[top_key],
                    section_id=sec_id,
                    material_id="CONC",
                )
                proj.model.elements.append(elem)
                elem_counter += 1

    # 7. Generate Beams
    beam_elements: List[BeamElement] = []

    if draft.beams:
        for b_draft in draft.beams:
            g1x, g1y = _find_grid_pair(b_draft.from_grid, coords_x, coords_y)
            g2x, g2y = _find_grid_pair(b_draft.to_grid, coords_x, coords_y)
            b_sec_name = b_draft.section or f"BEAM_{int((b_draft.b or 0.30)*1000)}x{int((b_draft.h or 0.50)*1000)}"
            b_sec_id = _get_or_create_section(b_sec_name, float(b_draft.b or 0.30), float(b_draft.h or 0.50))

            for lvl_idx in range(1, num_levels):
                k1 = (g1x, g1y, lvl_idx)
                k2 = (g2x, g2y, lvl_idx)
                if k1 in node_id_map and k2 in node_id_map:
                    bid = f"BEAM_{g1x}{g1y}_{g2x}{g2y}_L{lvl_idx}"
                    beam = BeamElement(
                        id=bid,
                        start_node=node_id_map[k1],
                        end_node=node_id_map[k2],
                        section_id=b_sec_id,
                        material_id="CONC",
                    )
                    proj.model.elements.append(beam)
                    beam_elements.append(beam)
                    elem_counter += 1
    else:
        # Default: Framing beams connecting adjacent columns along X and Y grid lines
        default_beam_sec = _get_or_create_section("BEAM_300x500", 0.30, 0.50)

        for lvl_idx in range(1, num_levels):
            # Beams along X grid lines (for each gy, connect consecutive gx)
            for gy in y_labels:
                for i in range(len(x_labels) - 1):
                    gx1 = x_labels[i]
                    gx2 = x_labels[i + 1]
                    k1 = (gx1, gy, lvl_idx)
                    k2 = (gx2, gy, lvl_idx)
                    if k1 in node_id_map and k2 in node_id_map:
                        bid = f"BEAM_X_{gx1}-{gx2}_{gy}_L{lvl_idx}"
                        beam = BeamElement(
                            id=bid,
                            start_node=node_id_map[k1],
                            end_node=node_id_map[k2],
                            section_id=default_beam_sec,
                            material_id="CONC",
                        )
                        proj.model.elements.append(beam)
                        beam_elements.append(beam)
                        elem_counter += 1

            # Beams along Y grid lines (for each gx, connect consecutive gy)
            for gx in x_labels:
                for j in range(len(y_labels) - 1):
                    gy1 = y_labels[j]
                    gy2 = y_labels[j + 1]
                    k1 = (gx, gy1, lvl_idx)
                    k2 = (gx, gy2, lvl_idx)
                    if k1 in node_id_map and k2 in node_id_map:
                        bid = f"BEAM_Y_{gx}_{gy1}-{gy2}_L{lvl_idx}"
                        beam = BeamElement(
                            id=bid,
                            start_node=node_id_map[k1],
                            end_node=node_id_map[k2],
                            section_id=default_beam_sec,
                            material_id="CONC",
                        )
                        proj.model.elements.append(beam)
                        beam_elements.append(beam)
                        elem_counter += 1

    # 8. Load Cases & Distributed Floor Loading on Beams
    dead_case_id = draft.dead_case or "DEAD"
    live_case_id = draft.live_case or "LIVE"

    lc_dead = LoadCase(id=dead_case_id, name="Dead Load")
    lc_live = LoadCase(id=live_case_id, name="Live Load")

    dl_mag = float(draft.dead_load or 0.0)
    ll_mag = float(draft.live_load or 0.0)

    if dl_mag > 0:
        for b in beam_elements:
            lc_dead.element_loads.append(
                DistributedLoad(element_id=b.id, wz=-dl_mag)
            )

    if ll_mag > 0:
        for b in beam_elements:
            lc_live.element_loads.append(
                DistributedLoad(element_id=b.id, wz=-ll_mag)
            )

    proj.model.load_cases.extend([lc_dead, lc_live])

    # 9. Standard ASCE 7 Factored Load Combinations
    c1 = LoadCombination(
        id="COMB1",
        name="1.4D",
        factors=[LoadCaseFactor(load_case_id=dead_case_id, factor=1.4)],
    )
    c2 = LoadCombination(
        id="COMB2",
        name="1.2D+1.6L",
        factors=[
            LoadCaseFactor(load_case_id=dead_case_id, factor=1.2),
            LoadCaseFactor(load_case_id=live_case_id, factor=1.6),
        ],
    )
    proj.model.load_combinations.extend([c1, c2])

    return proj
