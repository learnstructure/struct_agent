"""
struct_agent/vision/compiler.py
══════════════════════════════
Deterministic structural grid compiler that converts intermediate semantic
FloorPlanDraft representations into mathematically exact struct_core.Project models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from struct_core import (
    BeamElement,
    DistributedLoad,
    LoadCase,
    LoadCaseFactor,
    LoadCombination,
    Metadata,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    conversion_factor,
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
    applied: Optional[List[Any]] = None,
) -> Project:
    """
    Compile a FloorPlanDraft into an exact struct_core.Project. The draft must give the concrete f'c; the defaults
    applied for what it omits (bars, cover, unit weight) are appended to ``applied`` when it is given.

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
    if draft.fc is None:
        raise ValueError("FloorPlanDraft requires the concrete f'c (fc, strictly in the draft's stress unit).")

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

    from .. import defaults
    from ..tools.materials import concrete_material, rebar_material
    from ..tools.units_input import check_realistic

    units = parse_units(draft.units)
    used_defaults: List[AppliedDefault] = []
    if "units" not in draft.model_fields_set:
        used_defaults.append(defaults.UNITS.record())

    # Section sizes for members without their own (0.45 m columns, 0.30 x 0.50 m beams in the draft's units)
    col_b0 = draft.default_col_b if draft.default_col_b is not None else float(defaults.TRIAL_COLUMN_SIZE.in_units(units))
    col_h0 = draft.default_col_h if draft.default_col_h is not None else float(defaults.TRIAL_COLUMN_SIZE.in_units(units))
    beam_b0 = draft.default_beam_b if draft.default_beam_b is not None else float(defaults.TRIAL_BEAM_WIDTH.in_units(units))
    beam_h0 = draft.default_beam_h if draft.default_beam_h is not None else float(defaults.TRIAL_BEAM_DEPTH.in_units(units))

    if draft.default_col_b is None:
        used_defaults.append(defaults.TRIAL_COLUMN_SIZE.record(applies_to="col_b", model_units=units))
    if draft.default_col_h is None:
        used_defaults.append(defaults.TRIAL_COLUMN_SIZE.record(applies_to="col_h", model_units=units))
    if draft.default_beam_b is None:
        used_defaults.append(defaults.TRIAL_BEAM_WIDTH.record(applies_to="beam_b", model_units=units))
    if draft.default_beam_h is None:
        used_defaults.append(defaults.TRIAL_BEAM_DEPTH.record(applies_to="beam_h", model_units=units))

    mm = conversion_factor(units, "N, mm", "length")

    def _sec_name(prefix: str, b: float, h: float, default: Tuple[float, float]) -> str:
        # Members of the default size share the stable ID 'COL' / 'BEAM' (resizing keeps the ID, so it
        # must not encode dimensions); explicitly sized groups are named by their size
        if (b, h) == default:
            return prefix
        return f"{prefix}_{round(b * mm)}x{round(h * mm)}"

    # 3. Resolve Column Grid Intersections
    target_columns: List[Tuple[str, str, float, float, str]] = []
    if draft.columns:
        for c in draft.columns:
            gx, gy = _find_grid_pair(c.grid, coords_x, coords_y)
            b, h = float(c.b if c.b is not None else col_b0), float(c.h if c.h is not None else col_h0)
            target_columns.append((gx, gy, b, h, c.section or _sec_name("COL", b, h, (col_b0, col_h0))))
    else:
        # Default: columns at all grid intersections
        for gx in x_labels:
            for gy in y_labels:
                target_columns.append((gx, gy, col_b0, col_h0, "COL"))

    # 4. Initialize Project and Materials
    if existing_project is not None:
        proj = existing_project
        proj.metadata.title = draft.title
        proj.units = parse_units(draft.units)
        proj.model = StructuralModel()
    else:
        proj = Project(
            metadata=Metadata(title=draft.title, units=parse_units(draft.units)),
            model=StructuralModel(),
        )

    # Materials: concrete (f'c required) and reinforcing steel, in the draft units
    concrete, used = concrete_material(
        "CONC", fc=draft.fc, units=units,
        unit_weight=draft.unit_weight)
    used = list(used_defaults) + used
    rebar, used_rebar = rebar_material("REBAR", units=units, fy=draft.fy)
    proj.model.materials.extend([concrete, rebar])
    used += used_rebar
    transverse_id = rebar.id
    if draft.fyt is not None:
        transverse, used_transverse = rebar_material(
            "REBAR_TRANSVERSE", units=units, fy=draft.fyt)
        proj.model.materials.append(transverse)
        used += used_transverse
        transverse_id = transverse.id
    cover = draft.cover
    if cover is None:
        cover = defaults.COVER.in_units(units)
        used.append(defaults.COVER.record(model_units=units))
    check_realistic("cover", cover, units)
    if applied is not None:
        applied.extend(used)
    proj.defaults_applied = list(used)
    reinforcement = dict(material_id=concrete.id, rebar_material_id=rebar.id,
                         transverse_rebar_material_id=transverse_id, cover=cover)

    # Sections map
    created_sections: Dict[str, RectangularSection] = {}

    def _get_or_create_section(sec_id: str, b_val: float, h_val: float) -> str:
        if sec_id not in created_sections:
            sec = RectangularSection(id=sec_id, name=sec_id, b=round(b_val, 4), h=round(h_val, 4), **reinforcement)
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
                )
                proj.model.elements.append(elem)
                elem_counter += 1

    # 7. Generate Beams
    beam_elements: List[BeamElement] = []

    if draft.beams:
        for b_draft in draft.beams:
            g1x, g1y = _find_grid_pair(b_draft.from_grid, coords_x, coords_y)
            g2x, g2y = _find_grid_pair(b_draft.to_grid, coords_x, coords_y)
            bb = float(b_draft.b if b_draft.b is not None else beam_b0)
            bh = float(b_draft.h if b_draft.h is not None else beam_h0)
            b_sec_id = _get_or_create_section(b_draft.section or _sec_name("BEAM", bb, bh, (beam_b0, beam_h0)), bb, bh)

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
                    )
                    proj.model.elements.append(beam)
                    beam_elements.append(beam)
                    elem_counter += 1
    else:
        # Default: Framing beams connecting adjacent columns along X and Y grid lines
        default_beam_sec = _get_or_create_section("BEAM", beam_b0, beam_h0)

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
                        )
                        proj.model.elements.append(beam)
                        beam_elements.append(beam)
                        elem_counter += 1

    # 8. Load Cases & Distributed Floor Loading on Beams
    dead_case_id = draft.dead_case or "DEAD"
    live_case_id = draft.live_case or "LIVE"

    dl_mag = float(draft.dead_load or 0.0)
    ll_mag = float(draft.live_load or 0.0)

    created_cases: List[LoadCase] = []
    if dl_mag > 0:
        lc_dead = LoadCase(id=dead_case_id, name="Dead Load", load_type="dead")
        for b in beam_elements:
            lc_dead.element_loads.append(
                DistributedLoad(element_id=b.id, wz=-dl_mag)
            )
        created_cases.append(lc_dead)

    if ll_mag > 0:
        lc_live = LoadCase(id=live_case_id, name="Live Load", load_type="live")
        for b in beam_elements:
            lc_live.element_loads.append(
                DistributedLoad(element_id=b.id, wz=-ll_mag)
            )
        created_cases.append(lc_live)

    proj.model.load_cases.extend(created_cases)

    # 9. Standard ASCE 7 Factored Load Combinations
    if created_cases:
        from struct_load import LoadGenerator
        gen = LoadGenerator(units=proj.units)
        gen.generate_combinations(proj)

    return proj
