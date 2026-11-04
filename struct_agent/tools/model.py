"""
Model tools.

These wrap ``struct_core`` operations for project container lifecycle,
structural model creation (both parametric frames and raw structured JSON),
inspection, and modification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from struct_core import (
    AppliedDefault,
    BeamElement,
    CircularSection,
    ElasticMaterial,
    GeneralSection,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    TrussElement,
    Units,
    conversion_factor,
    validate_structural_model,
    load_json,
    parse_units,
    save_json,
)

from ..state.project_state import ProjectState, WorkflowStage
from ..vision.compiler import build_project_from_draft
from ..vision.schemas import FloorPlanDraft, normalize_floor_plan_dict
from .base import AgentError, ToolResult
from .. import defaults
from .materials import concrete_material, rebar_material
from .units_input import check_realistic


class ModelInputError(ValueError):
    """Model data that cannot become a model; the message lists every problem."""


def _material_from_dict(data: Dict[str, Any], units: Any) -> Tuple[Any, List[AppliedDefault]]:
    """A material from tool input (model units): concrete when it is typed so or gives f'c, rebar when typed so or
    gives fy (their defaults applied and returned), otherwise an elastic material with the given E."""
    kind = str(data.get("type", "")).lower()
    mid = data.get("id")
    if mid is None:
        raise ModelInputError(f"A material has no id: {data}")
    if kind == "concrete" or (kind in ("", "elastic") and data.get("fc") is not None):
        if data.get("fc") is None:
            raise ModelInputError(f"Concrete material {mid!r} needs fc (f'c, model stress unit).")
        material, used = concrete_material(mid, fc=float(data["fc"]), units=units, E=data.get("E"), nu=data.get("nu"),
                                           unit_weight=data.get("unit_weight"), name=data.get("name"))
        if data.get("rho") is not None:
            material = material.model_copy(update={"rho": data["rho"]})
        return material, used
    if kind == "rebar" or (kind in ("", "elastic") and data.get("fy") is not None):
        material, used = rebar_material(mid, units=units, fy=data.get("fy"), fu=data.get("fu"), E=data.get("E"),
                                        unit_weight=data.get("unit_weight"), name=data.get("name"))
        if data.get("rho") is not None:
            material = material.model_copy(update={"rho": data["rho"]})
        return material, used
    if data.get("E") is None:
        raise ModelInputError(f"Material {mid!r} needs E (or type 'concrete' with fc, or 'rebar').")
    return ElasticMaterial(**{k: v for k, v in data.items() if k != "type"}), []


def _merged_material_input(current: Any, given: Dict[str, Any]) -> Dict[str, Any]:
    """Tool input for updating material ``current`` (None for a new one): the given values over the current name,
    ν, unit weight and mass density, and over its type unless a new one is given. Its E is kept when it was a stated
    modulus (an elastic material, e.g. imported from SAP2000 / ETABS, that is now given f'c); the E of a concrete
    follows a new f'c unless E is given."""
    if current is None:
        return given
    kept = {k: getattr(current, k, None) for k in ("name", "nu", "unit_weight", "rho")}
    if "type" not in given and current.type in ("concrete", "rebar"):
        kept["type"] = current.type
    if current.type == "elastic" or not ({"fc", "fy"} & set(given)):
        kept["E"] = current.E
    for key in ("fc", "fy", "fu"):
        if getattr(current, key, None) is not None:
            kept[key] = getattr(current, key)
    return {**{k: v for k, v in kept.items() if v is not None}, **given}


def _section_from_dict(data: Dict[str, Any]) -> Any:
    """A section from tool input: circular with d, general with A, otherwise rectangular (b, h)."""
    if data.get("material_id") is None:
        raise ModelInputError(f"Section {data.get('id')!r} needs material_id (a section owns its material).")
    fields = {k: v for k, v in data.items() if k != "type"}
    if data.get("type") == "circular" or "d" in data:
        return CircularSection(**fields)
    if data.get("type") == "general" or "A" in data:
        return GeneralSection(**fields)
    return RectangularSection(**fields)


def complete_concrete_sections(materials: List[Any], sections: List[Any], units: Any) -> List[AppliedDefault]:
    """Give concrete sections (a ``ConcreteMaterial``) without reinforcement data their bars and cover: the model's
    reinforcing steel when it has exactly one (e.g. the one just given), Grade 60 bars (material 'REBAR', added) when
    it has none, and the 1.5 in clear cover. With several rebar materials the section is left for the user to
    assign. Updates the lists in place and returns the defaults applied."""
    applied: List[AppliedDefault] = []
    concrete_ids = {m.id for m in materials if getattr(m, "type", None) == "concrete"}
    for i, sec in enumerate(sections):
        if sec.material_id not in concrete_ids or not hasattr(sec, "cover"):
            continue
        update: Dict[str, Any] = {}
        if sec.rebar_material_id is None:
            rebars = [m.id for m in materials if getattr(m, "type", None) == "rebar"]
            if not rebars:
                rebar, used = rebar_material("REBAR", units=units)
                materials.append(rebar)
                applied += used
                rebars = [rebar.id]
            if len(rebars) == 1:
                update["rebar_material_id"] = rebars[0]
        if sec.transverse_rebar_material_id is None:
            update["transverse_rebar_material_id"] = update.get("rebar_material_id", sec.rebar_material_id)
        if sec.cover is None:
            update["cover"] = defaults.COVER.in_units(units)
            applied.append(defaults.COVER.record(sec.id, model_units=units))
        if update:
            sections[i] = sec.model_copy(update=update)
    return applied


def missing_design_inputs(model: StructuralModel) -> List[str]:
    """What keeps the frame members from reinforced-concrete design: sections whose material is not a concrete with
    f'c, or that lack rebar materials or cover."""
    used = {e.section_id for e in model.elements}
    problems = []
    for sec in model.sections:
        if sec.id not in used:
            continue
        mat = next((m for m in model.materials if m.id == sec.material_id), None)
        if getattr(mat, "type", None) != "concrete":
            problems.append(f"section {sec.id!r}: material {sec.material_id!r} has no f'c (give it type 'concrete' "
                            f"and fc with modify_project)")
        elif getattr(sec, "rebar_material_id", None) is None or getattr(sec, "cover", None) is None:
            problems.append(f"section {sec.id!r}: no reinforcement (rebar material, cover); give the bars with "
                            f"modify_project(materials=[{{'id': 'REBAR', 'type': 'rebar', 'fy': <fy>}}])")
    return problems


def parse_model_dict(
    model_data: Dict[str, Any],
    units: Any = None,
) -> Tuple[StructuralModel, List[AppliedDefault]]:
    """
    Parse a dictionary of nodes, elements, materials, sections and supports (model units) into a validated
    ``StructuralModel``, and the defaults applied.

    Nothing is invented: every node needs x and y, every section its ``material_id``, every element its
    ``section_id``, and every material its properties (concrete f'c, an elastic E). Concrete materials and concrete
    sections take the documented defaults for what is not given (E from f'c, unit weight, Grade 60 bars, 1.5 in
    cover). Raises ``ModelInputError`` listing every problem.
    """
    applied: List[AppliedDefault] = []
    given_units = model_data.get("units") or units
    if given_units is None:
        units = parse_units(defaults.UNITS.value)
        applied.append(defaults.UNITS.record())
    else:
        units = parse_units(given_units)
    problems: List[str] = []

    nodes: List[Node] = []
    for n in model_data.get("nodes", []):
        if isinstance(n, Node):
            nodes.append(n)
        elif isinstance(n, dict):
            if n.get("x") is None or n.get("y") is None:
                problems.append(f"Node {n.get('id')!r} needs x and y")
                continue
            nodes.append(Node(id=n.get("id", len(nodes) + 1), x=float(n["x"]), y=float(n["y"]),
                              z=float(n.get("z") or 0.0)))

    materials: List[Any] = []
    for m in model_data.get("materials", []):
        if not isinstance(m, dict):
            materials.append(m)
            continue
        try:
            material, used = _material_from_dict(dict(m), units)
        except (ModelInputError, ValueError) as exc:
            problems.append(str(exc))
            continue
        materials.append(material)
        applied += used

    sections: List[Any] = []
    for sec in model_data.get("sections", []):
        if not isinstance(sec, dict):
            sections.append(sec)
            continue
        try:
            check_realistic("cover", sec.get("cover"), units)
            sections.append(_section_from_dict(sec))
        except (ModelInputError, ValueError) as exc:
            problems.append(str(exc))

    applied += complete_concrete_sections(materials, sections, units)

    elements: List[Any] = []
    for e in model_data.get("elements", []):
        if not isinstance(e, dict):
            elements.append(e)
            continue
        etype = str(e.get("type", "beam")).lower()
        eid = e.get("id", f"E{len(elements) + 1}")
        if etype in ("beam", "truss") and e.get("section_id") is None:
            problems.append(f"Element {eid!r} needs section_id")
            continue
        fields = {k: v for k, v in e.items() if k not in ("type", "id")}
        elements.append((TrussElement if etype == "truss" else BeamElement)(id=eid, **fields))

    has_explicit_dim = "dimension" in model_data and model_data["dimension"] is not None
    non_zero_z_nodes = [n for n in nodes if abs(n.z) > 1e-9]
    if has_explicit_dim:
        explicit_dim = str(model_data["dimension"]).strip().upper()
        if explicit_dim == "2D":
            if non_zero_z_nodes:
                bad_ids = [n.id for n in non_zero_z_nodes[:5]]
                problems.append(
                    f"dimension='2D' but nodes {bad_ids} have z != 0"
                )
            is_3d = False
        elif explicit_dim == "3D":
            is_3d = True
        else:
            problems.append(f"Invalid dimension {model_data['dimension']!r}; must be '2D' or '3D'")
            is_3d = bool(non_zero_z_nodes)
    else:
        is_3d = bool(non_zero_z_nodes)
        if not is_3d:
            applied.append(
                AppliedDefault(
                    name="dimension",
                    value="2D",
                    source="all nodes at z = 0: plane frame in X-Y with Y vertical (struct_core convention); give dimension='3D' for a horizontal grid",
                )
            )

    # 5. Parse Supports
    supports: List[Support] = []
    allowed_sup_keys = {
        "node_id", "fixity", "name",
        "ux", "uy", "uz", "rx", "ry", "rz",
        "kx", "ky", "kz", "krx", "kry", "krz",
    }
    for sp in model_data.get("supports", []):
        if isinstance(sp, Support):
            supports.append(sp)
        elif isinstance(sp, dict):
            if "node_id" not in sp or sp["node_id"] is None:
                problems.append("Support definition is missing required 'node_id'")
                continue
            nid = sp["node_id"]
            unknown = set(sp.keys()) - allowed_sup_keys
            if unknown:
                problems.append(f"Support for node {nid!r} contains unknown key(s): {sorted(unknown)}")
                continue

            fixity = str(sp.get("fixity", "")).lower()
            if fixity in ("fixed_3d", "fixed"):
                sup = Support.fixed_3d(node_id=nid) if (is_3d or fixity == "fixed_3d") else Support.fixed(node_id=nid)
            elif fixity in ("pinned_3d", "pinned"):
                sup = Support.pinned_3d(node_id=nid) if (is_3d or fixity == "pinned_3d") else Support.pinned(node_id=nid)
            elif fixity in ("roller", "roller_2d"):
                sup = Support.roller(node_id=nid)
            elif fixity:
                problems.append(f"Unknown support fixity {fixity!r} for node {nid!r}")
                continue
            else:
                sup = Support(
                    node_id=nid,
                    ux=bool(sp.get("ux", False)),
                    uy=bool(sp.get("uy", False)),
                    uz=bool(sp.get("uz", False)),
                    rx=bool(sp.get("rx", False)),
                    ry=bool(sp.get("ry", False)),
                    rz=bool(sp.get("rz", False)),
                )
            for k_attr in ("kx", "ky", "kz", "krx", "kry", "krz"):
                if k_attr in sp and sp[k_attr] is not None:
                    setattr(sup, k_attr, float(sp[k_attr]))
            if "name" in sp and sp["name"] is not None:
                sup.name = str(sp["name"])
            supports.append(sup)

    if problems:
        raise ModelInputError("; ".join(problems))
    dim = "3D" if is_3d else "2D"
    v_axis = "Z" if is_3d else "Y"
    model = StructuralModel(
        dimension=dim,
        vertical_axis=v_axis,
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
    )
    errors = validate_structural_model(model, raise_error=False)
    if errors:
        raise ModelInputError("; ".join(errors))
    return model, applied


def _material_summary(material: Any) -> Dict[str, Any]:
    """The design-relevant properties of a material, for tool results (model units)."""
    summary: Dict[str, Any] = {"id": material.id, "type": material.type, "E": material.E}
    for attr in ("fc", "fy", "fu", "unit_weight"):
        value = getattr(material, attr, None)
        if value is not None:
            summary[attr] = value
    return summary


def create_frame_model(
    num_bays: Optional[int] = None,
    bay_width: Optional[float] = None,
    num_stories: Optional[int] = None,
    story_height: Optional[float] = None,
    *,
    fc: float,
    fy: Optional[float] = None,
    fyt: Optional[float] = None,
    unit_weight: Optional[float] = None,
    cover: Optional[float] = None,
    bays_x: Optional[int] = None,
    bays_y: Optional[int] = None,
    bay_width_x: Optional[float] = None,
    bay_width_y: Optional[float] = None,
    bay_widths_x: Optional[List[float]] = None,
    bay_widths_y: Optional[List[float]] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    units: Optional[Union[str, Any, dict]] = None,
    dimension: str = "2d",
    story_heights: Optional[List[float]] = None,
    support_type: str = "fixed",
    col_b: Optional[float] = None,
    col_h: Optional[float] = None,
    beam_b: Optional[float] = None,
    beam_h: Optional[float] = None,
) -> ToolResult:
    """
    Generate a complete, fully connected reinforced-concrete frame model (2D or 3D): grid nodes, columns, beams,
    supports, the concrete and reinforcing-steel materials, and the column and beam sections that own them.

    Geometry is required: the X bays (``bay_widths_x``, or a count ``bays_x`` / ``num_bays`` with a spacing
    ``bay_width_x`` / ``bay_width``), in 3D the Y bays likewise (``bay_widths_y``, or ``bays_y`` with ``bay_width_y`` /
    ``bay_width``), and the stories (``story_heights``, or ``num_stories`` with ``story_height``). Lengths, including
    the section sizes and ``cover``, are in the model length unit.

    Materials: ``fc`` (concrete f'c) is required; ``fy`` (longitudinal bars), ``fyt`` (stirrups and ties),
    ``unit_weight`` (concrete) and ``cover`` (clear cover) take their defaults when omitted (Grade 60, fyt = fy,
    150 pcf, 1.5 in), and the result lists every default applied. Stresses and unit weights are strictly in the model
    units. Omitted section sizes are trial sizes (0.45 m columns, 0.30 x 0.50 m beams, in the model unit).
    """
    try:
        applied: List[Any] = []
        if units is None:
            active_units = parse_units(defaults.UNITS.value)
            applied.append(defaults.UNITS.record())
        else:
            active_units = parse_units(units)
        is_3d = str(dimension or "2d").lower() == "3d"
        missing = []
        if not bay_widths_x and ((bays_x if bays_x is not None else num_bays) is None
                                 or (bay_width_x if bay_width_x is not None else bay_width) is None):
            missing.append("X bays (bay_widths_x, or bays_x / num_bays with bay_width_x / bay_width)")
        if is_3d and not bay_widths_y and ((bays_y if bays_y is not None else num_bays) is None
                                           or (bay_width_y if bay_width_y is not None else bay_width) is None):
            missing.append("Y bays (bay_widths_y, or bays_y / num_bays with bay_width_y / bay_width)")
        if not story_heights and (num_stories is None or story_height is None):
            missing.append("stories (story_heights, or num_stories with story_height)")
        if missing:
            return ToolResult.failure("create_frame_model", AgentError(
                error_type="invalid_input",
                message="create_frame_model needs the frame geometry; missing: " + "; ".join(missing) + "."))

        sizes = {"col_b": col_b, "col_h": col_h, "beam_b": beam_b, "beam_h": beam_h}
        if sizes["col_b"] is None and sizes["col_h"] is None:
            val = float(defaults.TRIAL_COLUMN_SIZE.in_units(active_units))
            sizes["col_b"] = val
            sizes["col_h"] = val
            applied.append(defaults.TRIAL_COLUMN_SIZE.record(applies_to="COL", model_units=active_units))
        else:
            for key in ("col_b", "col_h"):
                if sizes[key] is None:
                    val = float(defaults.TRIAL_COLUMN_SIZE.in_units(active_units))
                    sizes[key] = val
                    applied.append(defaults.TRIAL_COLUMN_SIZE.record(applies_to=f"COL.{key}", model_units=active_units))

        for key, default_obj in [("beam_b", defaults.TRIAL_BEAM_WIDTH), ("beam_h", defaults.TRIAL_BEAM_DEPTH)]:
            if sizes[key] is None:
                sizes[key] = float(default_obj.in_units(active_units))
                applied.append(default_obj.record(applies_to="BEAM", model_units=active_units))
        col_b, col_h, beam_b, beam_h = sizes["col_b"], sizes["col_h"], sizes["beam_b"], sizes["beam_h"]

        # Materials: concrete and reinforcing steel (the transverse grade only when it differs)
        concrete, concrete_defaults = concrete_material(
            "CONCRETE", fc=fc, units=active_units, unit_weight=unit_weight)
        rebar, rebar_defaults = rebar_material(
            "REBAR", units=active_units, fy=fy)
        materials = [concrete, rebar]
        applied += concrete_defaults + rebar_defaults
        transverse_id = rebar.id
        if fyt is not None and abs(fyt - rebar.fy) > 1e-9 * rebar.fy:
            transverse, transverse_defaults = rebar_material(
                "REBAR_TRANSVERSE", units=active_units, fy=fyt)
            materials.append(transverse)
            applied += transverse_defaults
            transverse_id = transverse.id
        elif fyt is None:
            applied.append(AppliedDefault(name="fyt", value="fy", unit=None,
                                          source="transverse bars of the longitudinal grade", applies_to=rebar.id))
        if cover is None:
            cover = defaults.COVER.in_units(active_units)
            applied.append(defaults.COVER.record(model_units=active_units))
        check_realistic("cover", cover, active_units)
        rc = dict(material_id=concrete.id, rebar_material_id=rebar.id, transverse_rebar_material_id=transverse_id,
                  cover=cover)
        materials_summary = [_material_summary(m) for m in materials]

        # Determine bay coordinates along X
        if bay_widths_x:
            x_coords = [0.0]
            for w in bay_widths_x:
                x_coords.append(x_coords[-1] + float(w))
            eff_bays_x = len(bay_widths_x)
        else:
            eff_bays_x = bays_x if bays_x is not None else num_bays
            bw_x = bay_width_x if bay_width_x is not None else bay_width
            x_coords = [float(i * bw_x) for i in range(eff_bays_x + 1)]

        # Determine story elevations along Y (2D) or Z (3D)
        if story_heights and len(story_heights) > 0:
            elev_coords = [0.0]
            for h in story_heights:
                elev_coords.append(elev_coords[-1] + float(h))
            num_stories = len(story_heights)
        else:
            elev_coords = [float(j * story_height) for j in range(num_stories + 1)]
        y_coords = elev_coords

        # 2. Sections, each with its concrete, bars and cover. Stable section IDs: resizing keeps the ID, so it must
        # not encode the dimensions
        col_sec_id, beam_sec_id = "COL", "BEAM"
        sections = [
            RectangularSection(id=col_sec_id, b=col_b, h=col_h, **rc),
            RectangularSection(id=beam_sec_id, b=beam_b, h=beam_h, **rc),
        ]

        if is_3d:
            # 3D Building Frame: X and Y are floor plan, Z is vertical elevation
            eff_bays_y = bays_y if bays_y is not None else num_bays
            if bay_widths_y and len(bay_widths_y) > 0:
                y_coords_plan = [0.0]
                for w in bay_widths_y:
                    y_coords_plan.append(y_coords_plan[-1] + float(w))
                eff_bays_y = len(bay_widths_y)
            else:
                bw_y = bay_width_y if bay_width_y is not None else bay_width
                y_coords_plan = [float(iy * bw_y) for iy in range(eff_bays_y + 1)]

            bays_x = eff_bays_x
            bays_y = eff_bays_y
            z_coords = elev_coords

            nodes: List[Node] = []
            node_map_3d: Dict[tuple, int] = {}
            supports: List[Support] = []
            node_id_ctr = 1

            for s_idx in range(num_stories + 1):
                z = z_coords[s_idx]
                for iy in range(bays_y + 1):
                    y = y_coords_plan[iy]
                    for ix in range(bays_x + 1):
                        x = x_coords[ix]
                        nid = node_id_ctr
                        node_id_ctr += 1
                        nodes.append(Node(id=nid, x=x, y=y, z=z))
                        node_map_3d[(ix, iy, s_idx)] = nid
                        if s_idx == 0:
                            if support_type.lower() == "pinned":
                                supports.append(Support(node_id=nid, ux=True, uy=True, uz=True, rx=False, ry=False, rz=False))
                            else:
                                supports.append(Support.fixed_3d(nid))

            elements: List[BeamElement] = []
            # Columns along Z
            for s_idx in range(num_stories):
                for iy in range(bays_y + 1):
                    for ix in range(bays_x + 1):
                        cid = f"C_{ix+1}_{iy+1}_{s_idx+1}"
                        elements.append(
                            BeamElement(
                                id=cid,
                                name=f"Col X{ix+1}Y{iy+1} Story{s_idx+1}",
                                start_node=node_map_3d[(ix, iy, s_idx)],
                                end_node=node_map_3d[(ix, iy, s_idx + 1)],
                                section_id=col_sec_id,
                            )
                        )

            # Beams along X
            for s_idx in range(1, num_stories + 1):
                for iy in range(bays_y + 1):
                    for ix in range(bays_x):
                        bid = f"BX_{ix+1}_{iy+1}_{s_idx}"
                        elements.append(
                            BeamElement(
                                id=bid,
                                name=f"Beam X{ix+1} GridY{iy+1} Story{s_idx}",
                                start_node=node_map_3d[(ix, iy, s_idx)],
                                end_node=node_map_3d[(ix + 1, iy, s_idx)],
                                section_id=beam_sec_id,
                            )
                        )

            # Beams along Y
            for s_idx in range(1, num_stories + 1):
                for iy in range(bays_y):
                    for ix in range(bays_x + 1):
                        bid = f"BY_{ix+1}_{iy+1}_{s_idx}"
                        elements.append(
                            BeamElement(
                                id=bid,
                                name=f"Beam Y{iy+1} GridX{ix+1} Story{s_idx}",
                                start_node=node_map_3d[(ix, iy, s_idx)],
                                end_node=node_map_3d[(ix, iy + 1, s_idx)],
                                section_id=beam_sec_id,
                            )
                        )

            model = StructuralModel(
                dimension="3D",
                vertical_axis="Z",
                nodes=nodes,
                elements=elements,
                materials=materials,
                sections=sections,
                supports=supports,
                story_elevations=z_coords,
                units=active_units,
            )
            project_title = title or f"3D {bays_x}x{bays_y} Frame ({num_stories} Stories)"
            project = Project(model=model)
            project.metadata.title = project_title
            if active_units:
                project.units = active_units
            if author:
                project.metadata.author = author
            project.defaults_applied = list(applied)

            return ToolResult.success(
                "create_frame_model",
                data={
                    "project": project,
                    "title": project.metadata.title,
                    "units": active_units.short_label if active_units else "kN, m",
                    "n_nodes": len(nodes),
                    "n_elements": len(elements),
                    "n_columns": (bays_x + 1) * (bays_y + 1) * num_stories,
                    "n_beams": (bays_x * (bays_y + 1) + bays_y * (bays_x + 1)) * num_stories,
                    "n_supports": len(supports),
                    "section_ids": {"columns": col_sec_id, "beams": beam_sec_id},
                    "materials": materials_summary,
                    "cover": cover,
                    "defaults_applied": [d.model_dump() for d in applied],
                    "bays_x": bays_x,
                    "bays_y": bays_y,
                    "stories": num_stories,
                    "total_height": z_coords[-1],
                    "total_width_x": x_coords[-1],
                    "total_width_y": y_coords_plan[-1],
                },
                next_suggested_stage="MODEL_CREATED",
            )

        # 2D Frame Model
        nodes: List[Node] = []
        node_map: Dict[tuple, int] = {}
        supports: List[Support] = []

        # 1. Grid Nodes & Ground Supports
        for j in range(num_stories + 1):
            for i in range(eff_bays_x + 1):
                nid = j * (eff_bays_x + 1) + i + 1
                node_map[(i, j)] = nid
                nodes.append(Node(id=nid, x=x_coords[i], y=y_coords[j], z=0.0))
                if j == 0:
                    if support_type.lower() == "pinned":
                        supports.append(Support.pinned(node_id=nid))
                    else:
                        supports.append(Support.fixed(node_id=nid))

        # 3. Elements (Columns & Beams)
        elements: List[BeamElement] = []
        # Vertical Columns
        for j in range(num_stories):
            for i in range(eff_bays_x + 1):
                cid = f"C_{i+1}_{j+1}"
                elements.append(
                    BeamElement(
                        id=cid,
                        name=f"Column Bay {i+1} Story {j+1}",
                        start_node=node_map[(i, j)],
                        end_node=node_map[(i, j + 1)],
                        section_id=col_sec_id,
                    )
                )

        # Horizontal Beams
        for j in range(1, num_stories + 1):
            for i in range(eff_bays_x):
                bid = f"B_{i+1}_{j}"
                elements.append(
                    BeamElement(
                        id=bid,
                        name=f"Beam Bay {i+1} Level {j}",
                        start_node=node_map[(i, j)],
                        end_node=node_map[(i + 1, j)],
                        section_id=beam_sec_id,
                    )
                )

        model = StructuralModel(
            dimension="2D",
            vertical_axis="Y",
            nodes=nodes,
            elements=elements,
            materials=materials,
            sections=sections,
            supports=supports,
            story_elevations=y_coords,
            units=active_units,
        )

        project_title = title or f"{eff_bays_x}-Bay {num_stories}-Story Frame"
        project = Project(model=model)
        project.metadata.title = project_title
        if active_units:
            project.units = active_units
        if author:
            project.metadata.author = author
        project.defaults_applied = list(applied)

        return ToolResult.success(
            "create_frame_model",
            data={
                "project": project,
                "title": project.metadata.title,
                "units": active_units.short_label if active_units else "kN, m",
                "n_nodes": len(nodes),
                "n_elements": len(elements),
                "n_columns": (eff_bays_x + 1) * num_stories,
                "n_beams": eff_bays_x * num_stories,
                "n_supports": len(supports),
                "section_ids": {"columns": col_sec_id, "beams": beam_sec_id},
                "materials": materials_summary,
                "cover": cover,
                "defaults_applied": [d.model_dump() for d in applied],
                "bays": eff_bays_x,
                "stories": num_stories,
                "total_height": y_coords[-1],
                "total_width": x_coords[-1],
            },
            next_suggested_stage="MODEL_CREATED",
        )
    except ValueError as exc:  # an input outside its realistic range, or another invalid value
        return ToolResult.failure("create_frame_model", AgentError(error_type="invalid_input", message=str(exc)))
    except Exception as exc:
        return ToolResult.failure(
            "create_frame_model",
            AgentError(
                error_type="model_creation_error",
                message=f"Failed to generate frame model: {exc}",
                possible_causes=["Invalid dimension parameters or numeric values"],
            ),
        )


def _model_check(tool: str, model: StructuralModel) -> Optional[ToolResult]:
    """A failure for elements or supports on undefined nodes (minimax C04 built a model of one element between two
    nonexistent nodes, reported as a success), else None."""
    node_ids = {str(n.id) for n in model.nodes}
    dangling = sorted({str(n) for e in model.elements for n in (e.start_node, e.end_node) if str(n) not in node_ids}
                      | {str(s.node_id) for s in model.supports if str(s.node_id) not in node_ids})
    if dangling:
        return ToolResult.failure(tool, AgentError(
            error_type="invalid_input",
            message=f"Elements or supports refer to nodes that are not defined: {dangling[:8]}. Define every node in "
                    f"'nodes' (id, x, y, z); the model was not changed.",
        ))
    return None


def _parse_or_fail(
    tool: str,
    data: Dict[str, Any],
    units: Any,
) -> Tuple[Optional[StructuralModel], List[Any], Optional[ToolResult]]:
    """``parse_model_dict`` for a tool: (model, defaults applied, None), or (None, [], failure) naming the problems."""
    try:
        model, applied = parse_model_dict(data, units)
    except ValueError as exc:
        return None, [], ToolResult.failure(tool, AgentError(
            error_type="invalid_input",
            message=f"{exc}. The model was not changed.",
            possible_causes=["Define every material ('concrete' with fc, 'rebar' with fy, or E), every section with "
                             "its material_id, and every element with its section_id."]))
    return model, applied, None


def create_project(
    structural_model: Optional[Union[StructuralModel, Dict[str, Any]]] = None,
    *,
    title: Optional[str] = None,
    author: Optional[str] = None,
    units: Optional[Union[str, Any, dict]] = None,
    dimension: Optional[str] = None,
    model: Optional[Dict[str, Any]] = None,
    nodes: Optional[List[Dict[str, Any]]] = None,
    elements: Optional[List[Dict[str, Any]]] = None,
    materials: Optional[List[Dict[str, Any]]] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
    supports: Optional[List[Dict[str, Any]]] = None,
) -> ToolResult:
    """
    Create a fresh ``struct_core.Project``.
    Accepts an existing StructuralModel, a full structured model dictionary,
    or individual geometry lists (nodes, elements, materials, sections, supports).
    """
    model_obj: Optional[StructuralModel] = None
    given: Optional[Dict[str, Any]] = None
    applied: List[Any] = []

    # 1. StructuralModel directly supplied
    if isinstance(structural_model, StructuralModel):
        model_obj = structural_model
    # 2. Model dictionary passed via structural_model or model parameter
    elif isinstance(structural_model, dict):
        given = structural_model
    elif isinstance(model, dict):
        given = model
    # 3. Direct geometry lists supplied as kwargs
    elif nodes or elements or materials or sections or supports:
        given = {
            "nodes": nodes or [],
            "elements": elements or [],
            "materials": materials or [],
            "sections": sections or [],
            "supports": supports or [],
        }
    # 4. Default empty model
    else:
        model_obj = StructuralModel()
    if given is not None:
        if dimension is not None and "dimension" not in given:
            given["dimension"] = dimension
        model_obj, applied, refused = _parse_or_fail("create_project", given, units)
        if refused is not None:
            return refused
    elif dimension is not None:
        model_obj.dimension = "3D" if str(dimension).upper() == "3D" else "2D"
        model_obj.vertical_axis = "Z" if model_obj.dimension == "3D" else "Y"
    refused = _model_check("create_project", model_obj)
    if refused is not None:
        return refused

    project = Project(model=model_obj)
    project.defaults_applied = list(applied)
    if title:
        project.metadata.title = title
    if author:
        project.metadata.author = author

    if units is not None:
        active_u = parse_units(units) if isinstance(units, (str, dict)) else units
        project.units = active_u
    elif model_obj and getattr(model_obj, "units", None):
        project.units = model_obj.units

    return ToolResult.success(
        "create_project",
        data={
            "project": project,
            "title": project.metadata.title,
            "author": project.metadata.author,
            "units": project.metadata.units.short_label if (project.metadata and project.metadata.units) else None,
            "n_nodes": len(model_obj.nodes),
            "n_elements": len(model_obj.elements),
            "n_supports": len(model_obj.supports),
            "materials": [_material_summary(m) for m in model_obj.materials],
            "defaults_applied": [d.model_dump() for d in applied],
        },
        next_suggested_stage="MODEL_CREATED",
    )



def create_model_from_plan_draft(
    draft: Optional[Union[FloorPlanDraft, Dict[str, Any]]] = None,
    project: Optional[Project] = None,
    **kwargs: Any,
) -> ToolResult:
    """
    Create or populate a 3D structural model from an architectural/structural
    floor plan draft (grid lines, story elevations, column/beam definitions).
    """
    try:
        if draft is None:
            draft_obj = normalize_floor_plan_dict(kwargs)
        elif isinstance(draft, dict):
            draft_obj = normalize_floor_plan_dict(draft)
        elif isinstance(draft, FloorPlanDraft):
            draft_obj = draft
        else:
            return ToolResult.failure(
                "create_model_from_plan_draft",
                AgentError(
                    error_type="invalid_input",
                    message=f"Expected FloorPlanDraft or dict, got {type(draft).__name__}",
                ),
            )

        applied: List[Any] = []
        compiled_project = build_project_from_draft(draft_obj, existing_project=project, applied=applied)

        return ToolResult.success(
            "create_model_from_plan_draft",
            data={
                "project": compiled_project,
                "title": compiled_project.metadata.title,
                "n_nodes": len(compiled_project.model.nodes),
                "n_elements": len(compiled_project.model.elements),
                "n_supports": len(compiled_project.model.supports),
                "n_stories": len(draft_obj.stories),
                "section_ids": sorted({str(e.section_id) for e in compiled_project.model.elements}),
                "units": compiled_project.metadata.units.label,
                "story_elevations": compiled_project.model.story_elevations,
                "materials": [_material_summary(m) for m in compiled_project.model.materials],
                "defaults_applied": [d.model_dump() for d in applied],
            },
            next_suggested_stage="MODEL_CREATED",
        )
    except Exception as exc:
        return ToolResult.failure(
            "create_model_from_plan_draft",
            AgentError(
                error_type="compiler_error",
                message=f"Failed to compile model from plan draft: {exc}",
                possible_causes=[
                    "Invalid grid line coordinates or missing stories",
                    str(exc),
                ],
            ),
        )


def load_project(path: str) -> ToolResult:
    """Load a previously serialized Project from disk."""
    try:
        project = load_json(Project, path)
    except Exception as exc:
        return ToolResult.failure(
            "load_project",
            AgentError(
                error_type="io_error",
                message=f"Failed to load project from {path!r}: {exc}",
                possible_causes=["Path is wrong", "JSON is malformed", "Schema drift"],
            ),
        )

    return ToolResult.success(
        "load_project",
        data={"project": project, "path": path},
        next_suggested_stage="MODEL_CREATED",
    )


def save_project(project: Project, path: str) -> ToolResult:
    """Serialize a Project to disk."""
    try:
        save_json(project, path, indent=2)
    except Exception as exc:
        return ToolResult.failure(
            "save_project",
            AgentError(
                error_type="io_error",
                message=f"Failed to save project to {path!r}: {exc}",
                possible_causes=["Path is not writable", "Project is not serializable"],
            ),
        )

    return ToolResult.success(
        "save_project",
        data={"project": project, "path": path},
    )


def inspect_project(project: Project) -> ToolResult:
    """
    Produce a human-readable summary of the project for the agent layer.

    The summary intentionally returns structured data rather than prose
    so the agent can reason about it without parsing text.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "inspect_project",
            AgentError(
                error_type="missing_input",
                message="No Project provided to inspect_project.",
                possible_causes=["Project was never created or loaded"],
            ),
        )

    units_val = getattr(project.metadata, "units", "kN, m")
    units_label = str(getattr(units_val, "label", units_val))

    model = project.model
    summary: Dict[str, Any] = {
        "title": project.metadata.title,
        "author": project.metadata.author,
        "units": units_label,
        "n_nodes": len(model.nodes),
        "n_elements": len(model.elements),
        "n_materials": len(model.materials),
        "n_sections": len(model.sections),
        "n_supports": len(model.supports),
        "n_load_cases": len(model.load_cases),
        "n_load_combinations": len(model.load_combinations),
        "n_analysis_cases": len(model.analysis_cases),
        "n_analysis_results": len(project.analysis_results),
        "n_design_results": len(project.design_results),
        "n_detailing_results": len(project.detailing_results),
    }

    return ToolResult.success("inspect_project", data=summary)


def modify_project(
    project: Any,
    mutator: Optional[Any] = None,
    *,
    sections: Optional[List[Dict[str, Any]]] = None,
    member_assignments: Optional[List[Dict[str, Any]]] = None,
    materials: Optional[List[Dict[str, Any]]] = None,
) -> ToolResult:
    """
    Modify an active structural engineering project.

    Supports either declarative JSON modifications (for LLM tool calling and API clients)
    or an arbitrary programmatic Python mutator function.

    Parameters
    ----------
    project : Project
        The project container holding the structural model.
    mutator : callable, optional
        A callable taking `StructuralModel` and returning the modified model.
    sections : list of dict, optional
        Cross-sections to update or create (lengths in the model length unit). Each dict specifies:
        - 'id': section identifier (e.g. 'COL' or 'Col_1')
        - 'b', 'h' (rectangular), 'd' (circular), or 'A', 'I3', ... (general)
        - 'material_id': the section's material (required for a new section; a section owns its material)
        - 'rebar_material_id', 'transverse_rebar_material_id', 'cover': reinforcement of a concrete section
        An existing section keeps every property that is not given.
    member_assignments : list of dict, optional
        Reassign structural elements to different sections:
        - 'element_ids': list of member IDs (e.g. ['C1', 'C2']) or single ID ('C1')
        - 'section_id': target section ID (to change a member's material, give it a section of that material)
    materials : list of dict, optional
        Materials to update or create, strictly in the model units:
        - 'id': material identifier
        - concrete: 'type': 'concrete', 'fc' (required), optional 'E' (default from f'c), 'nu', 'unit_weight'
        - reinforcing steel: 'type': 'rebar', optional 'fy', 'fu', 'E', 'unit_weight' (default Grade 60)
        - other: 'E' (required), optional 'nu', 'unit_weight'

    Returns
    -------
    ToolResult
        Result containing update summary and updated Project.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "modify_project",
            AgentError(
                error_type="missing_input",
                message="No Project provided to modify_project.",
            ),
        )

    # 1. Programmatic callable mutator fallback
    if callable(mutator):
        try:
            new_model = mutator(project.model)
            if new_model is not None:
                project.model = new_model
            _refresh_self_weight(project)
            _refresh_seismic_loads(project)
        except Exception as exc:
            return ToolResult.failure(
                "modify_project",
                AgentError(
                    error_type="mutator_error",
                    message=f"Mutator raised: {exc}",
                    possible_causes=["Agent supplied an invalid mutation"],
                ),
            )
        return ToolResult.success(
            "modify_project",
            data={"project": project, "n_elements": len(project.model.elements)},
            next_suggested_stage="MODEL_CREATED",
        )

    # 2. Declarative modifications. IDs are checked before anything changes: a new section or material
    # must be assigned to members in the same call, otherwise it would change nothing in the model.
    units = project.metadata.units
    materials = [dict(m) for m in materials or [] if isinstance(m, dict)]
    problem = _check_modification_ids(project.model, sections, member_assignments, materials)
    if problem:
        return ToolResult.failure("modify_project", AgentError(error_type="invalid_input", message=problem))

    # All or nothing: a refused modification leaves the model as it was
    snapshot = project.model.model_copy(deep=True)

    def refuse(message: str) -> ToolResult:
        project.model = snapshot
        return ToolResult.failure("modify_project", AgentError(error_type="invalid_input", message=message))

    # Materials first: sections defined in this call may name them
    updated_materials: List[Dict[str, Any]] = []
    applied: List[Any] = []
    for m_data in materials:
        current = next((m for m in project.model.materials if str(m.id) == str(m_data.get("id"))), None)
        try:
            new_mat, used = _material_from_dict(_merged_material_input(current, m_data), units)
        except ValueError as exc:
            return refuse(str(exc))
        existing = [i for i, m in enumerate(project.model.materials) if str(m.id) == str(new_mat.id)]
        if existing:
            project.model.materials[existing[0]] = new_mat
        else:
            project.model.materials.append(new_mat)
        applied += used
        updated_materials.append(_material_summary(new_mat))

    updated_sections: List[Dict[str, Any]] = []
    for s_data in sections or []:
        if not isinstance(s_data, dict) or not str(s_data.get("id", "")).strip():
            continue
        sec_id = str(s_data["id"]).strip()
        existing_idx = next((i for i, x in enumerate(project.model.sections) if str(x.id) == sec_id), None)
        existing_s = project.model.sections[existing_idx] if existing_idx is not None else None
        given = {k: v for k, v in s_data.items() if v is not None}
        kept = existing_s.model_dump() if existing_s is not None else {}
        shape = given.get("type") or ("circular" if "d" in given else "general" if "A" in given
                                      else "rectangular" if {"b", "h"} & set(given) else None)
        if existing_s is not None and shape is not None and shape != existing_s.type:
            # A new shape: the section keeps its materials and cover, not the old dimensions
            kept = {k: v for k, v in kept.items()
                    if k in ("id", "name", "material_id", "rebar_material_id", "transverse_rebar_material_id", "cover")
                    and (shape != "general" or k in ("id", "name", "material_id"))}
        merged = {**kept, **given}
        if existing_s is not None and "type" not in given:
            merged["type"] = shape or existing_s.type
        try:
            check_realistic("cover", given.get("cover"), units)
        except ValueError as exc:
            return refuse(f"Section {sec_id!r}: {exc}")
        try:
            new_sec = _section_from_dict(merged)
        except (ModelInputError, ValueError) as exc:
            return refuse(f"Section {sec_id!r}: {exc}. Give its dimensions (b, h / d / A, I3) and material_id.")
        sec_summary = {k: v for k, v in new_sec.model_dump().items() if v is not None and k != "name"}
        # A resize to the current shape and size changes nothing; say so in the result
        sec_summary["changed"] = existing_s is None or existing_s.model_dump() != new_sec.model_dump()
        if existing_idx is not None:
            project.model.sections[existing_idx] = new_sec
        else:
            project.model.sections.append(new_sec)
        updated_sections.append(sec_summary)

    reassigned_elements: List[str] = []
    if member_assignments:
        elems_by_id = {e.id: e for e in project.model.elements}
        for assign in member_assignments:
            e_ids = assign.get("element_ids") or assign.get("elements") or assign.get("element_id")
            if isinstance(e_ids, str):
                e_ids = [e_ids]
            elif not isinstance(e_ids, list):
                e_ids = []
            target_sec = assign.get("section_id")
            for eid in e_ids:
                elem = elems_by_id.get(eid)
                if elem and target_sec:
                    elem.section_id = str(target_sec)
                    reassigned_elements.append(eid)

    # Concrete sections take the default reinforcement (Grade 60, 1.5 in cover) for what is not given
    applied += complete_concrete_sections(project.model.materials, project.model.sections, units)
    errors = [e for e in validate_structural_model(project.model, raise_error=False) if "missing" in e or "rebar" in e]
    if errors:
        return refuse("The modification leaves the model inconsistent: " + "; ".join(errors))

    if not updated_sections and not reassigned_elements and not updated_materials:
        return ToolResult.failure(
            "modify_project",
            AgentError(
                error_type="missing_input",
                message="No valid modifications provided to modify_project. Provide 'sections', 'member_assignments', 'materials', or a callable 'mutator'.",
            ),
        )

    refreshed = _refresh_self_weight(project)
    lateral = _refresh_seismic_loads(project)
    overridden = set()
    for m_data in materials or []:
        mid = str(m_data.get("id", ""))
        for k, v in m_data.items():
            if v is not None:
                overridden.add((k, mid))
                if k == "unit_weight":
                    overridden.add(("concrete unit_weight", mid))
                    overridden.add(("rebar unit_weight", mid))
    for s_data in sections or []:
        if isinstance(s_data, dict):
            sid = str(s_data.get("id", ""))
            if s_data.get("cover") is not None:
                overridden.add(("cover", sid))
                overridden.add(("cover", None))
    prev_defaults = [
        d for d in (getattr(project, "defaults_applied", None) or [])
        if (d.name, d.applies_to) not in overridden
    ]
    seen_def = {(d.name, d.applies_to) for d in applied}
    merged_def = [d for d in prev_defaults if (d.name, d.applies_to) not in seen_def] + list(applied)
    project.defaults_applied = merged_def
    section_usage = _usage(project.model, "section_id")
    for sec_summary in updated_sections:
        sec_summary["n_elements"] = section_usage.get(sec_summary["id"], 0)
    from .loads import _case_total

    return ToolResult.success(
        "modify_project",
        data={
            "project": project,
            "sections_updated": updated_sections,
            "elements_reassigned": reassigned_elements,
            "materials_updated": updated_materials,
            "defaults_applied": [d.model_dump() for d in applied],
            "self_weight_updated_cases": refreshed,
            # Updated resultants (model force unit): the dead load W and the regenerated ELF base shear
            "total_dead_load": {cid: _case_total(project, cid) for cid in refreshed},
            **({"seismic_loads": lateral} if lateral is not None else {}),
            "summary": f"Modified project: {len(updated_sections)} section(s), {len(reassigned_elements)} member reassignment(s), {len(updated_materials)} material(s).",
            **({"note": "Sections " + ", ".join(s["id"] for s in updated_sections if not s["changed"])
                        + " already had these dimensions; the model did not change. Do not repeat this call: "
                          "continue with the next workflow step (run_analysis, then run_design to check these "
                          "sections). Only if a design check failed, resize with different dimensions (e.g. a "
                          "deeper beam h for beam flexure)."}
               if updated_sections and not reassigned_elements and not updated_materials
               and not any(s["changed"] for s in updated_sections) else {}),
        },
        next_suggested_stage="MODEL_CREATED",
    )


def _usage(model: StructuralModel, attr: str) -> Dict[str, int]:
    """Number of elements referencing each value of an element attribute (e.g. 'section_id')."""
    counts: Dict[str, int] = {}
    for e in model.elements:
        key = getattr(e, attr, None)
        if key is not None:
            counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


def _check_modification_ids(
    model: StructuralModel,
    sections: Optional[List[Dict[str, Any]]],
    member_assignments: Optional[List[Dict[str, Any]]],
    materials: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """Reason the declarative modification would not do what it says, or None."""
    element_ids = {str(e.id) for e in model.elements}
    known_sections = {str(s.id) for s in model.sections}
    new_sections = {str(s.get("id", "")).strip() for s in sections or [] if isinstance(s, dict)} - known_sections
    assigned = set()
    for assign in member_assignments or []:
        if assign.get("material_id") is not None:
            return ("Members take a material through their section: give the section its material_id (or assign "
                    "the members a section of that material).")
        e_ids = assign.get("element_ids") or assign.get("elements") or assign.get("element_id") or []
        e_ids = [e_ids] if isinstance(e_ids, str) else list(e_ids)
        missing = [str(e) for e in e_ids if str(e) not in element_ids]
        if missing:
            return f"Unknown element IDs {missing[:5]}; the model's elements are e.g. {sorted(element_ids)[:6]}."
        target = assign.get("section_id")
        if target is None:
            continue
        if str(target) not in known_sections and str(target) not in new_sections:
            return f"member_assignments refers to section '{target}', which neither exists nor is defined in this call."
        assigned.add(str(target))
    unused = sorted(i for i in new_sections - assigned if i)
    if unused:
        usage = ", ".join(f"'{k}' ({n} elements)" for k, n in sorted(_usage(model, "section_id").items()))
        return (f"Section {unused} does not exist and no member is assigned to it, so the model would not change. "
                f"Update an existing section by its ID ({usage}), or add member_assignments for the new one.")
    known_materials = {str(m.id) for m in model.materials}
    referenced = {str(v) for s in sections or [] if isinstance(s, dict)
                  for v in (s.get("material_id"), s.get("rebar_material_id"), s.get("transverse_rebar_material_id"))
                  if v is not None}
    referenced |= {str(v) for sec in model.sections
                   for v in (sec.material_id, getattr(sec, "rebar_material_id", None),
                             getattr(sec, "transverse_rebar_material_id", None)) if v is not None}
    concrete = {str(m.id) for m in model.materials if getattr(m, "type", None) == "concrete"}
    concrete |= {str(m.get("id")) for m in materials or [] if str(m.get("type", "")).lower() == "concrete"}
    bare = any(str(sec.material_id) in concrete and getattr(sec, "rebar_material_id", None) is None
               for sec in model.sections)  # its bars will be the new rebar material (complete_concrete_sections)
    new_unused = sorted(str(m.get("id")) for m in materials or []
                        if str(m.get("id")) not in known_materials and str(m.get("id")) not in referenced
                        and not (bare and str(m.get("type", "")).lower() == "rebar"))
    if new_unused:
        return (f"Material {new_unused} is new and no section uses it, so the model would not change. Give a section "
                f"its material_id (or rebar_material_id) in the same call.")
    return None


def _refresh_seismic_loads(project: Project) -> Optional[Dict[str, Any]]:
    """
    Regenerate ELF story forces after a member change: they are proportional to the seismic weight,
    which includes the member self-weight. Returns a summary, or None when no ELF loads exist.
    """
    basis = getattr(project, "seismic_basis", None)
    if basis is None or basis.procedure != "elf":
        return None
    if not basis.elf_inputs:
        return {"updated": False,
                "reason": "The ELF loads were generated from a SeismicParameters object or a custom mass source; "
                          "call apply_lateral_loads again so the story forces follow the new seismic weight."}
    from .loads import apply_lateral_loads

    res = apply_lateral_loads(project, **basis.elf_inputs)
    if not res.ok:
        return {"updated": False, "reason": res.error.message if res.error else "apply_lateral_loads failed"}
    return {"updated": True, "seismic_base_shear": res.data.get("seismic_base_shear"),
            "case_ids": res.data.get("seismic_case_ids") or [res.data.get("seismic_case_id")]}


def _refresh_self_weight(project: Project) -> List[str]:
    """
    Recompute the stored member self-weight loads (``is_self_weight``) from the current sections and
    materials, so analyses after a resize carry the new weight. Returns the updated load case IDs.
    """
    from struct_load.engine.self_weight import calculate_self_weight

    units = getattr(project.metadata, "units", None)
    refreshed: List[str] = []
    for lc in project.model.load_cases:
        if not any(getattr(dl, "is_self_weight", False) for dl in lc.element_loads):
            continue
        fresh = calculate_self_weight(model=project.model, load_case_id=lc.id, units=units)
        lc.element_loads = [dl for dl in lc.element_loads if not getattr(dl, "is_self_weight", False)] + list(
            fresh.element_loads
        )
        refreshed.append(lc.id)
    return refreshed


def set_project_units(
    project: Project,
    units: str,
    convert_existing: bool = True
) -> ToolResult:
    """
    Set or convert the project unit system (e.g. 'kips, in, sec', 'kN, m, s, C', etc.).
    If convert_existing is True, automatically converts all existing coordinates,
    loads, material properties, sections, and analysis results to the new units.
    """
    if project is None:
        return ToolResult.failure(
            "set_project_units",
            AgentError(
                error_type="missing_input",
                message="No Project provided to set_project_units.",
            ),
        )
    try:
        from struct_core.units import convert_project_units, parse_units
        u_target = parse_units(units)
        if convert_existing:
            convert_project_units(project, u_target, in_place=True)
        else:
            project.units = u_target
        return ToolResult.success(
            "set_project_units",
            data={"units": u_target.short_label, "converted": convert_existing},
        )
    except Exception as exc:
        return ToolResult.failure(
            "set_project_units",
            AgentError(
                error_type="unit_conversion_error",
                message=f"Failed to set/convert units: {exc}",
                possible_causes=["Invalid unit specification string", str(exc)],
            ),
        )


def attach_state(project: Project) -> ProjectState:
    """Create an empty ProjectState for a freshly created/loaded project."""
    return ProjectState(project=project, stage=WorkflowStage.MODEL_CREATED)