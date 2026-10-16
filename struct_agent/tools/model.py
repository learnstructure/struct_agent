"""
Model tools.

These wrap ``struct_core`` operations for project container lifecycle,
structural model creation (both parametric frames and raw structured JSON),
inspection, and modification.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

from struct_core import (
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
    load_json,
    parse_units,
    save_json,
)

from ..state.project_state import ProjectState, WorkflowStage
from ..vision.compiler import build_project_from_draft
from ..vision.schemas import FloorPlanDraft, normalize_floor_plan_dict
from .base import AgentError, ToolResult


def parse_model_dict(model_data: Dict[str, Any]) -> StructuralModel:
    """
    Parse a flexible dictionary containing nodes, elements, materials, sections,
    and supports into a validated struct_core.StructuralModel.
    Automatically provisions standard defaults if optional fields are missing.
    """
    # 1. Parse Nodes
    nodes: List[Node] = []
    for n in model_data.get("nodes", []):
        if isinstance(n, Node):
            nodes.append(n)
        elif isinstance(n, dict):
            nodes.append(Node(
                id=n.get("id", len(nodes) + 1),
                x=float(n.get("x", 0.0)),
                y=float(n.get("y", 0.0)),
                z=float(n.get("z", 0.0)),
            ))

    # 2. Parse Materials (ensure at least one concrete material exists)
    materials: List[Any] = []
    existing_mat_ids = set()
    for m in model_data.get("materials", []):
        if isinstance(m, dict):
            mid = str(m.get("id", "Concrete_4ksi"))
            existing_mat_ids.add(mid)
            materials.append(ElasticMaterial(
                id=mid,
                E=float(m.get("E", 24800000.0)),
                nu=float(m.get("nu", 0.2)),
                rho=float(m.get("rho", 2400.0)),
            ))
        else:
            materials.append(m)
            existing_mat_ids.add(getattr(m, "id", ""))

    default_mat_id = "Concrete_4ksi"
    if default_mat_id not in existing_mat_ids:
        materials.append(ElasticMaterial(id=default_mat_id, E=24800000.0, nu=0.2, rho=2400.0))
        existing_mat_ids.add(default_mat_id)

    # 3. Parse Sections
    sections: List[Any] = []
    existing_sec_ids = set()
    for s in model_data.get("sections", []):
        if isinstance(s, dict):
            sid = str(s.get("id", "Default_Section"))
            existing_sec_ids.add(sid)
            sections.append(RectangularSection(
                id=sid,
                b=float(s.get("b", 0.3)),
                h=float(s.get("h", 0.5)),
            ))
        else:
            sections.append(s)
            existing_sec_ids.add(getattr(s, "id", ""))

    default_sec_id = "Default_Section"
    if default_sec_id not in existing_sec_ids:
        sections.append(RectangularSection(id=default_sec_id, b=0.35, h=0.45))
        existing_sec_ids.add(default_sec_id)

    # 4. Parse Elements
    elements: List[Any] = []
    for e in model_data.get("elements", []):
        if isinstance(e, dict):
            eid = e.get("id", f"E{len(elements)+1}")
            sn = e["start_node"]
            en = e["end_node"]
            mid = e.get("material_id") or default_mat_id
            sid = e.get("section_id") or default_sec_id
            if sid not in existing_sec_ids:
                sections.append(RectangularSection(id=sid, b=0.35, h=0.45))
                existing_sec_ids.add(sid)
            etype = str(e.get("type", "beam")).lower()
            if etype == "truss":
                elements.append(TrussElement(id=eid, start_node=sn, end_node=en, material_id=mid, section_id=sid))
            else:
                elements.append(BeamElement(id=eid, start_node=sn, end_node=en, material_id=mid, section_id=sid))
        else:
            elements.append(e)

    # 5. Parse Supports
    supports: List[Support] = []
    for sp in model_data.get("supports", []):
        if isinstance(sp, Support):
            supports.append(sp)
        elif isinstance(sp, dict):
            nid = sp["node_id"]
            fixity = str(sp.get("fixity", "")).lower()
            if fixity == "pinned":
                supports.append(Support.pinned(node_id=nid))
            elif fixity in ("fixed", "fixed_3d") or (sp.get("ux") and sp.get("uy") and sp.get("rz")):
                supports.append(Support.fixed(node_id=nid))
            else:
                supports.append(Support(
                    node_id=nid,
                    ux=bool(sp.get("ux", True)),
                    uy=bool(sp.get("uy", True)),
                    rz=bool(sp.get("rz", fixity == "fixed")),
                    uz=bool(sp.get("uz", False)),
                ))

    units_raw = model_data.get("units")
    parsed_u = parse_units(units_raw) if units_raw else None

    return StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
        units=parsed_u,
    )


def create_frame_model(
    num_bays: int = 2,
    bay_width: float = 5.0,
    num_stories: int = 2,
    story_height: float = 3.0,
    *,
    title: Optional[str] = None,
    author: Optional[str] = None,
    units: Optional[Union[str, Any, dict]] = "kN, m",
    dimension: str = "2d",
    bay_widths: Optional[List[float]] = None,
    story_heights: Optional[List[float]] = None,
    support_type: str = "fixed",
    col_b: float = 0.4,
    col_h: float = 0.4,
    beam_b: float = 0.3,
    beam_h: float = 0.5,
    fc_kpa: float = 27580.0,
    rho_kg_m3: float = 2400.0,
) -> ToolResult:
    """
    Generate a complete, fully connected structural frame model (2D or 3D).
    Automatically creates all grid nodes, columns, beams, supports, materials, and sections.
    """
    try:
        active_units = parse_units(units) if units is not None else None
        # Determine bay coordinates along X
        if bay_widths and len(bay_widths) > 0:
            x_coords = [0.0]
            for w in bay_widths:
                x_coords.append(x_coords[-1] + float(w))
            num_bays = len(bay_widths)
        else:
            x_coords = [float(i * bay_width) for i in range(num_bays + 1)]

        # Determine story elevations along Y (or Z)
        if story_heights and len(story_heights) > 0:
            y_coords = [0.0]
            for h in story_heights:
                y_coords.append(y_coords[-1] + float(h))
            num_stories = len(story_heights)
        else:
            y_coords = [float(j * story_height) for j in range(num_stories + 1)]

        # 2. Materials & Sections
        mat_id = "Concrete_4ksi"
        materials = [
            ElasticMaterial(
                id=mat_id,
                E=float(57000.0 * (fc_kpa * 0.145038)**0.5 * 6.89476) if fc_kpa else 24800000.0,
                nu=0.2,
                rho=rho_kg_m3,
            )
        ]
        col_sec_id = f"Col_{int(col_b*1000)}x{int(col_h*1000)}"
        beam_sec_id = f"Beam_{int(beam_b*1000)}x{int(beam_h*1000)}"
        sections = [
            RectangularSection(id=col_sec_id, b=col_b, h=col_h),
            RectangularSection(id=beam_sec_id, b=beam_b, h=beam_h),
        ]

        if str(dimension or "2d").lower() == "3d":
            # 3D Building Frame: X and Y are floor plan, Z is vertical elevation
            bays_x = num_bays
            bays_y = num_bays
            z_coords = y_coords
            y_coords_plan = x_coords

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
                                material_id=mat_id,
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
                                material_id=mat_id,
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
                                material_id=mat_id,
                                section_id=beam_sec_id,
                            )
                        )

            model = StructuralModel(
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
                project.metadata.units = active_units
            if author:
                project.metadata.author = author

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
            for i in range(num_bays + 1):
                nid = j * (num_bays + 1) + i + 1
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
            for i in range(num_bays + 1):
                cid = f"C_{i+1}_{j+1}"
                elements.append(
                    BeamElement(
                        id=cid,
                        name=f"Column Bay {i+1} Story {j+1}",
                        start_node=node_map[(i, j)],
                        end_node=node_map[(i, j + 1)],
                        material_id=mat_id,
                        section_id=col_sec_id,
                    )
                )

        # Horizontal Beams
        for j in range(1, num_stories + 1):
            for i in range(num_bays):
                bid = f"B_{i+1}_{j}"
                elements.append(
                    BeamElement(
                        id=bid,
                        name=f"Beam Bay {i+1} Level {j}",
                        start_node=node_map[(i, j)],
                        end_node=node_map[(i + 1, j)],
                        material_id=mat_id,
                        section_id=beam_sec_id,
                    )
                )

        model = StructuralModel(
            nodes=nodes,
            elements=elements,
            materials=materials,
            sections=sections,
            supports=supports,
            story_elevations=y_coords,
            units=active_units,
        )

        project_title = title or f"{num_bays}-Bay {num_stories}-Story Frame"
        project = Project(model=model)
        project.metadata.title = project_title
        if active_units:
            project.metadata.units = active_units
        if author:
            project.metadata.author = author

        return ToolResult.success(
            "create_frame_model",
            data={
                "project": project,
                "title": project.metadata.title,
                "units": active_units.short_label if active_units else "kN, m",
                "n_nodes": len(nodes),
                "n_elements": len(elements),
                "n_columns": (num_bays + 1) * num_stories,
                "n_beams": num_bays * num_stories,
                "n_supports": len(supports),
                "bays": num_bays,
                "stories": num_stories,
                "total_height": y_coords[-1],
                "total_width": x_coords[-1],
            },
            next_suggested_stage="MODEL_CREATED",
        )
    except Exception as exc:
        return ToolResult.failure(
            "create_frame_model",
            AgentError(
                error_type="model_creation_error",
                message=f"Failed to generate frame model: {exc}",
                possible_causes=["Invalid dimension parameters or numeric values"],
            ),
        )


def create_project(
    structural_model: Optional[Union[StructuralModel, Dict[str, Any]]] = None,
    *,
    title: Optional[str] = None,
    author: Optional[str] = None,
    units: Optional[Union[str, Any, dict]] = None,
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

    # 1. StructuralModel directly supplied
    if isinstance(structural_model, StructuralModel):
        model_obj = structural_model
    # 2. Model dictionary passed via structural_model or model parameter
    elif isinstance(structural_model, dict):
        model_obj = parse_model_dict(structural_model)
    elif isinstance(model, dict):
        model_obj = parse_model_dict(model)
    # 3. Direct geometry lists supplied as kwargs
    elif nodes or elements:
        model_dict = {
            "nodes": nodes or [],
            "elements": elements or [],
            "materials": materials or [],
            "sections": sections or [],
            "supports": supports or [],
        }
        model_obj = parse_model_dict(model_dict)
    # 4. Default empty model
    else:
        model_obj = StructuralModel()

    project = Project(model=model_obj)
    if title:
        project.metadata.title = title
    if author:
        project.metadata.author = author

    if units is not None:
        active_u = parse_units(units) if isinstance(units, (str, dict)) else units
        project.metadata.units = active_u
        if project.model:
            project.model.units = active_u
    elif model_obj and getattr(model_obj, "units", None):
        project.metadata.units = model_obj.units

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
        },
        next_suggested_stage="MODEL_CREATED",
    )


def build_structural_model(
    project: Optional[Project] = None,
    *,
    nodes: Optional[List[Dict[str, Any]]] = None,
    elements: Optional[List[Dict[str, Any]]] = None,
    materials: Optional[List[Dict[str, Any]]] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
    supports: Optional[List[Dict[str, Any]]] = None,
    title: Optional[str] = None,
) -> ToolResult:
    """
    Construct or populate a structural model from structured node/element/support JSON.
    """
    model_dict = {
        "nodes": nodes or [],
        "elements": elements or [],
        "materials": materials or [],
        "sections": sections or [],
        "supports": supports or [],
    }
    new_model = parse_model_dict(model_dict)

    if project is None:
        project = Project(model=new_model)
        if title:
            project.metadata.title = title
    else:
        project.model = new_model

    return ToolResult.success(
        "build_structural_model",
        data={
            "project": project,
            "n_nodes": len(new_model.nodes),
            "n_elements": len(new_model.elements),
            "n_supports": len(new_model.supports),
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

        compiled_project = build_project_from_draft(draft_obj, existing_project=project)

        return ToolResult.success(
            "create_model_from_plan_draft",
            data={
                "project": compiled_project,
                "title": compiled_project.metadata.title,
                "n_nodes": len(compiled_project.model.nodes),
                "n_elements": len(compiled_project.model.elements),
                "n_supports": len(compiled_project.model.supports),
                "n_stories": len(draft_obj.stories),
                "units": compiled_project.metadata.units.label,
                "story_elevations": compiled_project.model.story_elevations,
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
        Cross-sections to update or create. Each dict specifies:
        - 'id': section identifier (e.g. 'Col_350x350' or 'Col_1')
        - 'b': width in meters
        - 'h': depth in meters
    member_assignments : list of dict, optional
        Reassign structural elements to different sections or materials:
        - 'element_ids': list of member IDs (e.g. ['C1', 'C2']) or single ID ('C1')
        - 'section_id': target section ID
        - 'material_id': optional target material ID
    materials : list of dict, optional
        Materials to update or create:
        - 'id': material identifier
        - 'E': Young's modulus (kPa)
        - 'fc_kpa': concrete f'c (kPa) to auto-compute E if E is omitted
        - 'rho': density (kg/m3)

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

    # 2. Declarative modifications
    updated_sections: List[Dict[str, Any]] = []
    if sections:
        for s_data in sections:
            if not isinstance(s_data, dict):
                continue
            sec_id = str(s_data.get("id", "")).strip()
            if not sec_id:
                continue

            existing_idx = None
            existing_s = None
            for i, s_obj in enumerate(project.model.sections):
                if getattr(s_obj, "id", None) == sec_id:
                    existing_idx = i
                    existing_s = s_obj
                    break

            # Infer or read section shape type
            sec_type = str(s_data.get("type", "")).lower().strip()
            if not sec_type:
                if "d" in s_data or "diameter" in s_data or isinstance(existing_s, CircularSection):
                    sec_type = "circular"
                elif "A" in s_data or "area" in s_data or "I3" in s_data or "Iz" in s_data or isinstance(existing_s, GeneralSection):
                    sec_type = "general"
                else:
                    sec_type = "rectangular"

            new_sec = None
            sec_summary: Dict[str, Any] = {"id": sec_id, "type": sec_type}

            if sec_type == "circular":
                d_val = s_data.get("d") if s_data.get("d") is not None else s_data.get("diameter")
                if d_val is not None:
                    d = float(d_val)
                elif isinstance(existing_s, CircularSection):
                    d = float(existing_s.d)
                else:
                    d = 0.40
                if d <= 0.0:
                    continue
                new_sec = CircularSection(id=sec_id, d=round(d, 4))
                sec_summary["d"] = round(d, 4)

            elif sec_type == "general":
                a_val = s_data.get("A") if s_data.get("A") is not None else s_data.get("area")
                i3_val = s_data.get("I3") if s_data.get("I3") is not None else s_data.get("Iz")
                i2_val = s_data.get("I2") if s_data.get("I2") is not None else s_data.get("Iy")
                j_val = s_data.get("J")
                as2_val = s_data.get("As2") if s_data.get("As2") is not None else s_data.get("Ay")
                as3_val = s_data.get("As3") if s_data.get("As3") is not None else s_data.get("Az")

                A = float(a_val if a_val is not None else (getattr(existing_s, "A", 0.0) or 0.01))
                I3 = float(i3_val if i3_val is not None else (getattr(existing_s, "I3", 0.0) or 0.001))
                I2 = float(i2_val if i2_val is not None else (getattr(existing_s, "I2", 0.0) or 0.0))
                J = float(j_val if j_val is not None else (getattr(existing_s, "J", 0.0) or 0.0))
                As2 = float(as2_val) if as2_val is not None else getattr(existing_s, "As2", None)
                As3 = float(as3_val) if as3_val is not None else getattr(existing_s, "As3", None)

                if A <= 0.0 or I3 <= 0.0:
                    continue
                new_sec = GeneralSection(id=sec_id, A=round(A, 6), I3=round(I3, 8), I2=round(I2, 8), J=round(J, 8), As2=As2, As3=As3)
                sec_summary.update({"A": round(A, 6), "I3": round(I3, 8), "I2": round(I2, 8), "J": round(J, 8)})

            else:  # rectangular
                b_val = s_data.get("b") if s_data.get("b") is not None else s_data.get("width")
                h_val = s_data.get("h") if s_data.get("h") is not None else (s_data.get("depth") or s_data.get("height"))

                if b_val is not None:
                    b = float(b_val)
                elif isinstance(existing_s, RectangularSection):
                    b = float(existing_s.b)
                else:
                    b = 0.35

                if h_val is not None:
                    h = float(h_val)
                elif isinstance(existing_s, RectangularSection):
                    h = float(existing_s.h)
                else:
                    h = 0.45

                if b <= 0.0 or h <= 0.0:
                    continue
                new_sec = RectangularSection(id=sec_id, b=round(b, 4), h=round(h, 4))
                sec_summary.update({"b": round(b, 4), "h": round(h, 4)})

            if new_sec is not None:
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
            target_mat = assign.get("material_id")

            for eid in e_ids:
                elem = elems_by_id.get(eid)
                if elem:
                    if target_sec:
                        elem.section_id = str(target_sec)
                    if target_mat:
                        elem.material_id = str(target_mat)
                    reassigned_elements.append(eid)

    updated_materials: List[Dict[str, Any]] = []
    if materials:
        for m_data in materials:
            mat_id = str(m_data.get("id", "")).strip()
            if not mat_id:
                continue
            E = float(m_data.get("E", 0.0) or m_data.get("elastic_modulus", 0.0))
            fc_kpa = float(m_data.get("fc_kpa", 0.0) or m_data.get("fc", 0.0))
            rho = float(m_data.get("rho", 0.0) or m_data.get("density", 2400.0))
            if not E and fc_kpa > 0:
                fc_mpa = fc_kpa / 1000.0
                E = 4700.0 * math.sqrt(fc_mpa) * 1000.0
            E = E or 25000000.0

            existing_idx = None
            for i, existing_m in enumerate(project.model.materials):
                if getattr(existing_m, "id", None) == mat_id:
                    existing_idx = i
                    break

            new_mat = ElasticMaterial(id=mat_id, E=round(E, 1), rho=rho)
            if existing_idx is not None:
                project.model.materials[existing_idx] = new_mat
            else:
                project.model.materials.append(new_mat)
            updated_materials.append({"id": mat_id, "E": E, "rho": rho})

    if not updated_sections and not reassigned_elements and not updated_materials:
        return ToolResult.failure(
            "modify_project",
            AgentError(
                error_type="missing_input",
                message="No valid modifications provided to modify_project. Provide 'sections', 'member_assignments', 'materials', or a callable 'mutator'.",
            ),
        )

    return ToolResult.success(
        "modify_project",
        data={
            "project": project,
            "sections_updated": updated_sections,
            "elements_reassigned": reassigned_elements,
            "materials_updated": updated_materials,
            "summary": f"Modified project: {len(updated_sections)} section(s), {len(reassigned_elements)} member reassignment(s), {len(updated_materials)} material(s).",
        },
        next_suggested_stage="MODEL_CREATED",
    )


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
            project.metadata.units = u_target
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