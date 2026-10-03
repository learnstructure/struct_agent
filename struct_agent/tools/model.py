"""
Model tools.

These wrap ``struct_core`` operations for project container lifecycle,
structural model creation (both parametric frames and raw structured JSON),
inspection, and modification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from struct_core import (
    BeamElement,
    ElasticMaterial,
    Node,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    TrussElement,
    load_json,
    save_json,
)

from ..state.project_state import ProjectState, WorkflowStage
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

    return StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
    )


def create_frame_model(
    num_bays: int = 2,
    bay_width: float = 5.0,
    num_stories: int = 2,
    story_height: float = 3.0,
    *,
    title: Optional[str] = None,
    author: Optional[str] = None,
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
        )

        project_title = title or f"{num_bays}-Bay {num_stories}-Story Frame"
        project = Project(model=model)
        project.metadata.title = project_title
        if author:
            project.metadata.author = author

        return ToolResult.success(
            "create_frame_model",
            data={
                "project": project,
                "title": project.metadata.title,
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

    return ToolResult.success(
        "create_project",
        data={
            "project": project,
            "title": project.metadata.title,
            "author": project.metadata.author,
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


def modify_project(project: Project, mutator) -> ToolResult:
    """
    Apply a user-supplied mutator function to the project model.

    The mutator receives the underlying :class:`StructuralModel` and must
    return it (mutated in place or replaced). The agent should not edit
    the project directly via free-form code in production — every change
    should go through this tool so the orchestrator can audit it.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "modify_project",
            AgentError(
                error_type="missing_input",
                message="No Project provided to modify_project.",
            ),
        )
    if not callable(mutator):
        return ToolResult.failure(
            "modify_project",
            AgentError(
                error_type="bad_input",
                message="mutator is not callable.",
                possible_causes=["Agent passed a non-function value"],
            ),
        )

    try:
        new_model = mutator(project.model)
    except Exception as exc:
        return ToolResult.failure(
            "modify_project",
            AgentError(
                error_type="mutator_error",
                message=f"Mutator raised: {exc}",
                possible_causes=["Agent supplied an invalid mutation"],
            ),
        )

    project.model = new_model if new_model is not None else project.model
    return ToolResult.success(
        "modify_project",
        data={"project": project, "n_elements": len(project.model.elements)},
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
            data={"units": u_target.label, "converted": convert_existing},
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