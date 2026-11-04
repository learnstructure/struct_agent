"""
Commercial software interoperability tools for struct_agent.

Wraps struct_link for bidirectional model exchange with SAP2000 and ETABS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import AgentError, ToolResult


def import_from_sap2000(
    s2k_path: Union[str, Path],
    *,
    project_title: Optional[str] = None,
    materialize_self_weight: bool = False,
    fy: Optional[float] = None,
    fyt: Optional[float] = None,
    cover: Optional[float] = None,
    allow_partial_import: bool = False,
) -> ToolResult:
    """
    Import a SAP2000 model text file (.s2k / .$2k) into a new struct_core.Project.

    Parameters
    ----------
    s2k_path : str or Path
        Path to the SAP2000 .s2k or .$2k text file.
    project_title : str, optional
        Title for the new Project (defaults to filename stem).
    materialize_self_weight : bool, optional
        Whether to explicitly materialize element self-weight loads into the model (default: False).
        Note: fem2d and fem3d solvers automatically compute and apply self-weight during analysis
        for cases with include_self_weight=True without needing to mutate the project model.
        Leaving this False preserves a clean schema for roundtripping to SAP2000/ETABS without
        duplicated distributed loads.
    fy, fyt, cover : optional
        Reinforcement of the concrete frame sections the file gives none for: longitudinal and transverse yield
        strengths (strictly in the model stress unit; ``fyt`` defaults to ``fy``) and clear cover (model length
        unit; default 1.5 in, reported). Sections with reinforcement in the file keep it.
    allow_partial_import : bool, optional
        Allow importing partial model when the file contains unsupported tables (e.g. springs, areas),
        recording unhandled items in notes rather than raising ValueError.

    Returns
    -------
    ToolResult
        Success ToolResult containing the new Project in data["project"].
    """
    path = Path(s2k_path)
    if not path.exists():
        return ToolResult.failure(
            "import_from_sap2000",
            AgentError(
                error_type="file_not_found",
                message=f"SAP2000 file not found at {str(path)!r}",
                possible_causes=["Incorrect path or missing file"],
            ),
        )

    try:
        from struct_link import Sap2000FileReader
        from struct_core import Project
    except ImportError as exc:
        return ToolResult.failure(
            "import_from_sap2000",
            AgentError(
                error_type="dependency_error",
                message=f"struct_link is unavailable: {exc}",
                possible_causes=["struct_link is not installed in the active environment"],
            ),
        )

    try:
        reader = Sap2000FileReader()
        project = reader.read_project(path, title=project_title, allow_partial_import=allow_partial_import)
        model = project.model
        applied = _add_reinforcement(project, fy, fyt, cover)
        project.defaults_applied = list(applied)
        if materialize_self_weight:
            from struct_load import apply_self_weight
            apply_self_weight(project)
    except Exception as exc:
        return ToolResult.failure(
            "import_from_sap2000",
            AgentError(
                error_type="import_failure",
                message=f"Failed to parse SAP2000 file: {exc}",
                possible_causes=["Corrupt or unsupported .s2k syntax", "Missing required tables"],
            ),
        )

    return ToolResult.success(
        "import_from_sap2000",
        data={
            "project": project,
            "n_nodes": len(model.nodes),
            "n_elements": len(model.elements),
            "n_load_cases": len(getattr(model, "load_cases", [])),
            **_import_report(reader, model),
            **({"defaults_applied": [d.model_dump() for d in applied]} if applied else {}),
        },
        next_suggested_stage="MODEL_CREATED",
    )


def export_to_sap2000(
    project: Any,
    *,
    output_path: Union[str, Path],
    allow_partial_export: bool = False,
) -> ToolResult:
    """
    Export the structural model in a struct_core.Project into a SAP2000 .s2k text file.

    Parameters
    ----------
    project : struct_core.Project
        The Project holding the StructuralModel to export.
    output_path : str or Path
        Target filepath for the generated .s2k file.
    allow_partial_export : bool, optional
        Allow exporting without unsupported features (e.g. dropping support springs),
        recording dropped items in result data rather than raising ValueError.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "export_to_sap2000",
            AgentError(
                error_type="missing_input",
                message="No Project or StructuralModel provided to export_to_sap2000.",
                possible_causes=["Project was never created or has no model"],
            ),
        )

    try:
        from struct_link import Sap2000FileWriter
    except ImportError as exc:
        return ToolResult.failure(
            "export_to_sap2000",
            AgentError(
                error_type="dependency_error",
                message=f"struct_link is unavailable: {exc}",
                possible_causes=["struct_link is not installed in the active environment"],
            ),
        )

    try:
        writer = Sap2000FileWriter()
        saved_path = writer.write_model(project, output_path, allow_partial_export=allow_partial_export)
    except Exception as exc:
        return ToolResult.failure(
            "export_to_sap2000",
            AgentError(
                error_type="export_failure",
                message=f"Failed to export SAP2000 file: {exc}",
                possible_causes=["Invalid file path or permission error"],
            ),
        )

    return ToolResult.success(
        "export_to_sap2000",
        data={
            "output_path": str(saved_path),
            "n_nodes": len(project.model.nodes),
            "n_elements": len(project.model.elements),
            **({"dropped_items": writer.dropped_items} if getattr(writer, "dropped_items", None) else {}),
            **({"dropped_springs": writer.dropped_springs} if getattr(writer, "dropped_springs", None) else {}),
        },
    )


def import_from_etabs(
    e2k_path: Union[str, Path],
    *,
    project_title: Optional[str] = None,
    fy: Optional[float] = None,
    fyt: Optional[float] = None,
    cover: Optional[float] = None,
    allow_partial_import: bool = False,
) -> ToolResult:
    """
    Import an ETABS model text file (.e2k / .$et) into a new struct_core.Project.

    Parameters
    ----------
    e2k_path : str or Path
        Path to the ETABS .e2k text file.
    project_title : str, optional
        Title for the new Project (defaults to filename stem).
    fy, fyt, cover : optional
        Reinforcement of the concrete frame sections the file gives none for: longitudinal and transverse yield
        strengths (strictly in the model stress unit; ``fyt`` defaults to ``fy``) and clear cover (model length
        unit; default 1.5 in, reported). Sections with reinforcement in the file keep it.
    allow_partial_import : bool, optional
        Allow importing partial model when the file contains unsupported tables (e.g. springs, areas),
        recording unhandled items in notes rather than raising ValueError.
    """
    path = Path(e2k_path)
    if not path.exists():
        return ToolResult.failure(
            "import_from_etabs",
            AgentError(
                error_type="file_not_found",
                message=f"ETABS file not found at {str(path)!r}",
                possible_causes=["Incorrect path or missing file"],
            ),
        )

    try:
        from struct_link import EtabsFileReader
        from struct_core import Project
    except ImportError as exc:
        return ToolResult.failure(
            "import_from_etabs",
            AgentError(
                error_type="dependency_error",
                message=f"struct_link is unavailable: {exc}",
                possible_causes=["struct_link is not installed in the active environment"],
            ),
        )

    try:
        reader = EtabsFileReader()
        project = reader.read_project(path, title=project_title, allow_partial_import=allow_partial_import)
        model = project.model
        applied = _add_reinforcement(project, fy, fyt, cover)
        project.defaults_applied = list(applied)
    except Exception as exc:
        return ToolResult.failure(
            "import_from_etabs",
            AgentError(
                error_type="import_failure",
                message=f"Failed to parse ETABS file: {exc}",
                possible_causes=["Corrupt or unsupported .e2k syntax", "Missing required tables"],
            ),
        )

    return ToolResult.success(
        "import_from_etabs",
        data={
            "project": project,
            "n_nodes": len(model.nodes),
            "n_elements": len(model.elements),
            "n_load_cases": len(getattr(model, "load_cases", [])),
            **_import_report(reader, model),
            **({"defaults_applied": [d.model_dump() for d in applied]} if applied else {}),
        },
        next_suggested_stage="MODEL_CREATED",
    )


def export_to_etabs(
    project: Any,
    *,
    output_path: Union[str, Path],
    allow_partial_export: bool = False,
) -> ToolResult:
    """
    Export the structural model in a struct_core.Project into an ETABS .e2k text file.

    Parameters
    ----------
    project : struct_core.Project
        The Project holding the StructuralModel to export.
    output_path : str or Path
        Target filepath for the generated .e2k file.
    allow_partial_export : bool, optional
        Allow exporting without unsupported features (e.g. dropping support springs),
        recording dropped items in result data rather than raising ValueError.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "export_to_etabs",
            AgentError(
                error_type="missing_input",
                message="No Project or StructuralModel provided to export_to_etabs.",
                possible_causes=["Project was never created or has no model"],
            ),
        )

    nodes = getattr(project.model, "nodes", None) or []
    if nodes and not any(abs(getattr(n, "z", 0.0) or 0.0) > 1e-9 for n in nodes):
        # A 2D frame (Y vertical) has no stories in the ETABS sense: the file re-imported as 3 nodes and 0 elements
        return ToolResult.failure(
            "export_to_etabs",
            AgentError(
                error_type="unsupported_model",
                message="ETABS models are 3D and story-based, so this 2D frame (Y vertical) cannot be written as an "
                        ".e2k file. Export it with export_to_sap2000, or build the frame in 3D.",
            ),
        )

    try:
        from struct_link import EtabsFileWriter
    except ImportError as exc:
        return ToolResult.failure(
            "export_to_etabs",
            AgentError(
                error_type="dependency_error",
                message=f"struct_link is unavailable: {exc}",
                possible_causes=["struct_link is not installed in the active environment"],
            ),
        )

    try:
        writer = EtabsFileWriter()
        saved_path = writer.write_model(project, output_path, allow_partial_export=allow_partial_export)
    except Exception as exc:
        return ToolResult.failure(
            "export_to_etabs",
            AgentError(
                error_type="export_failure",
                message=f"Failed to export ETABS file: {exc}",
                possible_causes=["Invalid file path or permission error"],
            ),
        )

    return ToolResult.success(
        "export_to_etabs",
        data={
            "output_path": str(saved_path),
            "n_nodes": len(project.model.nodes),
            "n_elements": len(project.model.elements),
            **({"dropped_items": writer.dropped_items} if getattr(writer, "dropped_items", None) else {}),
            **({"dropped_springs": writer.dropped_springs} if getattr(writer, "dropped_springs", None) else {}),
        },
    )


def _import_report(reader: Any, model: Any) -> Dict[str, Any]:
    """The imported materials, the reader's notes (e.g. a concrete material without f'c), and what the model still
    needs before reinforced-concrete design."""
    from .model import _material_summary, missing_design_inputs

    report: Dict[str, Any] = {"materials": [_material_summary(m) for m in model.materials]}
    if getattr(reader, "messages", None):
        report["import_notes"] = list(reader.messages)
    if getattr(reader, "ignored_tables", None):
        report["ignored_tables"] = list(reader.ignored_tables)
    if getattr(reader, "dropped_items", None):
        report["dropped_items"] = list(reader.dropped_items)
    missing = missing_design_inputs(model)
    if missing:
        report["before_design"] = missing
    return report


def _add_reinforcement(project: Any, fy: Optional[float], fyt: Optional[float], cover: Optional[float]) -> list:
    """Give the concrete frame sections without reinforcement in the file the bars ``fy`` / ``fyt`` (materials 'REBAR'
    and, when ``fyt`` differs, 'REBAR_TRANSVERSE') and the clear ``cover`` in the model units; nothing when neither is given.
    Returns the defaults applied."""
    if fy is None and fyt is None and cover is None:
        return []
    from .. import defaults
    from .materials import rebar_material
    from .units_input import check_realistic

    model = project.model
    units = project.metadata.units
    concrete = {m.id for m in model.materials if getattr(m, "type", None) == "concrete"}
    bare = [i for i, sec in enumerate(model.sections)
            if sec.material_id in concrete and hasattr(sec, "cover") and sec.rebar_material_id is None]
    if not bare:
        return []
    taken = {str(m.id) for m in model.materials}
    longitudinal, applied = rebar_material("REBAR" if "REBAR" not in taken else "REBAR_IMPORT", units=units, fy=fy)
    model.materials.append(longitudinal)
    transverse = longitudinal
    if fyt is not None and fyt != longitudinal.fy:
        transverse, used = rebar_material("REBAR_TRANSVERSE", units=units, fy=fyt)
        model.materials.append(transverse)
        applied += used
    if cover is None:
        cover = defaults.COVER.in_units(units)
        applied.append(defaults.COVER.record(model_units=units))
    check_realistic("cover", cover, units)
    for i in bare:
        sec = model.sections[i]
        model.sections[i] = sec.model_copy(update={
            "rebar_material_id": longitudinal.id, "transverse_rebar_material_id": transverse.id,
            "cover": sec.cover if sec.cover is not None else cover})
    return applied
