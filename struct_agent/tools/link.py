"""
Commercial software interoperability tools for struct_agent.

Wraps struct_link for bidirectional model exchange with SAP2000 and ETABS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .base import AgentError, ToolResult


def import_from_sap2000(
    s2k_path: Union[str, Path],
    *,
    project_title: Optional[str] = None,
) -> ToolResult:
    """
    Import a SAP2000 model text file (.s2k / .$2k) into a new struct_core.Project.

    Parameters
    ----------
    s2k_path : str or Path
        Path to the SAP2000 .s2k or .$2k text file.
    project_title : str, optional
        Title for the new Project (defaults to filename stem).

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
        model = reader.read_model(path)
        project = Project(model=model)
        project.metadata.title = project_title or path.stem
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
        },
        next_suggested_stage="MODEL_CREATED",
    )


def export_to_sap2000(
    project: Any,
    *,
    output_path: Union[str, Path],
) -> ToolResult:
    """
    Export the structural model in a struct_core.Project into a SAP2000 .s2k text file.

    Parameters
    ----------
    project : struct_core.Project
        The Project holding the StructuralModel to export.
    output_path : str or Path
        Target filepath for the generated .s2k file.
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
        saved_path = writer.write_model(project, output_path)
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
        },
    )


def import_from_etabs(
    e2k_path: Union[str, Path],
    *,
    project_title: Optional[str] = None,
) -> ToolResult:
    """
    Import an ETABS model text file (.e2k / .$et) into a new struct_core.Project.

    Parameters
    ----------
    e2k_path : str or Path
        Path to the ETABS .e2k text file.
    project_title : str, optional
        Title for the new Project (defaults to filename stem).
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
        model = reader.read_model(path)
        project = Project(model=model)
        project.metadata.title = project_title or path.stem
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
        },
        next_suggested_stage="MODEL_CREATED",
    )


def export_to_etabs(
    project: Any,
    *,
    output_path: Union[str, Path],
) -> ToolResult:
    """
    Export the structural model in a struct_core.Project into an ETABS .e2k text file.

    Parameters
    ----------
    project : struct_core.Project
        The Project holding the StructuralModel to export.
    output_path : str or Path
        Target filepath for the generated .e2k file.
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
        saved_path = writer.write_model(project, output_path)
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
        },
    )
