"""
Model tools.

These wrap ``struct_core`` operations only. They do not run analysis or
design. The agent should construct a project here, inspect it, and then
hand it to the next stage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from struct_core import (
    Project,
    StructuralModel,
    load_json,
    save_json,
)

from ..state.project_state import ProjectState, WorkflowStage
from .base import AgentError, ToolResult


def create_project(
    structural_model: StructuralModel,
    *,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> ToolResult:
    """
    Create a fresh ``struct_core.Project`` wrapping the supplied model.

    The tool does not run any validation; the agent should call
    :func:`validate_model` next.
    """
    if structural_model is None:
        return ToolResult.failure(
            "create_project",
            AgentError(
                error_type="missing_input",
                message="No StructuralModel supplied to create_project.",
                possible_causes=["Caller forgot to construct a StructuralModel first"],
            ),
        )

    project = Project(model=structural_model)
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
            "n_elements": len(structural_model.elements),
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

    model = project.model
    summary: Dict[str, Any] = {
        "title": project.metadata.title,
        "author": project.metadata.author,
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


def attach_state(project: Project) -> ProjectState:
    """Create an empty ProjectState for a freshly created/loaded project."""
    return ProjectState(project=project, stage=WorkflowStage.MODEL_CREATED)