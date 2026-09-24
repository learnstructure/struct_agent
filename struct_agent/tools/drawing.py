"""
Drawing tools.

These wrap ``struct_draw`` through its struct_core adapter. The agent
does not produce DXF geometry itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import AgentError, ToolResult


def generate_drawings(
    project: Any,
    *,
    design_case_id: str,
    output_path: str,
    detailing_case_id: Optional[str] = None,
) -> ToolResult:
    """
    Render a DXF drawing from a DesignResult / DetailingResult pair.

    Parameters
    ----------
    project
        The ``struct_core.Project`` containing both design and detailing results.
    design_case_id
        Which ``DesignResult`` to render.
    detailing_case_id
        Which ``DetailingResult`` to render. Defaults to the same value
        as ``design_case_id`` (the most common pattern).
    output_path
        Where to write the DXF file.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to generate_drawings.",
            ),
        )

    if not project.design_results or design_case_id not in project.design_results:
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="workflow_error",
                message="Cannot generate drawings without a valid design result.",
                possible_causes=["run_design must succeed before generate_drawings"],
            ),
            next_suggested_stage="DESIGN_COMPLETE",
        )

    detail_id = detailing_case_id or design_case_id
    if detail_id not in project.detailing_results:
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="workflow_error",
                message=f"No DetailingResult for case {detail_id!r}.",
                possible_causes=["run_design must produce a DetailingResult"],
            ),
        )

    try:
        from struct_draw.adapters import draw_from_core  # local import
    except Exception as exc:  # pragma: no cover - environment guard
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="dependency_error",
                message=f"struct_draw is unavailable: {exc}",
                possible_causes=["struct_draw is not installed"],
            ),
        )

    try:
        output = draw_from_core(
            project,
            design_result=project.design_results[design_case_id],
        )
    except Exception as exc:
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="drawing_failure",
                message=f"struct_draw adapter raised: {exc}",
                possible_causes=[
                    "DetailingResult has no element_results",
                    "struct_draw does not recognize a member type",
                ],
            ),
        )

    document = output["drawing"]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document.saveas(str(out))

    return ToolResult.success(
        "generate_drawings",
        data={
            "output_path": str(out),
            "elements_drawn": int(output.get("elements_drawn", 0) or 0),
            "design_case_id": design_case_id,
            "detailing_case_id": detail_id,
        },
        next_suggested_stage="DRAWINGS_GENERATED",
    )