"""
Drawing tools.

These wrap ``struct_draw`` through its struct_core adapter. The agent
does not produce DXF geometry itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .base import AgentError, ToolResult


def _render_framing_drawings(
    project: Any,
    output_path: str,
    view_mode: str = "both",
    story_level: Optional[Union[str, int, float]] = None,
    grid_line: Optional[Union[str, float]] = None,
    title: Optional[str] = None,
    scale: Optional[Union[str, float]] = None,
    unit: Optional[str] = None,
) -> ToolResult:
    mode = str(view_mode or "both").lower()
    if mode not in ("both", "plan", "elevation"):
        mode = "both"

    try:
        from struct_draw.adapters import draw_general_frame_from_core
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
        output = draw_general_frame_from_core(
            project,
            view_mode=mode,
            story_level=story_level,
            grid_line=grid_line,
            title=title,
            scale=scale,
            unit=unit,
            output_path=output_path,
        )
    except Exception as exc:
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="drawing_failure",
                message=f"struct_draw framing adapter raised: {exc}",
                possible_causes=[
                    "StructuralModel has no valid nodes or elements",
                    "Invalid grid line or story level identifier",
                ],
            ),
        )

    out = Path(output_path).resolve()
    return ToolResult.success(
        "generate_drawings",
        data={
            "output_path": str(out),
            "drawing_type": "framing",
            "view_mode": mode,
            "story_level": story_level,
            "grid_line": grid_line,
            "elements_drawn": int(output.get("elements_drawn", 0) or 0),
        },
        next_suggested_stage="DRAWINGS_GENERATED",
    )


def generate_drawings(
    project: Any,
    *,
    output_path: str,
    drawing_type: str = "auto",  # "auto", "detailing", "framing"
    design_case_id: Optional[str] = None,
    detailing_case_id: Optional[str] = None,
    view_mode: str = "both",     # "both", "plan", "elevation"
    story_level: Optional[Union[str, int, float]] = None,
    grid_line: Optional[Union[str, float]] = None,
    title: Optional[str] = None,
    scale: Optional[Union[str, float]] = None,
    unit: Optional[str] = None,
) -> ToolResult:
    """
    Unified drawing generator for the structural engineering pipeline.

    Generates either:
    1. General Arrangement (GA) framing plans / elevations directly from the
       structural model (when ``drawing_type="framing"`` or ``drawing_type="auto"``
       without design results).
    2. Production reinforced concrete detailing drawings (cross-sections, rebar
       mats, stirrups, beam/column schedules) when ``drawing_type="detailing"``
       or ``drawing_type="auto"`` with valid design results.

    Parameters
    ----------
    project
        The ``struct_core.Project`` containing the model and optional design/detailing results.
    output_path
        Destination path for the .dxf drawing file.
    drawing_type
        "auto" (default: uses detailing if available, else framing arrangement),
        "framing" (always produces GA framing plans/elevations), or
        "detailing" (produces rebar detailing for the specified design case).
    design_case_id
        Which ``DesignResult`` to render when generating detailing drawings.
    detailing_case_id
        Which ``DetailingResult`` to render. Defaults to ``design_case_id``.
    view_mode
        For framing drawings: "both" (plan + elevation sheet), "plan" (floor framing plan),
        or "elevation" (frame elevation along a grid line).
    story_level
        Story name, index, or elevation to filter framing plan (e.g. "FL1", "ROOF").
    grid_line
        Grid line name or coordinate to filter frame elevation (e.g. "1", "A").
    title
        Optional drawing title.
    scale
        Optional drawing scale (e.g. "1:50", "1:100", "1/4\"=1'-0\"").
    unit
        Optional drawing target unit (e.g. "mm", "m", "in", "ft").
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to generate_drawings.",
            ),
        )

    mode_type = str(drawing_type or "auto").lower()

    # Determine whether to generate framing or detailing drawings
    use_framing = False
    if mode_type == "framing":
        use_framing = True
    elif mode_type == "detailing":
        use_framing = False
    else:  # "auto"
        if design_case_id is not None:
            use_framing = False
        elif getattr(project, "detailing_results", None) and getattr(project, "design_results", None):
            use_framing = False
        else:
            use_framing = True

    if use_framing:
        return _render_framing_drawings(
            project=project,
            output_path=output_path,
            view_mode=view_mode,
            story_level=story_level,
            grid_line=grid_line,
            title=title,
            scale=scale,
            unit=unit,
        )

    target_case_id = design_case_id
    if target_case_id is None and getattr(project, "design_results", None):
        target_case_id = next(iter(project.design_results.keys()))

    if not getattr(project, "design_results", None) or target_case_id not in project.design_results:
        if mode_type == "auto" and design_case_id is None:
            return _render_framing_drawings(
                project=project,
                output_path=output_path,
                view_mode=view_mode,
                story_level=story_level,
                grid_line=grid_line,
                title=title,
                scale=scale,
                unit=unit,
            )
        return ToolResult.failure(
            "generate_drawings",
            AgentError(
                error_type="workflow_error",
                message="Cannot generate drawings without a valid design result.",
                possible_causes=["run_design must succeed before generate_drawings"],
            ),
            next_suggested_stage="DESIGN_COMPLETE",
        )

    detail_id = detailing_case_id or target_case_id
    if not getattr(project, "detailing_results", None) or detail_id not in project.detailing_results:
        if getattr(project, "detailing_results", None):
            detail_id = next(iter(project.detailing_results.keys()))
        elif mode_type == "auto" and design_case_id is None:
            return _render_framing_drawings(
                project=project,
                output_path=output_path,
                view_mode=view_mode,
                story_level=story_level,
                grid_line=grid_line,
                title=title,
                scale=scale,
                unit=unit,
            )
        else:
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
        view_t = "detailing" if mode_type == "detailing" else "auto"
        det_obj = project.detailing_results.get(detail_id) if getattr(project, "detailing_results", None) else project
        output = draw_from_core(
            det_obj or project,
            design_result=project.design_results[target_case_id],
            model_or_project=project,
            view_type=view_t,
            scale=scale,
            unit=unit,
            view_mode=view_mode,
            story_level=story_level,
            grid_line=grid_line,
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
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    document.saveas(str(out))

    has_framing = bool(output.get("has_framing", False) or "plan" in output.get("data", {}) or "elevation" in output.get("data", {}))
    return ToolResult.success(
        "generate_drawings",
        data={
            "output_path": str(out),
            "drawing_type": "composite" if has_framing and mode_type == "auto" else ("detailing" if mode_type == "detailing" else "auto"),
            "has_framing": has_framing,
            "has_detailing": True,
            "elements_drawn": int(output.get("elements_drawn", 0) or 0),
            "design_case_id": target_case_id,
            "detailing_case_id": detail_id,
        },
        next_suggested_stage="DRAWINGS_GENERATED",
    )


def generate_framing_drawings(
    project: Any,
    *,
    output_path: str,
    view_mode: str = "both",
    story_level: Optional[Union[str, int, float]] = None,
    grid_line: Optional[Union[str, float]] = None,
    title: Optional[str] = None,
    scale: Optional[Union[str, float]] = None,
    unit: Optional[str] = None,
) -> ToolResult:
    """
    Convenience wrapper around ``generate_drawings(..., drawing_type="framing")``.
    """
    return generate_drawings(
        project,
        output_path=output_path,
        drawing_type="framing",
        view_mode=view_mode,
        story_level=story_level,
        grid_line=grid_line,
        title=title,
        scale=scale,
        unit=unit,
    )