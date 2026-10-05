"""
Design tools.

These wrap ``aci318`` through its struct_core adapter. The agent does not
perform ACI 318 calculations itself.
"""

from __future__ import annotations

from typing import Any, Optional

from ..validation.design import validate_design
from .base import AgentError, ToolResult


def run_design(
    project: Any,
    *,
    design_case_id: str,
    design_code: str = "ACI 318-19",
    analysis_case_id: Optional[str] = None,
) -> ToolResult:
    """
    Run ACI 318 design + detailing for every element in the project model.

    The result is stored on the project under ``design_case_id`` and
    :func:`validate_design` is run immediately so the agent never sees
    an unvalidated design.
    """
    if project is None or getattr(project, "model", None) is None:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="missing_input",
                message="No Project supplied to run_design.",
            ),
        )

    # Engineering gate: don't design without analysis.
    if not project.analysis_results:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="workflow_error",
                message="Cannot run design before a valid analysis exists.",
                possible_causes=["run_analysis must succeed before run_design"],
            ),
            next_suggested_stage="MODEL_VALIDATED",
        )

    try:
        from aci318.adapters import design_from_core  # local import
    except Exception as exc:  # pragma: no cover - environment guard
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="dependency_error",
                message=f"aci318 adapters are unavailable: {exc}",
                possible_causes=["aci318 is not installed in the active environment"],
            ),
        )

    try:
        target_analysis_res = None
        if analysis_case_id:
            target_analysis_res = project.analysis_results.get(analysis_case_id)
            if target_analysis_res is not None:
                project.active_analysis_case = analysis_case_id

        design_res, detail_res = design_from_core(
            project,
            analysis_result=target_analysis_res,
            design_code=design_code,
        )
    except Exception as exc:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="design_failure",
                message=f"aci318 design_from_core raised: {exc}",
                possible_causes=[
                    "Analysis result has no element internal forces",
                    "Section schema is missing required dimensions",
                    "Material schema lacks fc",
                ],
            ),
        )

    project.design_results[design_case_id] = design_res
    project.detailing_results[design_case_id] = detail_res

    validation = validate_design(design_res, detail_res)
    if not validation.ok:
        return ToolResult.failure(
            "run_design",
            AgentError(
                error_type="design_failure",
                message="Design completed but produced no element results.",
                possible_causes=validation.error.possible_causes if validation.error else [],
            ),
            data={"design_case_id": design_case_id},
        )

    return ToolResult.success(
        "run_design",
        data={
            "design_case_id": design_case_id,
            "design_code": design_code,
            "analysis_case_id": analysis_case_id,
            "governing_dcr": float(design_res.governing_dcr or 0.0),
            "passed": bool(design_res.passed),
            "n_element_results": len(design_res.element_results),
            "n_detailing_results": len(detail_res.element_results),
        },
        next_suggested_stage="DESIGN_COMPLETE",
    )


def inspect_design_results(project: Any, design_case_id: str) -> ToolResult:
    """Return a structured summary of a design case for the agent."""
    if project is None or not getattr(project, "design_results", None):
        return ToolResult.failure(
            "inspect_design_results",
            AgentError(
                error_type="missing_result",
                message="No design results attached to the project.",
                possible_causes=["run_design was never called"],
            ),
        )

    design_res = project.design_results.get(design_case_id)
    if design_res is None:
        available = list(project.design_results.keys())
        return ToolResult.failure(
            "inspect_design_results",
            AgentError(
                error_type="unknown_case",
                message=f"Design case {design_case_id!r} not found.",
                possible_causes=[f"Available cases: {available}"],
            ),
        )

    elements: dict[str, Any] = {}
    for elem_id, elem_des in design_res.element_results.items():
        elements[str(elem_id)] = {
            "member_type": getattr(elem_des, "member_type", None),
            "governing_dcr": float(getattr(elem_des, "governing_dcr", 0.0) or 0.0),
            "passed": bool(getattr(elem_des, "passed", False)),
        }

    return ToolResult.success(
        "inspect_design_results",
        data={
            "design_case_id": design_case_id,
            "design_code": getattr(design_res, "design_code", None),
            "governing_dcr": float(getattr(design_res, "governing_dcr", 0.0) or 0.0),
            "passed": bool(getattr(design_res, "passed", False)),
            "elements": elements,
        },
    )


def inspect_detailing_results(
    project: Any,
    design_case_id: Optional[str] = None,
) -> ToolResult:
    """Return a structured summary of detailing results and rebar schedules for the agent."""
    if project is None or not getattr(project, "detailing_results", None):
        return ToolResult.failure(
            "inspect_detailing_results",
            AgentError(
                error_type="missing_result",
                message="No detailing results attached to the project.",
                possible_causes=["run_design was never called"],
            ),
        )

    target_case_id = design_case_id
    if target_case_id is None:
        target_case_id = next(iter(project.detailing_results.keys()))

    detail_res = project.detailing_results.get(target_case_id)
    if detail_res is None:
        available = list(project.detailing_results.keys())
        return ToolResult.failure(
            "inspect_detailing_results",
            AgentError(
                error_type="unknown_case",
                message=f"Detailing case {target_case_id!r} not found.",
                possible_causes=[f"Available cases: {available}"],
            ),
        )

    agg_schedule = getattr(detail_res, "aggregate_schedule", None)
    schedule_summary = []
    total_weight = 0.0
    if agg_schedule is not None:
        total_weight = float(getattr(agg_schedule, "total_weight", 0.0) or 0.0)
        for item in getattr(agg_schedule, "items", []):
            schedule_summary.append({
                "mark": getattr(item, "mark", ""),
                "bar_size": getattr(item, "bar_size", 0),
                "quantity": getattr(item, "quantity", 0),
                "cut_length": float(getattr(item, "cut_length", 0.0) or 0.0),
                "total_length": float(getattr(item, "total_length", 0.0) or 0.0),
                "weight": float(getattr(item, "weight", 0.0) or 0.0),
                "shape_code": getattr(item, "shape_code", ""),
                "location": getattr(item, "location", ""),
            })

    elements: dict[str, Any] = {}
    for elem_id, elem_det in getattr(detail_res, "element_results", {}).items():
        elem_items = []
        elem_sched = getattr(elem_det, "bar_schedule", None)
        if elem_sched is not None:
            for item in getattr(elem_sched, "items", []):
                elem_items.append({
                    "mark": getattr(item, "mark", ""),
                    "bar_size": getattr(item, "bar_size", 0),
                    "quantity": getattr(item, "quantity", 0),
                    "cut_length": float(getattr(item, "cut_length", 0.0) or 0.0),
                    "weight": float(getattr(item, "weight", 0.0) or 0.0),
                })
        elements[str(elem_id)] = {
            "member_type": getattr(elem_det, "member_type", None),
            "total_weight": float(getattr(elem_sched, "total_weight", 0.0) or 0.0) if elem_sched else 0.0,
            "items": elem_items,
        }

    return ToolResult.success(
        "inspect_detailing_results",
        data={
            "design_case_id": target_case_id,
            "total_weight": total_weight,
            "n_schedule_items": len(schedule_summary),
            "schedule": schedule_summary,
            "elements": elements,
        },
    )