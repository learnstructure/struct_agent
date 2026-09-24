"""Design-result validation tool."""

from __future__ import annotations

from typing import Any

from ..tools.base import AgentError, ToolResult


def validate_design(design_result: Any, detailing_result: Any = None) -> ToolResult:
    """Check that a DesignResult (and optional DetailingResult) are usable."""
    if design_result is None:
        return ToolResult.failure(
            "validate_design",
            AgentError(
                error_type="missing_result",
                message="No DesignResult supplied for validation.",
                possible_causes=["run_design was not called yet"],
            ),
        )

    element_results = getattr(design_result, "element_results", None)
    if not element_results:
        return ToolResult.failure(
            "validate_design",
            AgentError(
                error_type="empty_result",
                message="DesignResult contains no element design results.",
                possible_causes=[
                    "StructuralModel had no elements",
                    "aci318 could not design any members",
                ],
            ),
        )

    failed = [eid for eid, res in element_results.items() if not getattr(res, "passed", False)]
    governing_dcr = float(getattr(design_result, "governing_dcr", 0.0) or 0.0)

    payload: dict[str, Any] = {
        "design_code": getattr(design_result, "design_code", None),
        "n_element_results": len(element_results),
        "governing_dcr": governing_dcr,
        "failed_elements": failed,
    }

    if detailing_result is not None:
        detail_elements = getattr(detailing_result, "element_results", None) or {}
        payload["n_detailing_results"] = len(detail_elements)

    return ToolResult.success("validate_design", data=payload)