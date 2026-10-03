"""Analysis-result validation tool."""

from __future__ import annotations

from typing import Any

from ..tools.base import AgentError, ToolResult


def validate_analysis(analysis_result: Any) -> ToolResult:
    """Check that an analysis result looks engineering-meaningful."""
    if analysis_result is None:
        return ToolResult.failure(
            "validate_analysis",
            AgentError(
                error_type="missing_result",
                message="No AnalysisResult supplied for validation.",
                possible_causes=["run_analysis was not called yet"],
            ),
        )

    element_results = getattr(analysis_result, "element_results", None)
    modal_result = getattr(analysis_result, "modal_result", None)
    has_modes = bool(modal_result and getattr(modal_result, "modes", None))

    if not element_results and not has_modes:
        return ToolResult.failure(
            "validate_analysis",
            AgentError(
                error_type="empty_result",
                message="AnalysisResult contains neither element results nor modal results.",
                possible_causes=[
                    "Solver did not run any element or modal solves",
                    "Load case had no applied loads",
                    "Element schema mismatch between solver and struct_core",
                ],
            ),
        )

    return ToolResult.success(
        "validate_analysis",
        data={
            "analysis_case_id": getattr(analysis_result, "analysis_case_id", None),
            "n_element_results": len(element_results) if element_results else 0,
        },
    )