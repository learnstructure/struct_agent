"""Structural-model validation tool."""

from __future__ import annotations

from typing import Any, List

from struct_core.structural_model.validation import (
    StructuralValidationError,
    validate_structural_model,
)

from ..tools.base import AgentError, ToolResult


def validate_model(model_or_project: Any = None, *, project: Any = None) -> ToolResult:
    """
    Validate a :class:`struct_core.StructuralModel` (or a Project that
    owns one) using struct_core's schema-level validator.

    This is intentionally the only kind of validation performed at this
    stage — it does not check analysis, design, or detailing results.
    Those have their own validators.
    """
    target = project if project is not None else model_or_project
    model = getattr(target, "model", target)

    try:
        errors: List[str] = validate_structural_model(model, raise_error=False)
    except StructuralValidationError as exc:
        return ToolResult.failure(
            "validate_model",
            AgentError(
                error_type="schema_error",
                message=str(exc),
                possible_causes=[
                    "Missing nodes, elements, materials, or sections",
                    "Element references unknown node/material/section IDs",
                    "Support references unknown node ID",
                ],
            ),
            data={"errors": [str(exc)]},
            next_suggested_stage="MODEL_CREATED",
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return ToolResult.failure(
            "validate_model",
            AgentError(
                error_type="validation_error",
                message=f"Unexpected error during model validation: {exc}",
                possible_causes=["Internal error in struct_core validator"],
            ),
        )

    if errors:
        return ToolResult.failure(
            "validate_model",
            AgentError(
                error_type="schema_error",
                message=f"Model has {len(errors)} schema problem(s).",
                possible_causes=errors[:5],
            ),
            data={"errors": errors},
            next_suggested_stage="MODEL_CREATED",
        )

    return ToolResult.success(
        "validate_model",
        data={
            "n_nodes": len(getattr(model, "nodes", [])),
            "n_elements": len(getattr(model, "elements", [])),
            "n_materials": len(getattr(model, "materials", [])),
            "n_sections": len(getattr(model, "sections", [])),
            "n_supports": len(getattr(model, "supports", [])),
            "errors": [],
        },
        next_suggested_stage="MODEL_VALIDATED",
    )