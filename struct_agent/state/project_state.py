"""
Workflow state machine for struct_agent.

Tracks where the agent currently is in the canonical engineering workflow:

    MODEL_CREATED
        ↓
    MODEL_VALIDATED
        ↓
    ANALYSIS_COMPLETE
        ↓
    DESIGN_COMPLETE
        ↓
    DETAILING_COMPLETE
        ↓
    DRAWINGS_GENERATED

The state is intentionally small. The engineering data itself lives in the
``struct_core.Project``; this object holds workflow metadata, progress tracking,
and an audit trail of tool invocations. Preconditions for specific engineering
operations are enforced directly by the respective tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowStage(str, Enum):
    """Canonical engineering workflow stages."""

    UNKNOWN = "UNKNOWN"
    MODEL_CREATED = "MODEL_CREATED"
    LOADS_APPLIED = "LOADS_APPLIED"
    MODEL_VALIDATED = "MODEL_VALIDATED"
    ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
    DESIGN_COMPLETE = "DESIGN_COMPLETE"
    DETAILING_COMPLETE = "DETAILING_COMPLETE"
    DRAWINGS_GENERATED = "DRAWINGS_GENERATED"


# Allowed transitions in the workflow (including iterative reflection/optimization loops).
_ALLOWED: Dict[WorkflowStage, List[WorkflowStage]] = {
    WorkflowStage.UNKNOWN: [
        WorkflowStage.MODEL_CREATED,
    ],
    WorkflowStage.MODEL_CREATED: [
        WorkflowStage.MODEL_CREATED,
        WorkflowStage.LOADS_APPLIED,
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,  # Modal analysis from self-mass, imported CAE models, or re-analysis after modify_project
        WorkflowStage.DRAWINGS_GENERATED, # General Arrangement framing drawings (plans/elevations) directly from model geometry
    ],
    WorkflowStage.LOADS_APPLIED: [
        WorkflowStage.MODEL_CREATED,
        WorkflowStage.LOADS_APPLIED,
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,  # Direct static or dynamic analysis after loads
        WorkflowStage.DRAWINGS_GENERATED, # GA framing drawings
    ],
    WorkflowStage.MODEL_VALIDATED: [
        WorkflowStage.MODEL_CREATED,
        WorkflowStage.LOADS_APPLIED,
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,
        WorkflowStage.DESIGN_COMPLETE,    # Validation after the analysis changes nothing: its results stay current
        WorkflowStage.DRAWINGS_GENERATED, # GA framing drawings
    ],
    WorkflowStage.ANALYSIS_COMPLETE: [
        WorkflowStage.MODEL_CREATED,      # Reflection: resizing/modifying model
        WorkflowStage.LOADS_APPLIED,      # Modifying load definitions
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,  # Re-running analysis (modal/RSA/static)
        WorkflowStage.DESIGN_COMPLETE,
        WorkflowStage.DRAWINGS_GENERATED, # GA framing drawings
    ],
    WorkflowStage.DESIGN_COMPLETE: [
        WorkflowStage.MODEL_CREATED,      # Closed-loop reflection: resizing overstressed members
        WorkflowStage.LOADS_APPLIED,      # Redesign with revised loads
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,  # Re-solve after section change
        WorkflowStage.DESIGN_COMPLETE,    # Re-check design
        WorkflowStage.DETAILING_COMPLETE, # Compliant design -> detailing
        WorkflowStage.DRAWINGS_GENERATED, # Direct drawing generation (rebar detailing or framing)
    ],
    WorkflowStage.DETAILING_COMPLETE: [
        WorkflowStage.MODEL_CREATED,      # Reflection if detailing congestion requires resizing
        WorkflowStage.LOADS_APPLIED,
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,
        WorkflowStage.DESIGN_COMPLETE,
        WorkflowStage.DETAILING_COMPLETE,
        WorkflowStage.DRAWINGS_GENERATED,
    ],
    WorkflowStage.DRAWINGS_GENERATED: [
        WorkflowStage.MODEL_CREATED,
        WorkflowStage.LOADS_APPLIED,
        WorkflowStage.MODEL_VALIDATED,
        WorkflowStage.ANALYSIS_COMPLETE,
        WorkflowStage.DESIGN_COMPLETE,
        WorkflowStage.DETAILING_COMPLETE,
        WorkflowStage.DRAWINGS_GENERATED,
    ],
}


class StageTransitionError(Exception):
    """Raised when a workflow stage transition is invalid."""

    def __init__(self, current: WorkflowStage, requested: WorkflowStage) -> None:
        self.current = current
        self.requested = requested
        super().__init__(
            f"Invalid workflow transition: cannot move from {current.value!r} to {requested.value!r}."
        )


@dataclass
class ProjectState:
    """
    Workflow state for a single engineering project.

    Attributes
    ----------
    project
        The underlying ``struct_core.Project`` (the source of engineering truth).
    stage
        Current workflow stage. Note that `stage` and `WorkflowStage` are informational
        metadata for workflow progress tracking and logging; each engineering tool
        independently and deterministically validates its own actual engineering preconditions
        (e.g. existing model geometry, applied loads, solved analysis results, or code design).
    history
        Ordered list of visited stages.
    pending_questions
        Free-form list of clarifications the agent should ask the user
        (geometry, materials, loads, design code, etc.).
    tool_log
        Ordered list of tool invocations and their statuses for traceability.
    last_errors
        Last structured error dict per stage, useful for retry decisions.
    """

    project: Optional[Any] = None
    stage: WorkflowStage = WorkflowStage.UNKNOWN
    history: List[WorkflowStage] = field(default_factory=list)
    pending_questions: List[str] = field(default_factory=list)
    tool_log: List[Dict[str, Any]] = field(default_factory=list)
    last_errors: Dict[WorkflowStage, Dict[str, Any]] = field(default_factory=dict)

    def advance_to(self, stage: WorkflowStage) -> None:
        """Advance the workflow to ``stage`` if the transition is allowed."""
        if self.stage == stage:
            return

        allowed_targets = _ALLOWED.get(self.stage, [WorkflowStage.DRAWINGS_GENERATED])
        if self.stage == WorkflowStage.DRAWINGS_GENERATED and stage == WorkflowStage.DRAWINGS_GENERATED:
            return

        if stage not in allowed_targets:
            raise StageTransitionError(self.stage, stage)

        if not self.history or self.history[-1] != stage:
            self.history.append(stage)

        self.stage = stage

    def requires(self, stage: WorkflowStage) -> bool:
        """Return True if the workflow has already reached or passed ``stage``."""
        if stage in self.history:
            return True

        order = [
            WorkflowStage.MODEL_CREATED,
            WorkflowStage.LOADS_APPLIED,
            WorkflowStage.MODEL_VALIDATED,
            WorkflowStage.ANALYSIS_COMPLETE,
            WorkflowStage.DESIGN_COMPLETE,
            WorkflowStage.DETAILING_COMPLETE,
            WorkflowStage.DRAWINGS_GENERATED,
        ]

        if stage not in order:
            return True

        current_idx = order.index(self.stage) if self.stage in order else -1
        needed_idx = order.index(stage)
        return current_idx >= needed_idx

    def log_tool(self, name: str, status: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.tool_log.append({
            "tool": name,
            "status": status,
            "stage": self.stage.value,
            "payload": payload or {},
        })

    def record_error(self, stage: WorkflowStage, error: Dict[str, Any]) -> None:
        self.last_errors[stage] = error

    def clear_error(self, stage: WorkflowStage) -> None:
        self.last_errors.pop(stage, None)

