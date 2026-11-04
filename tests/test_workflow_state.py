"""Tests for the workflow state machine."""

import pytest

from struct_agent.state.project_state import (
    ProjectState,
    StageTransitionError,
    WorkflowStage,
)


def test_initial_stage_is_unknown():
    state = ProjectState()
    assert state.stage is WorkflowStage.UNKNOWN


def test_valid_transitions_follow_canonical_pipeline():
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    state.advance_to(WorkflowStage.MODEL_VALIDATED)
    state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
    state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    state.advance_to(WorkflowStage.DETAILING_COMPLETE)
    state.advance_to(WorkflowStage.DRAWINGS_GENERATED)
    assert state.stage is WorkflowStage.DRAWINGS_GENERATED


def test_skipping_a_stage_raises():
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    # Cannot jump directly to DESIGN_COMPLETE without structural analysis
    with pytest.raises(StageTransitionError):
        state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    # Cannot jump directly to DETAILING_COMPLETE
    with pytest.raises(StageTransitionError):
        state.advance_to(WorkflowStage.DETAILING_COMPLETE)


def test_requires_reports_completed_stages():
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    state.advance_to(WorkflowStage.MODEL_VALIDATED)

    assert state.requires(WorkflowStage.MODEL_CREATED)
    assert state.requires(WorkflowStage.MODEL_VALIDATED)
    assert not state.requires(WorkflowStage.ANALYSIS_COMPLETE)


def test_tool_log_records_each_call():
    state = ProjectState()
    state.log_tool("create_project", "success", {"k": "v"})
    state.log_tool("validate_model", "success")
    assert len(state.tool_log) == 2
    assert state.tool_log[0]["tool"] == "create_project"


def test_record_and_clear_error():
    state = ProjectState()
    state.record_error(WorkflowStage.MODEL_VALIDATED, {"type": "schema_error"})
    assert state.last_errors[WorkflowStage.MODEL_VALIDATED]["type"] == "schema_error"
    state.clear_error(WorkflowStage.MODEL_VALIDATED)
    assert WorkflowStage.MODEL_VALIDATED not in state.last_errors


def test_reflection_loop_transitions():
    """Verify that closed-loop reflection allows looping back from DESIGN_COMPLETE to MODEL_CREATED."""
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    state.advance_to(WorkflowStage.MODEL_VALIDATED)
    state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
    state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    assert state.stage is WorkflowStage.DESIGN_COMPLETE

    # Reflection: resize members after failing DCR
    state.advance_to(WorkflowStage.MODEL_CREATED)
    assert state.stage is WorkflowStage.MODEL_CREATED

    # Re-validate and re-solve
    state.advance_to(WorkflowStage.MODEL_VALIDATED)
    state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
    state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    assert state.stage is WorkflowStage.DESIGN_COMPLETE

    # Compliant: advance to detailing and drawings
    state.advance_to(WorkflowStage.DETAILING_COMPLETE)
    state.advance_to(WorkflowStage.DRAWINGS_GENERATED)
    assert state.stage is WorkflowStage.DRAWINGS_GENERATED
    assert state.requires(WorkflowStage.MODEL_CREATED)
    assert state.requires(WorkflowStage.DESIGN_COMPLETE)


def test_direct_transition_from_loads_applied_to_analysis():
    """Verify that models can transition directly from LOADS_APPLIED to ANALYSIS_COMPLETE and DESIGN_COMPLETE."""
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    state.advance_to(WorkflowStage.LOADS_APPLIED)
    # Direct transition to analysis without explicit validate_model call
    state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
    assert state.stage is WorkflowStage.ANALYSIS_COMPLETE
    state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    assert state.stage is WorkflowStage.DESIGN_COMPLETE


def test_reflection_and_framing_transitions_from_model_created():
    """Verify reflection re-analysis and direct framing drawing generation from MODEL_CREATED."""
    # 1. Direct GA framing drawings from model geometry
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    state.advance_to(WorkflowStage.DRAWINGS_GENERATED)
    assert state.stage is WorkflowStage.DRAWINGS_GENERATED

    # 2. Reflection cycle: modify_project resets to MODEL_CREATED, followed directly by re-analysis
    state = ProjectState()
    state.advance_to(WorkflowStage.MODEL_CREATED)
    state.advance_to(WorkflowStage.LOADS_APPLIED)
    state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
    state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    # Modify project resizes sections
    state.advance_to(WorkflowStage.MODEL_CREATED)
    assert state.stage is WorkflowStage.MODEL_CREATED
    # Re-analysis proceeds without error
    state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
    assert state.stage is WorkflowStage.ANALYSIS_COMPLETE
    state.advance_to(WorkflowStage.DESIGN_COMPLETE)
    assert state.stage is WorkflowStage.DESIGN_COMPLETE
    state.advance_to(WorkflowStage.DRAWINGS_GENERATED)
    assert state.stage is WorkflowStage.DRAWINGS_GENERATED

def test_loads_and_analysis_can_follow_a_design_or_drawing():
    """Revised loads after a design check, and loads and analysis after early GA drawings, move the stage back."""
    state = ProjectState()
    for stage in (WorkflowStage.MODEL_CREATED, WorkflowStage.LOADS_APPLIED, WorkflowStage.ANALYSIS_COMPLETE,
                  WorkflowStage.DESIGN_COMPLETE, WorkflowStage.LOADS_APPLIED):
        state.advance_to(stage)
    assert state.stage is WorkflowStage.LOADS_APPLIED

    state = ProjectState()
    for stage in (WorkflowStage.MODEL_CREATED, WorkflowStage.DRAWINGS_GENERATED, WorkflowStage.LOADS_APPLIED,
                  WorkflowStage.ANALYSIS_COMPLETE, WorkflowStage.DESIGN_COMPLETE):
        state.advance_to(stage)
    assert state.stage is WorkflowStage.DESIGN_COMPLETE

    # minimax validated the model after its analysis, then checked the design on that analysis
    state = ProjectState()
    for stage in (WorkflowStage.MODEL_CREATED, WorkflowStage.ANALYSIS_COMPLETE, WorkflowStage.MODEL_VALIDATED,
                  WorkflowStage.DESIGN_COMPLETE):
        state.advance_to(stage)
    assert state.stage is WorkflowStage.DESIGN_COMPLETE
