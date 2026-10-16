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
    with pytest.raises(StageTransitionError):
        state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)


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