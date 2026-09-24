"""Tests for the tool result base types."""

from struct_agent.tools.base import AgentError, ToolResult


def test_tool_result_success_factory():
    result = ToolResult.success("test_tool", data={"x": 1})
    assert result.ok is True
    assert result.status == "success"
    assert result.tool == "test_tool"
    assert result.error is None


def test_tool_result_failure_factory():
    err = AgentError(error_type="schema_error", message="bad", possible_causes=["x"])
    result = ToolResult.failure("test_tool", err, data={"y": 2})
    assert result.ok is False
    assert result.error is err
    d = result.to_dict()
    assert d["status"] == "failed"
    assert d["error"]["error_type"] == "schema_error"
    assert d["data"] == {"y": 2}


def test_tool_result_round_trip_to_dict():
    result = ToolResult.success("noop", data={"hello": "world"}, next_suggested_stage="MODEL_CREATED")
    d = result.to_dict()
    assert d["tool"] == "noop"
    assert d["data"] == {"hello": "world"}
    assert d["next_suggested_stage"] == "MODEL_CREATED"


def test_agent_error_to_dict():
    err = AgentError(
        error_type="analysis_failure",
        message="matrix singular",
        possible_causes=["Insufficient boundary conditions"],
    )
    assert err.to_dict() == {
        "error_type": "analysis_failure",
        "message": "matrix singular",
        "possible_causes": ["Insufficient boundary conditions"],
    }