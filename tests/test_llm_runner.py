"""
Unit and mocked execution tests for struct_agent.llm.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import pytest

from struct_agent.llm import (
    AgentRunner,
    BaseLLMProvider,
    CompleteEvent,
    LLMEvent,
    LLMResponse,
    StateUpdateEvent,
    TextChunkEvent,
    ToolCallEvent,
    ToolCallRequest,
    ToolResultEvent,
    get_provider,
    get_tool_schemas,
)


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider that replays predetermined turns of responses and tool calls.
    Allows testing the entire cognitive orchestration loop without external network or API keys.
    """

    def __init__(self, turns: List[LLMResponse]):
        super().__init__(model="mock-agent")
        self.turns = list(turns)
        self.turn_idx = 0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if self.turn_idx < len(self.turns):
            resp = self.turns[self.turn_idx]
            self.turn_idx += 1
            return resp
        return LLMResponse(content="Design loop finished.")

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Generator[LLMEvent, None, LLMResponse]:
        resp = self.generate(messages, tools, temperature)
        if resp.content:
            yield TextChunkEvent(resp.content)
        return resp


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent.parent.parent / "struct_link" / "tests" / "fixtures"


def test_tool_schemas():
    schemas = get_tool_schemas()
    assert len(schemas) >= 15
    names = {s["function"]["name"] for s in schemas}
    assert "create_project" in names
    assert "apply_gravity_loads" in names
    assert "run_analysis" in names
    assert "run_design" in names
    assert "generate_drawings" in names
    assert "import_from_sap2000" in names


def test_provider_factory():
    prov = get_provider("native", model="gpt-4o")
    assert prov.model == "gpt-4o"

    with pytest.raises(ValueError):
        get_provider("invalid_provider")


def test_mocked_autonomous_agent_pipeline(fixtures_dir, tmp_path):
    s2k_path = str(fixtures_dir / "sample_frame.s2k")
    dxf_path = str(tmp_path / "final_frame.dxf")

    # Scripted sequence of actions the Mock LLM will execute:
    # Turn 1: Import SAP2000 file
    turn_1 = LLMResponse(
        content="I will import the existing structural frame from SAP2000.",
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="import_from_sap2000",
                arguments={"s2k_path": s2k_path},
            )
        ],
    )
    # Turn 2: Apply ASCE 7 gravity loads
    turn_2 = LLMResponse(
        content="Now applying building code gravity loads for office occupancy.",
        tool_calls=[
            ToolCallRequest(
                id="call_2",
                name="apply_gravity_loads",
                arguments={"occupancy": "office", "tributary_width": 5.0},
            )
        ],
    )
    # Turn 3: Run 3D/2D FEM static analysis
    turn_3 = LLMResponse(
        content="Running finite element structural analysis.",
        tool_calls=[
            ToolCallRequest(
                id="call_3",
                name="run_analysis",
                arguments={"analysis_case_id": "Static_1", "solver": "auto"},
            )
        ],
    )
    # Turn 4: Run ACI 318-19 concrete member design
    turn_4 = LLMResponse(
        content="Running ACI 318 code design checks.",
        tool_calls=[
            ToolCallRequest(
                id="call_4",
                name="run_design",
                arguments={"design_case_id": "ACI_Design_1"},
            )
        ],
    )
    # Turn 5: Generate DXF production drawings
    turn_5 = LLMResponse(
        content="Generating production CAD DXF structural drawings.",
        tool_calls=[
            ToolCallRequest(
                id="call_5",
                name="generate_drawings",
                arguments={"design_case_id": "ACI_Design_1", "output_path": dxf_path},
            )
        ],
    )
    # Turn 6: Conclude and report results
    turn_6 = LLMResponse(
        content="All structural checks pass ACI 318. DXF drawing has been successfully saved to final_frame.dxf."
    )

    mock_provider = MockLLMProvider([turn_1, turn_2, turn_3, turn_4, turn_5, turn_6])
    runner = AgentRunner(provider=mock_provider, max_turns=10)

    events: List[LLMEvent] = []
    stream = runner.run_stream("Design and detail my structural frame from the SAP2000 model.")
    for ev in stream:
        events.append(ev)

    # Verify event types and sequence
    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    state_updates = [e for e in events if isinstance(e, StateUpdateEvent)]
    complete_events = [e for e in events if isinstance(e, CompleteEvent)]

    assert len(tool_calls) == 5
    assert len(tool_results) == 5
    assert all(tr.ok for tr in tool_results)

    # Check that tool call names match the workflow
    called_tools = [tc.tool_name for tc in tool_calls]
    assert called_tools == [
        "import_from_sap2000",
        "apply_gravity_loads",
        "run_analysis",
        "run_design",
        "generate_drawings",
    ]

    # Check project state update
    assert runner.project is not None
    assert len(runner.project.model.nodes) == 4
    assert len(state_updates) > 0

    # Check final completion
    assert len(complete_events) == 1
    final_ev = complete_events[0]
    assert "All structural checks pass" in final_ev.final_message
    assert final_ev.total_turns == 6


def test_runner_reset():
    mock_provider = MockLLMProvider([])
    runner = AgentRunner(provider=mock_provider)
    runner.messages.append({"role": "user", "content": "hello"})
    runner.reset()
    assert len(runner.messages) == 1
    assert runner.messages[0]["role"] == "system"
    assert runner.project is None


def test_apply_self_weight_project_injection(tmp_path):
    turn_1 = LLMResponse(
        content="Create frame model",
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="create_frame_model",
                arguments={"num_bays": 1, "bay_width": 4.0, "num_stories": 1, "story_height": 3.0},
            )
        ],
    )
    turn_2 = LLMResponse(
        content="Apply self weight",
        tool_calls=[
            ToolCallRequest(
                id="call_2",
                name="apply_self_weight",
                arguments={},
            )
        ],
    )
    mock_provider = MockLLMProvider([turn_1, turn_2])
    runner = AgentRunner(provider=mock_provider, max_turns=3)
    events = list(runner.run_stream("Create frame and apply self weight"))
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2
    assert tool_results[0].ok is True
    assert tool_results[1].ok is True
    assert tool_results[1].tool_name == "apply_self_weight"

