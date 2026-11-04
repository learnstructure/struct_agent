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
    ErrorEvent,
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
    # Turn 1b: The file has no f'c (the import result says so): give the concrete its strength
    turn_materials = LLMResponse(
        content="The imported material has no f'c; setting the concrete strength.",
        tool_calls=[
            ToolCallRequest(
                id="call_1b",
                name="modify_project",
                arguments={"materials": [{"id": "CONC", "type": "concrete", "fc": 27580.0}]},
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
                arguments={"occupancy": "office", "tributary_width": 5.0, "superimposed_dead": 1.5},
            )
        ],
    )
    # Turn 3: Generate the ASCE 7-22 factored load combinations
    turn_combinations = LLMResponse(
        content="Generating the factored load combinations.",
        tool_calls=[ToolCallRequest(id="call_2b", name="generate_load_combinations", arguments={})],
    )
    # Turn 4: Run 3D/2D FEM static analysis
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

    mock_provider = MockLLMProvider([turn_1, turn_materials, turn_2, turn_combinations, turn_3, turn_4, turn_5, turn_6])
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

    assert len(tool_calls) == 7
    assert len(tool_results) == 7
    assert all(tr.ok for tr in tool_results), [(tr.tool_name, tr.error_message) for tr in tool_results if not tr.ok]

    # Check that tool call names match the workflow
    called_tools = [tc.tool_name for tc in tool_calls]
    assert called_tools == [
        "import_from_sap2000",
        "modify_project",
        "apply_gravity_loads",
        "generate_load_combinations",
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
    assert final_ev.total_turns == 8


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
                arguments={"num_bays": 1, "bay_width": 4.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0},
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



def test_repeated_identical_calls_are_flagged_not_blocked():
    """A call repeating an earlier one with the same arguments and result gets a notice; the tool still runs."""
    args = {"num_bays": 1, "bay_width": 4.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0}
    resize = {"sections": [{"id": "BEAM", "b": 0.3, "h": 0.3}]}
    turns = [
        LLMResponse(content="", tool_calls=[ToolCallRequest(id="c1", name="create_frame_model", arguments=args)]),
        LLMResponse(content="", tool_calls=[ToolCallRequest(id="c2", name="modify_project", arguments=resize)]),
        LLMResponse(content="", tool_calls=[ToolCallRequest(id="c3", name="modify_project", arguments=resize)]),
        LLMResponse(content="", tool_calls=[ToolCallRequest(id="c4", name="modify_project", arguments=resize)]),
        LLMResponse(content="", tool_calls=[ToolCallRequest(id="c5", name="inspect_project", arguments={})]),
    ]
    runner = AgentRunner(provider=MockLLMProvider(turns), max_turns=10)
    results = [e for e in runner.run_stream("Resize") if isinstance(e, ToolResultEvent)]
    assert all(r.ok for r in results)
    tool_msgs = {m["tool_call_id"]: m["content"] for m in runner.messages if m.get("role") == "tool"}
    # c2 resizes; c3 asks for the size the beam already has (the tool says nothing changed); c4 repeats c3
    assert "repeated_call_notice" not in tool_msgs["c2"] and "did not change" not in tool_msgs["c2"]
    assert "did not change" in tool_msgs["c3"] and "repeated_call_notice" not in tool_msgs["c3"]
    assert "repeated_call_notice" in tool_msgs["c4"] and runner.repeated_calls == 1
    assert "repeated_call_notice" not in tool_msgs["c5"]


def test_turn_limit_ends_with_a_tool_free_report():
    """When the turn budget runs out mid-loop, the model gets one tool-free turn to report its results."""
    loop = [
        LLMResponse(content="", tool_calls=[ToolCallRequest(id=f"c{i}", name="create_frame_model", arguments={})])
        for i in range(2)
    ]

    class Recorder(MockLLMProvider):
        tools_seen: List[Any] = []

        def generate(self, messages, tools=None, temperature=0.0):
            self.tools_seen.append(tools)
            return super().generate(messages, tools, temperature)

    provider = Recorder(loop + [LLMResponse(content="W = 100 kN; resizing unfinished.")])
    runner = AgentRunner(provider=provider, max_turns=2)
    complete = [e for e in runner.run_stream("Design a frame") if isinstance(e, CompleteEvent)][0]
    assert complete.turn_limit_reached is True
    assert complete.final_message == "W = 100 kN; resizing unfinished."
    assert provider.tools_seen[-1] is None and provider.tools_seen[0]

    concluded = AgentRunner(provider=MockLLMProvider([]), max_turns=2)
    done = [e for e in concluded.run_stream("Hello") if isinstance(e, CompleteEvent)][0]
    assert done.turn_limit_reached is False


def test_seed_reaches_every_llm_call_and_is_never_dropped():
    """The seed goes to every call, the tool-free wrap-up turn included; a provider that rejects it is an error."""
    loop = [
        LLMResponse(content="", tool_calls=[ToolCallRequest(id=f"c{i}", name="create_frame_model", arguments={})])
        for i in range(2)
    ]

    class SeedRecorder(MockLLMProvider):
        seeds_seen: List[Any] = []

        def generate(self, messages, tools=None, temperature=0.0, seed=None):
            self.seeds_seen.append(seed)
            return super().generate(messages, tools, temperature)

    provider = SeedRecorder(loop + [LLMResponse(content="Done.")])
    list(AgentRunner(provider=provider, max_turns=2, seed=43).run_stream("Design a frame"))
    assert provider.seeds_seen == [43, 43, 43]

    events = list(AgentRunner(provider=MockLLMProvider([]), seed=43).run_stream("Hello"))
    assert any(isinstance(e, ErrorEvent) and "seed" in e.message for e in events)


def test_ablated_runner_neither_offers_nor_runs_the_resizing_tool():
    """allow_resizing=False removes modify_project from the tools and the prompt, and refuses a call by name."""
    from struct_agent.llm import DEFAULT_SYSTEM_PROMPT

    full = AgentRunner(provider=MockLLMProvider([]))
    assert "modify_project" in {t["function"]["name"] for t in full.tools}
    assert "modify_project" in full.system_prompt and "{reflection_stage}" not in DEFAULT_SYSTEM_PROMPT

    ablated = AgentRunner(provider=MockLLMProvider([]), allow_resizing=False)
    assert "modify_project" not in {t["function"]["name"] for t in ablated.tools}
    assert "modify_project" not in ablated.system_prompt
    # gemma's ablated runs called modify_project by name from the prompt, and it ran
    ablated._execute_tool("create_frame_model", {"num_bays": 1, "bay_width": 5.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0})
    res = ablated._execute_tool("modify_project", {"sections": [{"id": "COL", "b": 0.6, "h": 0.6}]})
    assert not res.ok and "not available in this run" in res.error.message



def test_unknown_argument_name_is_refused_with_the_valid_names():
    """minimax A02: 'supimposed_dead' failed with Python's bare TypeError nine times; the refusal now names the
    closest valid parameter and lists the others."""
    runner = AgentRunner(provider=MockLLMProvider([]))
    runner.apply_tool_call("create_frame_model", {"num_bays": 1, "bay_width": 5.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0})
    res = runner._execute_tool("apply_gravity_loads", {"occupancy": "office", "supimposed_dead": 5.1})
    assert not res.ok and res.error.error_type == "invalid_arguments"
    assert "'supimposed_dead' (did you mean 'superimposed_dead'?)" in res.error.message
    assert "custom_live_load" in res.error.message and "project" not in res.error.message
    assert not runner.project.model.load_cases  # nothing was applied
    res = runner._execute_tool("apply_gravity_loads", {"occupancy": "office", "Superimposed_dead": 5.1})
    assert "did you mean 'superimposed_dead'" in res.error.message
    assert runner._execute_tool("apply_gravity_loads",
                                {"occupancy": "office", "superimposed_dead": 5.1, "tributary_width": 5.0}).ok


def test_a_tool_call_the_server_cannot_parse_is_refused_back_to_the_model():
    """qwen B04: Ollama answered 500 to a malformed tool call at the same turn of every attempt; the refusal
    now goes back to the model and costs the turn, while any other provider error still ends the run."""
    xml = ('LLM API request failed with status 500: {"error":{"message":"XML syntax error on line 8: '
           'element <function> closed by </parameter>","type":"api_error"}}')

    class Rejecting(MockLLMProvider):
        def __init__(self, turns, failures):
            super().__init__(turns)
            self.failures = list(failures)

        def generate(self, messages, tools=None, temperature=0.0):
            if self.failures:
                raise RuntimeError(self.failures.pop(0))
            return super().generate(messages, tools, temperature)

    runner = AgentRunner(provider=Rejecting([LLMResponse(content="W = 100 kN")], [xml]), max_turns=3)
    events = list(runner.run_stream("Design a frame"))
    assert not [e for e in events if isinstance(e, ErrorEvent)]
    complete = [e for e in events if isinstance(e, CompleteEvent)][0]
    assert complete.final_message == "W = 100 kN" and complete.total_turns == 2
    assert runner.rejected_tool_calls == 1
    assert "could not parse" in runner.messages[-2]["content"] and "XML syntax error" in runner.messages[-2]["content"]

    down = AgentRunner(provider=Rejecting([], ["LLM API request failed with status 500: Internal Server Error"]))
    events = list(down.run_stream("Design a frame"))
    assert [e.message for e in events if isinstance(e, ErrorEvent)][0].startswith("LLM Provider error")
    assert down.rejected_tool_calls == 0


def test_ablated_runner_blocks_resizing_by_preliminary_sizing_after_a_design_check():
    """nemotron A07 (ablated): preliminary_sizing with chosen minimum dimensions resized the frame after its
    design check, which the ablation forbids like a model re-creation."""
    from struct_agent.tools.analysis import run_analysis
    from struct_agent.tools.loads import apply_gravity_loads, generate_load_combinations

    runner = AgentRunner(provider=MockLLMProvider([]), allow_resizing=False, allow_model_recreation=False)
    runner.apply_tool_call("create_frame_model", {"num_bays": 1, "bay_width": 5.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0})
    apply_gravity_loads(runner.project, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    generate_load_combinations(runner.project)
    run_analysis(runner.project, analysis_case_id="env")
    assert runner.apply_tool_call("preliminary_sizing", {}).ok  # before any design check: allowed
    run_analysis(runner.project, analysis_case_id="env")  # the resized model (run_design refuses the old analysis)
    assert runner.apply_tool_call("run_design", {"design_case_id": "d", "analysis_case_id": "env"}).ok
    blocked = runner.apply_tool_call("preliminary_sizing", {"min_col_dim": 0.6})
    assert not blocked.ok and blocked.error.error_type == "ablation_guard"
    # minimax A08 (ablated) exported its model and re-imported it to get a fresh one
    reimport = runner.apply_tool_call("import_from_sap2000", {"s2k_path": "frame.s2k"})
    assert not reimport.ok and reimport.error.error_type == "ablation_guard"


def test_a_2d_frame_is_not_exported_to_etabs(tmp_path):
    """ETABS is story-based 3D: a 2D frame's .e2k re-imported as 3 nodes and 0 elements. SAP2000 keeps it."""
    runner = AgentRunner(provider=MockLLMProvider([]))
    runner.apply_tool_call("create_frame_model", {"num_bays": 2, "bay_width": 5.0, "num_stories": 2, "story_height": 3.0, "fc": 27580.0})
    refused = runner.apply_tool_call("export_to_etabs", {"output_path": str(tmp_path / "f.e2k")})
    assert not refused.ok and "export_to_sap2000" in refused.error.message and not (tmp_path / "f.e2k").exists()
    assert runner.apply_tool_call("export_to_sap2000", {"output_path": str(tmp_path / "f.s2k")}).ok


def test_thinking_model_reasoning_goes_back_with_its_tool_calls():
    """gpt-oss ended 5 of 6 runs mid-workflow with an empty reply while its reasoning was dropped from the
    history; the chat templates of gpt-oss and Qwen3 expect the reasoning of the current tool-call chain."""
    args = {"num_bays": 1, "bay_width": 4.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0}
    turns = [
        LLMResponse(content="", reasoning="Create the frame first.",
                    tool_calls=[ToolCallRequest(id="c1", name="create_frame_model", arguments=args)]),
        LLMResponse(content="Done.", reasoning="The frame exists; report."),
    ]
    runner = AgentRunner(provider=MockLLMProvider(turns), max_turns=5)
    list(runner.run_stream("Build a portal frame"))
    assistant = [m for m in runner.messages if m.get("role") == "assistant"]
    assert assistant[0]["reasoning"] == "Create the frame first." and assistant[0]["tool_calls"]
    assert "reasoning" not in assistant[1]  # a final answer ends the chain: nothing to carry forward


def test_http_provider_reads_the_reasoning_field(monkeypatch):
    import io
    import json

    from struct_agent.llm.providers.http_client import NativeHTTPProvider

    body = {"choices": [{"message": {"role": "assistant", "content": "", "reasoning": "Think.",
                                     "tool_calls": []}, "finish_reason": "stop"}], "usage": {}}

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Reply(json.dumps(body).encode()))
    assert NativeHTTPProvider(model="m", api_base="http://x").generate([]).reasoning == "Think."


def test_runner_caps_the_tokens_of_each_response():
    from struct_agent.llm.providers.http_client import NativeHTTPProvider

    runner = AgentRunner(provider="native", model="m", api_base="http://x", max_tokens=48000, reasoning_effort="high")
    assert isinstance(runner.provider, NativeHTTPProvider)
    payload = runner.provider._prepare_payload([])
    assert payload["max_tokens"] == 48000 and payload["reasoning_effort"] == "high"
    bare = NativeHTTPProvider(model="m", api_base="http://x")._prepare_payload([])
    assert "max_tokens" not in bare and "reasoning_effort" not in bare  # gemma4 then does not think


def test_create_project_keeps_the_model_it_is_given():
    """The runner passed an empty structural_model to every create_project call, so a model given as 'model'
    (the system prompt's route for custom geometry) was dropped and an empty project reported as created."""
    runner = AgentRunner(provider=MockLLMProvider([]))
    model = {"nodes": [{"id": "N1", "x": 0, "y": 0}, {"id": "N2", "x": 0, "y": 3}],
             "elements": [{"id": "C1", "start_node": "N1", "end_node": "N2", "section_id": "COL"}],
             "supports": [{"node_id": "N1", "fixity": "fixed"}]}
    # Nothing is invented: an element naming an undefined section is refused, naming it
    refused = runner.apply_tool_call("create_project", {"title": "custom", "model": model})
    assert not refused.ok and "COL" in refused.error.message and runner.project is None
    model["materials"] = [{"id": "C", "type": "concrete", "fc": 4.0}]
    model["sections"] = [{"id": "COL", "b": 12.0, "h": 12.0, "material_id": "C"}]
    res = runner.apply_tool_call("create_project", {"title": "custom", "model": model, "units": "kip, in"})
    assert res.ok and (res.data["n_nodes"], res.data["n_elements"], res.data["n_supports"]) == (2, 1, 1)
    # The concrete and the concrete section took the documented defaults, and the agent is told so
    applied = {d["name"] for d in res.data["defaults_applied"]}
    assert {"E", "fy", "cover"} <= applied
    sec = runner.project.model.sections[0]
    assert (sec.rebar_material_id, sec.cover) == ("REBAR", pytest.approx(1.5))
    # minimax C04: an element between two undefined nodes is refused, and the model stays as it was
    bad = runner.apply_tool_call("create_project", {"elements": [{"id": "T", "start_node": "X", "end_node": "Y", "section_id": "COL"}]})
    assert not bad.ok and "X" in bad.error.message and len(runner.project.model.nodes) == 2
    assert runner.apply_tool_call("create_project", {"title": "empty"}).data["n_nodes"] == 0
