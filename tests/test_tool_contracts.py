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

# -----------------------------------------------------------------------------
# LLM schema <-> tool implementation contract
# -----------------------------------------------------------------------------
# Parameters a tool accepts but the LLM schema intentionally does not offer (programmatic use only).
# Anything else missing from a schema is a bug: the model cannot pass it (e.g. a hidden `sds` silently
# dropped the vertical seismic effect from every agent-built ELF combination).
_INTERNAL = {"project", "mutator", "params", "mass_source", "units", "draft", "kwargs", "existing_project",
             "model_or_project"}
_HIDDEN_ON_PURPOSE = {
    "create_frame_model": {"author"},
    "create_project": {"nodes", "elements", "materials", "sections", "supports", "structural_model"},
    "preliminary_sizing": {"rho_kg_m3"},
    "apply_lateral_loads": {"exposure_category", "tributary_width", "storage_live_cases", "partition_cases",
                            "snow_cases", "flat_roof_snow_pressure"},  # wind and snow are outside the agent scope
    "generate_load_combinations": {"seismic_design_category", "reduce_live_in_seismic"},
    "run_analysis": {"include_shear_deformation"},
    "run_time_history": {"include_shear_deformation", "scale_factor"},
    "run_response_spectrum": {"include_shear_deformation", "spectrum_curve", "Sa", "Sd", "scale_factor"},
    "generate_drawings": {"detailing_case_id", "scale", "unit"},
    "import_from_sap2000": {"materialize_self_weight"},
}

_NOT_OFFERED = {
    "apply_roof_loads": "Wind and snow are outside the primary LLM conversational workflow scope",
    "define_mass_source": "Low-level mass source configuration; handled automatically via seismic tools",
    "distribute_area_loads": "Low-level area load distributor; handled via apply_gravity_loads",
    "generate_framing_drawings": "Specialized drafting function; agent uses generate_drawings",
}


def _idle_runner():
    from struct_agent.llm import AgentRunner, BaseLLMProvider, LLMResponse

    class _Idle(BaseLLMProvider):
        def __init__(self):
            super().__init__(model="contract")

        def generate(self, messages, tools=None, temperature=0.0):
            return LLMResponse(content="")

        def generate_stream(self, messages, tools=None, temperature=0.0):
            yield self.generate(messages)

    return AgentRunner(provider=_Idle())


def _runner_tool_map():
    return _idle_runner()._tool_map


def test_every_schema_tool_is_executable_and_every_parameter_is_exposed():
    import inspect

    from struct_agent.llm.schemas import get_tool_schemas

    tool_map = _runner_tool_map()
    problems = []
    for schema in get_tool_schemas():
        fn = schema["function"]
        name, props = fn["name"], set(fn["parameters"].get("properties", {}))
        impl = tool_map.get(name)
        if impl is None:
            problems.append(f"{name}: in the schema but not registered in AgentRunner")
            continue
        sig = inspect.signature(impl)
        params = {p for p, v in sig.parameters.items() if v.kind in (v.KEYWORD_ONLY, v.POSITIONAL_OR_KEYWORD)}
        hidden = params - props - _INTERNAL - _HIDDEN_ON_PURPOSE.get(name, set())
        if hidden:
            problems.append(f"{name}: parameters not in the LLM schema: {sorted(hidden)}")
        if not any(v.kind == v.VAR_KEYWORD for v in sig.parameters.values()):
            unknown = props - params - {"units"}
            if unknown:
                problems.append(f"{name}: schema properties the tool does not accept: {sorted(unknown)}")
    assert not problems, "\n".join(problems)


def test_project_tools_run_through_the_runner_and_tool_exceptions_are_returned():
    """Every tool with a ``project`` parameter receives the current project (gemma's ablated runs called
    set_project_units, which was not injected); a tool that raises returns a failure instead of ending the run."""
    runner = _idle_runner()
    created = runner._execute_tool("create_frame_model", {"num_bays": 1, "bay_width": 5.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0})
    runner.project = created.data["project"]
    assert runner._execute_tool("set_project_units", {"units": "kN, m"}).ok
    refused = runner._execute_tool("create_frame_model", {"not_a_parameter": 1})
    assert not refused.ok and refused.error.error_type == "invalid_arguments"

    def raises(project: object) -> None:
        raise ValueError("boom")
    runner._tool_map["set_project_units"] = raises
    failed = runner._execute_tool("set_project_units", {})
    assert not failed.ok and failed.error.error_type == "tool_execution_error" and "boom" in failed.error.message


def test_string_encoded_scalar_arguments_are_decoded_or_refused():
    """nemotron sent "sds": "None" (a TypeError inside the tool) and "cracked_sections": "False" (a string a tool
    would read as true): scalars are decoded by their schema type, and an undecodable one is refused."""
    from struct_agent.llm.schemas import normalize_tool_arguments

    args, problem = normalize_tool_arguments(
        "generate_load_combinations", {"sds": "None", "orthogonal_seismic": "False", "rho": "1.0", "standard": "ASCE7_LRFD"})
    assert problem is None
    assert args == {"orthogonal_seismic": False, "rho": 1.0, "standard": "ASCE7_LRFD"}
    assert normalize_tool_arguments("run_analysis", {"cracked_sections": "TRUE"}) == ({"cracked_sections": True}, None)

    runner = _idle_runner()
    runner.project = runner._execute_tool("create_frame_model", {"num_bays": 1, "bay_width": 5.0, "num_stories": 1, "story_height": 3.0, "fc": 27580.0}).data["project"]
    refused = runner.apply_tool_call("run_analysis", {"analysis_case_id": "S", "cracked_sections": "maybe"})
    assert not refused.ok and refused.error.error_type == "invalid_input" and "true or false" in refused.error.message


def test_all_tools_in_module_are_in_tool_map_or_explicitly_not_offered():
    import struct_agent.tools as tools
    tool_map = _runner_tool_map()
    for name in tools.__all__:
        if name in ("ToolResult", "AgentError"):
            continue
        assert name in tool_map or name in _NOT_OFFERED, (
            f"Tool {name!r} is neither in runner tool_map nor in _NOT_OFFERED"
        )
