"""
Autonomous agent runner and multi-turn execution loop.

Coordinates the conversation between the LLM brain and the deterministic
structural engineering tool layer.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from struct_core import Project

from ..state.project_state import ProjectState, WorkflowStage
from ..tools import (
    apply_gravity_loads,
    apply_lateral_loads,
    build_structural_model,
    create_frame_model,
    create_project,
    export_to_etabs,
    export_to_sap2000,
    generate_drawings,
    generate_load_combinations,
    import_from_etabs,
    import_from_sap2000,
    inspect_analysis_results,
    inspect_design_results,
    inspect_project,
    modify_project,
    preliminary_sizing,
    run_analysis,
    inspect_analysis_results,
    run_time_history,
    inspect_dynamic_results,
    run_design,
)
from ..validation.model import validate_model
from .events import (
    CompleteEvent,
    ErrorEvent,
    LLMEvent,
    StateUpdateEvent,
    TextChunkEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .prompts import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from .providers import BaseLLMProvider, get_provider
from .schemas import get_tool_schemas


def _serialize_for_llm(obj: Any) -> Any:
    """Format tool output into a clean, token-efficient, JSON-safe structure for the LLM."""
    if hasattr(obj, "to_dict"):
        return _serialize_for_llm(obj.to_dict())
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k == "project" and hasattr(v, "metadata"):
                # Summarize Project to avoid wasting context tokens
                title = getattr(v.metadata, "title", "Untitled")
                n_nodes = len(getattr(v.model, "nodes", []))
                n_elems = len(getattr(v.model, "elements", []))
                cleaned[k] = f"Project(title={title!r}, nodes={n_nodes}, elements={n_elems})"
            else:
                cleaned[k] = _serialize_for_llm(v)
        return cleaned
    if isinstance(obj, list):
        return [_serialize_for_llm(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


class AgentRunner:
    """
    Autonomous Structural Engineering Agent orchestrator.

    Runs an event-driven multi-turn loop where the LLM reasons, calls tools,
    inspects results, self-corrects on failures, and completes structural designs.
    """

    def __init__(
        self,
        provider: Union[str, BaseLLMProvider] = "auto",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        base_url: Optional[str] = None,
        custom_instructions: str = "",
        max_turns: int = 15,
        temperature: float = 0.0,
    ):
        target_base = base_url or api_base
        if isinstance(provider, BaseLLMProvider):
            self.provider = provider
        else:
            self.provider = get_provider(
                provider=provider, model=model, api_key=api_key, api_base=target_base
            )

        self.max_turns = max_turns
        self.temperature = temperature
        self.system_prompt = build_system_prompt(custom_instructions)
        self.tools = get_tool_schemas()

        # State tracking
        self.project: Optional[Project] = None
        self.state = ProjectState()
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Register callable tools
        self._tool_map: Dict[str, Callable] = {
            "create_frame_model": create_frame_model,
            "build_structural_model": build_structural_model,
            "create_project": create_project,
            "inspect_project": inspect_project,
            "modify_project": modify_project,
            "preliminary_sizing": preliminary_sizing,
            "apply_gravity_loads": apply_gravity_loads,
            "apply_lateral_loads": apply_lateral_loads,
            "generate_load_combinations": generate_load_combinations,
            "validate_model": validate_model,
            "run_analysis": run_analysis,
            "inspect_analysis_results": inspect_analysis_results,
            "run_time_history": run_time_history,
            "inspect_dynamic_results": inspect_dynamic_results,
            "run_design": run_design,
            "inspect_design_results": inspect_design_results,
            "generate_drawings": generate_drawings,
            "import_from_sap2000": import_from_sap2000,
            "export_to_sap2000": export_to_sap2000,
            "import_from_etabs": import_from_etabs,
            "export_to_etabs": export_to_etabs,
        }

    def reset(self) -> None:
        """Reset agent conversation and project state."""
        self.project = None
        self.state = ProjectState()
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with automatic project injection."""
        fn = self._tool_map.get(tool_name)
        if not fn:
            from ..tools.base import AgentError, ToolResult
            return ToolResult.failure(
                tool_name,
                AgentError(
                    error_type="unknown_tool",
                    message=f"Tool {tool_name!r} is not registered in struct_agent.",
                ),
            )

        # Tools that need project passed as first parameter or kwarg
        project_consumer_tools = {
            "build_structural_model",
            "inspect_project",
            "modify_project",
            "preliminary_sizing",
            "apply_gravity_loads",
            "apply_lateral_loads",
            "generate_load_combinations",
            "validate_model",
            "run_analysis",
            "inspect_analysis_results",
            "run_time_history",
            "inspect_dynamic_results",
            "run_design",
            "inspect_design_results",
            "generate_drawings",
            "export_to_sap2000",
            "export_to_etabs",
        }

        call_args = dict(arguments)
        if tool_name in project_consumer_tools and "project" not in call_args:
            call_args["project"] = self.project

        # Special case: create_project might be called with no structural_model, build empty model
        if tool_name == "create_project" and "structural_model" not in call_args:
            from struct_core import StructuralModel
            call_args["structural_model"] = StructuralModel()

        return fn(**call_args)

    def run_stream(self, user_prompt: str) -> Generator[LLMEvent, None, str]:
        """
        Stream multi-turn agent execution, yielding LLMEvents in real time.
        """
        self.messages.append({"role": "user", "content": user_prompt})
        turn_count = 0
        final_answer = ""
        governing_dcr: Optional[float] = None
        passed: Optional[bool] = None

        while turn_count < self.max_turns:
            turn_count += 1

            # 1. Ask LLM for next action
            try:
                llm_response = self.provider.generate(
                    messages=self.messages,
                    tools=self.tools,
                    temperature=self.temperature,
                )
            except Exception as exc:
                yield ErrorEvent(message=f"LLM Provider error: {exc}")
                return f"Error during generation: {exc}"

            # 2. Append assistant response to messages
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if llm_response.content:
                assistant_msg["content"] = llm_response.content
                yield TextChunkEvent(chunk=llm_response.content)

            if llm_response.has_tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in llm_response.tool_calls
                ]

            self.messages.append(assistant_msg)

            # 3. If no tool calls, the model concluded its turn or is asking a question
            if not llm_response.has_tool_calls:
                final_answer = llm_response.content or ""
                break

            # 4. Dispatch requested tool calls
            for tool_call in llm_response.tool_calls:
                t_name = tool_call.name
                t_args = tool_call.arguments
                t_id = tool_call.id

                yield ToolCallEvent(tool_name=t_name, call_id=t_id, arguments=t_args)

                # Execute
                result = self._execute_tool(t_name, t_args)

                # Record tool execution in state
                status_str = "success" if result.ok else "failed"
                self.state.log_tool(t_name, status=status_str, payload=result.data if result.ok else None)

                # Track project updates
                if result.ok and isinstance(result.data, dict):
                    if "project" in result.data:
                        self.project = result.data["project"]
                    if "governing_dcr" in result.data:
                        governing_dcr = float(result.data["governing_dcr"])
                    if "passed" in result.data:
                        passed = bool(result.data["passed"])

                # Advance workflow state if suggested
                if result.ok and result.next_suggested_stage:
                    try:
                        self.state.advance_to(WorkflowStage(result.next_suggested_stage))
                    except Exception:
                        pass

                err_msg = result.error.message if (not result.ok and result.error) else None
                yield ToolResultEvent(
                    tool_name=t_name,
                    call_id=t_id,
                    ok=result.ok,
                    data=result.data,
                    error_message=err_msg,
                    next_stage=result.next_suggested_stage,
                )

                if self.project is not None:
                    yield StateUpdateEvent(
                        project=self.project, stage=self.state.stage.value
                    )

                # Append tool result to conversation history
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": t_id,
                        "content": json.dumps(_serialize_for_llm(result.to_dict())),
                    }
                )

        yield CompleteEvent(
            final_message=final_answer,
            total_turns=turn_count,
            governing_dcr=governing_dcr,
            passed=passed,
            project=self.project,
        )

        return final_answer

    def run(self, user_prompt: str) -> str:
        """
        Synchronous blocking execution. Runs multi-turn loop and returns the final answer.
        """
        generator = self.run_stream(user_prompt)
        last_result = ""
        for event in generator:
            if isinstance(event, CompleteEvent):
                last_result = event.final_message
        return last_result
