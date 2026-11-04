"""
Autonomous agent runner and multi-turn execution loop.

Coordinates the conversation between the LLM brain and the deterministic
structural engineering tool layer.
"""

from __future__ import annotations

import difflib
import inspect
import json
import logging
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Union

logger = logging.getLogger(__name__)

# A server that cannot parse the model's tool call answers 500 (Ollama: "XML syntax error ..." from its qwen parser)
_UNPARSED_TOOL_CALL = re.compile(r"status 500:.*(?:XML syntax error|error parsing tool call)", re.I | re.S)

from struct_core import Project

from ..state.project_state import ProjectState, StageTransitionError, WorkflowStage
from ..tools import (
    apply_gravity_loads,
    apply_lateral_loads,
    apply_self_weight,
    create_frame_model,
    create_model_from_plan_draft,
    create_project,
    export_to_etabs,
    export_to_sap2000,
    generate_drawings,
    generate_load_combinations,
    import_from_etabs,
    import_from_sap2000,
    inspect_analysis_results,
    inspect_design_results,
    inspect_detailing_results,
    inspect_project,
    modify_project,
    preliminary_sizing,
    run_analysis,
    run_time_history,
    run_response_spectrum,
    inspect_dynamic_results,
    run_design,
    set_project_units,
)
from ..tools.base import AgentError, ToolResult
from ..validation.model import validate_model
from ..vision.encoding import build_multimodal_message
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
from .schemas import get_tool_schemas, normalize_tool_arguments


def _serialize_for_llm(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return _serialize_for_llm(obj.to_dict())
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k == "project" and hasattr(v, "metadata"):
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
        max_turns: int = 20,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
        allow_model_recreation: bool = True,
        allow_resizing: bool = True,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.allow_model_recreation = allow_model_recreation
        self.allow_resizing = allow_resizing

        target_base = base_url or api_base
        if isinstance(provider, BaseLLMProvider):
            self.provider = provider
        else:
            self.provider = get_provider(
                provider,
                model=model,
                api_key=api_key,
                api_base=target_base,
                timeout=timeout,
            )

        if max_tokens:
            self.provider.max_tokens = max_tokens
        if reasoning_effort:
            self.provider.reasoning_effort = reasoning_effort

        self.max_turns = max_turns
        self.temperature = temperature
        self.seed = seed
        self.system_prompt = build_system_prompt(
            custom_instructions=custom_instructions,
            allow_resizing=allow_resizing,
        )

        self.tools = [
            t
            for t in get_tool_schemas()
            if allow_resizing or t["function"]["name"] != "modify_project"
        ]

        self.project = None
        self.state = ProjectState()

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self._last_results = {}
        self.repeated_calls = 0
        self.rejected_tool_calls = 0

        self._tool_map = {
            "create_frame_model": create_frame_model,
            "create_model_from_plan_draft": create_model_from_plan_draft,
            "create_project": create_project,
            "inspect_project": inspect_project,
            "modify_project": modify_project,
            "set_project_units": set_project_units,
            "preliminary_sizing": preliminary_sizing,
            "apply_gravity_loads": apply_gravity_loads,
            "apply_lateral_loads": apply_lateral_loads,
            "apply_self_weight": apply_self_weight,
            "generate_load_combinations": generate_load_combinations,
            "validate_model": validate_model,
            "run_analysis": run_analysis,
            "inspect_analysis_results": inspect_analysis_results,
            "run_time_history": run_time_history,
            "run_response_spectrum": run_response_spectrum,
            "inspect_dynamic_results": inspect_dynamic_results,
            "run_design": run_design,
            "inspect_design_results": inspect_design_results,
            "inspect_detailing_results": inspect_detailing_results,
            "generate_drawings": generate_drawings,
            "import_from_sap2000": import_from_sap2000,
            "export_to_sap2000": export_to_sap2000,
            "import_from_etabs": import_from_etabs,
            "export_to_etabs": export_to_etabs,
        }

    def reset(self) -> None:
        self.project = None
        self.state = ProjectState()
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self._last_results = {}
        self.repeated_calls = 0
        self.rejected_tool_calls = 0

    def apply_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        arguments, problem = normalize_tool_arguments(tool_name, arguments)
        if problem:
            result = ToolResult.failure(
                tool_name,
                AgentError(error_type="invalid_input", message=problem),
            )
        else:
            result = self._execute_tool(tool_name, arguments)

        self.state.log_tool(
            tool_name,
            status="success" if result.ok else "failed",
            payload=result.data if result.ok else None,
        )

        if result.ok and isinstance(result.data, dict) and "project" in result.data:
            self.project = result.data["project"]

        if result.ok and result.next_suggested_stage:
            try:
                self.state.advance_to(WorkflowStage(result.next_suggested_stage))
            except (ValueError, KeyError, StageTransitionError) as e:
                logger.warning("Could not advance to suggested stage '%s': %s", result.next_suggested_stage, e)

        return result

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        fn = self._tool_map.get(tool_name)
        if not fn:
            return ToolResult.failure(
                tool_name,
                AgentError(
                    error_type="unknown_tool",
                    message=f"Tool {tool_name!r} is not registered in struct_agent.",
                ),
            )

        if tool_name not in {t["function"]["name"] for t in self.tools}:
            return ToolResult.failure(
                tool_name,
                AgentError(
                    error_type="tool_not_available",
                    message=f"Tool {tool_name!r} is not available in this run.",
                ),
            )

        resizing_tools = frozenset({
            "preliminary_sizing",
            "create_model_from_plan_draft",
            "create_project",
            "import_from_etabs",
            "import_from_sap2000",
            "create_frame_model",
        })

        if not self.allow_model_recreation and tool_name in resizing_tools:
            has_run_design = any(
                entry.get("tool") == "run_design" and entry.get("status") == "success"
                for entry in getattr(self.state, "tool_log", [])
            )
            if has_run_design:
                return ToolResult.failure(
                    tool_name,
                    AgentError(
                        error_type="ablation_guard",
                        message=(
                            f"Tool {tool_name!r} is blocked: re-creating or resizing the model "
                            "is prohibited in ablated (single-pass) mode after design checks have been performed."
                        ),
                    ),
                )

        params = inspect.signature(fn).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            valid = [name for name in params if name != "project"]
            unknown = [key for key in arguments if key not in params]
            if unknown:
                names = []
                for key in unknown:
                    close = difflib.get_close_matches(key.lower(), valid, n=1, cutoff=0.6)
                    names.append(repr(key) + (f" (did you mean {close[0]!r}?)" if close else ""))
                return ToolResult.failure(
                    tool_name,
                    AgentError(
                        error_type="invalid_arguments",
                        message=(
                            f"Tool {tool_name!r} has no parameter {', '.join(names)}, "
                            f"so the call was not executed. Its parameters are: {', '.join(valid)}."
                        ),
                    ),
                )

        call_args = dict(arguments)
        if "project" in params and "project" not in call_args:
            call_args["project"] = self.project

        try:
            return fn(**call_args)
        except Exception as exc:
            return ToolResult.failure(
                tool_name,
                AgentError(
                    error_type="tool_execution_error",
                    message=(
                        f"Execution of tool '{tool_name}' failed: {exc}. "
                        "Please verify the arguments against the tool schema."
                    ),
                ),
            )

    def run_stream(
        self,
        user_prompt: str,
        attachments: Optional[List[str]] = None,
    ) -> Generator[LLMEvent, None, str]:
        user_msg = build_multimodal_message(user_prompt, attachments)
        self.messages.append(user_msg)
        turn_count = 0
        final_answer = ""
        governing_dcr = None
        passed = None
        concluded = False

        while turn_count < self.max_turns:
            turn_count += 1

            gen_kwargs = {
                "messages": self.messages,
                "tools": self.tools,
                "temperature": self.temperature,
            }
            if self.seed is not None:
                gen_kwargs["seed"] = self.seed

            try:
                llm_response = self.provider.generate(**gen_kwargs)
            except Exception as exc:
                if _UNPARSED_TOOL_CALL.search(str(exc)):
                    # The model wrote a tool call the server could not parse; like a call with invalid
                    # arguments, it is refused back to the model and costs the turn.
                    self.rejected_tool_calls += 1
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response contained a tool call that the server could not parse "
                            f"({exc}), so it was not executed. Re-issue the call with well-formed arguments."
                        ),
                    })
                    continue
                yield ErrorEvent(f"LLM Provider error: {exc}")
                return f"Error during generation: {exc}"

            assistant_msg = {"role": "assistant"}
            if llm_response.content:
                assistant_msg["content"] = llm_response.content
                yield TextChunkEvent(llm_response.content)

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
                if llm_response.reasoning:
                    assistant_msg["reasoning"] = llm_response.reasoning

            self.messages.append(assistant_msg)

            if not llm_response.has_tool_calls:
                final_answer = llm_response.content or ""
                concluded = True
                break

            for tool_call in llm_response.tool_calls:
                t_name = tool_call.name
                t_args = normalize_tool_arguments(t_name, tool_call.arguments or {})[0]
                t_id = tool_call.id

                yield ToolCallEvent(tool_name=t_name, call_id=t_id, arguments=t_args)

                result = self.apply_tool_call(t_name, t_args)

                if result.ok and isinstance(result.data, dict):
                    if result.data.get("governing_dcr") is not None:
                        governing_dcr = float(result.data["governing_dcr"])
                    if "passed" in result.data:
                        passed = bool(result.data["passed"])

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
                        project=self.project,
                        stage=self.state.stage.value,
                    )

                content = json.dumps(_serialize_for_llm(result.to_dict()))
                signature = f"{t_name}:{json.dumps(t_args, sort_keys=True, default=str)}"

                if self._last_results.get(signature) == content:
                    self.repeated_calls += 1
                    content = json.dumps({
                        "repeated_call_notice": (
                            f"This {t_name} call repeats an earlier call with the same arguments "
                            "and returned the same result, so it changed nothing. Do not repeat it; "
                            "continue with the next workflow step or report the results."
                        ),
                        "result": json.loads(content),
                    })
                else:
                    self._last_results[signature] = content

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": t_id,
                    "content": content,
                })

        if not concluded:
            self.messages.append({
                "role": "user",
                "content": (
                    "The tool-call budget for this task is exhausted. Do not call any more tools. "
                    "Report the requested results from the tool outputs above, and state what remains unfinished."
                ),
            })

            wrap_kwargs = {
                "messages": self.messages,
                "tools": None,
                "temperature": self.temperature,
            }
            if self.seed is not None:
                wrap_kwargs["seed"] = self.seed

            try:
                wrap_up = self.provider.generate(**wrap_kwargs)
                final_answer = wrap_up.content or ""
                self.messages.append({"role": "assistant", "content": final_answer})
                if final_answer:
                    yield TextChunkEvent(final_answer)
            except Exception as exc:
                yield ErrorEvent(f"LLM Provider error during the final report: {exc}")

        yield CompleteEvent(
            final_message=final_answer,
            total_turns=turn_count,
            governing_dcr=governing_dcr,
            passed=passed,
            project=self.project,
            turn_limit_reached=not concluded,
        )

        return final_answer

    def run(self, user_prompt: str) -> str:
        generator = self.run_stream(user_prompt)
        last_result = ""
        for event in generator:
            if isinstance(event, CompleteEvent):
                last_result = event.final_message
        return last_result
