"""
struct_agent LLM Streaming GUI Simulation Example
==================================================

Demonstrates how a future Desktop GUI (PyQt/Electron) or Web Dashboard
(FastAPI/React/WebSockets) connects to `AgentRunner.run_stream()` to build
a real-time interactive structural engineering assistant.

Architecture demonstrated:
- The UI layer does NOT know any internal engineering logic.
- The UI simply registers event listeners for:
  - `ThoughtEvent`: Live stream of the LLM's engineering reasoning
  - `ToolCallEvent`: Showing active tool execution indicators & spinners
  - `ToolResultEvent`: Updating tool status logs & error toasts
  - `StateUpdateEvent`: Live 3D model viewport refresh & stage timeline tracker
  - `CompleteEvent`: Final design summary card, governing DCR gauge, pass/fail badge

Usage:
    python struct_agent/examples/llm_streaming_gui_simulation.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from struct_agent import (
    AgentRunner,
    BaseLLMProvider,
    CompleteEvent,
    ErrorEvent,
    LLMEvent,
    LLMResponse,
    StateUpdateEvent,
    TextChunkEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolCallRequest,
    ToolResultEvent,
)


class MockMultiTurnAgent(BaseLLMProvider):
    """
    Simulates a multi-turn structural engineering AI session for GUI preview.
    Runs completely offline and self-contained.
    """

    def __init__(self, s2k_path: str, dxf_path: str):
        super().__init__(model="gui-stream-agent")
        self.s2k_path = s2k_path
        self.dxf_path = dxf_path
        self.turn = 0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.turn += 1

        if self.turn == 1:
            return LLMResponse(
                content="[Analyzing structural project requirements and loading commercial SAP2000 model]",
                tool_calls=[
                    ToolCallRequest(
                        id="c1",
                        name="import_from_sap2000",
                        arguments={"s2k_path": self.s2k_path, "project_title": "Modern Commercial Center"},
                    )
                ],
            )
        elif self.turn == 2:
            return LLMResponse(
                content="[Imported geometry. Performing ASCE 7-22 gravity load calculations and tributary area distribution]",
                tool_calls=[
                    ToolCallRequest(
                        id="c2",
                        name="apply_gravity_loads",
                        arguments={
                            "occupancy": "office",
                            "tributary_width": 4.5,
                            "superimposed_dead": 1.2,
                            "include_self_weight": True,
                        },
                    )
                ],
            )
        elif self.turn == 3:
            return LLMResponse(
                content="[Generating ASCE 7-22 / ACI 318-19 ultimate strength load combinations]",
                tool_calls=[
                    ToolCallRequest(
                        id="c3",
                        name="generate_load_combinations",
                        arguments={"standard": "ACI318_LRFD"},
                    )
                ],
            )
        elif self.turn == 4:
            return LLMResponse(
                content="[Validating model completeness, materials, and support conditions]",
                tool_calls=[
                    ToolCallRequest(
                        id="c4",
                        name="validate_model",
                        arguments={},
                    )
                ],
            )
        elif self.turn == 5:
            return LLMResponse(
                content="[Model validated. Executing linear static FEM solver with automatic dimension detection]",
                tool_calls=[
                    ToolCallRequest(
                        id="c5",
                        name="run_analysis",
                        arguments={"analysis_case_id": "CASE_STATIC", "solver": "auto"},
                    )
                ],
            )
        elif self.turn == 6:
            return LLMResponse(
                content="[Analysis converged. Running ACI 318-19 cross-section design & reinforcement optimization]",
                tool_calls=[
                    ToolCallRequest(
                        id="c6",
                        name="run_design",
                        arguments={"design_case_id": "ACI318_DES_01", "design_code": "ACI 318-19"},
                    )
                ],
            )
        elif self.turn == 7:
            return LLMResponse(
                content="[Generating construction-grade detailing drawings and schedules in DXF format]",
                tool_calls=[
                    ToolCallRequest(
                        id="c7",
                        name="generate_drawings",
                        arguments={
                            "design_case_id": "ACI318_DES_01",
                            "output_path": self.dxf_path,
                        },
                    )
                ],
            )
        else:
            return LLMResponse(
                content=(
                    "Structural design optimization complete! The frame satisfies ACI 318-19 "
                    "with all demand-capacity ratios (DCR) within code allowable limits. "
                    f"Production DXF drawings exported to {self.dxf_path}"
                )
            )

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


class TerminalGUIDashboard:
    """
    Simulates a GUI client dashboard consuming streaming events from the agent runner.
    """

    def __init__(self):
        self.stage_history = []
        self.tool_log = []
        self.active_project = None

    def on_thought(self, thought: str):
        print(f"\n[AI THOUGHT] {thought}")

    def on_tool_call(self, tool_name: str, arguments: dict):
        print(f"  --> [RUNNING TOOL] {tool_name}")
        for k, v in arguments.items():
            print(f"        * {k}: {v}")

    def on_tool_result(self, tool_name: str, ok: bool, data: dict, error: Optional[str] = None):
        badge = "[SUCCESS]" if ok else "[FAILED]"
        print(f"  <-- [TOOL FINISHED] {badge}: {tool_name}")
        if error:
            print(f"        ! Error: {error}")

    def on_state_update(self, project: Any, stage: str):
        self.active_project = project
        self.stage_history.append(stage)
        num_nodes = len(project.model.nodes) if hasattr(project, "model") else 0
        num_elements = len(project.model.elements) if hasattr(project, "model") else 0

        print(f"  [UI STATE SYNC] Stage: [{stage}] | Model Nodes: {num_nodes} | Elements: {num_elements}")

    def on_complete(self, event: CompleteEvent):
        print("\n" + "+" + "=" * 68 + "+")
        print("|              STRUCTURAL DESIGN WORKFLOW COMPLETED                  |")
        print("+" + "=" * 68 + "+")
        print(f"|  Workflow Turns Executed : {event.total_turns:<40} |")
        pass_str = "PASSED (Code Compliant)" if event.passed else "REVIEW / SAFE"
        print(f"|  ACI 318 Design Status   : {pass_str:<40} |")
        if event.governing_dcr is not None:
            print(f"|  Governing Max DCR       : {event.governing_dcr:<40.3f} |")
        print("+" + "=" * 68 + "+")
        print(f"|  Summary: {event.final_message[:56]:<57} |")
        print("+" + "=" * 68 + "+")


def main():
    print("=" * 70)
    print(" struct_agent: Live Streaming GUI / Frontend Integration Simulation")
    print("=" * 70)

    workspace_root = Path(__file__).resolve().parent.parent.parent
    sample_s2k = str(workspace_root / "struct_link" / "tests" / "fixtures" / "sample_frame.s2k")
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dxf_output = str(output_dir / "gui_sim_detailing.dxf")

    # Connect to custom provider (or pass api_key to connect live to OpenAI/Ollama)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("[INFO] Live OPENAI_API_KEY detected. Using OpenAI GPT-4o...")
        runner = AgentRunner(provider="native", model="gpt-4o", api_key=api_key)
    else:
        print("[INFO] Running in GUI Simulation mode (no external API key required).")
        sim_provider = MockMultiTurnAgent(s2k_path=sample_s2k, dxf_path=dxf_output)
        runner = AgentRunner(provider=sim_provider)

    dashboard = TerminalGUIDashboard()

    prompt = (
        f"Design the building frame from '{sample_s2k}'. Apply office gravity loads, "
        f"generate LRFD load combinations, run analysis, check ACI 318-19 design, "
        f"and output drawings to '{dxf_output}'."
    )

    print(f"\nUser Input: \"{prompt}\"\n")
    print("Connecting event stream to GUI Dashboard...")
    time.sleep(0.5)

    # Event consumption loop: this is the exact loop a GUI or WebSocket server implements
    for event in runner.run_stream(prompt):
        if isinstance(event, ThoughtEvent):
            dashboard.on_thought(event.thought)
        elif isinstance(event, ToolCallEvent):
            dashboard.on_tool_call(event.tool_name, event.arguments)
        elif isinstance(event, ToolResultEvent):
            dashboard.on_tool_result(event.tool_name, event.ok, event.data, event.error_message)
        elif isinstance(event, StateUpdateEvent):
            dashboard.on_state_update(event.project, event.stage)
        elif isinstance(event, CompleteEvent):
            dashboard.on_complete(event)
        elif isinstance(event, ErrorEvent):
            print(f"[AGENT ERROR] {event.message}")


if __name__ == "__main__":
    main()
