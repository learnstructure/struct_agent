"""
struct_agent LLM Wrapper Quickstart (Google Gemini 3.8 Flash)
=============================================================

Demonstrates using Google Gemini (gemini-3.8-flash) with the autonomous
`AgentRunner` to orchestrate structural engineering workflows through
function-calling and deterministic tools.

Features:
- Directly connects to Google Gemini via Google's official OpenAI-compatible endpoint.
- Uses `gemini-3.8-flash` by default (can be overridden with GEMINI_MODEL).
- Zero external package dependencies (uses Python's standard `urllib`).
- Multi-turn interactive conversation loop: continue chatting with the agent
  after the initial design without losing structural project state in memory.

Prerequisites:
    Set your Gemini API key in your terminal:
    PowerShell:
        $env:GEMINI_API_KEY = "AIzaSy..."
    Bash/Linux:
        export GEMINI_API_KEY="AIzaSy..."

Usage:
    python struct_agent/examples/llm_quickstart.py
"""

import os
import sys
from pathlib import Path

# Ensure Windows terminal stdout handles emojis and unicode characters cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from struct_agent import (
    AgentRunner,
    CompleteEvent,
    ErrorEvent,
    StateUpdateEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def stream_events(runner: AgentRunner, user_prompt: str) -> None:
    """Stream and print events from the autonomous agent runner."""
    for event in runner.run_stream(user_prompt):
        if isinstance(event, ThoughtEvent):
            print(f"\n[GEMINI REASONING] {event.thought}")
        elif isinstance(event, ToolCallEvent):
            print(f"  --> [TOOL CALL] {event.tool_name}({event.arguments})")
        elif isinstance(event, ToolResultEvent):
            status_symbol = "OK" if event.ok else "FAILED"
            print(f"  <-- [TOOL RESULT] [{status_symbol}] {event.tool_name}")
            if event.error_message:
                print(f"        ! Error: {event.error_message}")
        elif isinstance(event, StateUpdateEvent):
            print(f"  [STAGE SYNC] -> Workflow advanced to: {event.stage}")
        elif isinstance(event, ErrorEvent):
            print(f"  [AGENT ERROR] {event.message}")
        elif isinstance(event, CompleteEvent):
            print("\n" + "=" * 70)
            print("AGENT EXECUTION TURN COMPLETE")
            print(f"Turns: {event.total_turns} | Passed: {event.passed}")
            if event.governing_dcr is not None:
                print(f"Governing Demand-Capacity Ratio (DCR): {event.governing_dcr:.3f}")
            try:
                print(f"Summary: {event.final_message}")
            except UnicodeEncodeError:
                print(f"Summary: {event.final_message.encode('ascii', errors='replace').decode('ascii')}")
            print("=" * 70)


def main():
    print("=" * 70)
    print(" struct_agent: Autonomous Cognitive LLM Wrapper (Gemini 3.8 Flash)")
    print("=" * 70)

    # 1. Check for Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("\n[ERROR] GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in environment!")
        print("\nTo set your Gemini API key in Windows PowerShell, run:")
        print('    $env:GEMINI_API_KEY = "your-gemini-api-key-here"')
        print("\nThen re-run this script:")
        print("    python struct_agent/examples/llm_quickstart.py")
        sys.exit(1)

    # 2. Configure model and Google OpenAI-compatible endpoint
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    print(f"[INFO] Connecting to Google Gemini API...")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Endpoint: {gemini_base_url}\n")

    # 3. Initialize AgentRunner with NativeHTTPProvider
    runner = AgentRunner(
        provider="native",
        model=model_name,
        base_url=gemini_base_url,
        api_key=gemini_key,
        max_turns=12,
    )

    # Resolve paths to sample files
    workspace_root = Path(__file__).resolve().parent.parent.parent
    sample_s2k = str(workspace_root / "struct_link" / "tests" / "fixtures" / "sample_frame.s2k")
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dxf = str(output_dir / "quickstart_detailing.dxf")

    initial_prompt = (
        f"Import the structural model from '{sample_s2k}', validate the geometry, "
        f"run static analysis, design all concrete elements under ACI 318-19, "
        f"and export detailing drawings to '{output_dxf}'."
    )

    print(f"Initial Prompt: \"{initial_prompt}\"\n")
    print("-" * 70)
    print("Streaming Event Loop:")
    print("-" * 70)

    # Run initial pipeline
    stream_events(runner, initial_prompt)

    if Path(output_dxf).exists():
        print(f"\nGenerated DXF drawing verified at: {output_dxf}")

    # 4. Interactive Continuous Conversation Loop
    print("\n" + "=" * 70)
    print(" Interactive Continuous Conversation Mode")
    print(" (Ask follow-up questions, request model changes, or type 'exit' to quit)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting chat session.")
            break

        stream_events(runner, user_input)


if __name__ == "__main__":
    main()
