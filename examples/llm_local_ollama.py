"""
struct_agent Local Offline LLM Example (via Ollama) — Interactive Chat
=======================================================================

Runs `struct_agent` with completely local, open-weights models (Llama 3.1,
Nemotron, Qwen, Mistral, etc.) via Ollama.

Features:
- Zero API costs, zero data leaving your machine, zero cloud dependencies.
- After the initial structural design prompt, drops into an interactive
  continuous conversation loop — ask follow-up questions and the agent
  remembers the current project, stage, and full message history.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Start Ollama (it may already be running as a background service):
       ollama serve
    3. Pull a function-calling capable model if needed:
       ollama pull llama3.1

Usage:
    python struct_agent/examples/llm_local_ollama.py
"""

import os
import sys
import urllib.request
import json
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
    """Stream and print all agent events for a single user turn."""
    for event in runner.run_stream(user_prompt):
        if isinstance(event, ThoughtEvent):
            try:
                print(f"\n[LOCAL LLM REASONING] {event.thought}")
            except UnicodeEncodeError:
                print(f"\n[LOCAL LLM REASONING] {event.thought.encode('ascii', errors='replace').decode('ascii')}")
        elif isinstance(event, ToolCallEvent):
            print(f"  --> [TOOL CALL] {event.tool_name}({event.arguments})")
        elif isinstance(event, ToolResultEvent):
            badge = "OK" if event.ok else "FAILED"
            print(f"  <-- [TOOL RESULT] [{badge}] {event.tool_name}")
            if event.error_message:
                print(f"        ! Error: {event.error_message}")
        elif isinstance(event, StateUpdateEvent):
            print(f"  [STAGE SYNC] -> Workflow advanced to: {event.stage}")
        elif isinstance(event, ErrorEvent):
            print(f"  [AGENT ERROR] {event.message}")
        elif isinstance(event, CompleteEvent):
            print("\n" + "=" * 70)
            print("TURN COMPLETE")
            print(f"Turns used: {event.total_turns}")
            try:
                print(f"Summary: {event.final_message}")
            except UnicodeEncodeError:
                print(f"Summary: {event.final_message.encode('ascii', errors='replace').decode('ascii')}")
            print("=" * 70)


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if the local Ollama server is active and accessible."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "struct_agent"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_available_models(base_url: str = "http://localhost:11434") -> list:
    """Retrieve the list of locally pulled models from Ollama."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "struct_agent"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name") for m in data.get("models", [])]
    except Exception:
        return []


def main():
    print("=" * 70)
    print(" struct_agent: Local Offline LLM Engine (Ollama / Llama 3.1)")
    print("=" * 70)

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_api_base = f"{ollama_host}/v1"

    print(f"\n[1/3] Checking Ollama status at: {ollama_host}...")
    is_running = check_ollama_running(ollama_host)

    if not is_running:
        print("\n[!] Local Ollama server is not running.")
        print("To run structural models locally and completely offline:")
        print("  1. Download Ollama from https://ollama.com")
        print("  2. In your terminal, run:")
        print("       ollama serve")
        print("  3. In another terminal, pull a model:")
        print("       ollama pull llama3.1")
        print("  4. Re-run this script!")
        print("\nExiting gracefully. (For simulated offline test, run llm_quickstart.py)")
        return

    models = get_available_models(ollama_host)
    print(f"[OK] Ollama is active! Available local models: {models}")

    # Select preferred model
    preferred_models = ["nemotron-3-ultra:cloud", "qwen2.5-coder:7b", "mistral:latest", "mistral"]
    chosen_model = None
    for m in preferred_models:
        if m in models:
            chosen_model = m
            break

    if not chosen_model:
        chosen_model = models[0] if models else "llama3.1"
    print(f"[2/3] Selected local model: '{chosen_model}'")

    # Initialize runner with NativeHTTPProvider targeting Ollama's OpenAI-compatible endpoint
    runner = AgentRunner(
        provider="native",
        model=chosen_model,
        base_url=ollama_api_base,
        api_key="ollama",  # Ollama ignores api_key, but native provider accepts any placeholder string
        max_turns=10,
    )

    workspace_root = Path(__file__).resolve().parent.parent.parent
    sample_s2k = str(workspace_root / "struct_link" / "tests" / "fixtures" / "sample_frame.s2k")
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dxf_output = str(output_dir / "ollama_detailing.dxf")

    initial_prompt = (
        f"Import the structural model from '{sample_s2k}', validate the geometry, "
        f"and run linear static analysis."
    )

    print(f"\n[3/3] Initial prompt:\n\"{initial_prompt}\"\n")
    print("-" * 70)
    stream_events(runner, initial_prompt)

    if Path(dxf_output).exists():
        print(f"\nGenerated DXF drawing verified at: {dxf_output}")

    # Interactive continuous conversation loop
    print("\n" + "=" * 70)
    print(" Interactive Chat Mode (local model: " + chosen_model + ")")
    print(" The agent remembers the current structural project and workflow stage.")
    print(" Ask follow-up questions or issue new commands. Type 'exit' to quit.")
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
