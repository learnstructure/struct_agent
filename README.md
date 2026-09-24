# struct_agent

`struct_agent` is the **AI orchestration layer** for the
[Structural Engineering Ecosystem](../README.md).

Its job is to make the engineering libraries
(`struct_core`, `struct_load`, `struct_link`, `fem2d`, `fem3d`, `aci318`, `struct_draw`) usable as a
small, validated, deterministic toolset that any LLM framework
can call.

> **The AI decides *what* to do; the engineering libraries decide *how* the engineering is done.**

This package contains no engineering calculations of its own.

---

## The Bigger Picture: Full Ecosystem

`struct_agent` sits at the top of the structural engineering ecosystem. All packages
are in the parent `structural_engineering/` workspace and installed in
the `structeng` conda environment.

```
                    User / LLM / Script
                           |
                           v
                      struct_agent          <- YOU ARE HERE (v0.1.0)
              (Orchestration + Tool Layer)
                           |
     +---------------------+------+--------------------+
     |                     |                           |
     v                     v                           v
struct_load           struct_link                 struct_core
(Building code      (SAP2000 & ETABS            (Common Engineering
 load engine)        .s2k / .e2k link)                Schemas)
     |                     |                           |
     +---------------------+---------------------------+
                           |
           +---------------+---------------+
           |               |               |
           v               v               v
         fem2d           fem3d         structdyn
    (2D FEM solver) (3D FEM solver)  (Dynamics)
           |               |
           +-------+-------+
                   |
                 aci318
            (RC design & detailing)
                   |
              struct_draw
         (DXF drawing generation)
```

### Package Inventory

| Package | Location | Version | Responsibility |
|---|---|---|---|
| `struct_core` | `../struct_core/` | 0.1.0 | Common schemas: `Project`, `StructuralModel`, `AnalysisResult`, `DesignResult`, `DetailingResult`, serialization |
| `struct_load` | `../struct_load/` | 0.1.0 | ASCE 7 / IBC code load calculations (gravity, wind, seismic ELF, combinations) |
| `struct_link` | `../struct_link/` | 0.1.0 | Commercial CAE link: SAP2000 & ETABS .s2k / .e2k file import/export and live COM |
| `fem2d` | `../fem2d/` | 0.4.0 | 2D linear static FEM & combination envelopes — matrix assembly, direct stiffness solver |
| `fem3d` | `../fem3d/` | 0.1.0 | 3D general frame/truss FEM — 6 DOF per node, 3D static and eigenvalue modal analysis |
| `structdyn` | `../structdyn/` | — | Structural dynamics — modal, time-history, SDF/MDF analysis, ground motions |
| `aci318` | `../aci318/` | 0.1.0 | ACI 318-19 RC member design + reinforcement detailing |
| `struct_draw` | `../struct_draw/` | 0.1.0 | DXF drawing generation — beams, columns, footings, staircases |
| `struct_agent` | `./` | 0.1.0 | Tool layer + workflow state machine for LLM integration |

---

## Package Layout

```
struct_agent/
+-- struct_agent/
|   +-- __init__.py           # Public API
|   +-- agent.py              # StructAgent orchestrator (deterministic, no LLM)
|   +-- tools/
|   |   +-- base.py           # ToolResult, AgentError
|   |   +-- model.py          # create_project, inspect_project, modify_project, save_project, load_project
|   |   +-- loads.py          # apply_gravity_loads, apply_lateral_loads, generate_load_combinations
|   |   +-- link.py           # import_from_sap2000, export_to_sap2000, import_from_etabs, export_to_etabs
|   |   +-- sizing.py         # preliminary_sizing
|   |   +-- analysis.py       # run_analysis, inspect_analysis_results
|   |   +-- design.py         # run_design, inspect_design_results
|   |   +-- drawing.py        # generate_drawings
|   +-- validation/
|   |   +-- model.py          # validate_model
|   |   +-- analysis.py       # validate_analysis
|   |   +-- design.py         # validate_design
|   +-- state/
|       +-- project_state.py  # WorkflowStage enum + ProjectState dataclass
|   +-- llm/                  # Autonomous LLM Orchestration & Streaming Engine
|   |   +-- events.py         # Streaming events (Thought, ToolCall, ToolResult, StateUpdate, etc.)
|   |   +-- schemas.py        # OpenAI/Ollama function calling schema definitions for 15+ tools
|   |   +-- prompts.py        # Structural engineering system prompts and safety grounding rules
|   |   +-- runner.py         # Multi-turn autonomous agent runner (run, run_stream)
|   |   +-- providers/        # LLM Provider adapters
|   |       +-- base.py       # BaseLLMProvider interface
|   |       +-- http_client.py# Zero-dependency NativeHTTPProvider (urllib, OpenAI/Ollama)
|   |       +-- litellm_client.py # LiteLLM multi-provider adapter (Claude, Gemini, etc.)
+-- tests/
|   +-- test_pipeline_smoke.py    # End-to-end pipeline smoke test
|   +-- test_tool_contracts.py    # Tool input validation tests
|   +-- test_workflow_state.py    # State machine transition tests
|   +-- test_fem3d_wiring.py      # 3D solver auto-detection, static & modal analysis tests
|   +-- test_link_tools.py        # SAP2000 & ETABS import/export tool tests
|   +-- test_loads_integration.py # Preliminary sizing & load generation tests
|   +-- test_llm_runner.py        # Autonomous agent loop, streaming events & mock tests
+-- examples/
    +-- demo_pipeline.py                   # Full pipeline via StructAgent (start here)
    +-- end_to_end_ecosystem_pipeline.py   # Low-level adapter integration test
    +-- general_frame.py                   # struct_draw standalone: framing plan + truss DXFs
    +-- llm_quickstart.py                  # Autonomous LLM runner quickstart (OpenAI/simulated)
    +-- llm_streaming_gui_simulation.py    # GUI/dashboard event streaming consumer simulation
    +-- llm_local_ollama.py                # 100% offline local model execution (Ollama / Llama 3.1)
```

---

## Workflow State Machine

`ProjectState` / `WorkflowStage` enforces stage ordering. Invalid
transitions raise `StageTransitionError` before any engineering library
is called.

```
UNKNOWN
   |
   +--> create_project() / preliminary_sizing()  [scratch model]
   |    OR
   +--> import_from_sap2000()                    [SAP2000 .s2k file]
   |    OR
   +--> import_from_etabs()                      [ETABS .e2k file]
   |
   v
MODEL_CREATED
   |
   v
LOADS_APPLIED       <- apply_gravity_loads(), apply_lateral_loads(), generate_load_combinations()
   |
   v
MODEL_VALIDATED     <- validate_model()
   |
   v
ANALYSIS_COMPLETE   <- run_analysis()          [fem2d or fem3d static / envelope / modal]
   |
   v
DESIGN_COMPLETE     <- run_design()            [aci318]
   |
   v
DETAILING_COMPLETE  <- (auto-advanced inside run_design)
   |
   v
DRAWINGS_GENERATED  <- generate_drawings()     [struct_draw]
```

`ProjectState` also carries:

- `tool_log` — ordered record of every tool call and its status
- `pending_questions` — clarifications the agent should ask the user
- `last_errors` — last structured error per stage (for LLM retry logic)
- `history` — ordered list of visited stages

---

## Public API

```python
from struct_agent import (
    StructAgent,              # High-level orchestrator
    WorkflowStage,            # Enum of canonical workflow stages
    ProjectState,             # Workflow state dataclass
    StageTransitionError,     # Raised on invalid stage jump
    ToolResult,               # Return type for every tool
    AgentError,               # Structured error inside ToolResult
    create_project,           # Wrap a StructuralModel in a Project
    inspect_project,          # Summary of project contents
    modify_project,           # Apply a mutator to the model
    preliminary_sizing,       # Heuristic RC member sizing from spans
    apply_gravity_loads,      # ASCE 7 occupancy live load & self-weight
    apply_lateral_loads,      # ASCE 7 seismic ELF & wind lateral loads
    generate_load_combinations,# ACI 318 / ASCE 7 factored load combinations
    import_from_sap2000,      # Ingest SAP2000 .s2k file into Project
    export_to_sap2000,        # Export Project model to SAP2000 .s2k
    import_from_etabs,         # Ingest ETABS .e2k file into Project
    export_to_etabs,          # Export Project model to ETABS .e2k
    validate_model,           # struct_core schema validation
    run_analysis,             # 2D/3D FEM analysis via fem2d or fem3d (static / modal)
    inspect_analysis_results, # Structured force/displacement/modal summary
    run_design,               # ACI 318-19 RC design via aci318
    inspect_design_results,   # Structured DCR / pass-fail summary
    generate_drawings,        # DXF export via struct_draw
    AgentRunner,              # Autonomous multi-turn LLM reasoning loop
    get_provider,             # Provider factory ('native', 'litellm', 'auto')
    LLMEvent,                 # Streaming event base class
    ThoughtEvent, ToolCallEvent, ToolResultEvent, StateUpdateEvent, # Granular events
)
```

### Tool Signatures

```python
# Project & Model tools
create_project(structural_model, *, title=None, author=None) -> ToolResult
inspect_project(project) -> ToolResult
modify_project(project, mutator) -> ToolResult
save_project(project, path) -> ToolResult
load_project(path) -> ToolResult

# Commercial CAE link tools (struct_link)
import_from_sap2000(s2k_path, *, project_title=None) -> ToolResult
export_to_sap2000(project, *, output_path) -> ToolResult
import_from_etabs(e2k_path, *, project_title=None) -> ToolResult
export_to_etabs(project, *, output_path) -> ToolResult

# Building load tools (struct_load)
preliminary_sizing(project, *, material_id="Concrete_4ksi", fc_kpa=27580.0) -> ToolResult
apply_gravity_loads(project, *, occupancy="office", tributary_width=5.0, superimposed_dead=1.5, include_self_weight=True) -> ToolResult
apply_lateral_loads(project, *, code="ASCE7-22", sds=0.75, sd1=0.40, r_factor=8.0, wind_speed=None) -> ToolResult
generate_load_combinations(project, *, standard="ACI318_LRFD") -> ToolResult

# Validation tool
validate_model(model_or_project) -> ToolResult

# Analysis tools (fem2d & fem3d)
run_analysis(project, *, analysis_case_id, analysis_type="static", solver="auto", load_case_id=None, num_modes=6) -> ToolResult
inspect_analysis_results(project, analysis_case_id) -> ToolResult

# Design tools (aci318)
run_design(project, *, design_case_id, design_code="ACI 318-19", analysis_case_id=None) -> ToolResult
inspect_design_results(project, design_case_id) -> ToolResult

# Drawing tools (struct_draw)
generate_drawings(project, *, design_case_id, output_path, detailing_case_id=None) -> ToolResult
```

Every tool returns `ToolResult`. `result.ok` is `True` only when
`result.status == "success"`. Failures carry a structured `AgentError`
with `error_type`, `message`, and `possible_causes`.

---

## Quick Start

### High-level (via StructAgent)

```python
from struct_agent import StructAgent

agent = StructAgent()
state = agent.run_demo_pipeline(
    output_dxf="out.dxf",
    save_json_path="project.json",
)
print(state.stage)        # "DRAWINGS_GENERATED"
print(state.tool_log)     # list of all tool calls with status
```

### Autonomous LLM Agent (via AgentRunner)

```python
import os
from struct_agent import AgentRunner

# Initialize autonomous runner (works with OpenAI, Ollama, DeepSeek, or any OpenAI-compatible endpoint)
runner = AgentRunner(
    provider="native",           # Zero external dependencies (uses standard urllib)
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 1. Synchronous execution
response = runner.run("Create a 2-story RC frame in Reno, NV with office live loads, run analysis and design.")
print(response)

# 2. Event-driven streaming (ideal for GUI / web frontend integration)
for event in runner.run_stream("Apply wind lateral loads and export DXF drawings"):
    if event.type == "thought":
        print(f"[THOUGHT] {event.thought}")
    elif event.type == "tool_call":
        print(f"[TOOL] {event.tool_name}({event.arguments})")
    elif event.type == "tool_result":
        print(f"[RESULT] {event.tool_name} -> ok={event.ok}")
    elif event.type == "state_update":
        print(f"[STAGE] Advanced to: {event.stage}")
```

---

## Tests

```powershell
# From workspace root with structeng conda environment active
pytest struct_agent/tests/ -v
```

| Test file | Coverage |
|---|---|
| `test_pipeline_smoke.py` | Full end-to-end pipeline; checks all 8 tools are called and DXF/JSON are produced |
| `test_fem3d_wiring.py` | Auto solver detection, 3D static and modal analysis, combination envelopes |
| `test_link_tools.py` | SAP2000 & ETABS file import, export, and error handling |
| `test_loads_integration.py` | Preliminary member sizing and ASCE 7 load generator tool integration |
| `test_tool_contracts.py` | Individual tool input validation (missing project, wrong types, etc.) |
| `test_workflow_state.py` | State machine: valid transitions, invalid transitions, idempotency |
| `test_llm_runner.py` | Autonomous multi-turn agent loop, event streaming, provider factory, and mock tool reasoning |

---

## Installation

```powershell
# From struct_agent/ with structeng conda environment active
pip install -e .
```

All library dependencies (`struct_core`, `struct_load`, `struct_link`, `fem2d`, `fem3d`, `aci318`, `struct_draw`)
must be installed first. Use the workspace-level `environment.yml` to
create the full `structeng` environment.

---

## What Is Already Built

| Feature | Owner |
|---|---|
| `StructuralModel` schema (nodes, elements, sections, materials, supports, loads) | struct_core |
| `Project` container with `AnalysisResult`, `DesignResult`, `DetailingResult` | struct_core |
| JSON serialization / deserialization of full `Project` | struct_core |
| Building code load generation (gravity, lateral ELF/wind, combinations) | struct_load |
| Commercial software link (SAP2000 & ETABS .s2k / .e2k import & export) | struct_link |
| 2D linear static FEM solver & envelope combinations | fem2d |
| `fem2d` <-> `struct_core` adapter (`model_from_core`, `result_to_core`) | fem2d |
| 3D linear static & eigenvalue modal analysis | fem3d |
| `fem3d` <-> `struct_core` adapter (`model_from_core`, `result_to_core`) | fem3d |
| ACI 318-19 RC member design (beams + columns) | aci318 |
| Reinforcement detailing (`DetailingResult` with bar schedules) | aci318 |
| `aci318` <-> `struct_core` adapter (`design_from_core`) | aci318 |
| DXF drawing generation (beams, columns, footings, staircases) | struct_draw |
| `struct_draw` <-> `struct_core` adapter (`draw_from_core`) | struct_draw |
| General framing plan + truss outline DXF generation (examples) | struct_draw |
| Structural dynamics — modal, time-history, SDF, MDF, ground motions | structdyn (standalone) |
| Workflow state machine (7 stages, transition enforcement) | struct_agent |
| Agent tool layer (15 tools, structured `ToolResult`) | struct_agent |
| Automated solver selection (`auto`, `fem2d`, `fem3d`) | struct_agent |
| Deterministic demo orchestrator (`StructAgent`) | struct_agent |
| Autonomous cognitive LLM orchestration wrapper (`AgentRunner`) | struct_agent |
| Zero-dependency native HTTP client for OpenAI/Ollama API | struct_agent |
| Multi-provider LiteLLM adapter (Claude, Gemini, OpenAI, etc.) | struct_agent |
| Granular event streaming interface (`ThoughtEvent`, `ToolCallEvent`, `StateUpdateEvent`, etc.) | struct_agent |
| Grounded system prompt & 15+ tool schema function signatures | struct_agent |
| End-to-end pipeline & LLM mock tests | struct_agent |

---

## Future Evolution & Gaps

The cognitive LLM wrapper is fully functional via `AgentRunner`! The remaining ecosystem roadmap items are:

| Feature / Integration | Description & Impact |
|---|---|
| **Future GUI Application** | Building a desktop (e.g. PyQt/Electron) or web (FastAPI/React) interface consuming `AgentRunner.run_stream()` events, with interactive 3D rendering and live DXF preview |
| **`structdyn` direct time-history integration** | Non-linear time-history requires calling `structdyn` directly, bypassing the tool layer |

---

## License

MIT License.
