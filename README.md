# struct_agent

`struct_agent` is the **AI orchestration layer** for the
[Structural Engineering Ecosystem](../README.md).

Its job is to make the engineering libraries
(`struct_core`, `struct_load`, `struct_link`, `fem2d`, `fem3d`, `structdyn`, `aci318`, `struct_draw`) usable as a
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
           |               |               |
           +-------+-------+---------------+
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
| `structdyn` | `../structdyn/` | 0.1.0 | Structural dynamics — modal, time-history, SDF/MDF analysis, ground motions, FEM bridges (`fem2d`/`fem3d`) |
| `aci318` | `../aci318/` | 0.1.0 | ACI 318-19 RC member design + reinforcement detailing |
| `struct_draw` | `../struct_draw/` | 0.1.0 | DXF drawing generation — beams, columns, footings, staircases |
| `struct_agent` | `./` | 0.1.0 | Tool layer + workflow state machine for LLM integration (static, modal, and dynamic time-history) |

---

## Package Layout

```
struct_agent/
+-- struct_agent/
|   +-- __init__.py           # Public API
|   +-- agent.py              # StructAgent orchestrator (deterministic, no LLM)
|   +-- tools/
|   |   +-- base.py           # ToolResult, AgentError
|   |   +-- model.py          # create_project, create_frame_model, build_structural_model, create_model_from_plan_draft, inspect_project, modify_project, set_project_units, save_project, load_project
|   |   +-- loads.py          # apply_gravity_loads, apply_self_weight, apply_lateral_loads, generate_load_combinations
|   |   +-- link.py           # import_from_sap2000, export_to_sap2000, import_from_etabs, export_to_etabs
|   |   +-- sizing.py         # preliminary_sizing
|   |   +-- analysis.py       # run_analysis, inspect_analysis_results
|   |   +-- dynamics.py       # run_time_history, inspect_dynamic_results
|   |   +-- design.py         # run_design, inspect_design_results, inspect_detailing_results
|   |   +-- drawing.py        # generate_drawings (unified: GA framing & RC detailing)
|   +-- vision/               # Multimodal Drawing Ingestion & Deterministic Grid Compiler
|   |   +-- schemas.py        # Pydantic schemas (FloorPlanDraft, GridLineDraft, StoryDraft, ColumnDraft, BeamDraft)
|   |   +-- compiler.py       # Pure-Python deterministic grid compiler (draft -> struct_core.Project)
|   |   +-- encoding.py       # Drawing rasterization & multimodal message builder (pypdfium2 PDF, PNG, JPG)
|   +-- validation/
|   |   +-- model.py          # validate_model
|   |   +-- analysis.py       # validate_analysis
|   |   +-- design.py         # validate_design
|   +-- state/
|   |   +-- project_state.py  # WorkflowStage enum + ProjectState dataclass
|   +-- llm/                  # Autonomous LLM Orchestration & Streaming Engine
|   |   +-- events.py         # Streaming events (Thought, ToolCall, ToolResult, StateUpdate, etc.)
|   |   +-- schemas.py        # OpenAI/Ollama function calling schema definitions for 18+ tools
|   |   +-- prompts.py        # Structural engineering system prompts and safety grounding rules
|   |   +-- runner.py         # Multi-turn autonomous agent runner with multimodal attachments (run, run_stream)
|   |   +-- providers/        # LLM Provider adapters
|   |       +-- base.py       # BaseLLMProvider interface
|   |       +-- http_client.py# Zero-dependency NativeHTTPProvider (urllib, OpenAI/Ollama/Gemini)
|   |       +-- litellm_client.py # LiteLLM multi-provider adapter (Claude, Gemini, etc.)
+-- tests/
|   +-- test_pipeline_smoke.py    # End-to-end pipeline smoke test
|   +-- test_tool_contracts.py    # Tool input validation tests
|   +-- test_workflow_state.py    # State machine transition tests
|   +-- test_fem3d_wiring.py      # 3D solver auto-detection, static & modal analysis tests
|   +-- test_dynamics.py          # Dynamic time-history, modal response & peak force envelope tests
|   +-- test_link_tools.py        # SAP2000 & ETABS import/export tool tests
|   +-- test_loads_integration.py # Preliminary sizing & load generation tests
|   +-- test_vision_compiler.py   # Deterministic grid snapping, column generation & load combination tests
|   +-- test_vision_agent.py      # Multimodal attachment encoding, pypdfium2 PDF rendering & VLM loop tests
|   +-- test_llm_runner.py        # Autonomous agent loop, streaming events & mock tests
+-- examples/
|   +-- demo_pipeline.py                   # Full pipeline via StructAgent (start here)
|   +-- dynamic_design_pipeline.py         # End-to-end seismic time-history & ACI 318 design pipeline
|   +-- end_to_end_ecosystem_pipeline.py   # Low-level adapter integration test
|   +-- general_frame.py                   # struct_draw standalone: framing plan + truss DXFs
|   +-- llm_quickstart.py                  # Autonomous LLM runner quickstart (OpenAI/simulated)
|   +-- llm_streaming_gui_simulation.py    # GUI/dashboard event streaming consumer simulation
|   +-- llm_local_ollama.py                # 100% offline local model execution (Ollama / Llama 3.1)
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
                    OR
                    <- run_time_history()      [structdyn dynamic ground motion time-history + design envelope]
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
    create_frame_model,       # Quick parametric frame generator
    build_structural_model,   # Dictionary/schema-based model builder
    create_model_from_plan_draft, # Build 3D Project from VLM FloorPlanDraft
    inspect_project,          # Summary of project contents
    modify_project,           # Apply a mutator to the model
    preliminary_sizing,       # Heuristic RC member sizing from spans
    apply_gravity_loads,      # ASCE 7 occupancy live load & self-weight
    apply_lateral_loads,      # ASCE 7 seismic ELF & wind lateral loads
    apply_self_weight,        # Compute and materialize element self-weight into member line loads
    generate_load_combinations,# ACI 318 / ASCE 7 factored load combinations
    import_from_sap2000,      # Ingest SAP2000 .s2k file into Project
    export_to_sap2000,        # Export Project model to SAP2000 .s2k
    import_from_etabs,         # Ingest ETABS .e2k file into Project
    export_to_etabs,          # Export Project model to ETABS .e2k
    validate_model,           # struct_core schema validation
    run_analysis,             # 2D/3D FEM analysis via fem2d or fem3d (static / modal)
    inspect_analysis_results, # Structured force/displacement/modal summary
    run_time_history,         # Dynamic ground motion time-history via structdyn + force envelope
    inspect_dynamic_results,  # Peak dynamic response summary (displacements, base shear, envelope)
    run_design,               # ACI 318-19 RC design via aci318
    inspect_design_results,   # Structured DCR / pass-fail summary
    inspect_detailing_results,# Structured rebar schedule & BBS summary
    generate_drawings,        # DXF export via struct_draw
    AgentRunner,              # Autonomous multi-turn LLM reasoning loop
    get_provider,             # Provider factory ('native', 'litellm', 'auto')
    LLMEvent,                 # Streaming event base class
    ThoughtEvent, ToolCallEvent, ToolResultEvent, StateUpdateEvent, # Granular events
)

# Multimodal Drawing Ingestion & Deterministic Compiler Subsystem
from struct_agent.vision import (
    FloorPlanDraft,           # Pydantic schema for floor plans extracted by VLMs
    GridLineDraft,            # Grid axes with coordinate/dimension spacing
    StoryDraft,               # Story definition (height, level, floor loads)
    ColumnDraft, BeamDraft,   # Individual column/beam overrides & section assignments
    build_project_from_draft, # Pure-Python deterministic 3D grid compiler
    encode_attachments,       # Multi-format drawing encoder (pypdfium2 PDF, PNG, JPG)
)
```

### Tool Architecture & Return Convention

Every tool in `struct_agent` returns a deterministic [`ToolResult`](file:///c:/Users/lenovo/OneDrive%20-%20University%20of%20Nevada,%20Reno/extra%20research/structural_engineering/struct_agent/struct_agent/tools/base.py) object:

- `result.ok` (`bool`): `True` only when `result.status == "success"`.
- `result.data` (`dict`): Structured outputs (e.g. modified `project`, summary statistics, file paths).
- `result.summary` (`str`): Concise human- and LLM-readable summary string of what was performed.
- `result.next_suggested_stage` (`str`): Recommended workflow stage advancement.
- `result.error` (`AgentError` or `None`): Populated on failure with structured `error_type`, `message`, and recovery `possible_causes`.

---

## Tool Reference & Documentation

### 1. Model & Project Lifecycle Tools

#### `create_project`
Initializes a new `struct_core.Project` container wrapping an existing `StructuralModel`, dictionary, or creates an empty container.
- **Signature:**
  ```python
  create_project(structural_model=None, *, title=None, author=None, model=None, nodes=None, elements=None, materials=None, sections=None, supports=None) -> ToolResult
  ```
- **Parameters:**
  - `structural_model`: Optional existing `StructuralModel` instance or dictionary.
  - `title`: Project title metadata string.
  - `author`: Engineer / author name metadata string.
  - `nodes`, `elements`, `materials`, `sections`, `supports`: Optional lists to build model geometry inline.
- **Returns (`data`):** `{"project": Project, "title": str, "author": str, "n_nodes": int, "n_elements": int, "n_supports": int}`
- **Example:**
  ```python
  res = create_project(title="Office Tower Study", author="Structural Team")
  project = res.data["project"]
  ```

#### `create_frame_model`
High-level parametric generator for regular orthogonal 2D plane frames or 3D space frames. Automatically provisions grid nodes, column/beam connectivity, standard concrete materials, rectangular sections, and base supports.
- **Signature:**
  ```python
  create_frame_model(num_bays=2, bay_width=5.0, num_stories=2, story_height=3.0, *, title=None, author=None, dimension="2d", bay_widths=None, story_heights=None, support_type="fixed", col_b=0.4, col_h=0.4, beam_b=0.3, beam_h=0.5, fc_kpa=27580.0, rho_kg_m3=2400.0) -> ToolResult
  ```
- **Parameters:**
  - `num_bays`: Number of horizontal bays (or list `bay_widths` for non-uniform spacing).
  - `bay_width`: Uniform bay width in meters.
  - `num_stories`: Number of vertical stories (or list `story_heights` for non-uniform floor heights).
  - `story_height`: Uniform story height in meters.
  - `dimension`: `"2d"` for plane frame, `"3d"` for space frame.
  - `support_type`: Base boundary conditions (`"fixed"` or `"pinned"`).
  - `col_b`, `col_h`, `beam_b`, `beam_h`: Column and beam cross-section dimensions (m).
- **Returns (`data`):** `{"project": Project, "n_nodes": int, "n_elements": int, "n_columns": int, "n_beams": int, "stories": int, "bays": int, ...}`
- **Example:**
  ```python
  res = create_frame_model(num_bays=3, bay_width=6.0, num_stories=4, story_height=3.5, dimension="3d")
  project = res.data["project"]
  ```

#### `build_structural_model`
Constructs or augments arbitrary/irregular structural geometry (trusses, braced frames, angled members, bridges) directly from structured JSON lists.
- **Signature:**
  ```python
  build_structural_model(project=None, *, nodes=None, elements=None, materials=None, sections=None, supports=None, title=None) -> ToolResult
  ```
- **Parameters:**
  - `project`: Optional existing `Project` to populate or update; if `None`, creates a new one.
  - `nodes`: List of dicts `[{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}, ...]`.
  - `elements`: List of dicts `[{"id": "E1", "type": "beam"|"truss", "start_node": 1, "end_node": 2}, ...]`.
  - `supports`: List of dicts `[{"node_id": 1, "fixity": "fixed"|"pinned"}, ...]`.
- **Returns (`data`):** `{"project": Project, "n_nodes": int, "n_elements": int, "n_supports": int}`
- **Example:**
  ```python
  res = build_structural_model(
      nodes=[{"id": 1, "x": 0, "y": 0}, {"id": 2, "x": 6, "y": 0}, {"id": 3, "x": 3, "y": 2.5}],
      elements=[{"id": "T1", "type": "truss", "start_node": 1, "end_node": 2}],
      supports=[{"node_id": 1, "fixity": "pinned"}, {"node_id": 2, "fixity": "pinned"}]
  )
  ```

#### `create_model_from_plan_draft`
Compiles a complete 3D structural model from an architectural or structural floor plan draft (`FloorPlanDraft`) extracted from drawings (PDF, PNG, JPG) by Vision-Language Models.
- **Signature:**
  ```python
  create_model_from_plan_draft(draft=None, project=None, **kwargs) -> ToolResult
  ```
- **Parameters:**
  - `draft`: `FloorPlanDraft` instance or dictionary with `x_grids`, `y_grids`, `stories`, `columns`, `beams`.
  - `project`: Optional existing `Project` container.
- **Returns (`data`):** `{"project": Project, "title": str, "n_nodes": int, "n_elements": int, "n_columns": int, "n_beams": int, ...}`

#### `inspect_project`
Provides a human- and LLM-readable summary of project status, geometry counts, applied load cases, analysis cases, design checks, and detailing status.
- **Signature:** `inspect_project(project) -> ToolResult`
- **Returns (`data`):** `{"title": str, "units": dict, "model": dict, "load_cases": list, "analysis_cases": list, "design_cases": list, ...}`

#### `modify_project`
Applies declarative model modifications (updating rectangular, circular, or general section dimensions, member reassignments, material properties) or an optional programmatic callback mutator function safely to a `Project`.
- **Signature:** `modify_project(project, mutator=None, *, sections=None, member_assignments=None, materials=None) -> ToolResult`
- **Examples:**
  - Rectangular section: `modify_project(project, sections=[{"id": "Col_350x350", "b": 0.45, "h": 0.45}])`
  - Circular section: `modify_project(project, sections=[{"id": "Pier_D500", "d": 0.60}])`
  - General section: `modify_project(project, sections=[{"id": "W14x90", "type": "general", "A": 0.0171, "I3": 0.000416}])`
  - Member reassignment: `modify_project(project, member_assignments=[{"element_ids": ["C1", "C2"], "section_id": "Col_450x450"}])`

#### `set_project_units`
Explicitly defines or converts project units across all entities via `struct_core.units`.
- **Signature:**
  ```python
  set_project_units(project, units: str, convert_existing: bool = True) -> ToolResult
  ```
- **Example:**
  ```python
  set_project_units(project, units="kip, in", convert_existing=True)
  # or:
  set_project_units(project, units="kN, m")
  ```

#### `save_project` & `load_project`
Full JSON serialization and deserialization round-trip for `struct_core.Project` containers.
- **Signatures:**
  ```python
  save_project(project, path) -> ToolResult
  load_project(path) -> ToolResult
  ```

---

### 2. Commercial CAE Interoperability Tools (`struct_link`)

#### `import_from_sap2000`
Ingests SAP2000 `.s2k` or `.$2k` text files into a validated `Project`. Automatically extracts nodes, frame elements, sections, materials, supports, load cases/patterns, and unit system metadata.
- **Signature:**
  ```python
  import_from_sap2000(s2k_path, *, project_title=None, materialize_self_weight=False) -> ToolResult
  ```
- **Parameters:**
  - `s2k_path`: File path to the SAP2000 `.s2k` file.
  - `project_title`: Optional title override.
  - `materialize_self_weight`: When `True`, automatically computes and adds member self-weight line loads into the dead load case.
- **Returns (`data`):** `{"project": Project, "n_nodes": int, "n_elements": int, "n_supports": int, "load_cases": list, "units": str}`
- **Example:**
  ```python
  res = import_from_sap2000("frame.s2k", materialize_self_weight=True)
  project = res.data["project"]
  ```

#### `export_to_sap2000`
Exports a `Project` model to a native SAP2000 `.s2k` text file with full coordinate systems, joint restraints, frame section assignments, and material properties.
- **Signature:** `export_to_sap2000(project, *, output_path) -> ToolResult`
- **Returns (`data`):** `{"output_path": str, "file_size_bytes": int}`

#### `import_from_etabs` & `export_to_etabs`
Ingests from and exports to ETABS `.e2k` / `.$et` text files with full floor-story, grid, and framing connectivity preservation.
- **Signatures:**
  ```python
  import_from_etabs(e2k_path, *, project_title=None) -> ToolResult
  export_to_etabs(project, *, output_path) -> ToolResult
  ```

---

### 3. Building Code Loading & Sizing Tools (`struct_load`)

#### `preliminary_sizing`
Performs rule-of-thumb span-to-depth structural member dimensioning for beams ($L/12$ to $L/16$) and columns based on tributary area and concrete compressive strength.
- **Signature:**
  ```python
  preliminary_sizing(project, *, material_id="Concrete_4ksi", fc_kpa=27580.0) -> ToolResult
  ```
- **Returns (`data`):** `{"project": Project, "n_beams_sized": int, "n_columns_sized": int, "summary": dict}`

#### `apply_gravity_loads`
Applies ASCE 7 / IBC occupancy-based live loads (e.g. `"office"`, `"residential"`, `"classroom"`, `"storage"`) and superimposed dead loads distributed across member tributary widths.
- **Signature:**
  ```python
  apply_gravity_loads(project, *, occupancy="office", tributary_width=5.0, superimposed_dead=1.5, include_self_weight=True) -> ToolResult
  ```
- **Parameters:**
  - `occupancy`: ASCE 7 occupancy category name.
  - `tributary_width`: Tributary floor width in meters (or feet).
  - `superimposed_dead`: Additional dead load in kPa (or psf).
  - `include_self_weight`: When `True`, automatically triggers self-weight calculations.
- **Returns (`data`):** `{"project": Project, "load_cases_created": list, "total_gravity_load_kN": float}`

#### `apply_self_weight`
Computes the exact volumetric weight of all frame elements ($w = -\rho \cdot A \cdot g$) using material densities and section cross-sectional areas, and materializes continuous downward line loads directly into the target load case in the `Project` schema. (Note: `run_analysis` automatically computes and applies self-weight on the fly for any cases with `include_self_weight=True` without needing to mutate the project model; use `apply_self_weight` when you want to permanently bake element line loads into the schema).
- **Signature:**
  ```python
  apply_self_weight(project, *, target_case_id=None, units=None, direction=None) -> ToolResult
  ```
- **Parameters:**
  - `target_case_id`: Load case name to append loads to (defaults to `"DEAD"` or `"SelfWeight"`).
  - `units`: Units specification (e.g. `"kN, m"`, `"kip, in"`), or auto-detected from project metadata.
  - `direction`: Gravity axis (`"z"` or `"-z"` for 3D, `"y"` or `"-y"` for 2D).
- **Returns (`data`):** `{"project": Project, "target_case_id": str, "elements_loaded": int, "total_downward_load": float, "unit": str}`

#### `apply_lateral_loads`
Computes and applies ASCE 7-22 seismic Equivalent Lateral Force (ELF) vertical distribution and Main Wind Force Resisting System (MWFRS) wind pressures as nodal story forces.
- **Signature:**
  ```python
  apply_lateral_loads(project, *, code="ASCE7-22", sds=0.75, sd1=0.40, r_factor=8.0, wind_speed=None) -> ToolResult
  ```
- **Parameters:**
  - `sds`, `sd1`: Design spectral acceleration parameters ($S_{DS}, S_{D1}$).
  - `r_factor`: Seismic response modification coefficient $R$.
  - `wind_speed`: Basic wind speed in m/s (or mph).
- **Returns (`data`):** `{"project": Project, "seismic_base_shear_kN": float, "wind_base_shear_kN": float, ...}`

#### `generate_load_combinations`
Generates standard factored strength design (LRFD) and serviceability (ASD) load combinations per ACI 318 / ASCE 7 (e.g. $1.4D$, $1.2D + 1.6L$, $1.2D + 1.0E + 1.0L$).
- **Signature:**
  ```python
  generate_load_combinations(project, *, standard="ACI318_LRFD") -> ToolResult
  ```
- **Returns (`data`):** `{"project": Project, "combinations_generated": list}`

---

### 4. Validation Tool

#### `validate_model`
Validates model integrity per `struct_core` schema rules before solver execution: checks referential integrity (element start/end nodes, material and section references), flags duplicate IDs, catches zero-length elements, and confirms support fixity.
- **Signature:** `validate_model(model_or_project) -> ToolResult`
- **Returns (`data`):** `{"valid": bool, "warnings": list, "errors": list}`

---

### 5. Static & Modal FEM Analysis Tools (`fem2d` & `fem3d`)

#### `run_analysis`
Solves linear static equilibrium ($\mathbf{K}\mathbf{u} = \mathbf{f}$) or natural eigenvalue modal dynamics ($\mathbf{K}\boldsymbol{\phi} = \omega^2\mathbf{M}\boldsymbol{\phi}$). Automatically detects whether the structure is 2D or 3D and dispatches to `fem2d` or `fem3d`. Results are packaged into `AnalysisResult` and attached to `project.analysis_results`.
- **Signature:**
  ```python
  run_analysis(project, *, analysis_case_id, analysis_type="static", solver="auto", load_case_id=None, num_modes=6, include_self_weight=True) -> ToolResult
  ```
- **Parameters:**
  - `analysis_case_id`: Name identifier for this analysis run (e.g. `"Dead_Static"`, `"Modal"`).
  - `analysis_type`: `"static"` or `"modal"`.
  - `solver`: `"auto"` (recommended), `"fem2d"`, or `"fem3d"`.
  - `load_case_id`: Target load case or combination to solve (for static).
  - `num_modes`: Number of natural vibration modes to extract (for modal).
  - `include_self_weight`: When `True` (default), automatically computes and includes member self-weight line loads via `struct_load` for any load cases with `include_self_weight=True` without mutating the project schema.
- **Returns (`data`):** `{"project": Project, "analysis_case_id": str, "summary": dict}`
- **Example:**
  ```python
  res = run_analysis(project, analysis_case_id="1.2D+1.6L", solver="auto")
  ```

#### `inspect_analysis_results`
Extracts a structured engineering summary of analysis results: max nodal displacements, total base reactions, equilibrium checks, and peak member internal forces (axial $P$, shear $V$, bending moment $M$).
- **Signature:** `inspect_analysis_results(project, analysis_case_id) -> ToolResult`
- **Returns (`data`):** `{"analysis_case_id": str, "max_disp": float, "total_reactions": dict, "peak_forces": dict}`

---

### 6. Structural Dynamics Tools (`structdyn`)

#### `run_time_history`
Executes dynamic earthquake response history integration using `structdyn`. Bridges the model directly to an MDF dynamic system, runs numerical integration (modal superposition or Newmark-Beta), and envelopes peak member dynamic internal forces into an `AnalysisResult` ready for code design.
- **Signature:**
  ```python
  run_time_history(project, *, analysis_case_id, ground_motion="elcentro", component=None, direction="x", dt=None, scale_factor=1.0, damping_ratio=0.05, n_modes_damping=2, method="newmark_beta", solver="auto", envelope_to_design=True) -> ToolResult
  ```
- **Parameters:**
  - `ground_motion`: Ground motion name (`"elcentro"`, `"northridge"`, `"northridge:hor2"`), `GroundMotion` object, or array.
  - `component`: Specific component (`"hor1"`, `"hor2"`, `"up"`). Inferred from `direction` if omitted.
  - `direction`: Ground motion component (`"x"`, `"y"`, or `"z"`).
  - `damping_ratio`: Modal or Rayleigh damping ratio $\zeta$ (default: $0.05$).
  - `envelope_to_design`: When `True`, synthesizes peak dynamic $P, V_y, V_z, M_y, M_z$ into `AnalysisResult` for downstream ACI 318 design.
- **Returns (`data`):** `{"project": Project, "analysis_case_id": str, "max_disp_m": float, "max_base_shear_kN": float, ...}`

#### `inspect_dynamic_results`
Summarizes peak dynamic responses, floor displacement envelopes, base shear time histories, and inter-story drift ratios.
- **Signature:** `inspect_dynamic_results(project, analysis_case_id) -> ToolResult`

---

### 7. RC Design & Detailing Tools (`aci318`)

#### `run_design`
Runs reinforced concrete code checks per **ACI 318-19** using `aci318`. Evaluates beam flexure ($M_u \le \phi M_n$), beam shear ($V_u \le \phi V_n$), column axial-flexural P-M interaction, and shear ties. Synthesizes longitudinal bar layouts, stirrup schedules, and demand-capacity ratios (DCR).
- **Signature:**
  ```python
  run_design(project, *, design_case_id, design_code="ACI 318-19", analysis_case_id=None) -> ToolResult
  ```
- **Returns (`data`):** `{"project": Project, "design_case_id": str, "summary": {"status": "PASS"|"FAIL", "max_dcr": float, ...}}`

#### `inspect_design_results`
Returns detailed member-by-member Demand-Capacity Ratios (DCR), governing load combinations, and pass/fail statuses.
- **Signature:** `inspect_design_results(project, design_case_id) -> ToolResult`

#### `inspect_detailing_results`
Extracts reinforcement detailing summaries, rebar mark schedules, stirrup spacing per zone (hinging vs field), and total calculated reinforcement steel weight/tonnage.
- **Signature:** `inspect_detailing_results(project, design_case_id=None) -> ToolResult`
- **Returns (`data`):** `{"total_rebar_weight_kg": float, "total_rebar_weight_tons": float, "members": dict}`

---

### 8. Production CAD Drawing Generation (`struct_draw`)

#### `generate_drawings`
Produces production-ready DXF drawings compatible with AutoCAD, LibreCAD, and BricsCAD. Supports both **General Arrangement (GA)** framing plans/elevations and **Reinforced Concrete Rebar Detailing** with Bar Bending Schedule (BBS) tables.
- **Signature:**
  ```python
  generate_drawings(
      project,
      *,
      output_path,
      drawing_type="auto",        # "framing", "detailing", or "auto"
      view_mode="both",           # "plan", "elevation", or "both"
      story_level=None,           # Specific story for framing plan (e.g. "FL1", 1)
      grid_line=None,             # Specific grid line for elevation (e.g. "1", "A")
      title=None,                 # Sheet title block text
      scale=None,                 # "1:50", "1:100", "1/4\"=1'-0\""
      unit=None,                  # "mm", "m", "in", "ft"
      design_case_id=None,        # Design case ID for RC detailing
      detailing_case_id=None,
  ) -> ToolResult
  ```
- **Returns (`data`):** `{"output_path": str, "drawing_type": str, "file_size_bytes": int}`
- **Examples:**
  ```python
  # 1. GA Framing drawing directly from model geometry
  generate_drawings(project, output_path="framing.dxf", drawing_type="framing", view_mode="both")

  # 2. RC Rebar Detailing sheet with BBS schedule table
  generate_drawings(project, output_path="detailing.dxf", drawing_type="detailing", design_case_id="ACI318_Design")
  ```

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

### Dynamic Time-History Analysis & Seismic Design

`struct_agent` connects `structdyn` directly into the structural workflow. You can run modal analysis or earthquake time-history integration (with El Centro, Kobe, or custom acceleration records) and automatically envelope the peak dynamic internal forces into an `AnalysisResult` that directly feeds downstream ACI 318-19 RC design:

```python
from struct_agent import (
    create_project,
    create_frame_model,
    apply_gravity_loads,
    validate_model,
    run_analysis,
    run_time_history,
    inspect_dynamic_results,
    run_design,
    inspect_design_results,
)

# 1. Build parametric model & apply gravity loads
model = create_frame_model(bays=2, stories=3, bay_width=6.0, story_height=3.5)
res_proj = create_project(model, title="3-Story Seismic RC Frame")
project = res_proj.data["project"]
apply_gravity_loads(project, occupancy="office", tributary_width=4.0)
validate_model(project)

# 2. Modal analysis (natural frequencies, periods, mode shapes)
res_modal = run_analysis(project, analysis_case_id="Modal", analysis_type="modal", num_modes=3)
T1 = res_modal.data["summary"]["fundamental_period_s"]
print(f"Fundamental Period T1 = {T1:.3f} s")

# 3. Dynamic ground motion time-history analysis (El Centro 1940 record)
# Automatically builds MDF dynamic model, integrates response, and extracts peak forces
res_th = run_time_history(
    project,
    analysis_case_id="Earthquake_ElCentro",
    ground_motion="elcentro",
    direction="x",
    scale_factor=1.0,
    damping_ratio=0.05,
    envelope_to_design=True,  # Saves peak forces into Project as an AnalysisResult
)

# Inspect peak dynamic response
dyn_summary = inspect_dynamic_results(project, "Earthquake_ElCentro")
print("Max dynamic base shear:", dyn_summary.data["summary"]["max_base_shear_kN"], "kN")
print("Max roof displacement:", dyn_summary.data["summary"]["max_disp_m"], "m")

# 4. Downstream ACI 318-19 RC member design using dynamic peak force envelope
res_design = run_design(
    project,
    design_case_id="ACI318_Dynamic_Design",
    analysis_case_id="Earthquake_ElCentro",
)
print("Design status:", res_design.data["summary"]["status"])

# 5. Production CAD drawings via unified generate_drawings tool
# General Arrangement (GA) framing plans & elevations:
res_framing = generate_drawings(
    project,
    output_path="framing_sheet.dxf",
    drawing_type="framing",
    view_mode="both",  # "plan", "elevation", or "both"
)
print("GA Framing drawing generated:", res_framing.data["output_path"])

# Reinforced concrete rebar detailing (cross-sections, rebar mats, stirrups):
res_detailing = generate_drawings(
    project,
    output_path="rebar_detailing.dxf",
    drawing_type="detailing",
    design_case_id="ACI318_Dynamic_Design",
)
print("RC Detailing drawing generated:", res_detailing.data["output_path"])
```


### Multimodal Drawing-to-3D-Model Engine (VLM + Grid Compiler)

`struct_agent` features a robust, two-step hybrid pipeline for converting architectural floor plans and structural drawings (PNG, JPG, multi-page vector PDF) directly into analytical 3D structural models:

```
 User Drawing (PDF/PNG/JPG)
         |
         v
 [ VLM Semantic Transcription ]  <- Google Gemini 2.0, Ollama (gemma4:31b), OpenAI (gpt-4o)
         |
         | Transcribes written dimension strings ("24'-0"", "6.5 m") & grid labels (A-B-C / 1-2-3)
         v
  FloorPlanDraft (Pydantic)
         |
         v
 [ Deterministic Grid Compiler ]  <- Pure-Python compiler (struct_agent.vision.compiler)
         |
         | - Computes exact 3D nodal intersections
         | - Spans continuous columns across all stories
         | - Frames perimeter and interior floor beams
         | - Sets 6-DOF ground supports (Fixed/Pinned)
         | - Generates ASCE 7 load cases (1.4D, 1.2D + 1.6L)
         v
 struct_core.Project (Ready for 3D FEA, modal analysis & ACI 318 design)
```

#### Why a 2-Step Hybrid Pipeline?
1. **Transcribe Dimensions, Don't Estimate Pixels**: VLMs excel at reading handwritten and CAD dimension annotations, identifying structural column grids, and noting story heights. However, relying on raw pixel coordinates or asking an LLM to guess 3D nodal arrays leads to non-coplanar framing, misaligned joints, and singular stiffness matrices.
2. **Pure-Python Deterministic Compiler**: By capturing dimensions in a strict semantic intermediate representation (`FloorPlanDraft`), the deterministic compiler handles exact coordinate math, column continuity, perimeter and interior beam topology, boundary conditions, and code load cases with mathematical precision.
3. **Multi-Turn Story Clarification**: When a user uploads a floor plan drawing without elevations, the agent detects the missing vertical geometry and prompts the user for clarification (*"How many stories is this building and what is the typical story height?"*). Because multi-turn conversation history is preserved across prompts, the agent remembers the previously extracted floor plan once the user responds and compiles the complete 3D model.
4. **Multi-Format Drawing Ingestion**: Multimodal attachments (PNG, JPG, WEBP) and multi-page vector drawings (PDF) are ingested seamlessly. Vector PDFs are rendered into high-fidelity image buffers at 144 DPI using `pypdfium2` before transmission to local or cloud VLMs.

#### Programmatic Usage

```python
from struct_agent import create_model_from_plan_draft
from struct_agent.vision import (
    FloorPlanDraft,
    GridLineDraft,
    StoryDraft,
    build_project_from_draft,
)

# 1. Structured floor plan draft (produced by VLM or defined programmatically)
draft = FloorPlanDraft(
    title="2-Story Commercial Office",
    units="kN, m",
    grid_lines_x=[
        GridLineDraft(id="A", coordinate=0.0),
        GridLineDraft(id="B", coordinate=6.0),
        GridLineDraft(id="C", coordinate=12.0),
    ],
    grid_lines_y=[
        GridLineDraft(id="1", coordinate=0.0),
        GridLineDraft(id="2", coordinate=5.0),
        GridLineDraft(id="3", coordinate=10.0),
    ],
    stories=[
        StoryDraft(id="Story1", level=1, height=3.5),
        StoryDraft(id="Story2", level=2, height=3.0),
    ],
    support_type="fixed",
    default_fc_kpa=27580.0,  # 4 ksi concrete
)

# 2. Build 3D Project via ToolResult wrapper (or directly via build_project_from_draft)
result = create_model_from_plan_draft(draft)
project = result.data["project"]
print(f"Generated 3D Model: {len(project.model.nodes)} nodes, {len(project.model.elements)} elements.")
```

#### Multimodal Autonomous Execution via `AgentRunner`

```python
from struct_agent import AgentRunner

# Works with local Ollama (gemma4:31b, qwen2-vl) or cloud (Gemini, OpenAI)
runner = AgentRunner(
    provider="native",
    model="gemini/gemini-2.0-flash",  # or "ollama/gemma4:31b"
)

# Stream thought process and tool execution with drawing attachments
events = runner.run_stream(
    "Please read this architectural drawing, verify the dimensions, and create a 3D structural model.",
    attachments=["path/to/floor_plan.pdf"],
)

for event in events:
    if event.type == "thought":
        print(f"[REASONING] {event.thought}")
    elif event.type == "tool_call":
        print(f"[TOOL] Calling {event.tool_name} with {event.arguments}")
    elif event.type == "tool_result":
        print(f"[RESULT] {event.tool_name} -> success: {event.ok}")
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
| `test_dynamics.py` | Dynamic time-history analysis, ground motion integration, modal analysis, peak force enveloping to ACI 318 |
| `test_link_tools.py` | SAP2000 & ETABS file import, export, and error handling |
| `test_loads_integration.py` | Preliminary member sizing and ASCE 7 load generator tool integration |
| `test_vision_compiler.py` | Deterministic grid snapping, column generation, perimeter/interior beam framing, and ASCE 7 load combinations |
| `test_vision_agent.py` | Multimodal attachment encoding, pypdfium2 PDF rendering, and VLM multi-turn drafting |
| `test_tool_contracts.py` | Individual tool input validation (missing project, wrong types, etc.) |
| `test_workflow_state.py` | State machine: valid transitions, invalid transitions, idempotency |
| `test_llm_runner.py` | Autonomous multi-turn agent loop, event streaming, provider factory, and mock tool reasoning |
| `test_framing_drawing_tool.py` | GA framing plan & elevation generation directly from StructuralModel |

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
| Structural dynamics — modal, time-history, SDF, MDF, ground motions, FEM bridges | structdyn |
| Dynamic time-history tool (`run_time_history`) with El Centro/Kobe/custom ground motions | struct_agent |
| Dynamic peak force envelope synthesis for downstream ACI 318 RC design | struct_agent |
| Dynamic results inspection tool (`inspect_dynamic_results`) | struct_agent |
| Multimodal drawing ingestion (PDF rasterization via `pypdfium2`, PNG, JPG, WEBP) | struct_agent |
| Deterministic grid compiler (`FloorPlanDraft` -> 3D `struct_core.Project`) | struct_agent |
| VLM plan drafting tool (`create_model_from_plan_draft`) | struct_agent |
| Multi-turn conversational memory & drawing context preservation across prompts | struct_agent |
| Workflow state machine (7 stages, transition enforcement) | struct_agent |
| Agent tool layer (18+ tools, structured `ToolResult`) | struct_agent |
| Automated solver selection (`auto`, `fem2d`, `fem3d`) | struct_agent |
| Deterministic demo orchestrator (`StructAgent`) | struct_agent |
| Autonomous cognitive LLM orchestration wrapper (`AgentRunner`) | struct_agent |
| Zero-dependency native HTTP client for OpenAI/Ollama API | struct_agent |
| Multi-provider LiteLLM adapter (Claude, Gemini, OpenAI, etc.) | struct_agent |
| Granular event streaming interface (`ThoughtEvent`, `ToolCallEvent`, `StateUpdateEvent`, etc.) | struct_agent |
| Grounded system prompt & 18+ tool schema function signatures | struct_agent |
| End-to-end pipeline, dynamics, vision compiler & LLM mock tests | struct_agent |

---

## Future Evolution & Gaps

The cognitive LLM wrapper and multimodal drawing engine are fully functional via `AgentRunner`! The remaining ecosystem roadmap items are:

| Feature / Integration | Description & Impact | Status |
|---|---|---|
| **StructStudio Desktop GUI** | Full-featured PySide6/PyVista desktop studio (`struct_gui`) consuming `AgentRunner.run_stream()` events with 3D CAD modeling, dynamic modal animations, drawing uploads, and real-time viewport synchronization | ✅ Available (`struct_gui`) |
| **Nonlinear dynamic analysis** | Material and geometric nonlinearity, plastic hinges, and inelastic time-history analysis | In Design |

---

## License

MIT License.
