"""
struct_agent - AI orchestration layer for the Structural Engineering Ecosystem.

This package provides a thin, framework-independent tool layer on top of
``struct_core``, ``fem2d``, ``aci318``, and ``struct_draw``. It performs no
engineering calculations itself.

Public entry points:
- :class:`StructAgent`         — top-level orchestrator (deterministic, no LLM)
- :func:`create_project`        — Model tool
- :func:`inspect_project`       — Model tool
- :func:`modify_project`        — Model tool
- :func:`validate_model`        — Model validation tool
- :func:`run_analysis`          — Analysis tool (fem2d → struct_core)
- :func:`inspect_analysis_results`
- :func:`run_design`            — Design tool (aci318 → struct_core)
- :func:`inspect_design_results`
- :func:`generate_drawings`     — Drawing tool (struct_draw)
- :class:`WorkflowStage`        — Workflow state machine
- :class:`ToolResult`           — Structured tool return type
- :class:`AgentError`           — Engineered, machine-readable errors
"""

from .agent import StructAgent
from .state.project_state import (
    WorkflowStage,
    ProjectState,
    StageTransitionError,
)
from .tools.base import ToolResult, AgentError
from .tools.model import (
    create_project,
    create_frame_model,
    create_model_from_plan_draft,
    build_structural_model,
    inspect_project,
    modify_project,
    set_project_units,
    save_project,
    load_project,
)
from .tools.analysis import (
    run_analysis,
    inspect_analysis_results,
)
from .tools.dynamics import (
    run_time_history,
    inspect_dynamic_results,
)
from .tools.design import (
    run_design,
    inspect_design_results,
    inspect_detailing_results,
)
from .tools.loads import (
    apply_gravity_loads,
    apply_lateral_loads,
    apply_roof_loads,
    apply_self_weight,
    distribute_area_loads,
    generate_load_combinations,
)
from .tools.sizing import preliminary_sizing
from .tools.drawing import generate_drawings, generate_framing_drawings
from .tools.link import (
    import_from_sap2000,
    export_to_sap2000,
    import_from_etabs,
    export_to_etabs,
)
from .validation.model import validate_model
from .llm import (
    AgentRunner,
    LLMEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    StateUpdateEvent,
    TextChunkEvent,
    CompleteEvent,
    ErrorEvent,
    BaseLLMProvider,
    LLMResponse,
    ToolCallRequest,
    get_provider,
)

__version__ = "0.1.0"

__all__ = [
    "StructAgent",
    "AgentRunner",
    "LLMEvent",
    "ThoughtEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StateUpdateEvent",
    "TextChunkEvent",
    "CompleteEvent",
    "ErrorEvent",
    "BaseLLMProvider",
    "LLMResponse",
    "ToolCallRequest",
    "get_provider",
    "WorkflowStage",
    "ProjectState",
    "StageTransitionError",
    "ToolResult",
    "AgentError",
    "create_project",
    "create_frame_model",
    "create_model_from_plan_draft",
    "build_structural_model",
    "inspect_project",
    "modify_project",
    "set_project_units",
    "save_project",
    "load_project",
    "preliminary_sizing",
    "apply_gravity_loads",
    "apply_lateral_loads",
    "apply_roof_loads",
    "distribute_area_loads",
    "apply_self_weight",
    "generate_load_combinations",
    "validate_model",
    "run_analysis",
    "inspect_analysis_results",
    "run_time_history",
    "inspect_dynamic_results",
    "run_design",
    "inspect_design_results",
    "inspect_detailing_results",
    "generate_drawings",
    "generate_framing_drawings",
    "import_from_sap2000",
    "export_to_sap2000",
    "import_from_etabs",
    "export_to_etabs",
]