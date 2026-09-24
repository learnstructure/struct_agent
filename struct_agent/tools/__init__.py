"""
struct_agent.tools - High-level engineering operations exposed to the agent.

Each tool is a small, explicit, strongly-typed function that wraps the
underlying engineering libraries (struct_core, fem2d, aci318, struct_draw).
Tools return structured :class:`ToolResult` objects and never invent
engineering data.
"""

from .base import ToolResult, AgentError
from .model import create_project, inspect_project, modify_project
from .sizing import preliminary_sizing
from .loads import (
    apply_gravity_loads,
    apply_lateral_loads,
    generate_load_combinations,
)
from .analysis import run_analysis, inspect_analysis_results
from .design import run_design, inspect_design_results
from .drawing import generate_drawings
from .link import (
    import_from_sap2000,
    export_to_sap2000,
    import_from_etabs,
    export_to_etabs,
)

__all__ = [
    "ToolResult",
    "AgentError",
    "create_project",
    "inspect_project",
    "modify_project",
    "preliminary_sizing",
    "apply_gravity_loads",
    "apply_lateral_loads",
    "generate_load_combinations",
    "run_analysis",
    "inspect_analysis_results",
    "run_design",
    "inspect_design_results",
    "generate_drawings",
    "import_from_sap2000",
    "export_to_sap2000",
    "import_from_etabs",
    "export_to_etabs",
]