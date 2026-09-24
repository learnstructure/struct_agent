"""
OpenAI-compatible JSON Schema definitions for all struct_agent engineering tools.

These schemas are passed to any function-calling capable LLM (OpenAI, Gemini, Claude, Ollama).
"""

from __future__ import annotations

from typing import Any, Dict, List

ENGINEERING_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Initialize a new structural engineering Project container with an initial StructuralModel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Project title (e.g. '2-Story RC Office Frame')"},
                    "author": {"type": "string", "description": "Engineer or author name"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_project",
            "description": "Inspect the active Project: lists node counts, element connectivity, section profiles, materials, and load cases.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preliminary_sizing",
            "description": "Automatically calculate heuristic member sizes (e.g. beam depth h = span/12, column size) based on spans and concrete compressive strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material_id": {"type": "string", "description": "Identifier for the concrete material", "default": "Concrete_4ksi"},
                    "fc_kpa": {"type": "number", "description": "Concrete compressive strength f'c in kPa (default: 27580 kPa = 4000 psi)", "default": 27580.0},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_gravity_loads",
            "description": "Calculate and apply building-code-compliant gravity dead and live loads per ASCE 7 / IBC based on building occupancy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "occupancy": {
                        "type": "string",
                        "description": "Building occupancy type",
                        "enum": ["office", "residential", "classroom", "retail", "storage", "corridor", "roof"],
                    },
                    "tributary_width": {"type": "number", "description": "Tributary width in meters for converting area pressure to line loads", "default": 5.0},
                    "superimposed_dead": {"type": "number", "description": "Superimposed dead load intensity in kPa (finishes, MEP)", "default": 1.5},
                    "include_self_weight": {"type": "boolean", "description": "Whether to automatically compute and apply member self-weight", "default": True},
                },
                "required": ["occupancy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_lateral_loads",
            "description": "Calculate and apply seismic Equivalent Lateral Force (ELF) and/or wind lateral story forces per ASCE 7.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Building standard", "default": "ASCE7-22"},
                    "sds": {"type": "number", "description": "Design spectral acceleration parameter SDS (g)", "default": 0.75},
                    "sd1": {"type": "number", "description": "Design spectral acceleration parameter SD1 (g)", "default": 0.40},
                    "r_factor": {"type": "number", "description": "Response modification coefficient R (e.g. 8.0 for Special RC Moment Frame)", "default": 8.0},
                    "wind_speed": {"type": "number", "description": "Basic wind speed in m/s (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_load_combinations",
            "description": "Generate standard factored load combinations per ACI 318 / ASCE 7 (e.g. 1.4D, 1.2D+1.6L, 1.2D+1.0EQ+0.5L).",
            "parameters": {
                "type": "object",
                "properties": {
                    "standard": {
                        "type": "string",
                        "description": "Combination design standard",
                        "enum": ["ACI318_LRFD", "ASCE7_LRFD", "ASCE7_ASD"],
                        "default": "ACI318_LRFD",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_model",
            "description": "Run strict engineering validation on the structural model (referential integrity, supports, zero-length elements, boundary mechanisms).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": "Execute finite element analysis (static forces or eigenvalue modal analysis) via fem2d or fem3d.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_case_id": {"type": "string", "description": "Unique ID for storing results (e.g. 'Static_1' or 'Modal_1')"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["static", "modal"],
                        "description": "'static' for linear static / envelope, or 'modal' for natural periods & mode shapes",
                        "default": "static",
                    },
                    "solver": {
                        "type": "string",
                        "enum": ["auto", "fem2d", "fem3d"],
                        "description": "Solver selection. 'auto' selects fem2d for planar models and fem3d for 3D/modal.",
                        "default": "auto",
                    },
                    "load_case_id": {"type": "string", "description": "Specific load case or 'envelope' over all combinations"},
                    "num_modes": {"type": "integer", "description": "Number of modes for modal analysis (default: 6)", "default": 6},
                },
                "required": ["analysis_case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_analysis_results",
            "description": "Retrieve maximum axial forces, shear forces, bending moments, and modal periods from an analysis case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_case_id": {"type": "string", "description": "Identifier of the analysis case to inspect"},
                },
                "required": ["analysis_case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_design",
            "description": "Run reinforced concrete code checks per ACI 318-19 (flexure, shear, axial/P-M interaction) and detailing reinforcement schedules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "design_case_id": {"type": "string", "description": "Identifier for the design check results", "default": "ACI318_19"},
                    "design_code": {"type": "string", "description": "Design code edition", "default": "ACI 318-19"},
                    "analysis_case_id": {"type": "string", "description": "Which analysis case internal forces to design for"},
                },
                "required": ["design_case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_design_results",
            "description": "Inspect the ACI 318 design evaluation: checks governing Demand-Capacity Ratio (DCR), pass/fail status, and failing members.",
            "parameters": {
                "type": "object",
                "properties": {
                    "design_case_id": {"type": "string", "description": "Identifier of the design check to inspect"},
                },
                "required": ["design_case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_drawings",
            "description": "Generate production-grade CAD structural drawings in DXF format (cross-sections, rebar mats, stirrups, beam/column elevations).",
            "parameters": {
                "type": "object",
                "properties": {
                    "design_case_id": {"type": "string", "description": "Design case providing reinforcement detailing"},
                    "output_path": {"type": "string", "description": "Target file path for the .dxf drawing"},
                },
                "required": ["design_case_id", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "import_from_sap2000",
            "description": "Import an existing CSI SAP2000 model text file (.s2k / .$2k) into the Project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s2k_path": {"type": "string", "description": "Path to the .s2k file"},
                    "project_title": {"type": "string", "description": "Optional custom project title"},
                },
                "required": ["s2k_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_to_sap2000",
            "description": "Export the current structural model to a CSI SAP2000 text file (.s2k).",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Target .s2k file path"},
                },
                "required": ["output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "import_from_etabs",
            "description": "Import an existing CSI ETABS model text file (.e2k / .$et) into the Project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "e2k_path": {"type": "string", "description": "Path to the .e2k file"},
                    "project_title": {"type": "string", "description": "Optional custom project title"},
                },
                "required": ["e2k_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_to_etabs",
            "description": "Export the current structural model to a CSI ETABS text file (.e2k).",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Target .e2k file path"},
                },
                "required": ["output_path"],
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return all tool schemas formatted for LLM function calling."""
    return list(ENGINEERING_TOOL_SCHEMAS)
