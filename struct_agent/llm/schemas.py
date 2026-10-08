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
            "name": "create_frame_model",
            "description": "Create a fully connected 2D or 3D multi-bay, multi-story structural frame with grid nodes, columns, beams, supports, and standard material/section profiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_bays": {"type": "integer", "description": "Number of horizontal bays (e.g. 2)", "default": 2},
                    "bay_width": {"type": "number", "description": "Uniform bay width in meters (e.g. 5.0)", "default": 5.0},
                    "num_stories": {"type": "integer", "description": "Number of vertical stories (e.g. 2)", "default": 2},
                    "story_height": {"type": "number", "description": "Uniform story height in meters (e.g. 3.0)", "default": 3.0},
                    "dimension": {"type": "string", "enum": ["2d", "3d"], "description": "Dimensionality: '2d' for plane frame, '3d' for space frame", "default": "2d"},
                    "support_type": {"type": "string", "enum": ["fixed", "pinned"], "description": "Base boundary conditions", "default": "fixed"},
                    "title": {"type": "string", "description": "Project title (e.g. '2-Bay 2-Story RC Frame')"},
                    "col_b": {"type": "number", "description": "Column cross-section width in meters (default: 0.4)", "default": 0.4},
                    "col_h": {"type": "number", "description": "Column cross-section depth in meters (default: 0.4)", "default": 0.4},
                    "beam_b": {"type": "number", "description": "Beam cross-section width in meters (default: 0.3)", "default": 0.3},
                    "beam_h": {"type": "number", "description": "Beam cross-section depth in meters (default: 0.5)", "default": 0.5},
                },
                "required": ["num_bays", "bay_width", "num_stories", "story_height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_model_from_plan_draft",
            "description": "Create a complete, exact 3D structural model from an architectural or structural floor plan and grid draft. Use this tool when interpreting drawings (PDF, PNG, JPG) or reading structural plans with grid lines, dimensions, story heights, columns, and beams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Project title (e.g. 'Office Building Structural Frame')"},
                    "unit_system": {"type": "string", "enum": ["metric", "us_customary"], "description": "Unit system (metric=meters/kN, us_customary=feet/inches/kips)", "default": "metric"},
                    "x_grids": {
                        "type": "array",
                        "description": "Grid lines along X axis (e.g. A, B, C or 1, 2, 3) with exact dimension coordinates",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Grid line callout (e.g. 'A', 'B', '1')"},
                                "coordinate": {"type": "number", "description": "Exact coordinate value along X axis"},
                            },
                            "required": ["name", "coordinate"],
                        },
                    },
                    "y_grids": {
                        "type": "array",
                        "description": "Grid lines along Y axis (e.g. 1, 2, 3 or A, B, C) with exact dimension coordinates",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Grid line callout (e.g. '1', '2', '3')"},
                                "coordinate": {"type": "number", "description": "Exact coordinate value along Y axis"},
                            },
                            "required": ["name", "coordinate"],
                        },
                    },
                    "stories": {
                        "type": "array",
                        "description": "Vertical stories from bottom to top. Story heights must be confirmed from elevations or user prompt.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Story name (e.g. 'Story 1', 'Roof')"},
                                "height": {"type": "number", "description": "Story height (in meters or feet depending on unit_system)"},
                            },
                            "required": ["name", "height"],
                        },
                    },
                    "columns": {
                        "type": "array",
                        "description": "Optional specific column locations at grid intersections. If omitted or empty, columns are placed at all grid intersections.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "grid_x": {"type": "string", "description": "X grid line name"},
                                "grid_y": {"type": "string", "description": "Y grid line name"},
                                "section_b": {"type": "number", "description": "Column width (optional)"},
                                "section_h": {"type": "number", "description": "Column depth (optional)"},
                            },
                            "required": ["grid_x", "grid_y"],
                        },
                    },
                    "beams": {
                        "type": "array",
                        "description": "Optional specific beam lines connecting grid points. If omitted or empty, beams are framed along all continuous grid lines on each floor.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_grid_x": {"type": "string"},
                                "start_grid_y": {"type": "string"},
                                "end_grid_x": {"type": "string"},
                                "end_grid_y": {"type": "string"},
                                "section_b": {"type": "number"},
                                "section_h": {"type": "number"},
                            },
                            "required": ["start_grid_x", "start_grid_y", "end_grid_x", "end_grid_y"],
                        },
                    },
                    "default_col_b": {"type": "number", "description": "Default column width (e.g. 0.4 m or 16 in)"},
                    "default_col_h": {"type": "number", "description": "Default column depth (e.g. 0.4 m or 16 in)"},
                    "default_beam_b": {"type": "number", "description": "Default beam width (e.g. 0.3 m or 12 in)"},
                    "default_beam_h": {"type": "number", "description": "Default beam depth (e.g. 0.5 m or 20 in)"},
                    "base_support": {"type": "string", "enum": ["fixed", "pinned"], "default": "fixed"},
                },
                "required": ["x_grids", "y_grids", "stories"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Initialize a new structural engineering Project container, optionally populated with complete structural model geometry (nodes, elements, materials, sections, supports).",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Project title (e.g. '2-Story RC Office Frame')"},
                    "author": {"type": "string", "description": "Engineer or author name"},
                    "model": {
                        "type": "object",
                        "description": "Structured structural model specification with nodes, elements, materials, sections, and supports.",
                        "properties": {
                            "nodes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "description": "Node identifier"},
                                        "x": {"type": "number", "description": "X coordinate in meters"},
                                        "y": {"type": "number", "description": "Y coordinate in meters"},
                                        "z": {"type": "number", "description": "Z coordinate in meters", "default": 0.0},
                                    },
                                    "required": ["id", "x", "y"],
                                },
                            },
                            "elements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "description": "Element identifier (e.g. 'C1', 'B1')"},
                                        "type": {"type": "string", "enum": ["beam", "truss"], "default": "beam"},
                                        "start_node": {"type": "string", "description": "Start node ID"},
                                        "end_node": {"type": "string", "description": "End node ID"},
                                        "section_id": {"type": "string", "description": "Cross-section ID"},
                                        "material_id": {"type": "string", "description": "Material ID"},
                                    },
                                    "required": ["id", "start_node", "end_node"],
                                },
                            },
                            "supports": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "node_id": {"type": "string", "description": "Target node ID"},
                                        "fixity": {"type": "string", "enum": ["fixed", "pinned"], "default": "fixed"},
                                    },
                                    "required": ["node_id"],
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_structural_model",
            "description": "Directly construct or augment structural geometry by defining nodes, elements, and supports from structured JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "List of grid nodes with id, x, y, z coordinates",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number", "default": 0.0},
                            },
                            "required": ["id", "x", "y"],
                        },
                    },
                    "elements": {
                        "type": "array",
                        "description": "List of structural members (columns, beams, trusses)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string", "enum": ["beam", "truss"], "default": "beam"},
                                "start_node": {"type": "string"},
                                "end_node": {"type": "string"},
                                "section_id": {"type": "string"},
                                "material_id": {"type": "string"},
                            },
                            "required": ["id", "start_node", "end_node"],
                        },
                    },
                    "supports": {
                        "type": "array",
                        "description": "Boundary condition supports",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "fixity": {"type": "string", "enum": ["fixed", "pinned"], "default": "fixed"},
                            },
                            "required": ["node_id"],
                        },
                    },
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
            "name": "apply_self_weight",
            "description": "Materialize element self-weight distributed gravity loads for any load case where include_self_weight=True (e.g. from SAP2000 SelfWtMult=1 or custom dead load cases).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_case_id": {"type": "string", "description": "Optional specific load case ID to receive self-weight if no case currently has include_self_weight=True (default: 'DEAD')"},
                    "unit_system": {"type": "string", "enum": ["si", "us"], "description": "Optional unit system override ('si' or 'us')"},
                    "direction": {"type": "string", "enum": ["-z", "-y"], "description": "Optional gravity direction ('-z' for 3D frames, '-y' for 2D plane frames)"},
                },
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
                    "include_self_weight": {
                        "type": "boolean",
                        "description": "Whether to automatically compute and include element self-weight for cases with include_self_weight=True (default: True)",
                        "default": True,
                    },
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
            "name": "run_time_history",
            "description": "Execute dynamic earthquake time-history analysis using structdyn (Newmark-Beta / Central Difference) on 2D or 3D models with peak dynamic force enveloping for ACI 318 design.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_case_id": {"type": "string", "description": "Unique ID for storing time-history results (e.g. 'TH_ElCentro')"},
                    "earthquake": {"type": "string", "description": "Earthquake record preset, e.g. 'elcentro'", "default": "elcentro"},
                    "direction": {"type": "string", "enum": ["x", "y", "z"], "description": "Excitation direction", "default": "x"},
                    "dt": {"type": "number", "description": "Time step in seconds (default: 0.02)"},
                    "scale_factor": {"type": "number", "description": "Acceleration scale factor (auto-detected from model length unit if omitted)"},
                    "damping_ratio": {"type": "number", "description": "Modal damping ratio (e.g. 0.05 for 5% damping)", "default": 0.05},
                    "n_modes_damping": {"type": "integer", "description": "Number of modes for damping matrix calculation", "default": 3},
                    "method": {"type": "string", "enum": ["newmark_beta", "central_difference"], "description": "Numerical time integration method", "default": "newmark_beta"},
                    "solver": {"type": "string", "enum": ["auto", "fem2d", "fem3d"], "description": "Solver selection", "default": "auto"},
                    "envelope_to_design": {"type": "boolean", "description": "Whether to envelope peak dynamic forces across all time steps for downstream ACI 318 design", "default": True},
                },
                "required": ["analysis_case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dynamic_results",
            "description": "Inspect dynamic time-history results: time duration, peak roof drift, base shear, peak member forces, and fundamental periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_case_id": {"type": "string", "description": "Identifier of the dynamic analysis case to inspect"},
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
            "name": "inspect_detailing_results",
            "description": "Inspect structural detailing results, bar bending schedules (BBS), and rebar tonnage breakdowns across members and footings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "design_case_id": {"type": "string", "description": "Optional identifier of the design check / detailing case to inspect"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_drawings",
            "description": "Generate production-grade CAD structural drawings in DXF format. In 'auto' mode (default), generates a complete composite structural drawing set containing the General Arrangement floor framing plan, frame elevations in each orthogonal direction (for 3D buildings), and reinforced concrete rebar detailing (if design was performed). Standard drafting scales (1:100 plan, 1:50 elevation, 1:20 detailing) and engineering unit systems are applied automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Target file path for the .dxf drawing (e.g. 'structural_drawings.dxf')"},
                    "drawing_type": {
                        "type": "string",
                        "enum": ["auto", "detailing", "framing"],
                        "description": "Drawing type: 'auto' (default: complete composite sheet with framing plan, elevations in each direction, and detailing if designed), 'framing' (GA framing plans & elevations only), or 'detailing' (RC rebar detailing only)",
                        "default": "auto",
                    },
                    "design_case_id": {"type": "string", "description": "Design case identifier when generating detailing drawings (e.g. 'ACI318_19')"},
                    "view_mode": {
                        "type": "string",
                        "enum": ["both", "plan", "elevation"],
                        "description": "View mode for GA framing drawings: 'both' (combined framing plan & elevation sheet), 'plan' (floor framing plan), 'elevation' (frame elevation along a grid line)",
                        "default": "both",
                    },
                    "story_level": {"type": "string", "description": "Optional story name or level index for framing plan (e.g. 'FL1', 'ROOF')"},
                    "grid_line": {"type": "string", "description": "Optional grid line name for frame elevation (e.g. '1', 'A')"},
                    "title": {"type": "string", "description": "Optional custom drawing title"},
                },
                "required": ["output_path"],
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
