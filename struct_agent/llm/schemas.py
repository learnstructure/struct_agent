"""
OpenAI-compatible JSON Schema definitions for all struct_agent engineering tools.

These schemas are passed to any function-calling capable LLM (OpenAI, Gemini, Claude, Ollama).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

ENGINEERING_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_frame_model",
            "description": "Create a fully connected 2D or 3D multi-bay, multi-story reinforced concrete frame with grid nodes, columns, beams and supports. The concrete strength fc is required; rebar (Grade 60), unit weight (150 pcf), clear cover (1.5 in) and the trial member sizes take documented defaults when not given, and the result lists every default used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_bays": {"type": "integer", "description": "Number of horizontal bays (or give bay_widths_x)"},
                    "bay_width": {"type": "number", "description": "Uniform bay width in model length units (or give bay_widths_x)"},
                    "num_stories": {"type": "integer", "description": "Number of vertical stories (or give story_heights)"},
                    "story_height": {"type": "number", "description": "Uniform story height in model length units (or give story_heights)"},
                    "dimension": {"type": "string", "enum": ["2d", "3d"], "description": "Dimensionality: '2d' for plane frame, '3d' for space frame", "default": "2d"},
                    "support_type": {"type": "string", "enum": ["fixed", "pinned"], "description": "Base boundary conditions", "default": "fixed"},
                    "title": {"type": "string", "description": "Project title (e.g. '2-Bay 2-Story RC Frame')"},
                    "col_b": {"type": "number", "description": "Column cross-section width in model length units (trial size 0.40 m when omitted)"},
                    "col_h": {"type": "number", "description": "Column cross-section depth in model length units (trial size 0.40 m when omitted)"},
                    "beam_b": {"type": "number", "description": "Beam cross-section width in model length units (trial size 0.30 m when omitted)"},
                    "beam_h": {"type": "number", "description": "Beam cross-section depth in model length units (trial size 0.50 m when omitted)"},
                    "bays_x": {"type": "integer", "description": "Number of bays along X axis (for 3D or 2D, defaults to num_bays)"},
                    "bays_y": {"type": "integer", "description": "Number of bays along Y axis (for 3D, defaults to num_bays)"},
                    "bay_width_x": {"type": "number", "description": "Bay width along X axis in model length units (defaults to bay_width)"},
                    "bay_width_y": {"type": "number", "description": "Bay width along Y axis in model length units (defaults to bay_width)"},
                    "bay_widths_x": {"type": "array", "items": {"type": "number"}, "description": "Non-uniform bay widths along X, left to right (e.g. [5.0, 6.0, 4.0]); overrides num_bays/bay_width"},
                    "bay_widths_y": {"type": "array", "items": {"type": "number"}, "description": "Non-uniform bay widths along Y for 3D frames (e.g. [6.0, 4.0]); overrides bays_y/bay_width_y"},
                    "story_heights": {"type": "array", "items": {"type": "number"}, "description": "Story heights from the ground story up (e.g. [4.0, 3.5, 3.5]); overrides num_stories/story_height"},
                    "fc": {"type": "number", "description": "Concrete compressive strength f'c, strictly in the model stress unit (e.g. kPa for 'kN, m'; ksi for 'kip, in'). Required."},
                    "fy": {"type": "number", "description": "Longitudinal rebar yield strength fy in the model stress unit (default Grade 60, 60 ksi)"},
                    "fyt": {"type": "number", "description": "Transverse (tie / stirrup) rebar yield strength in the model stress unit (default fy)"},
                    "unit_weight": {"type": "number", "description": "Concrete unit weight in the model weight-density unit (default 150 pcf / 23.6 kN/m3)"},
                    "cover": {"type": "number", "description": "Clear cover to the ties / stirrups in the model length unit, e.g. 0.04 for 40 mm in a kN, m model (default 1.5 in = 0.0381 m)"},
                    "units": {"type": "string", "description": "Canonical engineering unit pair, e.g. 'kN, m' or 'kip, in' (default: 'kN, m')", "default": "kN, m"},
                },
                "required": ["fc"],
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
                    "units": {"type": "string", "description": "Unit system specification, e.g. 'kN, m' or 'kip, in' (default: 'kN, m')", "default": "kN, m"},
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
                                "height": {"type": "number", "description": "Story height (in meters or feet depending on units)"},
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
                    "default_col_b": {"type": "number", "description": "Width of every column without its own section_b, in model length units (default 0.45 m)"},
                    "default_col_h": {"type": "number", "description": "Depth of every column without its own section_h, in model length units (default 0.45 m)"},
                    "default_beam_b": {"type": "number", "description": "Width of every beam without its own section_b, in model length units (default 0.30 m)"},
                    "default_beam_h": {"type": "number", "description": "Depth of every beam without its own section_h, in model length units (default 0.50 m)"},
                    "fc": {"type": "number", "description": "Concrete compressive strength f'c, strictly in the draft's stress unit (e.g. kPa for 'kN, m'). Required."},
                    "fy": {"type": "number", "description": "Longitudinal rebar yield strength in the draft's stress unit (default Grade 60)"},
                    "fyt": {"type": "number", "description": "Transverse rebar yield strength in the draft's stress unit (default fy)"},
                    "unit_weight": {"type": "number", "description": "Concrete unit weight in the draft's weight-density unit (default 150 pcf / 23.6 kN/m3)"},
                    "cover": {"type": "number", "description": "Clear cover in the model length unit, e.g. 0.04 for 40 mm in a kN, m model (default 1.5 in = 0.0381 m)"},
                    "base_support": {"type": "string", "enum": ["fixed", "pinned"], "default": "fixed"},
                },
                "required": ["x_grids", "y_grids", "stories", "fc"],
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
                    "units": {"type": "string", "description": "Canonical engineering unit pair, e.g. 'kN, m' or 'kip, in' (default: 'kN, m')", "default": "kN, m"},
                    "dimension": {"type": "string", "enum": ["2D", "3D"], "description": "Model dimension: '2D' (X-Y plane) or '3D' (X-Y-Z space)."},
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
                                        "x": {"type": "number", "description": "X coordinate in the model length unit"},
                                        "y": {"type": "number", "description": "Y coordinate in the model length unit"},
                                        "z": {"type": "number", "description": "Z coordinate in the model length unit", "default": 0.0},
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
                                        "section_id": {"type": "string", "description": "Cross-section ID (the section carries the material)"},
                                    },
                                    "required": ["id", "start_node", "end_node", "section_id"],
                                },
                            },
                            "materials": {
                                "type": "array",
                                "description": "Materials, in the model units: concrete (type 'concrete', fc), reinforcing steel (type 'rebar'), or elastic (E)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                    "id": {"type": "string", "description": "Material identifier (e.g. 'C35', 'A615Gr60')"},
                                    "type": {"type": "string", "enum": ["concrete", "rebar", "elastic"], "description": "'concrete' (needs fc), 'rebar' (reinforcing steel, Grade 60 unless fy is given) or 'elastic' (needs E)"},
                                    "fc": {"type": "number", "description": "Concrete compressive strength f'c (project stress unit)"},
                                    "fy": {"type": "number", "description": "Rebar yield strength fy (project stress unit)"},
                                    "fu": {"type": "number", "description": "Rebar tensile strength fu"},
                                    "E": {"type": "number", "description": "Elastic modulus (default for concrete: from f'c, ACI 318-25 §19.2.2.1; for rebar 29,000 ksi)"},
                                    "nu": {"type": "number", "description": "Poisson's ratio"},
                                    "unit_weight": {"type": "number", "description": "Unit weight (project weight-density unit; default 150 pcf concrete, 490 pcf steel)"},
                                    },
                                    "required": ["id"],
                                },
                            },
                            "sections": {
                                "type": "array",
                                "description": "Cross-sections, model length units: rectangular (b, h), circular (d) or general (A, I3, I2, J); each with its material_id. Concrete sections take Grade 60 rebar and 1.5 in cover unless given.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "type": {"type": "string", "enum": ["rectangular", "circular", "general"]},
                                        "b": {"type": "number"},
                                        "h": {"type": "number"},
                                        "d": {"type": "number"},
                                        "A": {"type": "number"},
                                        "I3": {"type": "number"},
                                        "I2": {"type": "number"},
                                        "J": {"type": "number"},
                                    "material_id": {"type": "string", "description": "The section's material (required for a new section)"},
                                    "rebar_material_id": {"type": "string", "description": "Longitudinal rebar material of a concrete section"},
                                    "transverse_rebar_material_id": {"type": "string", "description": "Tie / stirrup rebar material of a concrete section"},
                                    "cover": {"type": "number", "description": "Clear cover of a concrete section in the model length unit (e.g. 0.04 for 40 mm in a kN, m model)"},
                                    },
                                    "required": ["id", "material_id"],
                                },
                            },
                            "supports": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "node_id": {"type": "string", "description": "Target node ID"},
                                        "fixity": {"type": "string", "enum": ["fixed", "pinned", "roller"], "default": "fixed"},
                                        "kx": {"type": "number", "description": "Translational spring along X in model units (force/length)"},
                                        "ky": {"type": "number", "description": "Translational spring along Y in model units (force/length)"},
                                        "kz": {"type": "number", "description": "Translational spring along Z in model units (force/length)"},
                                        "krx": {"type": "number", "description": "Rotational spring about X in model units (moment/rad)"},
                                        "kry": {"type": "number", "description": "Rotational spring about Y in model units (moment/rad)"},
                                        "krz": {"type": "number", "description": "Rotational spring about Z in model units (moment/rad)"},
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
            "name": "set_project_units",
            "description": "Convert an existing project to a different unit system (e.g. from 'kip, in' to 'kN, m'). Automatically converts coordinates, loads, sections, materials, and results if convert_existing is True. Note: Declare initial units for new models directly via create_project or create_frame_model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "units": {"type": "string", "description": "Target canonical unit pair, e.g. 'kN, m' or 'kip, in'"},
                    "convert_existing": {"type": "boolean", "description": "Whether to convert existing numeric coordinates, loads, and properties (default: True)", "default": True},
                },
                "required": ["units"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_project",
            "description": "Modify the active structural model: update or enlarge cross-sections (rectangular, circular, or general shapes) for iterative reflection and member optimization loops, reassign members to different sections, or update materials. A section owns its material: to change a member's concrete, change its section's material (or the material itself). Values are strictly in the project units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "description": "Cross-sections to update or create (supports rectangular, circular, or general profiles)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "ID of an existing section to resize, as listed in the model creation result's section_ids (e.g. 'COL', 'BEAM'). Resizing keeps the ID. A new ID must be assigned to members via member_assignments in the same call."},
                                "type": {"type": "string", "enum": ["rectangular", "circular", "general"], "description": "Optional section shape type (default: inferred from provided dimensions)"},
                                "b": {"type": "number", "description": "Width in the model length unit (for rectangular sections, e.g. 0.45)"},
                                "h": {"type": "number", "description": "Depth/height in the model length unit (for rectangular sections, e.g. 0.55)"},
                                "d": {"type": "number", "description": "Diameter in the model length unit (for circular sections, e.g. 0.50 for 500 mm)"},
                                "A": {"type": "number", "description": "Cross-sectional area in length^2 of the model units (for general sections)"},
                                "I3": {"type": "number", "description": "Major axis second moment of area in length^4 of the model units (for general sections)"},
                                "I2": {"type": "number", "description": "Minor axis second moment of area in length^4 of the model units (for general sections)"},
                                "J": {"type": "number", "description": "Torsional constant in length^4 of the model units (for general sections)"},
                                "material_id": {"type": "string", "description": "The section's material (required for a new section)"},
                                "rebar_material_id": {"type": "string", "description": "Longitudinal rebar material of a concrete section"},
                                "transverse_rebar_material_id": {"type": "string", "description": "Tie / stirrup rebar material of a concrete section"},
                                "cover": {"type": "number", "description": "Clear cover of a concrete section in the model length unit (e.g. 0.04 for 40 mm in a kN, m model)"},
                            },
                            "required": ["id"],
                        },
                    },
                    "member_assignments": {
                        "type": "array",
                        "description": "Optional member reassignments to different section profiles",
                        "items": {
                            "type": "object",
                            "properties": {
                                "element_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of element IDs (e.g. ['C1', 'C2'])",
                                },
                                "section_id": {"type": "string", "description": "Target cross-section ID to assign"},
                            },
                            "required": ["element_ids", "section_id"],
                        },
                    },
                    "materials": {
                        "type": "array",
                        "description": "Optional materials to update or add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Material identifier (e.g. 'C35', 'A615Gr60')"},
                                "type": {"type": "string", "enum": ["concrete", "rebar", "elastic"], "description": "'concrete' (needs fc), 'rebar' (reinforcing steel, Grade 60 unless fy is given) or 'elastic' (needs E)"},
                                "fc": {"type": "number", "description": "Concrete compressive strength f'c strictly in the model stress unit"},
                                "fy": {"type": "number", "description": "Rebar yield strength fy strictly in the model stress unit"},
                                "fu": {"type": "number", "description": "Rebar tensile strength fu strictly in the model stress unit"},
                                "E": {"type": "number", "description": "Elastic modulus in the model stress unit (default for concrete: from f'c, ACI 318-25 §19.2.2.1; for rebar 29,000 ksi)"},
                                "nu": {"type": "number", "description": "Poisson's ratio"},
                                "unit_weight": {"type": "number", "description": "Unit weight in the model weight-density unit (default 150 pcf concrete, 490 pcf steel)"},
                            },
                            "required": ["id"],
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preliminary_sizing",
            "description": "Give the frame's column and beam sections rule-of-thumb starting sizes, keeping their IDs and materials: beams h = longest span / 14 and b = 0.55 h; square columns = tallest story / 10; rounded up to 0.05 m (2 in for US units). Self-weight and ELF loads are regenerated. Not for redesign after a design check.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_beam_width": {"type": "number", "description": "Least beam width, model length unit (default 0.25 m / 10 in)"},
                    "min_col_dim": {"type": "number", "description": "Least column dimension, model length unit (default 0.35 m / 14 in)"},
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
                        "description": "ASCE 7-22 Table 4.3-1 occupancy (sets the live load; 'office' adds the 15 psf partition allowance)",
                        "enum": [
                            "office", "residential", "classroom", "corridor_first_floor", "corridor_upper",
                            "retail_first_floor", "retail_upper", "storage_light", "storage_heavy",
                            "assembly_fixed", "assembly_movable", "parking", "roof_ordinary",
                        ],
                    },
                    "tributary_width": {"type": "number", "description": "Tributary width (model length units) for converting floor pressure to line loads; required for 2D frames, omit for 3D two-way floor panels"},
                    "one_way_span": {"type": "string", "enum": ["x", "y"], "description": "One-way slab span direction ('x' or 'y') for 3D frames; perpendicular beams receive load based on bay spacing."},
                    "superimposed_dead": {"type": "number", "description": "Superimposed (floor) dead load pressure, strictly in the model pressure unit (e.g. kPa for 'kN, m'; ksf for 'kip, ft', so 105 psf = 0.105 ksf); 0 for none. Required."},
                    "custom_live_load": {"type": "number", "description": "Live load pressure override in the model pressure unit; replaces the occupancy table value"},
                    "partition_weight": {"type": "number", "description": "Partition weight included in the effective seismic weight (ASCE 7-22 §12.7.2 item 2), model pressure unit. Defaults to the 10 psf (0.48 kPa) minimum for occupancies with a partition allowance; must be given when custom_live_load is used."},
                    "two_way": {"type": "boolean", "description": "Force two-way (45-degree yield-line) floor panel distribution in 3D frames"},
                    "include_self_weight": {"type": "boolean", "description": "Whether to automatically compute and apply member self-weight", "default": True},
                    "pressure_unit": {"type": "string", "description": "Pressure unit for superimposed_dead, custom_live_load, partition_weight (e.g. 'kPa', 'psf', 'ksf'); omit if in model pressure units"},
                    "length_unit": {"type": "string", "description": "Length unit for tributary_width (e.g. 'm', 'ft', 'in'); omit if in model length units"},
                    "allow_partial_floor": {"type": "boolean", "description": "Whether to allow floor panels that are not fully closed without raising an error", "default": False},
                },
                "required": ["occupancy", "superimposed_dead"],
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
                    "direction": {"type": "string", "enum": ["-z", "-y"], "description": "Optional gravity direction ('-z' for 3D frames, '-y' for 2D plane frames)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_lateral_loads",
            "description": "Calculate and apply seismic Equivalent Lateral Force (ELF) and/or wind lateral story forces per ASCE 7. Seismic loads need the site's sds and sd1; R comes from system_type (ASCE 7-22 Table 12.2-1) and Ie from risk_category (Table 1.5-2) unless given; risk_category defaults to II and site_class to D (reported). Wind loads need wind_speed and tributary_width.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Building standard", "default": "ASCE7-22"},
                    "sds": {"type": "number", "description": "Design spectral acceleration parameter SDS (g), from the site; required for seismic loads"},
                    "sd1": {"type": "number", "description": "Design spectral acceleration parameter SD1 (g), from the site; required for seismic loads"},
                    "s1": {"type": "number", "description": "Mapped MCE_R 1-second spectral acceleration parameter S1 (g)"},
                    "tl": {"type": "number", "description": "Mapped long-period transition period TL (seconds); needed only when the period exceeds 4 s"},
                    "r_factor": {"type": "number", "description": "Response modification coefficient R (default: ASCE 7-22 Table 12.2-1 value for system_type)"},
                    "importance_factor": {"type": "number", "description": "Seismic importance factor Ie (default: from risk_category, ASCE 7-22 Table 1.5-2)"},
                    "redundancy_factor": {"type": "number", "description": "Seismic redundancy factor rho (ASCE 7-22 §12.3.4, 1.0 or 1.3)", "default": 1.0},
                    "system_type": {"type": "string", "enum": ["rc_smf", "rc_imf", "rc_omf", "steel_smf", "steel_imf", "steel_omf", "steel_ebf", "steel_brbf", "all_other"], "description": "Seismic force-resisting system (ASCE 7-22 Tables 12.2-1 and 12.8-2): 'rc_smf', 'rc_imf', 'rc_omf' are special, intermediate and ordinary RC moment frames. It sets Ta, Cd for the drift checks, and the ACI 318 Chapter 18 frame type used by run_design.", "default": "rc_smf"},
                    "site_class": {"type": "string", "description": "ASCE 7 site class (A, B, C, D, E, F); default D (reported)"},
                    "risk_category": {"type": "string", "enum": ["I", "II", "III", "IV"], "description": "Risk Category (ASCE 7-22 Table 1.5-1); default II (reported)"},
                    "distribution": {"type": "string", "enum": ["face", "uniform"], "description": "Seismic force distribution across story nodes: 'face' for exterior perimeter nodes or 'uniform' across all floor nodes", "default": "face"},
                    "direction": {"type": "string", "enum": ["both", "x", "y"], "description": "Load direction", "default": "both"},
                    "wind_speed": {"type": "number", "description": "Basic wind speed in m/s (optional)"},
                    "tributary_width": {"type": "number", "description": "Loaded width of the frame for wind, model length unit (required with wind_speed)"},
                    "length_unit": {"type": "string", "description": "Length unit for tributary_width (e.g. 'm', 'ft', 'in'); omit if in model length units"},
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
                    "sds": {"type": "number", "description": "SDS (g) for the vertical seismic effect Ev = 0.2*SDS*D in the seismic combinations (ASCE 7-22 §12.4.2.2). Omit for no vertical seismic effect."},
                    "rho": {"type": "number", "description": "Redundancy factor rho on the horizontal seismic effect (ASCE 7-22 §12.3.4)", "default": 1.0},
                    "orthogonal_seismic": {"type": "boolean", "description": "Add the orthogonal combinations 1.0Ex ± 0.3Ey and 0.3Ex ± 1.0Ey (ASCE 7-22 §12.5) when seismic loads exist in both directions", "default": True},
                    "include_overstrength": {"type": "boolean", "description": "Add the overstrength combinations with Omega_0*QE (ASCE 7-22 §12.4.3); needs omega_0", "default": False},
                    "omega_0": {"type": "number", "description": "Overstrength factor Omega_0 (ASCE 7-22 Table 12.2-1), with include_overstrength"},
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
                        "description": "'static' for linear static / envelope, or 'modal' for natural periods, mode shapes and modal mass participation ratios (per mode and cumulative)",
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
                    "cracked_sections": {"type": "boolean", "description": "Use ACI 318-25 Table 6.6.3.1.1 cracked-section stiffness (0.70 Ig columns, 0.35 Ig beams), required for factored-load analysis of concrete frames and for seismic drift. With seismic loads from apply_lateral_loads, story drift (ASCE 7-22 §12.12.1) and stability (§12.8.7) are checked.", "default": False},
                    "second_order": {"type": "boolean", "description": "Include column slenderness by ACI 318-25 §6.6.4 moment magnification (sway delta_s and member delta, with §6.2.5.3 and §6.6.4.6.2 limits). Requires cracked_sections=true.", "default": False},
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
            "description": "Retrieve maximum axial forces, shear forces, bending moments, and modal periods and mass participation ratios from an analysis case.",
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
                    "ground_motion": {"type": "string", "description": "Ground motion record: a file path to a PEER '.AT2' or a two-column '.csv' (time [s], acceleration [g]) record, or a built-in event name, e.g. 'elcentro', 'loma_prieta', 'loma_prieta:hor2'"},
                    "component": {"type": "string", "enum": ["hor1", "hor2", "up"], "description": "Component of a built-in event (auto-inferred from direction if omitted; ignored for record files)"},
                    "direction": {"type": "string", "enum": ["x", "y", "z"], "description": "Excitation direction", "default": "x"},
                    "dt": {"type": "number", "description": "Time step in seconds for raw acceleration arrays only (record files carry their own dt)"},
                    "record_scale": {"type": "number", "description": "Dimensionless amplitude multiplier (default: 1.0; g to model-unit conversion is automatic)", "default": 1.0},
                    "damping_ratio": {"type": "number", "description": "Modal damping ratio (e.g. 0.05 for 5% damping)", "default": 0.05},
                    "n_modes_damping": {"type": "integer", "description": "Number of modes for damping matrix calculation", "default": 3},
                    "method": {"type": "string", "enum": ["newmark_beta", "central_difference"], "description": "Numerical time integration method", "default": "newmark_beta"},
                    "solver": {"type": "string", "enum": ["auto", "fem2d", "fem3d"], "description": "Solver selection", "default": "auto"},
                    "envelope_to_design": {"type": "boolean", "description": "Whether to envelope peak dynamic forces across all time steps for downstream ACI 318 design", "default": True},
                },
                "required": ["analysis_case_id", "ground_motion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_response_spectrum",
            "description": "Execute dynamic Response Spectrum Analysis (RSA) using structdyn (CQC / SRSS) on 2D or 3D models with ASCE 7 design spectra or ground motion records, with modal peak force enveloping for ACI 318 design.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_case_id": {"type": "string", "description": "Unique ID for storing RSA results (e.g. 'RSA_ASCE_X')", "default": "RSA_X"},
                    "sds": {"type": "number", "description": "ASCE 7 short-period design spectral acceleration SDS (in g)"},
                    "sd1": {"type": "number", "description": "ASCE 7 1-second design spectral acceleration SD1 (in g)"},
                    "tl": {"type": "number", "description": "Mapped long-period transition period TL (seconds); needed only for modes with periods above 4 s"},
                    "site_class": {"type": "string", "description": "ASCE 7 site class (A, B, C, D, E, F)", "default": "D"},
                    "ground_motion": {"type": "string", "description": "Ground motion record used to compute the 5%-damped spectrum: a file path to a PEER '.AT2' or a two-column '.csv' (time [s], acceleration [g]) record, or a built-in event name, e.g. 'elcentro', 'loma_prieta:hor2'"},
                    "component": {"type": "string", "enum": ["hor1", "hor2", "up"], "description": "Component of a built-in event (auto-inferred from direction if omitted; ignored for record files)"},
                    "direction": {"type": "string", "enum": ["x", "y", "z", "both"], "description": "Excitation direction. 'both' analyses X and Y separately and combines them in the design envelope with the 100%-30% rule (ASCE 7-22 §12.5.3, 1.0Ex ± 0.3Ey and 0.3Ex ± 1.0Ey); 'x', 'y', or 'z' gives a one-direction envelope", "default": "x"},
                    "combination": {"type": "string", "enum": ["CQC", "SRSS", "ABS"], "description": "Modal combination rule", "default": "CQC"},
                    "r_factor": {"type": "number", "description": "ASCE 7 seismic response modification coefficient R (e.g. 5.0 for Intermediate Moment Frames, 8.0 for Special Moment Frames). Elastic spectrum forces are divided by R/Ie. If omitted, forces stay elastic (no reduction). With an ASCE 7 design spectrum (sds/sd1), forces are then scaled up to the ELF base shear per ASCE 7-22 Section 12.9.1.4, which requires system_type."},
                    "importance_factor": {"type": "number", "description": "ASCE 7 seismic importance factor Ie (default: from risk_category, ASCE 7-22 Table 1.5-2: I/II 1.0, III 1.25, IV 1.5)"},
                    "system_type": {"type": "string", "enum": ["rc_smf", "rc_imf", "rc_omf", "steel_smf", "steel_imf", "steel_omf", "steel_ebf", "steel_brbf", "all_other"], "description": "Seismic force-resisting system (ASCE 7-22 Tables 12.2-1 and 12.8-2); required with r_factor. It sets Ta for the §12.9.1.4 scaling, Cd for the drift checks, and the ACI 318 Chapter 18 frame type used by run_design."},
                    "s1": {"type": "number", "description": "Mapped MCE_R spectral acceleration S1 (g), used for the Cs lower bound of ASCE 7-22 Eq. 12.8-7"},
                    "risk_category": {"type": "string", "enum": ["I", "II", "III", "IV"], "description": "Risk Category: the importance factor Ie (Table 1.5-2) and the allowable story drift (ASCE 7-22 Table 12.12-1)", "default": "II"},
                    "cracked_sections": {"type": "boolean", "description": "Use ACI 318-25 Table 6.6.3.1.1 cracked-section stiffness (also for periods, ASCE 7-22 §12.7.3); with r_factor and system_type, story drift (§12.12.1) and stability (§12.8.7) are checked", "default": False},
                    "second_order": {"type": "boolean", "description": "Include ACI 318-25 §6.6.4 moment magnification in the design envelope. Requires cracked_sections=true.", "default": False},
                    "n_modes": {"type": "integer", "description": "Number of modes to include (default: all or 12)"},
                    "damping_ratio": {"type": "number", "description": "Modal damping ratio (e.g. 0.05 for 5% damping)", "default": 0.05},
                    "spectrum_scale": {"type": "number", "description": "Dimensionless amplitude multiplier (default: 1.0; g to model-unit conversion is automatic)", "default": 1.0},
                    "solver": {"type": "string", "enum": ["auto", "fem2d", "fem3d"], "description": "Solver selection", "default": "auto"},
                    "envelope_to_design": {"type": "boolean", "description": "Whether to envelope modal forces for downstream ACI 318 design", "default": True},
                },
                "required": ["analysis_case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dynamic_results",
            "description": "Inspect dynamic time-history results: time duration, peak roof drift, base shear, peak member forces, and fundamental periods and modal mass participation ratios.",
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
            "description": "Design the reinforcement and run reinforced concrete code checks per ACI 318-25. Beams are reinforced for the analysis envelope; column bars are designed (1-8 % of Ag, 1-6 % for special moment frames) for axial/P-M strength and, in special moment frames, strong-column/weak-beam. Members use the materials and cover of their sections. Isolated footings are sized for allowable_bearing_pressure from the service-level reactions (ASCE 7-22 §2.4) and designed for strength from the factored ones. Checks: flexure, shear, torsion, axial/P-M interaction, beam ductility, beam-column joint shear (Chapter 15), the Chapter 18 checks of the seismic frame type recorded by apply_lateral_loads / run_response_spectrum (special: capacity-design shear, strong-column/weak-beam, confinement, joints; intermediate: capacity shear), the analysis frame checks (slenderness, story drift, stability), and detailing reinforcement schedules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "design_case_id": {"type": "string", "description": "Identifier for the design check results", "default": "ACI318_25"},
                    "design_code": {"type": "string", "description": "Design code edition", "default": "ACI 318-25"},
                    "analysis_case_id": {"type": "string", "description": "Which analysis case internal forces to design for"},
                    "allowable_bearing_pressure": {"type": "number", "description": "Allowable soil bearing pressure for the isolated footings under service loads, strictly in the model pressure unit (e.g. kPa for 'kN, m'; ksf for 'kip, ft'). Default 150 kPa, reported as a preliminary assumption."},
                    "footing_cover": {"type": "number", "description": "Clear cover of the footing reinforcement in the model length unit (default 3 in = 0.0762 m, concrete cast against earth)"},
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
            "description": "Import an existing CSI SAP2000 model text file (.s2k / .$2k) into the Project workspace. Materials are read as the file defines them; the result lists what design still needs (before_design). Give the reinforcement the file lacks with fy (and fyt, cover).",
            "parameters": {
                "type": "object",
                "properties": {
                    "s2k_path": {"type": "string", "description": "Path to the .s2k file"},
                    "project_title": {"type": "string", "description": "Optional custom project title"},
                    "fy": {"type": "number", "description": "Yield strength of the reinforcement for concrete sections the file gives no bars for, strictly in the model stress unit"},
                    "fyt": {"type": "number", "description": "Yield strength of their ties / stirrups in the model stress unit (default fy)"},
                    "cover": {"type": "number", "description": "Clear cover of those sections in the model length unit (default 1.5 in = 0.0381 m)"},
                    "allow_partial_import": {"type": "boolean", "description": "Allow importing partial model when the file contains unsupported tables (e.g. springs, areas), recording unhandled items in notes", "default": False},
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
                    "allow_partial_export": {"type": "boolean", "description": "Allow exporting without unsupported features (e.g. dropping support springs)", "default": False},
                },
                "required": ["output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "import_from_etabs",
            "description": "Import an existing CSI ETABS model text file (.e2k / .$et) into the Project workspace. Materials are read as the file defines them; the result lists what design still needs (before_design). Give the reinforcement the file lacks with fy (and fyt, cover).",
            "parameters": {
                "type": "object",
                "properties": {
                    "e2k_path": {"type": "string", "description": "Path to the .e2k file"},
                    "project_title": {"type": "string", "description": "Optional custom project title"},
                    "fy": {"type": "number", "description": "Yield strength of the reinforcement for concrete sections the file gives no bars for, strictly in the model stress unit"},
                    "fyt": {"type": "number", "description": "Yield strength of their ties / stirrups in the model stress unit (default fy)"},
                    "cover": {"type": "number", "description": "Clear cover of those sections in the model length unit (default 1.5 in = 0.0381 m)"},
                    "allow_partial_import": {"type": "boolean", "description": "Allow importing partial model when the file contains unsupported tables (e.g. springs, areas), recording unhandled items in notes", "default": False},
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
                    "allow_partial_export": {"type": "boolean", "description": "Allow exporting without unsupported features (e.g. dropping support springs)", "default": False},
                },
                "required": ["output_path"],
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return all tool schemas formatted for LLM function calling."""
    return list(ENGINEERING_TOOL_SCHEMAS)


_PARAMETER_TYPES: Dict[str, Dict[str, Any]] = {
    t["function"]["name"]: {k: v.get("type") for k, v in t["function"]["parameters"].get("properties", {}).items()}
    for t in ENGINEERING_TOOL_SCHEMAS
}
_OMITTED = {"none", "null", ""}


def normalize_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Top-level scalar arguments decoded by their schema type: models sometimes send JSON scalars as strings.

    - 'None', 'null', or '' for a boolean, number, or integer parameter: the argument is omitted;
    - 'true' / 'false' for a boolean, a numeric string for a number or integer: converted;
    - any other string for those types: an error message (the call is refused, never guessed).

    Returns (arguments, error message or None). Other parameters pass through unchanged.
    """
    declared = _PARAMETER_TYPES.get(tool_name, {})
    out: Dict[str, Any] = {}
    for key, value in arguments.items():
        kind = declared.get(key)
        if kind not in ("boolean", "number", "integer") or not isinstance(value, str):
            out[key] = value
            continue
        text = value.strip().lower()
        if text in _OMITTED:
            continue
        if kind == "boolean":
            if text not in ("true", "false"):
                return arguments, f"Argument {key!r} of {tool_name} must be true or false; got {value!r}."
            out[key] = text == "true"
            continue
        try:
            number = float(text)
        except ValueError:
            return arguments, f"Argument {key!r} of {tool_name} must be a number; got {value!r}."
        if kind == "integer":
            if not number.is_integer():
                return arguments, f"Argument {key!r} of {tool_name} must be an integer; got {value!r}."
            out[key] = int(number)
        else:
            out[key] = number
    return out, None
