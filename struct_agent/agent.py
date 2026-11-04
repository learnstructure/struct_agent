"""
struct_agent.agent — Thin orchestrator over the struct_agent toolset.

This is the user-facing entry point. It does **not** call an LLM and does
**not** perform engineering calculations. It exists to:

1. Carry the workflow state machine through the full canonical pipeline.
2. Validate intermediate results at every stage.
3. Build a canonical demonstration project so the prototype can be run
   deterministically without any AI framework attached.

When this orchestrator is wrapped by an LLM framework, the LLM only has
to decide which tool to call next; it never has to re-implement
engineering logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from struct_core import (
    BeamElement,
    ConcreteMaterial,
    DistributedLoad,
    LinearStaticAnalysis,
    LoadCase,
    Node,
    Project,
    RebarMaterial,
    RectangularSection,
    StructuralModel,
    Support,
)

from .state.project_state import (
    ProjectState,
    StageTransitionError,
    WorkflowStage,
)
from .tools.base import AgentError, ToolResult
from .tools.model import (
    create_project,
    inspect_project,
    modify_project,
    save_project,
    attach_state,
)
from .tools.loads import apply_gravity_loads, generate_load_combinations
from .tools.analysis import run_analysis, inspect_analysis_results
from .tools.design import run_design, inspect_design_results, inspect_detailing_results
from .tools.drawing import generate_drawings
from .validation.model import validate_model


class StructAgent:
    """
    Thin, deterministic orchestrator over the engineering libraries.

    Usage
    -----

    .. code-block:: python

        agent = StructAgent()
        state = agent.run_demo_pipeline(output_dxf="out.dxf")
        print(state.stage)          # "DRAWINGS_GENERATED"
        print(state.tool_log[-1])   # last tool invocation

    For LLM-driven usage, call the individual tools (which are also
    exposed at package level) and use :class:`ProjectState` to track
    progress and refuse invalid transitions.
    """

    DEFAULT_TITLE = "struct_agent Demo: RC Portal Frame"
    DEFAULT_AUTHOR = "struct_agent orchestrator"

    def __init__(self) -> None:
        self.state: Optional[ProjectState] = None

    # ------------------------------------------------------------------ #
    # High-level demo pipeline
    # ------------------------------------------------------------------ #
    def run_demo_pipeline(
        self,
        *,
        output_dxf: Optional[str] = None,
        analysis_case_id: str = "Static_1",
        design_case_id: str = "ACI318_19",
        save_json_path: Optional[str] = None,
    ) -> ProjectState:
        """
        Build a small RC portal-frame project, apply ASCE 7 loads, validate it,
        run fem2d (with combination enveloping), run aci318, and render a DXF
        via struct_draw — with validation at every stage.

        Returns the final :class:`ProjectState`. Raises an exception if
        any stage fails so the caller cannot silently ignore errors.
        """
        structural_model = self._build_demo_model()
        created = create_project(
            structural_model,
            title=self.DEFAULT_TITLE,
            author=self.DEFAULT_AUTHOR,
        )
        self._require(created, "create_project")
        state = attach_state(created.data["project"])
        state.advance_to(WorkflowStage.MODEL_CREATED)
        state.log_tool("create_project", created.status, created.data)

        # Apply realistic building code gravity loads
        gravity = apply_gravity_loads(
            state.project,
            occupancy="office",
            tributary_width=5.0,
            superimposed_dead=1.5,
            include_self_weight=True,
        )
        self._require(gravity, "apply_gravity_loads")
        state.advance_to(WorkflowStage.LOADS_APPLIED)
        state.log_tool("apply_gravity_loads", gravity.status, gravity.data)

        # Generate standard ACI 318 / ASCE 7 factored load combinations
        combinations = generate_load_combinations(state.project, standard="ACI318_LRFD")
        self._require(combinations, "generate_load_combinations")
        state.log_tool("generate_load_combinations", combinations.status, combinations.data)

        validated = validate_model(state.project)
        self._require(validated, "validate_model")
        state.advance_to(WorkflowStage.MODEL_VALIDATED)
        state.log_tool("validate_model", validated.status, validated.data)

        analyzed = run_analysis(
            state.project,
            analysis_case_id=analysis_case_id,
            analysis_type="static",
            load_case_id="Envelope",
        )
        self._require(analyzed, "run_analysis")
        state.advance_to(WorkflowStage.ANALYSIS_COMPLETE)
        state.log_tool("run_analysis", analyzed.status, analyzed.data)

        inspected = inspect_analysis_results(state.project, analysis_case_id)
        self._require(inspected, "inspect_analysis_results")
        state.log_tool("inspect_analysis_results", inspected.status, inspected.data)

        designed = run_design(
            state.project,
            design_case_id=design_case_id,
            design_code="ACI 318-19",
            analysis_case_id=analysis_case_id,
        )
        self._require(designed, "run_design")
        state.advance_to(WorkflowStage.DESIGN_COMPLETE)
        state.advance_to(WorkflowStage.DETAILING_COMPLETE)
        state.log_tool("run_design", designed.status, designed.data)

        inspected_design = inspect_design_results(state.project, design_case_id)
        self._require(inspected_design, "inspect_design_results")
        state.log_tool("inspect_design_results", inspected_design.status, inspected_design.data)

        inspected_detailing = inspect_detailing_results(state.project, design_case_id)
        self._require(inspected_detailing, "inspect_detailing_results")
        state.log_tool("inspect_detailing_results", inspected_detailing.status, inspected_detailing.data)

        if output_dxf:
            drawn = generate_drawings(
                state.project,
                design_case_id=design_case_id,
                output_path=output_dxf,
            )
            self._require(drawn, "generate_drawings")
            state.advance_to(WorkflowStage.DRAWINGS_GENERATED)
            state.log_tool("generate_drawings", drawn.status, drawn.data)

        if save_json_path:
            saved = save_project(state.project, save_json_path)
            self._require(saved, "save_project")
            state.log_tool("save_project", saved.status, saved.data)

        self.state = state
        return state

    # ------------------------------------------------------------------ #
    # Canonical demo model (matches the existing ecosystem example).
    # ------------------------------------------------------------------ #
    def _build_demo_model(self) -> StructuralModel:
        nodes = [
            Node(id=1, x=0.0, y=0.0),
            Node(id=2, x=0.0, y=3.6),
            Node(id=3, x=6.0, y=3.6),
            Node(id=4, x=6.0, y=0.0),
        ]

        # 4,000 psi concrete, Grade 60 bars, 1.5 in (38 mm) clear cover
        mat_conc = ConcreteMaterial(id="Concrete_4ksi", fc=27_579.0, E=24_800_000.0, rho=2400.0, unit_weight=23.56)
        mat_rebar = RebarMaterial(id="Grade60", fy=413_685.0, fu=620_528.0, E=199_948_000.0, unit_weight=76.97)
        rc = dict(material_id="Concrete_4ksi", rebar_material_id="Grade60", transverse_rebar_material_id="Grade60",
                  cover=0.0381)
        sec_beam = RectangularSection(id="Beam_12x20", b=0.30, h=0.50, **rc)
        sec_col = RectangularSection(id="Col_16x16", b=0.40, h=0.40, **rc)

        elements = [
            BeamElement(id=1, start_node=1, end_node=2, section_id="Col_16x16"),
            BeamElement(id=2, start_node=2, end_node=3, section_id="Beam_12x20"),
            BeamElement(id=3, start_node=4, end_node=3, section_id="Col_16x16"),
        ]

        supports = [Support.fixed(node_id=1), Support.fixed(node_id=4)]
        loads = LoadCase(
            id="Gravity",
            element_loads=[DistributedLoad(element_id=2, wy=-25.0)],
        )
        analysis_case = LinearStaticAnalysis(id="Static_1", load_case_id="Gravity")

        return StructuralModel(
            nodes=nodes,
            materials=[mat_conc, mat_rebar],
            sections=[sec_beam, sec_col],
            elements=elements,
            supports=supports,
            load_cases=[loads],
            analysis_cases=[analysis_case],
        )

    # ------------------------------------------------------------------ #
    # Failure handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require(result: ToolResult, tool_name: str) -> None:
        """Raise a structured error if a tool returned a non-success result."""
        if result.ok:
            return
        err = result.error or AgentError(
            error_type="tool_failure",
            message=f"{tool_name} returned status {result.status!r} without an error payload.",
        )
        raise RuntimeError(
            f"struct_agent stage {tool_name!r} failed: {err.error_type}: {err.message}"
        )