"""
Tests for structdyn dynamic analysis integration in struct_agent.

Covers:
- 2D & 3D modal analysis via structdyn MDF (run_analysis with analysis_type="modal")
- 2D & 3D earthquake time-history analysis (run_time_history with El Centro)
- Dynamic force enveloping across all time steps for downstream ACI 318 design
- Direct run_design execution against dynamic earthquake envelopes
- inspect_dynamic_results reporting
- AgentRunner tool dispatch for dynamics
"""

import numpy as np
import pytest

from struct_core import (
    BeamElement,
    ElasticMaterial,
    Node,
    PointLoad,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
)
from struct_agent.llm.runner import AgentRunner
from struct_agent.tools.analysis import inspect_analysis_results, run_analysis
from struct_agent.tools.design import inspect_design_results, run_design
from struct_agent.tools.dynamics import (
    inspect_dynamic_results,
    run_response_spectrum,
    run_time_history,
)
from struct_agent.tools.model import create_project


def _build_2d_frame_project():
    """Build a 2-story 1-bay 2D frame with lumped story mass."""
    # Nodes: Bay width = 5m, Story height = 3m
    nodes = [
        Node(id="N1", x=0.0, y=0.0),
        Node(id="N2", x=5.0, y=0.0),
        Node(id="N3", x=0.0, y=3.0, mass=5000.0),
        Node(id="N4", x=5.0, y=3.0, mass=5000.0),
        Node(id="N5", x=0.0, y=6.0, mass=4000.0),
        Node(id="N6", x=5.0, y=6.0, mass=4000.0),
    ]

    supports = [
        Support(node_id="N1", ux=True, uy=True, rz=True),
        Support(node_id="N2", ux=True, uy=True, rz=True),
    ]

    materials = [
        ElasticMaterial(id="CONC", E=2.5e7, nu=0.2, rho=2400.0),
    ]

    sections = [
        RectangularSection(id="COL", b=0.4, h=0.4),
        RectangularSection(id="BEAM", b=0.3, h=0.5),
    ]

    elements = [
        # Columns Story 1
        BeamElement(id="C1", start_node="N1", end_node="N3", material_id="CONC", section_id="COL"),
        BeamElement(id="C2", start_node="N2", end_node="N4", material_id="CONC", section_id="COL"),
        # Beams Story 1
        BeamElement(id="B1", start_node="N3", end_node="N4", material_id="CONC", section_id="BEAM"),
        # Columns Story 2
        BeamElement(id="C3", start_node="N3", end_node="N5", material_id="CONC", section_id="COL"),
        BeamElement(id="C4", start_node="N4", end_node="N6", material_id="CONC", section_id="COL"),
        # Beams Story 2
        BeamElement(id="B2", start_node="N5", end_node="N6", material_id="CONC", section_id="BEAM"),
    ]

    model = StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
    )
    return create_project(model, title="2D Dynamic Test Frame").data["project"]


def _build_3d_frame_project():
    """Build a 1-story 1-bay 3D space frame with lumped floor mass."""
    nodes = [
        Node(id="N1", x=0.0, y=0.0, z=0.0),
        Node(id="N2", x=5.0, y=0.0, z=0.0),
        Node(id="N3", x=0.0, y=4.0, z=0.0),
        Node(id="N4", x=5.0, y=4.0, z=0.0),
        Node(id="N5", x=0.0, y=0.0, z=3.0, mass=2000.0),
        Node(id="N6", x=5.0, y=0.0, z=3.0, mass=2000.0),
        Node(id="N7", x=0.0, y=4.0, z=3.0, mass=2000.0),
        Node(id="N8", x=5.0, y=4.0, z=3.0, mass=2000.0),
    ]

    supports = [
        Support(node_id="N1", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        Support(node_id="N2", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        Support(node_id="N3", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        Support(node_id="N4", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
    ]

    materials = [
        ElasticMaterial(id="CONC", E=2.5e7, nu=0.2, rho=2400.0),
    ]

    sections = [
        RectangularSection(id="COL", b=0.4, h=0.4),
        RectangularSection(id="BEAM", b=0.3, h=0.5),
    ]

    elements = [
        # Columns
        BeamElement(id="C1", start_node="N1", end_node="N5", material_id="CONC", section_id="COL"),
        BeamElement(id="C2", start_node="N2", end_node="N6", material_id="CONC", section_id="COL"),
        BeamElement(id="C3", start_node="N3", end_node="N7", material_id="CONC", section_id="COL"),
        BeamElement(id="C4", start_node="N4", end_node="N8", material_id="CONC", section_id="COL"),
        # Beams
        BeamElement(id="B1", start_node="N5", end_node="N6", material_id="CONC", section_id="BEAM"),
        BeamElement(id="B2", start_node="N7", end_node="N8", material_id="CONC", section_id="BEAM"),
        BeamElement(id="B3", start_node="N5", end_node="N7", material_id="CONC", section_id="BEAM"),
        BeamElement(id="B4", start_node="N6", end_node="N8", material_id="CONC", section_id="BEAM"),
    ]

    model = StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
    )
    return create_project(model, title="3D Dynamic Test Frame").data["project"]


def test_2d_modal_analysis_structdyn():
    """Verify 2D modal analysis solves via structdyn MDF."""
    proj = _build_2d_frame_project()
    res = run_analysis(proj, analysis_case_id="Modal_2D", analysis_type="modal", solver="auto", num_modes=3)
    assert res.ok, f"Modal analysis failed: {res.error}"
    assert res.data["solver_used"] == "fem2d"
    assert res.data["n_modes"] == 3
    assert res.data["fundamental_period"] > 0.0

    # Inspect modal result
    inspect = inspect_analysis_results(proj, "Modal_2D")
    assert inspect.ok
    assert "modal" in inspect.data
    periods = inspect.data["modal"]["periods"]
    assert len(periods) == 3
    # Fundamental period is highest
    assert periods[0] >= periods[1] >= periods[2]
    assert all(p > 0 for p in periods)


def test_3d_modal_analysis_structdyn():
    """Verify 3D modal analysis solves via structdyn MDF."""
    proj = _build_3d_frame_project()
    res = run_analysis(proj, analysis_case_id="Modal_3D", analysis_type="modal", solver="auto", num_modes=3)
    assert res.ok, f"Modal analysis failed: {res.error}"
    assert res.data["solver_used"] == "fem3d"
    assert res.data["n_modes"] == 3
    assert res.data["fundamental_period"] > 0.0


def test_2d_time_history_elcentro():
    """Verify 2D transient dynamic response under El Centro ground motion."""
    proj = _build_2d_frame_project()
    res = run_time_history(
        proj,
        analysis_case_id="TH_ElCentro_2D",
        ground_motion="elcentro",
        direction="x",
        damping_ratio=0.05,
        solver="auto",
        envelope_to_design=True,
    )
    assert res.ok, f"Time-history failed: {res.error}"
    assert res.data["solver_used"] == "fem2d"
    assert res.data["n_steps"] > 1000
    assert res.data["duration"] > 20.0
    assert res.data["peak_lateral_displacement"] > 0.0
    assert res.data["max_moment_envelope"] > 0.0
    assert res.data["n_element_results"] == 6

    # Verify AnalysisResult schema attachments
    ar = proj.analysis_results["TH_ElCentro_2D"]
    assert ar.time_history_result is not None
    assert len(ar.time_history_result.time_steps) > 0
    assert ar.modal_result is not None
    assert len(ar.element_results) == 6

    # Check element forces
    c1_res = ar.element_results["C1"]
    assert c1_res.max_moment > 0.0
    assert c1_res.max_shear > 0.0
    assert c1_res.max_axial >= 0.0


def test_3d_time_history_elcentro():
    """Verify 3D space frame transient dynamic response under El Centro ground motion."""
    proj = _build_3d_frame_project()
    res = run_time_history(
        proj,
        analysis_case_id="TH_ElCentro_3D",
        ground_motion="elcentro",
        direction="x",
        damping_ratio=0.05,
        solver="auto",
        envelope_to_design=True,
    )
    assert res.ok, f"3D Time-history failed: {res.error}"
    assert res.data["solver_used"] == "fem3d"
    assert res.data["peak_lateral_displacement"] > 0.0
    assert res.data["max_moment_envelope"] > 0.0
    assert res.data["n_element_results"] == 8


def test_inspect_dynamic_results():
    """Verify inspect_dynamic_results tool outputs."""
    proj = _build_2d_frame_project()
    run_time_history(
        proj,
        analysis_case_id="TH_Case1",
        ground_motion="elcentro",
        direction="x",
    )

    res = inspect_dynamic_results(proj, "TH_Case1")
    assert res.ok, f"Inspect failed: {res.error}"
    data = res.data
    assert data["has_time_history"] is True
    assert data["n_recorded_steps"] > 0
    assert data["duration"] > 20.0
    assert data["fundamental_period"] > 0.0
    assert data["max_displacement_x"] > 0.0
    assert "C1" in data["elements"]
    assert data["elements"]["C1"]["max_moment"] > 0.0


def test_dynamic_design_pipeline():
    """Verify downstream RC member design against peak dynamic earthquake envelopes."""
    proj = _build_2d_frame_project()

    # 1. Run transient time-history with envelope_to_design=True
    th_res = run_time_history(
        proj,
        analysis_case_id="Earthquake_Envelope",
        ground_motion="elcentro",
        direction="x",
        envelope_to_design=True,
    )
    assert th_res.ok

    # 2. Directly run ACI 318 design against the earthquake envelope!
    design_res = run_design(
        proj,
        design_case_id="ACI318_EQ",
        analysis_case_id="Earthquake_Envelope",
    )
    assert design_res.ok, f"Design failed: {design_res.error}"
    assert "governing_dcr" in design_res.data
    assert design_res.data["governing_dcr"] > 0.0

    # 3. Inspect design checks
    d_inspect = inspect_design_results(proj, "ACI318_EQ")
    assert d_inspect.ok
    assert "elements" in d_inspect.data
    assert "C1" in d_inspect.data["elements"]


def test_agent_runner_tool_dispatch():
    """Verify AgentRunner tool map contains dynamics tools and executes them."""
    from struct_agent.llm import BaseLLMProvider, LLMResponse

    class _MockProvider(BaseLLMProvider):
        def __init__(self):
            super().__init__(model="mock")
        def generate(self, messages, tools=None, temperature=0.0):
            return LLMResponse(content="done")
        def generate_stream(self, messages, tools=None, temperature=0.0):
            yield from ()
            return LLMResponse(content="done")

    proj = _build_2d_frame_project()
    runner = AgentRunner(provider=_MockProvider())
    runner.project = proj

    # Dispatch run_time_history through runner's execution layer
    res = runner._execute_tool(
        "run_time_history",
        {"analysis_case_id": "TH_Runner_Test", "ground_motion": "elcentro", "direction": "x"},
    )
    assert res.ok
    assert "TH_Runner_Test" in runner.project.analysis_results

    # Dispatch inspect_dynamic_results
    insp = runner._execute_tool(
        "inspect_dynamic_results",
        {"analysis_case_id": "TH_Runner_Test"},
    )
    assert insp.ok
    assert insp.data["has_time_history"] is True


def test_run_response_spectrum_2d_and_design():
    """Verify run_response_spectrum on 2D frame feeds into ACI 318 run_design."""
    proj = _build_2d_frame_project()

    rsa_res = run_response_spectrum(
        proj,
        analysis_case_id="RSA_Case_1",
        sds=1.0,
        sd1=0.6,
        direction="x",
        combination="CQC",
        envelope_to_design=True,
    )
    assert rsa_res.ok, f"RSA failed: {rsa_res.error}"
    assert "combined_base_shear" in rsa_res.data
    assert rsa_res.data["combined_base_shear"] > 0.0
    assert "cumulative_mass_participation" in rsa_res.data

    # Verify ACI 318 design against the RSA envelope
    design_res = run_design(
        proj,
        design_case_id="ACI318_RSA",
        analysis_case_id="RSA_Case_1",
    )
    assert design_res.ok, f"Design failed: {design_res.error}"
    assert "governing_dcr" in design_res.data
    assert design_res.data["governing_dcr"] > 0.0


def test_run_response_spectrum_3d():
    """Verify run_response_spectrum on 3D frame."""
    proj = _build_3d_frame_project()

    rsa_res = run_response_spectrum(
        proj,
        analysis_case_id="RSA_3D_Case",
        sds=1.0,
        sd1=0.6,
        direction="x",
        combination="CQC",
    )
    assert rsa_res.ok, f"RSA 3D failed: {rsa_res.error}"
    assert rsa_res.data["combined_base_shear"] > 0.0
    assert "RSA_3D_Case" in proj.analysis_results


def test_runner_response_spectrum_dispatch():
    """Verify AgentRunner dispatches run_response_spectrum."""
    from struct_agent.llm import BaseLLMProvider, LLMResponse

    class _MockProvider(BaseLLMProvider):
        def __init__(self):
            super().__init__(model="mock")
        def generate(self, messages, tools=None, temperature=0.0):
            return LLMResponse(content="done")
        def generate_stream(self, messages, tools=None, temperature=0.0):
            yield from ()
            return LLMResponse(content="done")

    proj = _build_2d_frame_project()
    runner = AgentRunner(provider=_MockProvider())
    runner.project = proj

    res = runner._execute_tool(
        "run_response_spectrum",
        {"analysis_case_id": "RSA_Runner", "sds": 1.0, "sd1": 0.6, "direction": "x"},
    )
    assert res.ok
    assert "RSA_Runner" in runner.project.analysis_results


def test_peer_earthquakes_integration():
    """Verify built-in PEER earthquake records (e.g. Northridge) in struct_agent tools."""
    proj = _build_2d_frame_project()

    # 1. run_time_history with Northridge record
    th_res = run_time_history(
        proj,
        analysis_case_id="TH_Northridge",
        ground_motion="northridge",
        direction="x",
    )
    assert th_res.ok
    assert "TH_Northridge" in proj.analysis_results

    # 2. run_response_spectrum with Northridge record
    rsa_res = run_response_spectrum(
        proj,
        analysis_case_id="RSA_Northridge",
        ground_motion="northridge",
        direction="x",
    )
    assert rsa_res.ok
    assert "RSA_Northridge" in proj.analysis_results
    assert rsa_res.data["combined_base_shear"] > 0.0


