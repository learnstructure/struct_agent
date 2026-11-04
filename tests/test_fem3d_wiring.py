"""
Tests for fem3d wiring and solver selection in struct_agent.
"""

import pytest
from struct_core import (
    StructuralModel,
    Node,
    BeamElement,
    ElasticMaterial,
    RectangularSection,
    Support,
    PointLoad,
    LoadCase,
    LoadCombination,
    LoadCaseFactor,
)
from struct_agent.tools.model import create_project
from struct_agent.tools.analysis import run_analysis, inspect_analysis_results, _detect_solver


def _create_3d_frame():
    """Build a minimal 3D frame (1 bay, 1 story in 3D)."""
    # Nodes: base at z=0, top at z=3.0
    nodes = [
        Node(id="N1", x=0.0, y=0.0, z=0.0),
        Node(id="N2", x=5.0, y=0.0, z=0.0),
        Node(id="N3", x=0.0, y=4.0, z=0.0),
        Node(id="N4", x=5.0, y=4.0, z=0.0),
        Node(id="N5", x=0.0, y=0.0, z=3.0, mass=1000.0),
        Node(id="N6", x=5.0, y=0.0, z=3.0, mass=1000.0),
        Node(id="N7", x=0.0, y=4.0, z=3.0, mass=1000.0),
        Node(id="N8", x=5.0, y=4.0, z=3.0, mass=1000.0),
    ]

    # Supports at base (all 6 DOFs fixed)
    supports = [
        Support(node_id="N1", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        Support(node_id="N2", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        Support(node_id="N3", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        Support(node_id="N4", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
    ]

    materials = [
        ElasticMaterial(id="M1", E=2.5e7, nu=0.2, rho=2400.0),
    ]

    sections = [
        RectangularSection(id="COL", material_id="M1", b=0.4, h=0.4),
        RectangularSection(id="BEAM", material_id="M1", b=0.3, h=0.5),
    ]

    elements = [
        # Columns
        BeamElement(id="C1", start_node="N1", end_node="N5", section_id="COL"),
        BeamElement(id="C2", start_node="N2", end_node="N6", section_id="COL"),
        BeamElement(id="C3", start_node="N3", end_node="N7", section_id="COL"),
        BeamElement(id="C4", start_node="N4", end_node="N8", section_id="COL"),
        # Beams
        BeamElement(id="B1", start_node="N5", end_node="N6", section_id="BEAM"),
        BeamElement(id="B2", start_node="N7", end_node="N8", section_id="BEAM"),
        BeamElement(id="B3", start_node="N5", end_node="N7", section_id="BEAM"),
        BeamElement(id="B4", start_node="N6", end_node="N8", section_id="BEAM"),
    ]

    load_cases = [
        LoadCase(
            id="Dead",
            point_loads=[
                PointLoad(node_id="N5", fz=-20.0),
                PointLoad(node_id="N6", fz=-20.0),
                PointLoad(node_id="N7", fz=-20.0),
                PointLoad(node_id="N8", fz=-20.0),
            ],
        ),
        LoadCase(
            id="Lateral_X",
            point_loads=[
                PointLoad(node_id="N5", fx=15.0),
                PointLoad(node_id="N7", fx=15.0),
            ],
        ),
    ]

    load_combinations = [
        LoadCombination(
            id="1.2D+1.0EQ",
            factors=[
                LoadCaseFactor(load_case_id="Dead", factor=1.2),
                LoadCaseFactor(load_case_id="Lateral_X", factor=1.0),
            ],
        )
    ]

    model = StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
        load_cases=load_cases,
        load_combinations=load_combinations,
    )
    return create_project(model, title="3D Frame Test").data["project"]


def test_solver_auto_detection():
    project_3d = _create_3d_frame()
    assert _detect_solver(project_3d) == "fem3d"

    # 2D model
    model_2d = StructuralModel(
        nodes=[Node(id="N1", x=0.0, y=0.0, z=0.0), Node(id="N2", x=5.0, y=0.0, z=0.0)],
        supports=[Support(node_id="N1", ux=True, uy=True, rz=True)],
    )
    proj_2d = create_project(model_2d).data["project"]
    assert _detect_solver(proj_2d) == "fem2d"


def test_fem3d_static_analysis_auto():
    project = _create_3d_frame()
    res = run_analysis(project, analysis_case_id="Static_3D", solver="auto", load_case_id="Dead")
    assert res.ok, f"Analysis failed: {res.error}"
    assert res.data["solver_used"] == "fem3d"
    assert res.data["n_element_results"] == 8

    # Inspect results
    inspect = inspect_analysis_results(project, "Static_3D")
    assert inspect.ok
    assert "C1" in inspect.data["elements"]
    # Axial force is tension positive (as in fem2d): a gravity-loaded column is in compression
    assert inspect.data["elements"]["C1"]["max_axial"] < 0.0


def test_fem3d_modal_analysis():
    project = _create_3d_frame()
    res = run_analysis(project, analysis_case_id="Modal_3D", analysis_type="modal", solver="auto", num_modes=3)
    assert res.ok, f"Modal analysis failed: {res.error}"
    assert res.data["solver_used"] == "fem3d"
    assert res.data["n_modes"] == 3
    assert res.data["fundamental_period"] > 0.0

    # Inspect results
    inspect = inspect_analysis_results(project, "Modal_3D")
    assert inspect.ok
    assert "modal" in inspect.data
    assert len(inspect.data["modal"]["periods"]) == 3
    assert all(p > 0 for p in inspect.data["modal"]["periods"])


def test_fem3d_combination_envelope():
    project = _create_3d_frame()
    res = run_analysis(project, analysis_case_id="Envelope_3D", solver="fem3d", load_case_id="envelope")
    assert res.ok, f"Envelope failed: {res.error}"
    assert res.data["n_element_results"] == 8


def test_create_frame_model_rectangular_3d():
    from struct_agent.tools.model import create_frame_model
    res = create_frame_model(
        fc=27580.0,
        bays_x=3,
        bays_y=2,
        bay_width_x=5.0,
        bay_width_y=6.0,
        num_stories=3,
        story_height=3.2,
        dimension="3d",
    )
    assert res.ok
    proj = res.data["project"]
    m = proj.model
    # 3 bays in X -> 4 grid lines; 2 bays in Y -> 3 grid lines; 3 stories -> 4 elevations
    # Nodes: 4 * 3 * 4 = 48 nodes
    assert len(m.nodes) == 48
    assert max(n.x for n in m.nodes) == 15.0
    assert max(n.y for n in m.nodes) == 12.0
    assert max(n.z for n in m.nodes) == pytest.approx(9.6)
    # Columns: 4 * 3 * 3 = 36 columns
    # Beams X: 3 * 3 * 3 = 27 beams
    # Beams Y: 4 * 2 * 3 = 24 beams
    # Total elements = 36 + 27 + 24 = 87
    assert len(m.elements) == 87
