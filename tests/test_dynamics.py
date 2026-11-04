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
    ConcreteMaterial,
    RebarMaterial,
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
        ConcreteMaterial(id="CONC", fc=27580.0, E=2.5e7, nu=0.2, rho=2400.0),
        RebarMaterial(id="GR60", fy=413685.0, E=2.0e8),
    ]

    sections = [
        RectangularSection(id="COL", material_id="CONC", rebar_material_id="GR60", transverse_rebar_material_id="GR60", cover=0.0381, b=0.4, h=0.4),
        RectangularSection(id="BEAM", material_id="CONC", rebar_material_id="GR60", transverse_rebar_material_id="GR60", cover=0.0381, b=0.3, h=0.5),
    ]

    elements = [
        # Columns Story 1
        BeamElement(id="C1", start_node="N1", end_node="N3", section_id="COL"),
        BeamElement(id="C2", start_node="N2", end_node="N4", section_id="COL"),
        # Beams Story 1
        BeamElement(id="B1", start_node="N3", end_node="N4", section_id="BEAM"),
        # Columns Story 2
        BeamElement(id="C3", start_node="N3", end_node="N5", section_id="COL"),
        BeamElement(id="C4", start_node="N4", end_node="N6", section_id="COL"),
        # Beams Story 2
        BeamElement(id="B2", start_node="N5", end_node="N6", section_id="BEAM"),
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
        ConcreteMaterial(id="CONC", fc=27580.0, E=2.5e7, nu=0.2, rho=2400.0),
        RebarMaterial(id="GR60", fy=413685.0, E=2.0e8),
    ]

    sections = [
        RectangularSection(id="COL", material_id="CONC", rebar_material_id="GR60", transverse_rebar_material_id="GR60", cover=0.0381, b=0.4, h=0.4),
        RectangularSection(id="BEAM", material_id="CONC", rebar_material_id="GR60", transverse_rebar_material_id="GR60", cover=0.0381, b=0.3, h=0.5),
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


def test_modal_analysis_reports_mass_participation():
    """A03 and B03 ask for the cumulative modal mass participation; with only the periods reported, minimax and
    nemotron invented seismic parameters to read it from a response spectrum analysis."""
    proj = _build_2d_frame_project()
    res = run_analysis(proj, analysis_case_id="Modal_2D", analysis_type="modal", solver="auto", num_modes=3)
    ratios, cumulative = res.data["mass_participation_x"], res.data["cumulative_mass_participation_x"]
    assert len(ratios) == 3 and "mass_participation_y" not in res.data  # 2D: Y is vertical
    assert cumulative == pytest.approx(np.cumsum(ratios), abs=1e-3) and 0.5 < cumulative[-1] <= 1.0
    assert inspect_dynamic_results(proj, "Modal_2D").data["cumulative_mass_participation_x"] == cumulative
    assert inspect_analysis_results(proj, "Modal_2D").data["modal"]["cumulative_mass_participation_x"] == cumulative
    proj3 = _build_3d_frame_project()
    res3 = run_analysis(proj3, analysis_case_id="Modal_3D", analysis_type="modal", solver="auto", num_modes=3)
    assert len(res3.data["cumulative_mass_participation_y"]) == 3


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


def test_time_history_record_file_base_shear_and_asce7_mass():
    """Record files load by path, seismic mass is ASCE 7 dead + partition weight (no live), and THA
    peak base shear agrees with RSA."""
    from structdyn import GroundMotion
    from struct_agent.tools.loads import apply_gravity_loads
    from struct_agent.tools.model import create_frame_model
    from struct_load import calculate_mass_source

    record = GroundMotion._data_dir() / "imperialValley_elCentro_1940" / "RSN6_IMPVALL.I_I-ELC180-hor1.AT2"
    proj = create_frame_model(fc=27580.0, num_bays=2, bay_width=5.0, num_stories=2, story_height=3.2, dimension="2d").data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)

    th = run_time_history(proj, analysis_case_id="TH_file", ground_motion=str(record), direction="x")
    assert th.ok, th.error
    rsa = run_response_spectrum(proj, analysis_case_id="RSA_file", ground_motion=str(record), direction="x")
    assert rsa.ok, rsa.error

    run_analysis(proj, analysis_case_id="D", load_case_id="Dead")
    run_analysis(proj, analysis_case_id="P", load_case_id="PartitionWeight")
    w_dead = sum(r.fy for r in proj.analysis_results["D"].node_reactions)
    w_partition = sum(r.fy for r in proj.analysis_results["P"].node_reactions)
    # office floors: 10 psf (0.48 kPa) minimum partition weight, ASCE 7-22 §12.7.2 item 2
    assert w_partition == pytest.approx(0.48 * 10.0 * 5.0 * 2, rel=1e-6)
    assert [(f.load_case_id, f.factor) for f in proj.model.mass_source.factors] == [("Dead", 1.0), ("PartitionWeight", 1.0)]
    assert calculate_mass_source(proj.model).total_seismic_weight == pytest.approx(w_dead + w_partition, rel=1e-6)

    vb_th = th.data["peak_base_shear"]
    vb_rsa = rsa.data["combined_base_shear"]
    assert vb_th > th.data["max_shear_envelope"]
    assert vb_th == pytest.approx(vb_rsa, rel=0.2)


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
    """Verify that downstream RC member design against a raw dynamic time-history result is refused (not factored combinations)."""
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

    # 2. Directly running ACI 318 design against a raw earthquake envelope is refused
    design_res = run_design(
        proj,
        design_case_id="ACI318_EQ",
        analysis_case_id="Earthquake_Envelope",
    )
    assert not design_res.ok
    assert "factored load combinations" in design_res.error.message or "combinations" in design_res.error.message


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
    """Verify run_response_spectrum on 2D frame feeds into ACI 318 run_design (gravity + E envelope)."""
    from struct_agent.tools.loads import apply_gravity_loads

    proj = _build_2d_frame_project()
    assert apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5).ok

    rsa_res = run_response_spectrum(
        proj,
        analysis_case_id="RSA_Case_1",
        sds=1.0,
        sd1=0.6,
        tl=8.0,
        direction="x",
        combination="CQC",
        envelope_to_design=True,
    )
    assert rsa_res.ok, f"RSA failed: {rsa_res.error}"
    assert "combined_base_shear" in rsa_res.data
    assert rsa_res.data["combined_base_shear"] > 0.0
    assert "cumulative_mass_participation" in rsa_res.data

    # The raw (seismic-only, sign-less) RSA result is not a design basis; its gravity envelope is
    raw = run_design(proj, design_case_id="ACI318_RSA", analysis_case_id="RSA_Case_1")
    assert not raw.ok and "RSA_Case_1_design" in raw.error.message
    assert not run_design(proj, design_case_id="x", analysis_case_id="Missing").ok
    design_res = run_design(
        proj,
        design_case_id="ACI318_RSA",
        analysis_case_id=rsa_res.data["design_case_id"],
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


def test_run_response_spectrum_refuses_a_record_and_a_design_spectrum():
    """qwen B06 passed the Loma Prieta record with sds/sd1 it chose; the design spectrum replaced the record's
    without notice. One analysis takes one spectrum."""
    proj = _build_2d_frame_project()
    res = run_response_spectrum(proj, analysis_case_id="RSA_Both", ground_motion="northridge", sds=0.8, sd1=0.3,
                                direction="x")
    assert not res.ok
    assert res.error.error_type == "invalid_input" and "one spectrum" in res.error.message
    assert "RSA_Both" not in proj.analysis_results
    half = run_response_spectrum(proj, analysis_case_id="RSA_Half", sds=0.8, direction="x")
    assert not half.ok and half.error.error_type == "missing_input"
    none = run_response_spectrum(proj, analysis_case_id="RSA_None", direction="x")
    assert not none.ok and none.error.error_type == "missing_input"


def test_run_response_spectrum_asce7_params():
    """Verify run_response_spectrum using SeismicParameters and site_class."""
    from struct_load import SeismicParameters
    proj = _build_3d_frame_project()
    params = SeismicParameters(
        sds=1.00,
        sd1=0.40,
        tl=8.0,
        site_class="D",
        risk_category="II",
        system_type="rc_smf",
        damping_ratio=0.05,
    )
    rsa_res = run_response_spectrum(
        proj,
        analysis_case_id="RSA_ASCE_Params",
        params=params,
        direction="x",
        combination="CQC",
    )
    assert rsa_res.ok, f"RSA with params failed: {rsa_res.error}"
    assert rsa_res.data["combined_base_shear"] > 0.0


def test_run_response_spectrum_multi_dir_and_r_factor():
    """Verify multi-directional RSA (direction='both') and R factor scaling."""
    proj = _build_3d_frame_project()
    rsa_res = run_response_spectrum(
        proj,
        analysis_case_id="RSA_MultiDir",
        sds=1.0,
        sd1=0.6,
        r_factor=5.0,
        system_type="rc_imf",
        direction="both",
        combination="CQC",
    )
    assert rsa_res.ok, f"Multi-directional RSA failed: {rsa_res.error}"
    assert "RSA_MultiDir" in proj.analysis_results
    assert rsa_res.data["combined_base_shear"] > 0.0


def test_response_spectrum_design_envelope_combines_gravity():
    """RSA design demand is the ASCE 7 envelope of gravity cases with the sign-less spectral response."""
    from struct_agent.tools.loads import apply_gravity_loads
    from struct_agent.tools.model import create_frame_model

    proj = create_frame_model(fc=27580.0, num_bays=2, bay_width=5.0, num_stories=2, story_height=3.2, dimension="2d").data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    res = run_response_spectrum(
        proj, analysis_case_id="RSA", sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", direction="x"
    )
    assert res.ok, res.error

    design_id = res.data["design_case_id"]
    assert design_id == "RSA_design"
    assert proj.active_analysis_case == design_id

    eid = next(iter(proj.analysis_results["RSA"].element_results))
    e_only = proj.analysis_results["RSA"].element_results[eid]
    dead = proj.analysis_results["Dead"].element_results[eid]
    design = proj.analysis_results[design_id].element_results[eid]
    # (1.2 + 0.2 SDS) D + L + E governs over E alone and over 1.4 D at the member ends
    assert abs(design.max_moment) >= abs(e_only.max_moment)
    assert abs(design.max_moment) >= 1.4 * max(abs(s.M3) for s in dead.forces) - 1e-9

    des = run_design(proj, design_case_id="aci")
    assert des.ok, des.error
    # Frame verdict covers beams, columns, and frame checks (joints); auto-sized footings are reported separately
    assert des.data["governing_member_type"] in ("beam", "column", "frame")
    assert des.data["governing_limit_state"]
    # Governing DCR covers flexure, shear, and the additional code checks (ductility, torsion)
    assert des.data["governing_dcr"] >= max(des.data["dcr_flexure"], des.data["dcr_shear"])
    assert des.data["foundation_governing_dcr"] is not None


def test_response_spectrum_bidirectional_reports_both_base_shears():
    from struct_agent.tools.loads import apply_gravity_loads
    from struct_agent.tools.model import create_frame_model

    proj = create_frame_model(
        fc=27580.0,
        num_stories=2, story_height=3.2, dimension="3d", bay_widths_x=[5.0, 6.0], bay_widths_y=[4.0]
    ).data["project"]
    apply_gravity_loads(proj, occupancy="office", superimposed_dead=1.5)
    res = run_response_spectrum(
        proj, analysis_case_id="RSA2", sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", direction="both"
    )
    assert res.ok, res.error
    vb = res.data["base_shear_by_direction"]
    assert set(vb) == {"x", "y"} and vb["x"] > 0.0 and vb["y"] > 0.0
    assert res.data["combined_base_shear"] == max(vb.values())
    assert res.data["design_case_id"] == "RSA2_design"


def test_response_spectrum_scaled_to_elf_base_shear():
    """ASCE 7-22 §12.9.1.4.1: design-spectrum RSA forces are scaled up to the ELF base shear V = Cs W."""
    from struct_agent.tools.loads import apply_gravity_loads
    from struct_agent.tools.model import create_frame_model
    from struct_load import (
        SeismicParameters,
        approximate_fundamental_period,
        calculate_mass_source,
        calculate_seismic_response_coefficient,
    )

    proj = create_frame_model(fc=27580.0, num_bays=3, bay_width=6.0, num_stories=4, story_height=3.5, dimension="2d").data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    res = run_response_spectrum(
        proj, analysis_case_id="RSA", sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", direction="x"
    )
    assert res.ok, res.error
    scaling = res.data["elf_scaling"]
    assert scaling["applied"]
    info = scaling["directions"]["x"]

    # Independent ELF base shear with T = min(T_modal, Cu * Ta)
    w = calculate_mass_source(proj).total_seismic_weight
    ta = approximate_fundamental_period(14.0, system_type="rc_smf", units="kN, m")
    t_used = min(info["modal_period"], 1.4 * ta)
    cs = calculate_seismic_response_coefficient(
        SeismicParameters(sds=1.0, sd1=0.6, site_class="D", risk_category="II", r_factor=8.0, system_type="rc_smf"), t_used
    )
    assert info["elf_base_shear"] == pytest.approx(cs * w, rel=1e-9)
    # The design base shear is never below the ELF base shear
    assert res.data["base_shear_by_direction"]["x"] == pytest.approx(
        max(info["modal_base_shear"], cs * w), rel=1e-9
    )


def test_response_spectrum_design_spectrum_with_r_requires_system_type():
    proj = _build_3d_frame_project()
    res = run_response_spectrum(proj, analysis_case_id="RSA", sds=1.0, sd1=0.6, r_factor=8.0, direction="x")
    assert not res.ok
    assert "system_type" in res.error.message
