"""run_design refuses analysis results computed from an earlier state of the model."""

import pytest

from struct_agent.tools.analysis import run_analysis
from struct_agent.tools.design import run_design
from struct_agent.tools.dynamics import run_response_spectrum
from struct_agent.tools.loads import apply_gravity_loads, generate_load_combinations
from struct_agent.tools.model import create_frame_model, modify_project


def _portal():
    project = create_frame_model(fc=27580.0, num_bays=1, num_stories=1, bay_width=6.0, story_height=3.5, col_b=0.3, col_h=0.3,
                                 beam_b=0.25, beam_h=0.35).data["project"]
    assert apply_gravity_loads(project, superimposed_dead=5.1, custom_live_load=3.12, tributary_width=4.0, occupancy="office").ok
    assert generate_load_combinations(project).ok
    return project


def test_design_refuses_an_analysis_of_resized_sections():
    project = _portal()
    assert run_analysis(project, analysis_case_id="S1").ok
    assert run_design(project, design_case_id="D1", analysis_case_id="S1").ok
    assert modify_project(project, sections=[{"id": "COL", "b": 0.5, "h": 0.5}]).ok
    stale = run_design(project, design_case_id="D2", analysis_case_id="S1")
    assert not stale.ok and "before the latest change" in stale.error.message
    assert not run_design(project, design_case_id="D3").ok  # the active case is S1 too
    assert run_analysis(project, analysis_case_id="S2").ok
    assert run_design(project, design_case_id="D4", analysis_case_id="S2").ok


def test_envelope_alias_is_the_latest_static_envelope():
    project = _portal()
    assert run_analysis(project, analysis_case_id="S1").ok
    assert modify_project(project, sections=[{"id": "COL", "b": 0.5, "h": 0.5}]).ok
    assert run_analysis(project, analysis_case_id="S2").ok
    assert project.analysis_results["envelope"] is project.analysis_results["S2"]
    assert run_design(project, design_case_id="D", analysis_case_id="envelope").ok


def test_unchanged_model_and_unrelated_combinations_keep_results_current():
    project = _portal()
    assert run_analysis(project, analysis_case_id="S1").ok
    assert modify_project(project, sections=[{"id": "COL", "b": 0.3, "h": 0.3}]).ok  # same dimensions
    assert run_design(project, design_case_id="D", analysis_case_id="S1").ok
    # A spectrum envelope builds its own combinations: regenerating the model's combinations leaves it current
    assert run_response_spectrum(project, analysis_case_id="R", sds=1.0, sd1=0.6, r_factor=8.0,
                                 system_type="rc_smf").ok
    assert generate_load_combinations(project, sds=1.0).ok
    assert run_design(project, design_case_id="DR", analysis_case_id="R_design").ok


def test_static_analysis_of_a_model_without_loads_is_refused():
    """qwen3.5:9b re-created a C06 model (which drops its loads) and analysed it: zero reactions, reported as success."""
    project = create_frame_model(fc=27580.0, num_bays=1, num_stories=1, bay_width=6.0, story_height=3.5).data["project"]
    res = run_analysis(project, analysis_case_id="S1")
    assert not res.ok and "no load cases" in res.error.message


def test_reaction_summaries_are_totals_only_for_one_combination():
    """An envelope's per-support reactions come from different combinations: nemotron reported their sum as the
    maximum support reaction (B01) and as a base shear in a kip-ft model labelled kN (D01)."""
    from struct_agent.tools.analysis import inspect_analysis_results

    project = _portal()
    env = run_analysis(project, analysis_case_id="S1").data["reactions"]
    assert "total_vertical_reaction" not in env and "not totals" in env["note"] and env["force_unit"] == "kN"
    comb = next(c.id for c in project.model.load_combinations if "1.6L" in c.id)
    one = inspect_analysis_results(project, comb).data["reactions_summary"]
    assert one["max_support_vertical_reaction"]["value"] == pytest.approx(one["total_vertical_reaction"] / 2, rel=1e-3)
    assert one["base_shear_x"] == pytest.approx(0.0, abs=1e-6)

    us = create_frame_model(fc=576.0, num_bays=1, num_stories=1, bay_width=20.0, story_height=12.0, units="kip, ft",
                            col_b=1.0, col_h=1.0, beam_b=1.0, beam_h=1.5).data["project"]
    assert apply_gravity_loads(us, superimposed_dead=0.1, custom_live_load=0.05, tributary_width=15.0, occupancy="office").ok
    assert run_analysis(us, analysis_case_id="S1").data["reactions"]["force_unit"] == "kip"


def test_a_spectrum_analysis_leaves_a_static_analysis_current():
    """nemotron (A05): static ELF analysis, then an RSA of the same frame system, then a design of the static
    result. The RSA replaced the recorded basis (procedure, case ids) but changed nothing the static result uses."""
    from struct_agent.tools.loads import apply_lateral_loads

    project = _portal()
    kw = dict(sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", risk_category="II")
    assert apply_lateral_loads(project, direction="x", distribution="uniform", **kw).ok
    assert generate_load_combinations(project, sds=1.0).ok
    assert run_analysis(project, analysis_case_id="S1").ok
    assert run_response_spectrum(project, analysis_case_id="R", direction="x", **kw).ok
    assert project.seismic_basis.procedure == "rsa"
    assert run_design(project, design_case_id="D", analysis_case_id="S1").ok


def test_aci_design_uses_strength_level_combinations():
    """gpt-oss (C02) asked for standard='ACI 318-25', got ASD combinations, and designed them (DCR 0.937 instead
    of 0.987): an ACI standard means strength design, and run_design refuses service-level combinations."""
    project = create_frame_model(fc=27580.0, num_bays=1, num_stories=1, bay_width=6.0, story_height=3.5, col_b=0.3, col_h=0.3,
                                 beam_b=0.25, beam_h=0.35).data["project"]
    assert apply_gravity_loads(project, superimposed_dead=5.1, custom_live_load=3.12, tributary_width=4.0, occupancy="office").ok
    assert generate_load_combinations(project, standard="ACI 318-25").data["method"] == "LRFD"
    assert generate_load_combinations(project, standard="ASCE7_ASD").data["method"] == "ASD"
    assert run_analysis(project, analysis_case_id="S").ok
    refused = run_design(project, design_case_id="D", analysis_case_id="S")
    assert not refused.ok and "ASD" in refused.error.message



def test_create_project_3d_horizontal_grid_has_3d_supports():
    """F4: create_project(dimension='3D') with nodes at z = 0 creates 3D supports (6-DOF) and solves."""
    from struct_agent.tools.model import create_project
    from struct_core.structural_model import PointLoad, LoadCase

    nodes = [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 5.0, "y": 0.0, "z": 0.0},
        {"id": 3, "x": 5.0, "y": 5.0, "z": 0.0},
        {"id": 4, "x": 0.0, "y": 5.0, "z": 0.0},
    ]
    elements = [
        {"id": "B1", "type": "beam", "start_node": 1, "end_node": 2, "section_id": "SEC"},
        {"id": "B2", "type": "beam", "start_node": 2, "end_node": 3, "section_id": "SEC"},
        {"id": "B3", "type": "beam", "start_node": 3, "end_node": 4, "section_id": "SEC"},
        {"id": "B4", "type": "beam", "start_node": 4, "end_node": 1, "section_id": "SEC"},
    ]
    materials = [{"id": "CONC", "type": "concrete", "E": 2.5e7, "nu": 0.2, "rho": 2400.0, "fc": 28000.0}]
    sections = [{"id": "SEC", "type": "rectangular", "b": 0.3, "h": 0.5, "material_id": "CONC"}]
    supports = [
        {"node_id": 1, "fixity": "fixed"},
        {"node_id": 2, "fixity": "fixed"},
        {"node_id": 3, "fixity": "fixed"},
        {"node_id": 4, "fixity": "fixed"},
    ]
    res = create_project(
        dimension="3D",
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=supports,
        units="kN, m",
    )
    assert res.ok
    proj = res.data["project"]
    assert proj.model.dimension == "3D"
    for s in proj.model.supports:
        assert s.ux and s.uy and s.uz and s.rx and s.ry and s.rz

    # Add vertical load and solve linear 3D analysis
    proj.model.load_cases = [
        LoadCase(id="LIVE", point_loads=[PointLoad(node_id=2, fz=-50.0)])
    ]
    ans = run_analysis(proj, analysis_case_id="LIVE")
    assert ans.ok
    # Sum of vertical reactions equals applied load (50 kN)
    tot_rz = sum(nr.reaction.fz for nr in proj.analysis_results["LIVE"].node_results.values() if nr.reaction)
    assert tot_rz == pytest.approx(50.0, rel=1e-3)


def test_create_project_support_springs_and_unknown_key_rejection():
    """F1: create_project accepts support spring stiffnesses and rejects unknown keys."""
    from struct_agent.tools.model import create_project

    nodes = [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}, {"id": 2, "x": 0.0, "y": 3.0, "z": 0.0}]
    elements = [{"id": "C1", "start_node": 1, "end_node": 2, "section_id": "S"}]
    materials = [{"id": "M", "type": "concrete", "E": 2.5e7, "nu": 0.2, "rho": 2400.0, "fc": 28000.0}]
    sections = [{"id": "S", "type": "rectangular", "b": 0.3, "h": 0.3, "material_id": "M"}]

    # Support with spring
    res = create_project(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=[{"node_id": 1, "ux": True, "ky": 5000.0}],
        units="kN, m",
    )
    assert res.ok
    sup = res.data["project"].model.supports[0]
    assert sup.ux is True and sup.ky == 5000.0

    # Support with unknown key rejected
    bad_res = create_project(
        nodes=nodes,
        elements=elements,
        materials=materials,
        sections=sections,
        supports=[{"node_id": 1, "invalid_key": 123}],
        units="kN, m",
    )
    assert not bad_res.ok
    assert "unknown key" in bad_res.error.message.lower()
