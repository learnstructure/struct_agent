"""
Cracked-section analysis, ACI 318-25 slenderness (Section 6.6.4), ASCE 7-22 drift and stability
checks, and the frame-level design checks (joints, Chapter 18).
"""

import pytest

from struct_core import frame_stories
from struct_agent.tools.analysis import run_analysis
from struct_agent.tools.design import run_design
from struct_agent.tools.dynamics import run_response_spectrum
from struct_agent.tools.frame_stability import (
    aci_cracked_stiffness_modifiers,
    story_drifts,
    story_shear,
)
from struct_agent.tools.loads import apply_gravity_loads, apply_lateral_loads, generate_load_combinations
from struct_agent.tools.model import create_frame_model, modify_project


def _elf_frame(system="rc_smf", r=8.0):
    proj = create_frame_model(fc=27580.0, num_bays=3, bay_widths_x=[5.0, 6.0, 4.0], num_stories=3, story_height=3.5,
                              dimension="2d", col_b=0.45, col_h=0.45, beam_b=0.30, beam_h=0.55).data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    res = apply_lateral_loads(proj, sds=1.0, sd1=0.6, r_factor=r, system_type=system, risk_category="II",
                              direction="x", distribution="uniform")
    assert res.ok, res.error
    generate_load_combinations(proj, sds=1.0)
    return proj


def test_cracked_modifiers_follow_aci_table_6_6_3_1_1():
    proj = _elf_frame()
    mods = aci_cracked_stiffness_modifiers(proj.model)
    assert set(mods.values()) == {0.70, 0.35}
    assert all(mods[e.id] == (0.70 if str(e.id).startswith("C") else 0.35) for e in proj.model.elements)


def test_elf_records_seismic_basis_with_table_12_2_1_coefficients():
    proj = _elf_frame("rc_imf", 5.0)
    basis = proj.seismic_basis
    assert (basis.system, basis.r_factor, basis.omega_0, basis.cd) == ("rc_imf", 5.0, 3.0, 4.5)
    assert basis.procedure == "elf" and basis.lateral_case_ids == ["EQ_X"] and basis.directions == ["x"]
    assert basis.seismic_design_category == "D"


def test_seismic_system_is_validated_not_dropped():
    """A system string outside the accepted keys fails the tool instead of silently skipping Chapter 18."""
    proj = create_frame_model(fc=27580.0, bay_widths_x=[5.0, 6.0], story_heights=[3.5, 3.5]).data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    bad = apply_lateral_loads(proj, sds=1.0, sd1=0.6, r_factor=8.0, system_type="special moment frame", direction="x")
    assert not bad.ok and "rc_smf" in bad.error.message
    assert proj.seismic_basis is None
    # Case and spaces are normalised to the canonical key that design reads
    ok = apply_lateral_loads(proj, sds=1.0, sd1=0.6, r_factor=8.0, system_type=" RC_SMF ", direction="x")
    assert ok.ok and proj.seismic_basis.system == "rc_smf"
    # Design-level RSA needs the system
    rsa = run_response_spectrum(proj, analysis_case_id="rsa", sds=1.0, sd1=0.6, r_factor=8.0, direction="x")
    assert not rsa.ok and "system_type" in rsa.error.message


def test_second_order_requires_cracked_sections():
    proj = _elf_frame()
    res = run_analysis(proj, analysis_case_id="env", second_order=True)
    assert not res.ok and "cracked_sections" in res.error.message


def test_drift_check_matches_hand_calculation():
    proj = _elf_frame()
    res = run_analysis(proj, analysis_case_id="env", cracked_sections=True)
    assert res.ok, res.error
    env = proj.analysis_results["env"]
    assert env.stiffness_basis == "cracked"
    story = frame_stories(proj.model)[0]
    eq = proj.analysis_results["EQ_X"]
    drift_e = max(story_drifts(proj.model, eq, story, "x"), key=abs)
    expected = 5.5 * abs(drift_e) / 1.0  # Cd delta_xe / Ie
    check = next(c for c in env.code_checks if c.clause == "ASCE 7-22 §12.12.1" and c.description.startswith("Story 1 "))
    assert check.capacity == pytest.approx(0.020 * story.height)  # Table 12.12-1, Risk Category II, rho = 1.0
    theta = next(c for c in env.code_checks if c.clause == "ASCE 7-22 §12.8.7" and c.description.startswith("Story 1 "))
    amplification = 1.0 / (1.0 - theta.demand) if theta.demand > 0.10 else 1.0
    assert check.demand == pytest.approx(expected * amplification, rel=1e-9)


def test_cracked_stiffness_increases_drift():
    gross = _elf_frame()
    run_analysis(gross, analysis_case_id="env")
    cracked = _elf_frame()
    run_analysis(cracked, analysis_case_id="env", cracked_sections=True)
    story = frame_stories(gross.model)[0]
    d_gross = max(map(abs, story_drifts(gross.model, gross.analysis_results["EQ_X"], story, "x")))
    d_cracked = max(map(abs, story_drifts(cracked.model, cracked.analysis_results["EQ_X"], story, "x")))
    assert d_cracked > 1.5 * d_gross


def test_second_order_magnifies_lateral_combinations():
    proj = _elf_frame()
    res = run_analysis(proj, analysis_case_id="env", cracked_sections=True, second_order=True)
    assert res.ok, res.error
    story = frame_stories(proj.model)[0]
    s1 = next(s for s in res.data["second_order"] if s["story"] == 1)
    # Q = sum(Pu) delta_o / (Vus lc) with delta_o / Vus from the ELF case
    eq_case = next(lc for lc in proj.model.load_cases if lc.id == "EQ_X")
    drifts = story_drifts(proj.model, proj.analysis_results["EQ_X"], story, "x")
    flexibility = abs(sum(drifts) / len(drifts)) / story_shear(proj.model, eq_case, story, "x")
    assert s1["Q"] == pytest.approx(s1["sum_pu"] * flexibility / story.height, rel=1e-9)
    assert s1["delta_s"] == pytest.approx(1.0 / (1.0 - s1["Q"]), rel=1e-9)
    clauses = {c.clause for c in proj.analysis_results["env"].code_checks}
    assert {"ACI 318-25 §6.6.4.6.2", "ACI 318-25 §6.6.4.5.2", "ACI 318-25 §6.2.5.3",
            "ASCE 7-22 §12.12.1", "ASCE 7-22 §12.8.7"} <= clauses


def test_design_reports_joint_and_chapter_18_checks():
    proj = _elf_frame()
    run_analysis(proj, analysis_case_id="env", cracked_sections=True, second_order=True)
    des = run_design(proj, design_case_id="d", analysis_case_id="env")
    assert des.ok, des.error
    design = proj.design_results["d"]
    system_clauses = {c.clause for c in design.system_checks}
    assert {"ACI 318-25 §18.8.4", "ACI 318-25 §18.7.3.2", "ASCE 7-22 §12.12.1"} <= system_clauses
    member_clauses = {c.clause for e in design.element_results.values() for c in e.code_checks}
    assert {"ACI 318-25 §18.6.5", "ACI 318-25 §18.7.6.1", "ACI 318-25 §18.7.5.4"} <= member_clauses
    worst = max([c.dcr for c in design.system_checks]
                + [e.governing_dcr for e in design.element_results.values() if e.member_type in ("beam", "column")])
    assert des.data["governing_dcr"] == pytest.approx(worst)


def test_smf_column_bars_are_designed_for_strength_and_strong_column():
    """Column bars rise above the 1 % minimum where strong-column/weak-beam needs it, within 6 % Ag."""
    proj = create_frame_model(fc=27580.0, num_bays=3, bay_widths_x=[5.0, 6.0, 4.0], num_stories=3, story_height=3.5,
                              dimension="2d", col_b=0.40, col_h=0.40, beam_b=0.30, beam_h=0.55).data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    apply_lateral_loads(proj, sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", risk_category="II",
                        direction="x", distribution="uniform")
    generate_load_combinations(proj, sds=1.0)
    run_analysis(proj, analysis_case_id="env", cracked_sections=True, second_order=True)
    assert run_design(proj, design_case_id="d", analysis_case_id="env").ok
    design = proj.design_results["d"]
    ag = (0.40 / 0.0254) ** 2  # in^2
    ratios = [e.flexure.provided_steel_area / ag for e in design.element_results.values() if e.member_type == "column"]
    assert min(ratios) >= 0.01 and max(ratios) <= 0.06 and max(ratios) > 0.012
    assert all(c.passed for c in design.system_checks if c.clause == "ACI 318-25 §18.7.3.2")


def test_rsa_cracked_reports_drift_from_modal_story_drifts():
    proj = create_frame_model(fc=27580.0, num_bays=3, bay_widths_x=[5.0, 6.0, 4.0], num_stories=3, story_height=3.5,
                              dimension="2d").data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    res = run_response_spectrum(proj, analysis_case_id="RSA", sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf",
                                direction="x", cracked_sections=True, second_order=True)
    assert res.ok, res.error
    assert res.data["seismic_basis"]["recorded"] and proj.seismic_basis.procedure == "rsa"
    checks = proj.analysis_results[res.data["design_case_id"]].code_checks
    assert sum(c.clause == "ASCE 7-22 §12.12.1" for c in checks) == 3
    assert sum(c.clause == "ACI 318-25 §6.6.4.6.2" for c in checks) == 3
    assert res.data["stiffness_basis"] == "cracked"


def test_resizing_regenerates_elf_story_forces():
    """ELF forces follow the seismic weight, so a member resize regenerates them (no stale lateral loads)."""
    proj = _elf_frame()
    v_before = sum(abs(pl.fx) for lc in proj.model.load_cases if lc.id == "EQ_X" for pl in lc.point_loads)
    col_id = next(s.id for s in proj.model.sections if s.id.startswith("COL"))
    res = modify_project(proj, sections=[{"id": col_id, "b": 0.60, "h": 0.60}])
    assert res.ok and res.data["seismic_loads"]["updated"] is True
    v_after = sum(abs(pl.fx) for lc in proj.model.load_cases if lc.id == "EQ_X" for pl in lc.point_loads)
    assert v_after > v_before
    # Story forces are rounded to 0.001 (struct_load) before they are split over the floor nodes
    assert res.data["seismic_loads"]["seismic_base_shear"] == pytest.approx(v_after, abs=1e-3)
    # Same result as applying the ELF loads to a model built with the new sections from the start
    fresh = create_frame_model(fc=27580.0, num_bays=3, bay_widths_x=[5.0, 6.0, 4.0], num_stories=3, story_height=3.5,
                               dimension="2d", col_b=0.60, col_h=0.60, beam_b=0.30, beam_h=0.55).data["project"]
    apply_gravity_loads(fresh, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    ref = apply_lateral_loads(fresh, sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", risk_category="II",
                              direction="x", distribution="uniform")
    assert v_after == pytest.approx(ref.data["seismic_base_shear"], abs=1e-3)
    assert [lc.id for lc in proj.model.load_cases].count("EQ_X") == 1


def test_resizing_regenerates_elf_forces_applied_one_direction_per_call():
    """nemotron applied EQ_X and EQ_Y in two calls; the basis kept only EQ_Y, so a resize regenerated EQ_Y
    (682 kN) and left EQ_X at the old seismic weight (511 kN), and the drift checks covered y only."""
    def frame(col):
        proj = create_frame_model(fc=27580.0, num_bays=2, num_stories=2, bay_width=5.0, story_height=3.5, dimension="3d",
                                  bays_y=1, col_b=col, col_h=col, beam_b=0.30, beam_h=0.50).data["project"]
        apply_gravity_loads(proj, occupancy="office", superimposed_dead=5.1, two_way=True)
        return proj

    kw = dict(sds=1.0, sd1=0.6, r_factor=8.0, system_type="rc_smf", risk_category="II", distribution="uniform")
    proj = frame(0.40)
    assert apply_lateral_loads(proj, direction="x", **kw).ok
    assert apply_lateral_loads(proj, direction="y", **kw).ok
    assert proj.seismic_basis.lateral_case_ids == ["EQ_X", "EQ_Y"] and proj.seismic_basis.directions == ["x", "y"]
    assert modify_project(proj, sections=[{"id": "COL", "b": 0.60, "h": 0.60}]).data["seismic_loads"]["updated"]

    ref = frame(0.60)
    v_ref = apply_lateral_loads(ref, direction="both", **kw).data["seismic_base_shear"]
    cases = {lc.id: lc for lc in proj.model.load_cases}
    assert sum(abs(pl.fx) for pl in cases["EQ_X"].point_loads) == pytest.approx(v_ref, rel=1e-4)
    assert sum(abs(pl.fy) for pl in cases["EQ_Y"].point_loads) == pytest.approx(v_ref, rel=1e-4)
    # Directions with different inputs cannot be regenerated by one call: the resize says so
    other = frame(0.40)
    apply_lateral_loads(other, direction="x", **kw)
    apply_lateral_loads(other, direction="y", **{**kw, "distribution": "face"})
    assert other.seismic_basis.directions == ["x", "y"]
    assert modify_project(other, sections=[{"id": "COL", "b": 0.60, "h": 0.60}]).data["seismic_loads"]["updated"] is False


def test_design_needs_factored_load_combinations():
    """nemotron (ablated): an analysis without load combinations is the unfactored sum of the load cases;
    designing it is refused instead of reported as a design check."""
    proj = create_frame_model(fc=27580.0, num_bays=1, bay_widths_x=[6.0], num_stories=1, story_height=3.5, dimension="2d",
                              col_b=0.40, col_h=0.40, beam_b=0.30, beam_h=0.55).data["project"]
    apply_gravity_loads(proj, occupancy="office", tributary_width=5.0, superimposed_dead=1.5)
    assert run_analysis(proj, analysis_case_id="raw").ok
    refused = run_design(proj, design_case_id="d", analysis_case_id="raw")
    assert not refused.ok and "generate_load_combinations" in refused.error.message
    assert not run_design(proj, design_case_id="d").ok  # the active analysis is the same unfactored one
    generate_load_combinations(proj)
    assert run_analysis(proj, analysis_case_id="env").ok
    assert run_design(proj, design_case_id="d", analysis_case_id="env").ok
