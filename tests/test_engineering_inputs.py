"""
Engineering inputs: user values pass through in model units, documented defaults are
applied only where a value is missing and are reported, and nothing is invented.
"""

from pathlib import Path

import pytest

from struct_agent.tools.design import _service_results, run_design
from struct_agent.tools.dynamics import run_response_spectrum
from struct_agent.tools.link import import_from_etabs, import_from_sap2000
from struct_agent.tools.loads import apply_gravity_loads, apply_lateral_loads, generate_load_combinations
from struct_agent.tools.analysis import run_analysis
from struct_agent.tools.model import create_frame_model, modify_project

S2K = Path(__file__).parent.parent.parent / "struct_link" / "tests" / "fixtures" / "sample_frame.s2k"


def _frame(**kwargs):
    args = dict(dimension="2d", bay_widths_x=[6.0, 6.0], story_heights=[3.5, 3.5], fc=30000.0)
    args.update(kwargs)
    project = create_frame_model(**args).data["project"]
    assert apply_gravity_loads(project, occupancy="office", superimposed_dead=1.5, tributary_width=5.0).ok
    return project


def test_create_frame_model_takes_the_given_materials_and_reports_the_defaults():
    # In kip, ft units: fc=5.0 ksi = 720 ksf, fy=75.0 ksi = 10800 ksf
    res = create_frame_model(dimension="2d", bay_widths_x=[20.0], story_heights=[12.0], units="kip, ft",
                             fc=720.0, fy=10800.0, cover=0.125)
    assert res.ok, res.error
    model = res.data["project"].model
    concrete = model.get_material("CONCRETE")
    rebar = model.get_material("REBAR")
    assert concrete.fc == pytest.approx(720.0) and rebar.fy == pytest.approx(10800.0)
    assert all(s.cover == 0.125 and s.rebar_material_id == "REBAR" for s in model.sections)
    applied = {d["name"] for d in res.data["defaults_applied"]}
    assert "fy" not in applied and "cover" not in applied
    assert {"E", "concrete unit_weight"} <= applied
    # fu belongs to the Grade 60 fy: a given fy leaves it unset rather than pairing it with 90 ksi
    assert rebar.fu is None and "fu" not in applied


def test_create_frame_model_needs_fc():
    with pytest.raises(TypeError):
        create_frame_model(dimension="2d", bay_widths_x=[6.0], story_heights=[3.5])


def test_run_design_reports_the_default_bearing_pressure_and_footing_cover():
    project = _frame()
    generate_load_combinations(project)
    assert run_analysis(project, analysis_case_id="ENV").ok
    res = run_design(project, design_case_id="D", analysis_case_id="ENV")
    assert res.ok, res.error
    foundations = res.data["foundations"]
    assert foundations["designed"] and foundations["allowable_bearing_pressure"] == pytest.approx(150.0)
    assert foundations["pressure_unit"] == "kPa"
    assert foundations["footing_cover"] == pytest.approx(3.0 * 0.0254)
    inputs = project.design_results["D"].inputs
    assert {d["name"] for d in foundations["defaults_applied"]} == {"allowable_bearing_pressure", "footing_cover"}
    assert {"allowable_bearing_pressure", "footing_cover", "fy", "cover", "E"} <= {d.name for d in inputs.defaults_applied}
    assert inputs.allowable_bearing_pressure == pytest.approx(150.0)


def test_run_design_takes_a_given_bearing_pressure():
    project = _frame()
    generate_load_combinations(project)
    assert run_analysis(project, analysis_case_id="ENV").ok
    res = run_design(project, design_case_id="D", analysis_case_id="ENV", allowable_bearing_pressure=200.0,
                     footing_cover=0.075)
    assert res.ok, res.error
    assert res.data["foundations"]["allowable_bearing_pressure"] == pytest.approx(200.0)
    assert res.data["foundations"]["defaults_applied"] == []
    assert not ({"allowable_bearing_pressure", "footing_cover"} & {d.name for d in project.design_results["D"].inputs.defaults_applied})
    assert {"fy", "cover", "E"} <= {d.name for d in project.design_results["D"].inputs.defaults_applied}
    # A smaller allowable pressure needs larger footings
    small = run_design(project, design_case_id="D2", analysis_case_id="ENV", allowable_bearing_pressure=100.0)
    big = {k: v for k, v in project.design_results["D"].element_results.items() if str(k).startswith("F")}
    larger = {k: v for k, v in project.design_results["D2"].element_results.items() if str(k).startswith("F")}
    assert small.ok and big and set(big) == set(larger)
    length = lambda r: r._length_in  # plan dimension of the footing (mm for a metric model)
    assert all(length(larger[k]) >= length(big[k]) for k in big) and any(length(larger[k]) > length(big[k]) for k in big)


def test_response_spectrum_service_reactions_are_gravity_plus_and_minus_0_7_e():
    project = _frame()
    generate_load_combinations(project)
    res = run_response_spectrum(project, analysis_case_id="RSA", sds=1.0, sd1=0.6, r_factor=8.0,
                                system_type="rc_smf", direction="x")
    assert res.ok, res.error
    e = project.analysis_results["RSA_EX"]
    dead = project.analysis_results["Dead"]
    node = next(iter(e.node_results))
    service = {r.analysis_case_id: r for r in _service_results(project)}
    # ASCE 7-22 §2.4.5 with Ev = 0.2 SDS D: (1.0 + 0.14 SDS) D + 0.7 E and (0.6 - 0.14 SDS) D + 0.7 E
    up = service["ASD_1.14D+0.7E"].node_results[node].reaction
    down = service["ASD_1.14D+0.7E_least_axial"].node_results[node].reaction
    g, ev = dead.node_results[node].reaction, e.node_results[node].reaction
    assert up.fy == pytest.approx(1.14 * g.fy + 0.7 * ev.fy)
    assert down.fy == pytest.approx(1.14 * g.fy - 0.7 * ev.fy)
    assert abs(up.mz) == pytest.approx(abs(1.14 * g.mz) + 0.7 * ev.mz)
    # +E and -E of a sign-less response are one combination
    assert not any("-0.7E" in cid for cid in service)
    design = run_design(project, design_case_id="D", analysis_case_id="RSA_design")
    assert design.ok and design.data["foundations"]["designed"]


def test_imported_model_without_fc_is_reported_and_completed_with_modify_project():
    imported = import_from_sap2000(S2K)
    assert imported.ok
    assert any("no f'c" in p for p in imported.data["before_design"])
    project = imported.data["project"]
    file_e = project.model.get_material("CONC").E
    # S2K is in kN, m: 28,000 kPa
    res = modify_project(project, materials=[{"id": "CONC", "type": "concrete", "fc": 28000.0}])
    assert res.ok, res.error
    concrete = project.model.get_material("CONC")
    assert concrete.type == "concrete" and concrete.fc == pytest.approx(28000.0)
    assert concrete.E == file_e  # the file's modulus is kept
    assert all(s.rebar_material_id == "REBAR" and s.cover == pytest.approx(0.0381) for s in project.model.sections)
    assert {"fy", "cover"} <= {d["name"] for d in res.data["defaults_applied"]}


def test_a_refused_modification_leaves_the_model_unchanged():
    project = create_frame_model(dimension="2d", bay_widths_x=[6.0], story_heights=[3.5], fc=30000.0).data["project"]
    before = project.model.model_dump()
    res = modify_project(project, materials=[{"id": "CONCRETE", "fc": 35000.0}],
                         sections=[{"id": "COL", "b": 0.5, "h": 0.5, "rebar_material_id": "NOPE"}])
    assert not res.ok
    assert project.model.model_dump() == before


def test_seismic_defaults_are_derived_and_reported():
    project = _frame()
    res = apply_lateral_loads(project, sds=1.0, sd1=0.6, system_type="rc_smf", direction="x")
    assert res.ok, res.error
    names = {d["name"] for d in res.data["defaults_applied"]}
    assert names == {"risk_category", "site_class"}
    basis = project.seismic_basis
    assert basis.r_factor == 8.0 and basis.importance_factor == 1.0 and basis.risk_category == "II"
    # Risk Category IV gives Ie = 1.5 (ASCE 7-22 Table 1.5-2)
    project = _frame()
    assert apply_lateral_loads(project, sds=1.0, sd1=0.6, system_type="rc_smf", risk_category="IV",
                               site_class="C", direction="x").ok
    assert project.seismic_basis.importance_factor == 1.5


def test_lateral_loads_need_site_accelerations():
    project = _frame()
    res = apply_lateral_loads(project, sds=1.0)
    assert not res.ok and "sd1" in res.error.message


E2K = Path(__file__).parent.parent.parent / "research_paper" / "benchmarks" / "fixtures" / "exported_seismic_frame.e2k"


def test_import_takes_the_reinforcement_the_file_lacks():
    # E2K is in kN, m: 414,000 kPa (= 414 MPa)
    res = import_from_etabs(E2K, fy=414000.0)
    assert res.ok and "before_design" not in res.data
    model = res.data["project"].model
    rebar = model.get_material("REBAR")
    assert rebar.fy == pytest.approx(414000.0) and model.get_material("C30_Concrete").fc == pytest.approx(30000.0)
    assert all(s.rebar_material_id == "REBAR" and s.cover == pytest.approx(0.0381) for s in model.sections)
    assert "cover" in {d["name"] for d in res.data["defaults_applied"]}


def test_one_new_rebar_material_reinforces_the_bare_concrete_sections():
    project = import_from_etabs(E2K).data["project"]
    res = modify_project(project, materials=[{"id": "G420", "type": "rebar", "fy": 420000.0}])
    assert res.ok, res.error
    assert all(s.rebar_material_id == "G420" for s in project.model.sections)


def test_a_custom_unit_weight_overrides_the_default():
    res = create_frame_model(dimension="2d", bay_widths_x=[6.0], story_heights=[3.5], fc=27600.0,
                             unit_weight=24.0)
    concrete = res.data["project"].model.get_material("CONCRETE")
    assert concrete.unit_weight == pytest.approx(24.0)
    assert "concrete unit_weight" not in {d["name"] for d in res.data["defaults_applied"]}


@pytest.mark.parametrize("kwargs, words", [
    (dict(fc=27.6, fy=414), ["f'c = 27.6 kPa", "All values must be given in the project's kPa unit."]),
    (dict(fc=30000.0, cover=1.5), ["cover = 1.5 m", "0.0127 to 0.1524 m"]),
    (dict(fc=4000, fy=60000, units="kip, ft"), ["f'c = 4000 ksf", "All values must be given in the project's ksf unit."]),
])
def test_inputs_outside_their_realistic_range_are_refused_as_unit_slips(kwargs, words):
    res = create_frame_model(dimension="2d", bay_widths_x=[6.0], story_heights=[3.5], **kwargs)
    assert not res.ok and res.error.error_type == "invalid_input"
    assert all(w in res.error.message for w in words), res.error.message


def test_bearing_pressure_and_section_cover_are_range_checked():
    project = _frame()
    generate_load_combinations(project)
    run_analysis(project, analysis_case_id="ENV")
    res = run_design(project, design_case_id="D", analysis_case_id="ENV", allowable_bearing_pressure=150000.0)
    assert not res.ok and "bearing pressure" in res.error.message
    before = project.model.model_dump()
    res = modify_project(project, sections=[{"id": "COL", "cover": 1.5}])
    assert not res.ok and "cover = 1.5 m" in res.error.message and project.model.model_dump() == before
