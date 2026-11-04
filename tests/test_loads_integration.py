"""
Tests for load determination and preliminary sizing tools in struct_agent.
"""

import pytest
from struct_core import (
    AreaLoad,
    BeamElement,
    ConcreteMaterial,
    ElasticMaterial,
    LoadCase,
    Metadata,
    Node,
    Project,
    RebarMaterial,
    RectangularSection,
    StructuralModel,
    Support,
    Units,
)
from struct_agent import (
    WorkflowStage,
    create_project,
    preliminary_sizing,
    apply_gravity_loads,
    apply_lateral_loads,
    apply_roof_loads,
    apply_self_weight,
    distribute_area_loads,
    generate_load_combinations,
    validate_model,
    run_analysis,
    run_design,
)


def _build_bare_frame():
    """Frame geometry with its concrete and trial (unsized) sections, and NO loads."""
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=0.0, y=3.5),
        Node(id=3, x=6.0, y=3.5),
        Node(id=4, x=6.0, y=0.0),
    ]
    materials = [
        ConcreteMaterial(id="CONC", fc=27580.0, E=2.5e7, nu=0.2, unit_weight=23.56),
        RebarMaterial(id="GR60", fy=413685.0, E=2.0e8),
    ]
    rc = dict(material_id="CONC", rebar_material_id="GR60", transverse_rebar_material_id="GR60", cover=0.0381)
    sections = [RectangularSection(id="COL", b=0.3, h=0.3, **rc), RectangularSection(id="BEAM", b=0.3, h=0.3, **rc)]
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, section_id="COL"),
        BeamElement(id=2, start_node=2, end_node=3, section_id="BEAM"),
        BeamElement(id=3, start_node=4, end_node=3, section_id="COL"),
    ]
    supports = [Support.fixed(node_id=1), Support.fixed(node_id=4)]
    model = StructuralModel(nodes=nodes, elements=elements, materials=materials, sections=sections, supports=supports)
    return Project(model=model)


def _build_3d_frame():
    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=6.0, y=5.0, z=0.0),
        Node(id=4, x=0.0, y=5.0, z=0.0),
        Node(id=5, x=0.0, y=0.0, z=3.5),
        Node(id=6, x=6.0, y=0.0, z=3.5),
        Node(id=7, x=6.0, y=5.0, z=3.5),
        Node(id=8, x=0.0, y=5.0, z=3.5),
    ]
    mat = ElasticMaterial(id="CONC", E=2.5e7, nu=0.2, rho=2400.0)
    col = RectangularSection(id="COL", material_id="CONC", name="C400", b=0.4, h=0.4)
    bm = RectangularSection(id="BEAM", material_id="CONC", name="B300", b=0.3, h=0.5)
    elements = [
        BeamElement(id="C1", start_node=1, end_node=5, section_id="COL"),
        BeamElement(id="C2", start_node=2, end_node=6, section_id="COL"),
        BeamElement(id="C3", start_node=3, end_node=7, section_id="COL"),
        BeamElement(id="C4", start_node=4, end_node=8, section_id="COL"),
        BeamElement(id="B1", start_node=5, end_node=6, section_id="BEAM"),
        BeamElement(id="B2", start_node=6, end_node=7, section_id="BEAM"),
        BeamElement(id="B3", start_node=7, end_node=8, section_id="BEAM"),
        BeamElement(id="B4", start_node=8, end_node=5, section_id="BEAM"),
    ]
    model = StructuralModel(nodes=nodes, elements=elements, materials=[mat], sections=[col, bm])
    return Project(model=model, metadata=Metadata(units=Units(length="m", force="kN")))


def test_preliminary_sizing_and_loading_flow():
    # 1. Bare model
    project = _build_bare_frame()

    # 2. Preliminary sizing tool sizes beam and column automatically
    size_res = preliminary_sizing(project)
    assert size_res.ok is True
    sized = {s["id"]: s for s in size_res.data["sections_sized"]}
    # 6.0 m beam: h = 6/14 -> 0.45 m, b = 0.55 h -> 0.25 m; 3.5 m columns: max(0.35, 0.35) m
    assert (sized["BEAM"]["b"], sized["BEAM"]["h"]) == pytest.approx((0.25, 0.45))
    assert (sized["COL"]["b"], sized["COL"]["h"]) == pytest.approx((0.35, 0.35))
    assert {e.section_id for e in project.model.elements} == {"COL", "BEAM"}
    assert len(project.model.sections) == 2
    assert len(project.model.materials) == 2  # sizing creates no material

    # 3. Apply gravity loads per ASCE 7 office occupancy
    grav_res = apply_gravity_loads(
        project,
        occupancy="office",
        tributary_width=5.0,
        superimposed_dead=1.5,
        include_self_weight=True,
    )
    assert grav_res.ok is True
    assert grav_res.next_suggested_stage == "LOADS_APPLIED"
    # Dead, Live, and (office partition allowance) the seismic-weight-only PartitionWeight case
    assert len(project.model.load_cases) == 3
    assert grav_res.data["partition_weight_case_id"] == "PartitionWeight"

    # 4. Apply seismic lateral loads
    lat_res = apply_lateral_loads(
        project,
        code="ASCE7-22",
        sds=0.75,
        sd1=0.40,
        r_factor=8.0,
        tributary_width=5.0,
    )
    assert lat_res.ok is True
    assert "seismic_base_shear" in lat_res.data

    # 5. Generate ACI 318 / ASCE 7 factored load combinations
    comb_res = generate_load_combinations(project, standard="ACI318_LRFD")
    assert comb_res.ok is True
    assert comb_res.data["n_combinations"] >= 3

    # 6. Validate model
    val_res = validate_model(project)
    assert val_res.ok is True

    # 7. Run FEM2D analysis on the governing envelope of combinations
    ana_res = run_analysis(project, analysis_case_id="Env", load_case_id="Envelope")
    assert ana_res.ok is True
    assert "Env" in project.analysis_results

    # 8. Run RC design against the governing envelope
    des_res = run_design(project, design_case_id="ACI318_19", analysis_case_id="Env")
    assert des_res.ok is True
    assert "ACI318_19" in project.design_results


def test_apply_self_weight_tool():
    project = _build_bare_frame()
    preliminary_sizing(project)
    project.model.load_cases = [LoadCase(id="DEAD", include_self_weight=True)]

    res = apply_self_weight(project)
    assert res.ok is True
    assert res.data["total_element_loads"] == 3
    assert project.model.load_cases[0].include_self_weight is False


def test_apply_roof_loads_tool():
    project = _build_3d_frame()
    res = apply_roof_loads(project, ground_snow_load=1.5, one_way_span="y")
    assert res.ok is True
    assert "cases_created" in res.data
    assert "snow" in res.data["cases_created"]
    assert "roof_live" in res.data["cases_created"]


def test_distribute_area_loads_tool():
    project = _build_3d_frame()
    al = AreaLoad(
        id="SLAB",
        pressure=3.0,
        load_type="live",
        node_ids=[5, 6, 7, 8],
        one_way=True,
        span_direction_axis="x",
    )
    project.model.load_cases = [LoadCase(id="Live", area_loads=[al])]

    res = distribute_area_loads(project, target_case_id="Live")
    assert res.ok is True
    assert res.data["distributed_loads_created"] == 2
    assert len(project.model.load_cases[0].element_loads) == 2


def test_define_mass_source_tool():
    from struct_agent import define_mass_source
    project = _build_3d_frame()
    preliminary_sizing(project)
    apply_gravity_loads(project, occupancy="office", superimposed_dead=1.5)

    res = define_mass_source(
        project,
        factors={"DEAD": 1.0, "LIVE": 0.25},
        include_element_mass=True,
    )
    assert res.ok is True
    assert res.data["total_weight"] > 0.0
    assert res.data["total_mass"] > 0.0
    assert len(res.data["story_masses"]) > 0
    assert project.model.mass_source is not None


def test_apply_lateral_loads_with_params_and_mass_source():
    from struct_load import SeismicParameters
    project = _build_3d_frame()
    preliminary_sizing(project)
    apply_gravity_loads(project, occupancy="office", superimposed_dead=1.5)

    params = SeismicParameters(
        sds=1.00,
        sd1=0.40,
        s1=0.35,
        tl=8.0,
        r_factor=8.0,
        importance_factor=1.0,
        system_type="rc_smf",
        site_class="D",
        risk_category="I",
        method=2,
    )

    lat_res = apply_lateral_loads(
        project,
        params=params,
        direction="X",
        distribution="face",
    )
    assert lat_res.ok is True
    assert "seismic_base_shear" in lat_res.data
    assert lat_res.data["seismic_base_shear"] > 0.0



def test_gravity_pressures_reject_unit_mistakes():
    """105 psf typed into a kip-ft model (105 ksf) fails; 0.105 ksf succeeds."""
    from struct_agent.tools.model import create_frame_model

    project = create_frame_model(fc=576.0, dimension="2d", bay_widths_x=[16.0, 20.0, 14.0], story_heights=[14.0, 12.0, 12.0],
                                 units="kip, ft").data["project"]
    bad = apply_gravity_loads(project, occupancy="office", tributary_width=16.0, superimposed_dead=105.0)
    assert not bad.ok and "ksf" in bad.error.message and "is not a floor pressure" in bad.error.message
    good = apply_gravity_loads(project, occupancy="office", tributary_width=16.0, superimposed_dead=0.105,
                               include_self_weight=False)
    assert good.ok, good.error
    # 0.105 ksf x 16 ft x 50 ft x 3 floors
    assert good.data["total_dead_load"] == pytest.approx(0.105 * 16.0 * 50.0 * 3, rel=1e-6)


def test_preliminary_sizing_classifies_3d_members_and_keeps_section_ids():
    """gemma B09: Y-direction beams of a 3D frame were sized as columns and the columns were skipped."""
    from struct_agent.tools.model import create_frame_model

    project = create_frame_model(fc=27580.0, dimension="3d", bay_widths_x=[5.0, 6.0, 4.5], bay_widths_y=[6.0, 4.0],
                                 story_heights=[4.0, 3.5]).data["project"]
    columns_before = {e.id for e in project.model.elements if e.section_id == "COL"}
    dead_before = apply_gravity_loads(project, occupancy="office", superimposed_dead=5.1).data["total_dead_load"]

    res = preliminary_sizing(project, min_col_dim=0.30)
    assert res.ok, res.error
    sized = {s["id"]: s for s in res.data["sections_sized"]}
    assert set(sized) == {"COL", "BEAM"} and {s.id for s in project.model.sections} == {"COL", "BEAM"}
    assert {e.id for e in project.model.elements if e.section_id == "COL"} == columns_before
    assert sized["COL"]["governing_length"] == pytest.approx(4.0) and sized["COL"]["b"] == pytest.approx(0.40)
    assert sized["BEAM"]["governing_length"] == pytest.approx(6.0) and sized["BEAM"]["h"] == pytest.approx(0.45)
    # applied through modify_project: the member self-weight follows the new sections
    assert "Dead" in res.data["self_weight_updated_cases"]
    assert res.data["total_dead_load"]["Dead"] != pytest.approx(dead_before)


def test_preliminary_sizing_uses_two_inch_steps_in_us_units():
    from struct_agent.tools.model import create_frame_model

    project = create_frame_model(fc=576.0, bay_widths_x=[16.0, 20.0], story_heights=[14.0, 12.0], units="kip, ft").data["project"]
    res = preliminary_sizing(project)
    assert res.ok, res.error
    for s in res.data["sections_sized"]:
        for dim in (s["b"], s["h"]):
            assert dim * 6.0 == pytest.approx(round(dim * 6.0), abs=1e-4)  # multiples of 2 in = 1/6 ft
    beam = next(s for s in res.data["sections_sized"] if s["id"] == "BEAM")
    assert beam["h"] == pytest.approx(1.5)  # 20 ft / 14 = 1.43 ft, rounded up to 18 in


def test_elf_weight_does_not_depend_on_an_earlier_modal_analysis():
    """A modal analysis attaches a default ASCE 7 mass source that lumps the lower half of the ground-story columns
    at the base; ELF then lost that weight (minimax B04: 715 kN instead of 751 kN) only because a modal ran first."""
    from struct_agent.tools.model import create_frame_model

    project = create_frame_model(fc=27580.0, bay_widths_x=[5.0, 6.0], bay_widths_y=[5.0], story_heights=[4.0, 3.5],
                                 dimension="3d").data["project"]
    apply_gravity_loads(project, occupancy="office", superimposed_dead=5.1, partition_weight=0.48)
    elf = dict(sds=1.2, sd1=0.6, s1=0.5, r_factor=8.0, system_type="rc_smf", direction="x", distribution="uniform")
    before = apply_lateral_loads(project, **elf).data["seismic_base_shear"]
    assert run_analysis(project, analysis_case_id="M", analysis_type="modal", num_modes=3).ok
    assert getattr(project.model.mass_source, "id", None) == "MS_ASCE7"
    assert apply_lateral_loads(project, **elf).data["seismic_base_shear"] == pytest.approx(before)
    assert getattr(project.model.mass_source, "id", None) == "MS_ASCE7"  # kept for later dynamic analyses


def test_h2_agent_apply_gravity_loads_open_panel():
    """H2: apply_gravity_loads raises on open floor panels unless allow_partial_floor=True."""
    from struct_core import Project, Metadata, parse_units
    from struct_core.structural_model import Node, BeamElement, StructuralModel, RectangularSection, ElasticMaterial
    from struct_agent.tools.loads import apply_gravity_loads

    nodes = []
    nid = 1
    for z in (0.0, 3.5):
        for y in (0.0, 5.0):
            for x in (0.0, 6.0, 12.0):
                nodes.append(Node(id=nid, x=x, y=y, z=z))
                nid += 1
    elements = []
    for i in range(1, 7):
        elements.append(BeamElement(id=f'C{i}', start_node=i, end_node=i+6, section_id='SEC'))
    elements.append(BeamElement(id='B_bot1', start_node=7, end_node=8, section_id='SEC'))
    elements.append(BeamElement(id='B_top1', start_node=10, end_node=11, section_id='SEC'))
    elements.append(BeamElement(id='B_left', start_node=7, end_node=10, section_id='SEC'))
    elements.append(BeamElement(id='B_mid', start_node=8, end_node=11, section_id='SEC'))
    elements.append(BeamElement(id='B_bot2', start_node=8, end_node=9, section_id='SEC'))
    # B_top2 omitted -> open panel in Bay 2
    elements.append(BeamElement(id='B_right', start_node=9, end_node=12, section_id='SEC'))

    sec = RectangularSection(id='SEC', b=0.3, h=0.5, material_id='MAT')
    mat = ElasticMaterial(id='MAT', E=2.5e7, nu=0.2, unit_weight=24.0, rho=2400.0)
    model = StructuralModel(dimension='3D', nodes=nodes, elements=elements, sections=[sec], materials=[mat])
    project = Project(model=model, metadata=Metadata(title='Open Bay Model', units=parse_units('kN, m')))

    # Refuses by default
    res = apply_gravity_loads(project, occupancy='office', superimposed_dead=1.5, two_way=True, allow_partial_floor=False)
    assert not res.ok
    assert "loaded panel area (30.00) is less than floor outline area (60.00)" in res.error.message

    # Succeeds with allow_partial_floor=True
    res_allow = apply_gravity_loads(project, occupancy='office', superimposed_dead=1.5, two_way=True, allow_partial_floor=True)
    assert res_allow.ok, res_allow.error
    assert res_allow.data["loaded_floor_area"] == {3.5: 30.0}
    assert set(res_allow.data["unloaded_beams"]) == {'B_bot2', 'B_right'}
