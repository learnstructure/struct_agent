"""
Tests for load determination and preliminary sizing tools in struct_agent.
"""

import pytest
from struct_core import (
    AreaLoad,
    BeamElement,
    ElasticMaterial,
    LoadCase,
    Metadata,
    Node,
    Project,
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
    """Frame geometry with nodes and elements, but NO sections, NO materials, and NO loads."""
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=0.0, y=3.5),
        Node(id=3, x=6.0, y=3.5),
        Node(id=4, x=6.0, y=0.0),
    ]
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="", section_id=""),
        BeamElement(id=2, start_node=2, end_node=3, material_id="", section_id=""),
        BeamElement(id=3, start_node=4, end_node=3, material_id="", section_id=""),
    ]
    supports = [Support.fixed(node_id=1), Support.fixed(node_id=4)]
    model = StructuralModel(nodes=nodes, elements=elements, supports=supports)
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
    col = RectangularSection(id="COL", name="C400", b=0.4, h=0.4)
    bm = RectangularSection(id="BEAM", name="B300", b=0.3, h=0.5)
    elements = [
        BeamElement(id="C1", start_node=1, end_node=5, material_id="CONC", section_id="COL"),
        BeamElement(id="C2", start_node=2, end_node=6, material_id="CONC", section_id="COL"),
        BeamElement(id="C3", start_node=3, end_node=7, material_id="CONC", section_id="COL"),
        BeamElement(id="C4", start_node=4, end_node=8, material_id="CONC", section_id="COL"),
        BeamElement(id="B1", start_node=5, end_node=6, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B2", start_node=6, end_node=7, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B3", start_node=7, end_node=8, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B4", start_node=8, end_node=5, material_id="CONC", section_id="BEAM"),
    ]
    model = StructuralModel(nodes=nodes, elements=elements, materials=[mat], sections=[col, bm])
    return Project(model=model, metadata=Metadata(units=Units(length="m", force="kN")))


def test_preliminary_sizing_and_loading_flow():
    # 1. Bare model
    project = _build_bare_frame()

    # 2. Preliminary sizing tool sizes beam and column automatically
    size_res = preliminary_sizing(project)
    assert size_res.ok is True
    assert size_res.data["beams_sized"] == 1
    assert size_res.data["columns_sized"] == 2
    assert len(project.model.sections) == 2
    assert len(project.model.materials) == 1

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
    assert len(project.model.load_cases) == 2

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
    res = apply_roof_loads(project, ground_snow_load=1.5, tributary_width=4.0)
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
    apply_gravity_loads(project, occupancy="office")

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
    apply_gravity_loads(project, occupancy="office")

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

