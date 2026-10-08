"""
Tests for load determination and preliminary sizing tools in struct_agent.
"""

import pytest
from struct_core import (
    BeamElement,
    Node,
    Project,
    StructuralModel,
    Support,
)
from struct_agent import (
    WorkflowStage,
    create_project,
    preliminary_sizing,
    apply_gravity_loads,
    apply_lateral_loads,
    apply_self_weight,
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
    from struct_core import LoadCase
    project = _build_bare_frame()
    preliminary_sizing(project)
    project.model.load_cases = [LoadCase(id="DEAD", include_self_weight=True)]

    res = apply_self_weight(project)
    assert res.ok is True
    assert res.data["total_element_loads"] == 3
    assert project.model.load_cases[0].include_self_weight is False
