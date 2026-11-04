"""
Tests for unit handling integration in struct_agent.
Verifies that canonical force-length unit pairs ('kN, m', 'kip, in', etc.)
properly propagate across project creation, frame generation, dynamics, and tools.
"""

import pytest
from struct_agent.tools.model import create_frame_model, create_project, set_project_units
from struct_agent.tools.dynamics import run_time_history, run_response_spectrum
from struct_core import Units


def test_create_frame_model_units_propagation():
    # Test SI units
    res_si = create_frame_model(fc=27580.0, num_bays=1, bay_width=4.0, num_stories=1, story_height=3.0, units="kN, m")
    assert res_si.ok is True
    assert res_si.data["units"] == "kN, m"
    project_si = res_si.data["project"]
    assert project_si.units.force == "kN"
    assert project_si.units.length == "m"
    assert project_si.metadata.units.force == "kN"
    assert project_si.metadata.units.length == "m"

    # Test US Customary units
    res_us = create_frame_model(fc=4.0, num_bays=1, bay_width=240.0, num_stories=1, story_height=144.0, units="kip, in")
    assert res_us.ok is True
    assert res_us.data["units"] == "kip, in"
    project_us = res_us.data["project"]
    assert project_us.units.force == "kip"
    assert project_us.units.length == "in"
    assert project_us.metadata.units.force == "kip"
    assert project_us.metadata.units.length == "in"


def test_create_project_units_propagation():
    res = create_project(
        title="Test US Frame",
        units="kip, ft",
        model={
            "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}],
            "supports": [{"node_id": 1, "fixity": "fixed"}],
        },
    )
    assert res.ok is True
    project = res.data["project"]
    assert project.units.force == "kip"
    assert project.units.length == "ft"
    assert project.metadata.units.force == "kip"
    assert project.metadata.units.length == "ft"


def test_set_project_units_conversion():
    # Start in kN, m
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=6.0, num_stories=1, story_height=3.5, units="kN, m")
    project = res.data["project"]
    assert project.units.short_label == "kN, m"

    # Convert to kip, in
    res_conv = set_project_units(project=project, units="kip, in", convert_existing=True)
    assert res_conv.ok is True
    assert res_conv.data["units"] == "kip, in"
    assert project.units.short_label == "kip, in"
    assert project.metadata.units.short_label == "kip, in"


def test_dynamics_gravity_acceleration_from_units():
    # Verify that dynamics tools leverage auto-derived gravity acceleration
    frame_si = create_frame_model(fc=27580.0, num_bays=1, bay_width=4.0, num_stories=1, story_height=3.0, units="kN, m")
    proj_si = frame_si.data["project"]
    assert pytest.approx(proj_si.units.gravity_acceleration, rel=1e-3) == 9.80665

    frame_us = create_frame_model(fc=4.0, num_bays=1, bay_width=240.0, num_stories=1, story_height=144.0, units="kip, in")
    proj_us = frame_us.data["project"]
    assert pytest.approx(proj_us.units.gravity_acceleration, rel=1e-3) == 386.08858


def test_create_frame_model_2d_follows_bay_widths():
    """The number of 2D bays comes from bay_widths_x, not the num_bays default."""
    widths = [4.0, 6.0, 3.5]
    res = create_frame_model(
        fc=27580.0,
        dimension="2d",
        bay_widths_x=widths,
        story_heights=[3.5],
    )
    assert res.ok is True
    project = res.data["project"]
    xs = sorted({n.x for n in project.model.nodes})
    assert len(xs) == len(widths) + 1
    assert res.data["bays"] == len(widths)
    assert xs == pytest.approx([0.0, 4.0, 10.0, 13.5])


def test_create_frame_model_us_units_material():
    """Concrete strength, stiffness and unit weight are expressed in the model units (kip, ft), and f'c may be given
    in another stress unit."""
    # Metric baseline (27.58 MPa = 27580 kPa)
    res_si = create_frame_model(
        fc=27580.0,
        dimension="2d",
        bay_widths_x=[6.0],
        story_heights=[3.5],
    )
    assert res_si.ok is True
    mat_si = res_si.data["project"].model.materials[0]

    # US Customary in kip, ft with fc in ksi (4.0 ksi = 576 ksf)
    res_us = create_frame_model(
        fc=4.0 * 144.0,
        dimension="2d",
        bay_widths_x=[20.0],
        story_heights=[12.0],
        units="kip, ft",
    )
    assert res_us.ok is True
    mat_us = res_us.data["project"].model.materials[0]
    assert mat_us.type == "concrete"
    assert mat_us.fc == pytest.approx(4.0 * 144.0)
    assert mat_si.fc == pytest.approx(27580.0)
    kpa_per_ksf = 47.880259
    assert mat_us.E == pytest.approx(mat_si.E / kpa_per_ksf, rel=1e-3)
    assert mat_us.unit_weight == pytest.approx(0.15, rel=1e-6)


def test_default_sections_follow_model_units():
    """Omitted sections are 0.40 m columns and 0.30 x 0.50 m beams in the model's own length unit."""
    res = create_frame_model(
        fc=4.0 * 144.0,
        dimension="2d",
        bay_widths_x=[16.0, 20.0],
        story_heights=[14.0],
        units="kip, ft",
    )
    assert res.ok is True
    project = res.data["project"]
    col = next(s for s in project.model.sections if "col" in str(s.id).lower())
    beam = next(s for s in project.model.sections if "beam" in str(s.id).lower())
    assert col.b == pytest.approx(0.45 / 0.3048)
    assert (beam.b, beam.h) == pytest.approx((0.30 / 0.3048, 0.50 / 0.3048))


def test_dynamic_tools_share_lumped_mass_periods():
    """Modal analysis, RSA, and time history report the same fundamental period."""
    from struct_agent.tools.analysis import run_analysis
    from struct_agent.tools.loads import apply_gravity_loads

    res_m = create_frame_model(
        fc=27580.0,
        dimension="2d",
        bay_widths_x=[5.0, 6.0],
        story_heights=[4.0, 3.5],
    )
    assert res_m.ok is True
    project = res_m.data["project"]

    res_g = apply_gravity_loads(
        project,
        occupancy="office",
        tributary_width=5.0,
        superimposed_dead=1.5,
    )
    assert res_g.ok is True

    res_modal = run_analysis(
        project,
        analysis_case_id="modal",
        analysis_type="modal",
        num_modes=2,
    )
    assert res_modal.ok is True
    t_modal = project.analysis_results["modal"].modal_result.modes[0].period

    res_rsa = run_response_spectrum(
        project,
        analysis_case_id="rsa",
        sds=1.0,
        sd1=0.6,
    )
    assert res_rsa.ok is True
    t_rsa = res_rsa.data["fundamental_period"]
    assert t_rsa == pytest.approx(t_modal, rel=1e-5)

    res_tha = run_time_history(
        project,
        analysis_case_id="tha",
        ground_motion="elcentro",
    )
    assert res_tha.ok is True
    t_tha = res_tha.data["fundamental_period"]
    assert t_tha == pytest.approx(t_modal, rel=1e-5)


def test_plan_draft_uses_requested_default_sections_and_no_implicit_loads():
    from struct_agent.tools.model import create_model_from_plan_draft
    from struct_agent.tools.materials import concrete_material

    res = create_model_from_plan_draft(
        x_grids=[{"name": "1", "coordinate": 0.0}, {"name": "2", "coordinate": 6.0}],
        y_grids=[{"name": "A", "coordinate": 0.0}, {"name": "B", "coordinate": 5.0}],
        stories=[{"name": "L1", "height": 3.2}],
        default_col_b=0.4,
        default_col_h=0.4,
        default_beam_b=0.3,
        default_beam_h=0.6,
        fc=27600.0,
    )
    assert res.ok is True
    project = res.data["project"]
    beam_dims = sorted({(round(s.b, 4), round(s.h, 4)) for s in project.model.sections if "beam" in str(s.id).lower()})
    col_dims = sorted({(round(s.b, 4), round(s.h, 4)) for s in project.model.sections if "col" in str(s.id).lower()})
    assert beam_dims == [(0.3, 0.6)]
    assert col_dims == [(0.4, 0.4)]
    assert all(len(lc.element_loads) == 0 and len(lc.point_loads) == 0 for lc in project.model.load_cases)
    ref_mat, _ = concrete_material("ref", fc=27600.0, units="kN, m")
    mat = project.model.materials[0]
    assert mat.E == pytest.approx(ref_mat.E)

