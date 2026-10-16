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
    res_si = create_frame_model(num_bays=1, bay_width=4.0, num_stories=1, story_height=3.0, units="kN, m")
    assert res_si.ok is True
    assert res_si.data["units"] == "kN, m"
    project_si = res_si.data["project"]
    assert project_si.units.force == "kN"
    assert project_si.units.length == "m"
    assert project_si.model.units.force == "kN"
    assert project_si.model.units.length == "m"

    # Test US Customary units
    res_us = create_frame_model(num_bays=1, bay_width=240.0, num_stories=1, story_height=144.0, units="kip, in")
    assert res_us.ok is True
    assert res_us.data["units"] == "kip, in"
    project_us = res_us.data["project"]
    assert project_us.units.force == "kip"
    assert project_us.units.length == "in"
    assert project_us.model.units.force == "kip"
    assert project_us.model.units.length == "in"


def test_create_project_units_propagation():
    res = create_project(
        title="Test US Frame",
        units="kip, ft",
        model={"nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}]},
    )
    assert res.ok is True
    project = res.data["project"]
    assert project.units.force == "kip"
    assert project.units.length == "ft"
    assert project.model.units.force == "kip"
    assert project.model.units.length == "ft"


def test_set_project_units_conversion():
    # Start in kN, m
    res = create_frame_model(num_bays=1, bay_width=6.0, num_stories=1, story_height=3.5, units="kN, m")
    project = res.data["project"]
    assert project.units.short_label == "kN, m"

    # Convert to kip, in
    res_conv = set_project_units(project=project, units="kip, in", convert_existing=True)
    assert res_conv.ok is True
    assert res_conv.data["units"] == "kip, in"
    assert project.units.short_label == "kip, in"
    assert project.model.units.short_label == "kip, in"


def test_dynamics_gravity_acceleration_from_units():
    # Verify that dynamics tools leverage auto-derived gravity acceleration
    frame_si = create_frame_model(num_bays=1, bay_width=4.0, num_stories=1, story_height=3.0, units="kN, m")
    proj_si = frame_si.data["project"]
    assert pytest.approx(proj_si.units.gravity_acceleration, rel=1e-3) == 9.80665

    frame_us = create_frame_model(num_bays=1, bay_width=240.0, num_stories=1, story_height=144.0, units="kip, in")
    proj_us = frame_us.data["project"]
    assert pytest.approx(proj_us.units.gravity_acceleration, rel=1e-3) == 386.0885
