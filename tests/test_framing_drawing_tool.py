"""
Tests for unified generate_drawings tool and GA drawing integration in struct_agent.
"""

from pathlib import Path
import pytest

from struct_agent.llm.runner import AgentRunner
from struct_agent.tools import create_frame_model, generate_drawings, generate_framing_drawings


def test_generate_drawings_framing_mode(tmp_path):
    """Verify generate_drawings with drawing_type='framing' generates valid DXF drawings."""
    res_model = create_frame_model(
        fc=27580.0,
        num_bays=3,
        bay_width=6.0,
        num_stories=2,
        story_height=3.5,
        title="3-Bay 2-Story Test Frame",
    )
    assert res_model.status == "success"
    project = res_model.data["project"]

    # 1. Combined Framing Sheet
    comb_dxf = tmp_path / "combined_test.dxf"
    res_comb = generate_drawings(
        project,
        output_path=str(comb_dxf),
        drawing_type="framing",
        view_mode="both",
    )
    assert res_comb.status == "success"
    assert comb_dxf.exists()
    assert comb_dxf.stat().st_size > 1000
    assert res_comb.data["drawing_type"] == "framing"
    assert res_comb.data["view_mode"] == "both"
    assert res_comb.data["elements_drawn"] == len(project.model.elements)

    # 2. Plan View
    plan_dxf = tmp_path / "plan_test.dxf"
    res_plan = generate_drawings(
        project,
        output_path=str(plan_dxf),
        drawing_type="framing",
        view_mode="plan",
        story_level="ROOF",
    )
    assert res_plan.status == "success"
    assert plan_dxf.exists()
    assert res_plan.data["view_mode"] == "plan"

    # 3. Elevation View
    elev_dxf = tmp_path / "elev_test.dxf"
    res_elev = generate_drawings(
        project,
        output_path=str(elev_dxf),
        drawing_type="framing",
        view_mode="elevation",
    )
    assert res_elev.status == "success"
    assert elev_dxf.exists()
    assert res_elev.data["view_mode"] == "elevation"


def test_generate_drawings_auto_fallback_to_framing(tmp_path):
    """Verify generate_drawings in default auto mode without design results generates GA framing drawings."""
    res_model = create_frame_model(fc=27580.0, num_bays=2, bay_width=5.0, num_stories=1, story_height=3.0)
    project = res_model.data["project"]

    dxf_out = tmp_path / "fallback_frame.dxf"
    res = generate_drawings(project, output_path=str(dxf_out))

    assert res.status == "success"
    assert dxf_out.exists()
    assert res.data["drawing_type"] == "framing"
    assert res.data["elements_drawn"] == len(project.model.elements)


def test_generate_framing_drawings_compatibility_wrapper(tmp_path):
    """Verify the backward-compatible Python wrapper generate_framing_drawings works."""
    res_model = create_frame_model(fc=27580.0, num_bays=2, bay_width=4.0, num_stories=1, story_height=3.0)
    project = res_model.data["project"]

    dxf_out = tmp_path / "wrapper_frame.dxf"
    res = generate_framing_drawings(project, output_path=str(dxf_out))

    assert res.status == "success"
    assert dxf_out.exists()
    assert res.data["drawing_type"] == "framing"


def test_agent_runner_unified_drawing_dispatch(tmp_path):
    """Verify AgentRunner executes unified generate_drawings with automatic project injection."""
    runner = AgentRunner(provider="auto", model="test")
    res_model = runner._execute_tool(
        "create_frame_model",
        {"num_bays": 2, "bay_width": 5.0, "num_stories": 2, "story_height": 3.0, "fc": 27580.0},
    )
    runner.project = res_model.data["project"]

    dxf_out = tmp_path / "runner_framing.dxf"
    res_draw = runner._execute_tool(
        "generate_drawings",
        {"output_path": str(dxf_out), "drawing_type": "framing", "view_mode": "both"},
    )

    assert res_draw.status == "success"
    assert dxf_out.exists()
    assert res_draw.data["drawing_type"] == "framing"
    assert res_draw.data["elements_drawn"] == len(runner.project.model.elements)


def test_generate_drawings_auto_with_3d_both_elevations(tmp_path):
    """Verify generate_drawings in 3D auto mode generates framing plan and elevations in both directions."""
    res_model = create_frame_model(fc=27580.0, num_bays=2, bay_width=6.0, num_stories=2, story_height=3.5, dimension="3d")
    project = res_model.data["project"]

    dxf_out = tmp_path / "auto_3d_composite.dxf"
    res = generate_drawings(project, output_path=str(dxf_out), drawing_type="auto")

    assert res.status == "success"
    assert dxf_out.exists()
    assert dxf_out.stat().st_size > 1000
    assert res.data["drawing_type"] == "framing"


def test_generate_drawings_auto_with_design_includes_framing_and_detailing(tmp_path):
    """Verify that when design results are available, generate_drawings produces general plan & elevation by default in addition to detailed drawings."""
    import ezdxf
    from struct_agent.tools.analysis import run_analysis
    from struct_agent.tools.design import run_design
    from struct_agent.tools.loads import apply_gravity_loads, generate_load_combinations

    res_model = create_frame_model(fc=27580.0, num_bays=2, bay_width=6.0, num_stories=2, story_height=3.5, dimension="2d")
    project = res_model.data["project"]

    # Design the factored load combinations
    assert apply_gravity_loads(project, occupancy="office", tributary_width=5.0, superimposed_dead=1.5).ok
    assert generate_load_combinations(project).ok
    res_ana = run_analysis(project, analysis_case_id="case1", analysis_type="static")
    assert res_ana.status == "success"

    res_des = run_design(project, design_case_id="case1", design_code="ACI 318-19")
    assert res_des.status == "success"

    # Default generate_drawings (no drawing_type passed, defaults to 'auto')
    dxf_out = tmp_path / "composite_framing_and_detailing.dxf"
    res_draw = generate_drawings(project, output_path=str(dxf_out))

    assert res_draw.status == "success"
    assert dxf_out.exists()
    assert res_draw.data["has_framing"] is True
    assert res_draw.data["has_detailing"] is True
    assert res_draw.data["drawing_type"] in ("composite", "detailing")
    assert res_draw.data["elements_drawn"] > 0

    # Inspect the DXF text content to ensure framing plan, frame elevation, and member sections are present
    doc = ezdxf.readfile(str(dxf_out))
    msp = doc.modelspace()
    texts = [e.dxf.text for e in msp.query("TEXT")]
    text_content = " ".join(texts)

    assert "FRAMING PLAN" in text_content
    assert "FRAME ELEVATION" in text_content
    assert any("COLUMN" in t or "BEAM" in t or "SEC" in t for t in texts)


