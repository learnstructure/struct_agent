"""
End-to-end smoke test for the struct_agent orchestrator.

This test runs the full canonical pipeline (Create → Validate → Analyze →
Design → Inspect → Draw) using the actual engineering libraries. It is
deliberately lightweight — it only checks that the agent reaches the
final workflow stage and produces the expected artifacts.
"""

import os
from pathlib import Path

import pytest

from struct_agent import StructAgent


@pytest.mark.skipif(
    os.environ.get("STRUCT_AGENT_SKIP_E2E") == "1",
    reason="E2E pipeline is opt-out for fast test runs",
)
def test_demo_pipeline_runs_to_completion(tmp_path: Path):
    agent = StructAgent()
    dxf = tmp_path / "out.dxf"
    json_path = tmp_path / "project.json"

    state = agent.run_demo_pipeline(
        output_dxf=str(dxf),
        save_json_path=str(json_path),
        analysis_case_id="Static_1",
        design_case_id="ACI318_19",
    )

    assert state.stage.value == "DRAWINGS_GENERATED"
    # The pipeline should have produced a DXF and a JSON file.
    assert dxf.exists()
    assert json_path.exists()
    # And there should be at least one tool call per stage.
    tool_names = {entry["tool"] for entry in state.tool_log}
    expected_tools = {
        "create_project",
        "validate_model",
        "run_analysis",
        "inspect_analysis_results",
        "run_design",
        "inspect_design_results",
        "inspect_detailing_results",
        "generate_drawings",
        "save_project",
    }
    assert expected_tools.issubset(tool_names)

    # Detailing and rebar schedule must be dynamically computed
    det_res = state.project.detailing_results.get("ACI318_19")
    assert det_res is not None
    assert det_res.aggregate_schedule is not None
    assert len(det_res.aggregate_schedule.items) > 0
    assert det_res.aggregate_schedule.total_weight > 0.0


def test_drawing_generation_requires_design():
    """Drawing generation must fail when there is no DesignResult."""
    from struct_core import Project, StructuralModel
    from struct_agent import generate_drawings

    project = Project(model=StructuralModel())
    result = generate_drawings(
        project,
        design_case_id="anything",
        output_path="should_not_be_created.dxf",
    )
    assert result.ok is False
    assert result.error.error_type == "workflow_error"