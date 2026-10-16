"""
Test the flagship end-to-end multi-story building pipeline.
Verifies integration across struct_link -> struct_load -> fem3d -> aci318 -> struct_draw.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import importlib.util

_pipeline_path = repo_root / "struct_link" / "examples" / "run_full_building_pipeline.py"
_spec = importlib.util.spec_from_file_location("full_building_pipeline_module", _pipeline_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_pipeline = _mod.run_pipeline


def test_flagship_full_building_pipeline(tmp_path: Path):
    """Run full building pipeline on the SAP2000 frame model and assert end-to-end correctness."""
    results = run_pipeline(output_dir=tmp_path, verbose=False)

    assert results["project"] is not None
    assert results["structure"] is not None
    assert results["total_reaction_z"] > 1000.0  # Non-trivial reaction

    # Check ACI 318 design results
    beam_res = results["beam_design"]
    assert beam_res.status == "Passed"
    assert beam_res.flexure.demand_capacity_ratio < 1.0
    assert beam_res.shear.demand_capacity_ratio < 1.0

    col_res = results["column_design"]
    assert col_res.status == "Passed"
    assert col_res.pm_interaction.demand_capacity_ratio < 1.0

    # Check generated CAD files
    for dxf_file in results["files_generated"]:
        assert dxf_file.exists()
        assert dxf_file.stat().st_size > 0


def test_flagship_pipeline_with_recalculated_gravity(tmp_path: Path):
    """Run pipeline with recalculate_gravity=True to verify load replacement logic."""
    results = run_pipeline(output_dir=tmp_path, recalculate_gravity=True, verbose=False)

    assert results["project"] is not None
    assert results["structure"] is not None
    assert results["total_reaction_z"] > 0.0

    # Ensure replaced cases are present and active
    case_ids = {lc.id for lc in results["project"].model.load_cases}
    assert "Dead" in case_ids or "DEAD" in case_ids
    assert "Live" in case_ids
    assert "EQ_X" in case_ids
    assert "EQ_Y" in case_ids
