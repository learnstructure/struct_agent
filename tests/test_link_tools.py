"""
Tests for struct_agent SAP2000 and ETABS link tools.
"""

from pathlib import Path
import pytest
from struct_agent import (
    import_from_sap2000,
    export_to_sap2000,
    import_from_etabs,
    export_to_etabs,
    validate_model,
)


@pytest.fixture
def fixtures_dir():
    # fixtures from struct_link
    return Path(__file__).parent.parent.parent / "struct_link" / "tests" / "fixtures"


def test_import_sap2000(fixtures_dir):
    s2k_path = fixtures_dir / "sample_frame.s2k"
    res = import_from_sap2000(s2k_path)
    assert res.ok, f"Import failed: {res.error}"
    assert res.data["n_nodes"] == 4
    assert res.data["n_elements"] == 3
    assert res.next_suggested_stage == "MODEL_CREATED"

    project = res.data["project"]
    assert project.model is not None

    # Validate imported model
    val = validate_model(project)
    assert val.ok, f"Validation of imported model failed: {val.error}"


def test_export_sap2000(fixtures_dir, tmp_path):
    s2k_path = fixtures_dir / "sample_frame.s2k"
    res = import_from_sap2000(s2k_path)
    assert res.ok
    project = res.data["project"]

    out_s2k = tmp_path / "exported.s2k"
    exp_res = export_to_sap2000(project, output_path=out_s2k)
    assert exp_res.ok, f"Export failed: {exp_res.error}"
    assert out_s2k.exists()


def test_import_etabs(fixtures_dir):
    e2k_path = fixtures_dir / "sample_frame.e2k"
    res = import_from_etabs(e2k_path)
    assert res.ok, f"Import failed: {res.error}"
    assert res.data["n_nodes"] == 4
    assert res.data["n_elements"] == 3
    assert res.next_suggested_stage == "MODEL_CREATED"

    project = res.data["project"]
    val = validate_model(project)
    assert val.ok


def test_export_etabs(fixtures_dir, tmp_path):
    e2k_path = fixtures_dir / "sample_frame.e2k"
    res = import_from_etabs(e2k_path)
    assert res.ok
    project = res.data["project"]

    out_e2k = tmp_path / "exported.e2k"
    exp_res = export_to_etabs(project, output_path=out_e2k)
    assert exp_res.ok
    assert out_e2k.exists()


def test_link_tool_error_handling():
    res = import_from_sap2000("non_existent_file.s2k")
    assert not res.ok
    assert res.error.error_type == "file_not_found"

    exp = export_to_sap2000(None, output_path="out.s2k")
    assert not exp.ok
    assert exp.error.error_type == "missing_input"
