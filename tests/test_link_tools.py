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


def test_import_sap2000_with_self_weight(fixtures_dir):
    s2k_path = fixtures_dir / "sample_frame.s2k"
    res = import_from_sap2000(s2k_path, materialize_self_weight=True)
    assert res.ok
    project = res.data["project"]
    # Element loads should have been augmented with self-weight
    dead_case = next(lc for lc in project.model.load_cases if lc.id == "DEAD")
    assert len(dead_case.element_loads) >= 1
    assert dead_case.include_self_weight is False


def test_import_sap2000_allow_partial_import_flag(tmp_path):
    """H3: allow_partial_import flag allows importing models with unsupported tables."""
    s2k_text = (
        'TABLE:  "PROGRAM CONTROL"\n'
        '   ProgramName=SAP2000   Version=24.0.0   CurrUnits="KN, m, C"\n'
        'TABLE:  "JOINT COORDINATES"\n'
        '   Joint=1   CoordSys=GLOBAL   X=0   Y=0   Z=0\n'
        'TABLE:  "OPTIONS - COLORS - DISPLAY"\n'
        '   ColorType=Screen   Color=Black\n'
        'TABLE:  "JOINT SPRING ASSIGNMENTS 1 - UNCOUPLED"\n'
        '   Joint=1   CoordSys=GLOBAL   U1=1000   U2=0   U3=0\n'
        'END TABLE DATA\n'
    )
    s2k_file = tmp_path / "with_spring.s2k"
    s2k_file.write_text(s2k_text, encoding="utf-8")

    # Default raises / fails
    res_fail = import_from_sap2000(s2k_file, allow_partial_import=False)
    assert not res_fail.ok
    assert "unsupported table" in res_fail.error.message

    # allow_partial_import=True succeeds
    res_ok = import_from_sap2000(s2k_file, allow_partial_import=True)
    assert res_ok.ok
    assert "OPTIONS - COLORS - DISPLAY" in res_ok.data.get("ignored_tables", [])
    assert any("JOINT SPRING ASSIGNMENTS" in msg for msg in res_ok.data.get("import_notes", []))


def test_export_sap2000_allow_partial_export_flag(tmp_path):
    """H3: allow_partial_export flag allows exporting models with support springs."""
    from struct_core import ElasticMaterial, Node, Project, RectangularSection, StructuralModel, Support

    nodes = [Node(id="1", x=0, y=0, z=0), Node(id="2", x=0, y=0, z=3)]
    mat = ElasticMaterial(id="E1", E=3e7, nu=0.2)
    sec = RectangularSection(id="S1", material_id="E1", b=0.3, h=0.3)
    supports = [Support(node_id="1", ux=True, uy=True, uz=True, rx=True, ry=True, rz=True, ky=1000.0)]
    model = StructuralModel(units="kN, m", nodes=nodes, materials=[mat], sections=[sec], supports=supports)
    proj = Project(model=model, units="kN, m")

    out_s2k = tmp_path / "spring_out.s2k"

    # Default export fails
    res_fail = export_to_sap2000(proj, output_path=out_s2k, allow_partial_export=False)
    assert not res_fail.ok
    assert "not supported for export" in res_fail.error.message

    # allow_partial_export=True succeeds and reports dropped items
    res_ok = export_to_sap2000(proj, output_path=out_s2k, allow_partial_export=True)
    assert res_ok.ok
    assert "dropped_items" in res_ok.data or "dropped_springs" in res_ok.data
