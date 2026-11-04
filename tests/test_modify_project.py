"""
Unit tests for declarative model modification tool (modify_project).
"""

import pytest
from struct_agent.tools import create_frame_model, modify_project
from struct_agent.llm import AgentRunner, BaseLLMProvider, LLMResponse, ToolCallRequest, ToolResultEvent


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, turns):
        super().__init__(model="mock-agent")
        self.turns = list(turns)
        self.turn_idx = 0

    def generate(self, messages, tools=None, temperature=0.0):
        if self.turn_idx < len(self.turns):
            resp = self.turns[self.turn_idx]
            self.turn_idx += 1
            return resp
        return LLMResponse(content="Done.")

    def generate_stream(self, messages, tools=None, temperature=0.0):
        resp = self.generate(messages, tools, temperature)
        yield resp




def test_modify_project_update_existing_section():
    # 1. Create initial 1-bay 1-story frame
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5, col_b=0.35, col_h=0.35)
    assert res.ok is True
    project = res.data["project"]

    col_sec = next(s for s in project.model.sections if s.id == "COL")
    assert col_sec.b == 0.35
    assert col_sec.h == 0.35

    # 2. Modify section dimensions via modify_project
    mod_res = modify_project(
        project,
        sections=[{"id": col_sec.id, "b": 0.45, "h": 0.50}],
    )
    assert mod_res.ok is True
    assert mod_res.next_suggested_stage == "MODEL_CREATED"
    assert len(mod_res.data["sections_updated"]) == 1

    updated_sec = next(s for s in project.model.sections if s.id == col_sec.id)
    assert updated_sec.b == 0.45
    assert updated_sec.h == 0.50


def test_modify_project_add_section_and_reassign():
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]
    assert len(project.model.elements) > 0

    first_elem = project.model.elements[0]
    initial_sec_id = first_elem.section_id

    # Add a new section and reassign the first element
    mod_res = modify_project(
        project,
        sections=[{"id": "Heavy_Col_600x600", "b": 0.60, "h": 0.60, "material_id": "CONCRETE"}],
        member_assignments=[{"element_ids": [first_elem.id], "section_id": "Heavy_Col_600x600"}],
    )
    assert mod_res.ok is True
    assert any(s.id == "Heavy_Col_600x600" for s in project.model.sections)
    assert first_elem.section_id == "Heavy_Col_600x600"
    assert first_elem.section_id != initial_sec_id


def test_modify_project_update_materials():
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    mat_id = project.model.materials[0].id
    mod_res = modify_project(
        project,
        materials=[{"id": mat_id, "fc": 41370.0, "rho": 2450.0}],
    )
    assert mod_res.ok is True
    mat = next((m for m in project.model.materials if m.id == mat_id), None)
    assert mat is not None
    assert mat.E > 25000000.0  # calculated from f'c
    assert mat.rho == 2450.0


def test_modify_project_rejects_ids_that_would_change_nothing():
    """A new section or material without member assignments, or unknown elements, fail instead of no-ops."""
    project = create_frame_model(fc=27580.0, bay_widths_x=[5.0, 6.0], story_heights=[3.5]).data["project"]
    before = [(s.id, s.b, s.h) for s in project.model.sections]
    res = modify_project(project, sections=[{"id": "Beam_350x600", "b": 0.35, "h": 0.6}])
    assert not res.ok and "BEAM" in res.error.message and "elements" in res.error.message
    assert [(s.id, s.b, s.h) for s in project.model.sections] == before
    assert not modify_project(project, materials=[{"id": "Concrete_6ksi", "fc_kpa": 41370.0}]).ok
    assert not modify_project(project, member_assignments=[{"element_ids": ["C1"], "section_id": "BEAM"}]).ok
    assert not modify_project(project, member_assignments=[{"element_ids": [project.model.elements[0].id],
                                                            "section_id": "Missing"}]).ok
    # Resizing by the existing ID reports how many elements carry the section
    ok = modify_project(project, sections=[{"id": "BEAM", "b": 0.35, "h": 0.6}])
    assert ok.ok and ok.data["sections_updated"][0]["n_elements"] == 2


def test_modify_project_callable_mutator_fallback():
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    def custom_mutator(model):
        model.elements[0].type = "beam"
        return model

    mod_res = modify_project(project, mutator=custom_mutator)
    assert mod_res.ok is True
    assert mod_res.next_suggested_stage == "MODEL_CREATED"


def test_modify_project_in_runner_pipeline():
    turn_1 = LLMResponse(
        content="Create initial frame",
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="create_frame_model",
                arguments={"num_bays": 1, "bay_width": 6.0, "num_stories": 1, "story_height": 3.5, "fc": 27580.0, "col_b": 0.3, "col_h": 0.3},
            )
        ],
    )
    turn_2 = LLMResponse(
        content="Reflection: columns overstressed, enlarging section dimensions via modify_project",
        tool_calls=[
            ToolCallRequest(
                id="call_2",
                name="modify_project",
                arguments={
                    "sections": [
                        {"id": "COL", "b": 0.45, "h": 0.45}
                    ]
                },
            )
        ],
    )
    mock_provider = MockLLMProvider([turn_1, turn_2])
    runner = AgentRunner(provider=mock_provider, max_turns=3)
    events = list(runner.run_stream("Create frame and reflectively resize overstressed columns"))

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2
    assert tool_results[0].ok is True
    assert tool_results[1].ok is True
    assert tool_results[1].tool_name == "modify_project"

    # Verify that the runner's project model reflects the enlarged section
    sec = next(s for s in runner.project.model.sections if s.id == "COL")
    assert sec.b == 0.45
    assert sec.h == 0.45


def test_modify_project_circular_section():
    from struct_core import CircularSection
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    # Add a circular column section and assign it to the columns
    columns = [e.id for e in project.model.elements if str(e.id).startswith("C")]
    mod_res = modify_project(
        project,
        sections=[{"id": "Pier_D500", "d": 0.50, "material_id": "CONCRETE"}],
        member_assignments=[{"element_ids": columns, "section_id": "Pier_D500"}],
    )
    assert mod_res.ok is True
    sec = next((s for s in project.model.sections if s.id == "Pier_D500"), None)
    assert isinstance(sec, CircularSection)
    assert sec.d == 0.50

    # In-place update of diameter
    mod_res2 = modify_project(
        project,
        sections=[{"id": "Pier_D500", "d": 0.65}],
    )
    assert mod_res2.ok is True
    sec2 = next((s for s in project.model.sections if s.id == "Pier_D500"), None)
    assert isinstance(sec2, CircularSection)
    assert sec2.d == 0.65


def test_modify_project_general_section():
    from struct_core import GeneralSection
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    # Add a general (e.g. steel W-shape) section
    beams = [e.id for e in project.model.elements if str(e.id).startswith("B")]
    mod_res = modify_project(
        project,
        sections=[{"id": "W14x90", "type": "general", "A": 0.0171, "I3": 0.000416, "I2": 0.000151, "J": 0.0000017, "material_id": "CONCRETE"}],
        member_assignments=[{"element_ids": beams, "section_id": "W14x90"}],
    )
    assert mod_res.ok is True
    sec = next((s for s in project.model.sections if s.id == "W14x90"), None)
    assert isinstance(sec, GeneralSection)
    assert sec.A == 0.0171
    assert sec.I3 == 0.000416


def test_modify_project_convert_rectangular_to_circular():
    from struct_core import CircularSection
    res = create_frame_model(fc=27580.0, num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5, col_b=0.35, col_h=0.35)
    project = res.data["project"]

    col_sec = next(s for s in project.model.sections if s.id == "COL")

    # Convert column to circular
    mod_res = modify_project(
        project,
        sections=[{"id": col_sec.id, "type": "circular", "d": 0.55}],
    )
    assert mod_res.ok is True
    updated = next(s for s in project.model.sections if s.id == col_sec.id)
    assert isinstance(updated, CircularSection)
    assert updated.d == 0.55



def test_resizing_updates_stored_self_weight():
    """Self-weight loads stored by apply_gravity_loads follow the new sections after modify_project."""
    from struct_agent.tools import apply_gravity_loads

    project = create_frame_model(fc=27580.0, dimension="2d", bay_widths_x=[5.0, 6.0, 4.0], story_heights=[4.0, 3.5, 3.5],
                                 col_b=0.45, col_h=0.45, beam_b=0.3, beam_h=0.5).data["project"]
    loads = apply_gravity_loads(project, occupancy="office", tributary_width=5.0, superimposed_dead=5.1).data
    # Member self-weight (150 pcf = 23.56 kN/m3, the concrete default) + 5.1 kPa x 5.0 m x 15 m x 3 floors
    gamma = project.model.materials[0].unit_weight
    assert gamma == pytest.approx(0.150 * 157.087464)
    self_weight = gamma * (4 * 11.0 * 0.45 ** 2 + 3 * 15.0 * 0.3 * 0.5)
    assert loads["total_dead_load"] == pytest.approx(self_weight + 5.1 * 5.0 * 15.0 * 3, rel=1e-4)
    assert loads["total_partition_weight"] == pytest.approx(0.48 * 5.0 * 15.0 * 3, rel=1e-4)
    assert loads["force_unit"] == "kN"

    col_id = next(s.id for s in project.model.sections if s.id.startswith("COL"))
    res = modify_project(project, sections=[{"id": col_id, "b": 0.6, "h": 0.6}])
    assert res.ok and res.data["self_weight_updated_cases"] == [loads["dead_case_id"]]
    dead = next(lc for lc in project.model.load_cases if lc.id == loads["dead_case_id"])
    sw = [dl for dl in dead.element_loads if dl.is_self_weight]
    assert len(sw) == len(project.model.elements)
    from struct_agent.tools.loads import _case_total
    grown = gamma * 4 * 11.0 * (0.6 ** 2 - 0.45 ** 2)
    assert _case_total(project, dead.id) == pytest.approx(loads["total_dead_load"] + grown, rel=1e-4)


def test_section_ids_are_stable_and_reported():
    """IDs do not encode sizes (a resized 'Beam_300x500' would mislead the model) and are listed on creation."""
    res = create_frame_model(fc=27580.0, bay_widths_x=[5.0, 6.0], story_heights=[3.5], beam_b=0.3, beam_h=0.6)
    assert res.data["section_ids"] == {"columns": "COL", "beams": "BEAM"}
    project = res.data["project"]
    assert modify_project(project, sections=[{"id": "BEAM", "b": 0.3, "h": 0.4}]).ok
    assert {s.id: (s.b, s.h) for s in project.model.sections}["BEAM"] == (0.3, 0.4)


def test_resize_to_current_size_reports_no_change():
    project = create_frame_model(fc=27580.0, bay_widths_x=[5.0], story_heights=[3.5], beam_b=0.3, beam_h=0.3).data["project"]
    same = modify_project(project, sections=[{"id": "BEAM", "b": 0.3, "h": 0.3}])
    assert same.ok and same.data["sections_updated"][0]["changed"] is False and "did not change" in same.data["note"]
    bigger = modify_project(project, sections=[{"id": "BEAM", "b": 0.3, "h": 0.45}])
    assert bigger.data["sections_updated"][0]["changed"] is True and "note" not in bigger.data
