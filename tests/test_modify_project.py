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
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5, col_b=0.35, col_h=0.35)
    assert res.ok is True
    project = res.data["project"]

    col_sec = next(s for s in project.model.sections if "Col" in s.id)
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
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]
    assert len(project.model.elements) > 0

    first_elem = project.model.elements[0]
    initial_sec_id = first_elem.section_id

    # Add a new section and reassign the first element
    mod_res = modify_project(
        project,
        sections=[{"id": "Heavy_Col_600x600", "b": 0.60, "h": 0.60}],
        member_assignments=[{"element_ids": [first_elem.id], "section_id": "Heavy_Col_600x600"}],
    )
    assert mod_res.ok is True
    assert any(s.id == "Heavy_Col_600x600" for s in project.model.sections)
    assert first_elem.section_id == "Heavy_Col_600x600"
    assert first_elem.section_id != initial_sec_id


def test_modify_project_update_materials():
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    mod_res = modify_project(
        project,
        materials=[{"id": "Concrete_6ksi", "fc_kpa": 41370.0, "rho": 2450.0}],
    )
    assert mod_res.ok is True
    mat = next((m for m in project.model.materials if m.id == "Concrete_6ksi"), None)
    assert mat is not None
    assert mat.E > 25000000.0  # calculated from f'c
    assert mat.rho == 2450.0


def test_modify_project_callable_mutator_fallback():
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
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
                arguments={"num_bays": 1, "bay_width": 6.0, "num_stories": 1, "story_height": 3.5, "col_b": 0.3, "col_h": 0.3},
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
                        {"id": "Col_300x300", "b": 0.45, "h": 0.45}
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
    sec = next(s for s in runner.project.model.sections if s.id == "Col_300x300")
    assert sec.b == 0.45
    assert sec.h == 0.45


def test_modify_project_circular_section():
    from struct_core import CircularSection
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    # Add a circular column section
    mod_res = modify_project(
        project,
        sections=[{"id": "Pier_D500", "d": 0.50}],
    )
    assert mod_res.ok is True
    sec = next((s for s in project.model.sections if s.id == "Pier_D500"), None)
    assert isinstance(sec, CircularSection)
    assert sec.d == 0.50

    # In-place update of diameter
    mod_res2 = modify_project(
        project,
        sections=[{"id": "Pier_D500", "diameter": 0.65}],
    )
    assert mod_res2.ok is True
    sec2 = next((s for s in project.model.sections if s.id == "Pier_D500"), None)
    assert isinstance(sec2, CircularSection)
    assert sec2.d == 0.65


def test_modify_project_general_section():
    from struct_core import GeneralSection
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5)
    project = res.data["project"]

    # Add a general (e.g. steel W-shape) section
    mod_res = modify_project(
        project,
        sections=[{"id": "W14x90", "type": "general", "A": 0.0171, "I3": 0.000416, "I2": 0.000151, "J": 0.0000017}],
    )
    assert mod_res.ok is True
    sec = next((s for s in project.model.sections if s.id == "W14x90"), None)
    assert isinstance(sec, GeneralSection)
    assert sec.A == 0.0171
    assert sec.I3 == 0.000416


def test_modify_project_convert_rectangular_to_circular():
    from struct_core import CircularSection
    res = create_frame_model(num_bays=1, bay_width=5.0, num_stories=1, story_height=3.5, col_b=0.35, col_h=0.35)
    project = res.data["project"]

    col_sec = next(s for s in project.model.sections if "Col" in s.id)

    # Convert column to circular
    mod_res = modify_project(
        project,
        sections=[{"id": col_sec.id, "type": "circular", "d": 0.55}],
    )
    assert mod_res.ok is True
    updated = next(s for s in project.model.sections if s.id == col_sec.id)
    assert isinstance(updated, CircularSection)
    assert updated.d == 0.55

