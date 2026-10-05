"""
Tests for VLM vision encoding, tool integration, and AgentRunner multi-modal execution.
"""

import io
from pathlib import Path
from PIL import Image
import pytest

from struct_agent.vision.encoding import (
    encode_file_to_data_urls,
    build_multimodal_message,
)
from struct_agent.tools.model import create_model_from_plan_draft
from struct_agent.llm import (
    AgentRunner,
    BaseLLMProvider,
    LLMResponse,
    TextChunkEvent,
    ToolCallRequest,
)


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a minimal PNG file for testing."""
    img_path = tmp_path / "test_drawing.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path, format="PNG")
    return img_path


def test_encode_png(sample_png: Path):
    urls = encode_file_to_data_urls(str(sample_png))
    assert len(urls) == 1
    assert urls[0].startswith("data:image/png;base64,")


def test_build_multimodal_message_text_only():
    msg = build_multimodal_message("Hello engineer", None)
    assert msg == {"role": "user", "content": "Hello engineer"}

    msg_empty = build_multimodal_message("Hello engineer", [])
    assert msg_empty == {"role": "user", "content": "Hello engineer"}


def test_build_multimodal_message_with_attachment(sample_png: Path):
    msg = build_multimodal_message("Interpret this plan", [str(sample_png)])
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 2
    assert msg["content"][0] == {"type": "text", "text": "Interpret this plan"}
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_create_model_from_plan_draft_direct():
    draft_dict = {
        "title": "3-Story RC Office",
        "unit_system": "metric",
        "x_grids": [
            {"name": "1", "coordinate": 0.0},
            {"name": "2", "coordinate": 6.0},
            {"name": "3", "coordinate": 12.0},
        ],
        "y_grids": [
            {"name": "A", "coordinate": 0.0},
            {"name": "B", "coordinate": 5.0},
        ],
        "stories": [
            {"name": "Ground to 1st", "height": 3.5},
            {"name": "1st to 2nd", "height": 3.0},
            {"name": "2nd to Roof", "height": 3.0},
        ],
        "default_col_b": 0.45,
        "default_col_h": 0.45,
        "default_beam_b": 0.3,
        "default_beam_h": 0.5,
    }

    result = create_model_from_plan_draft(**draft_dict)
    assert result.ok
    assert result.data["n_nodes"] == 3 * 2 * 4  # 3 x-grids, 2 y-grids, 4 vertical levels = 24
    assert result.data["n_supports"] == 6      # 6 base nodes
    assert result.data["n_stories"] == 3
    assert "kN, m" in result.data["units"]


class MockVLMProvider(BaseLLMProvider):
    def __init__(self, turns):
        super().__init__(model="mock-vlm")
        self.turns = list(turns)
        self.turn_idx = 0
        self.last_messages = []

    def generate(self, messages, tools=None, temperature=0.0):
        self.last_messages = list(messages)
        if self.turn_idx < len(self.turns):
            resp = self.turns[self.turn_idx]
            self.turn_idx += 1
            return resp
        return LLMResponse(content="Ready.")

    def generate_stream(self, messages, tools=None, temperature=0.0):
        resp = self.generate(messages, tools, temperature)
        if resp.content:
            yield TextChunkEvent(resp.content)
        return resp


def test_runner_multimodal_plan_interpretation(sample_png: Path):
    """
    Test full runner loop when user provides an image attachment and prompts the agent.
    """
    mock_turns = [
        # Turn 1: VLM responds by invoking create_model_from_plan_draft
        LLMResponse(
            content="I have read the dimensions from Grids 1-3 (0, 6, 12m) and Grids A-B (0, 5m), with 2 stories at 3.5m each. Creating 3D model...",
            tool_calls=[
                ToolCallRequest(
                    id="call_vlm_1",
                    name="create_model_from_plan_draft",
                    arguments={
                        "title": "VLM Modeled Frame",
                        "x_grids": [
                            {"name": "1", "coordinate": 0.0},
                            {"name": "2", "coordinate": 6.0},
                            {"name": "3", "coordinate": 12.0},
                        ],
                        "y_grids": [
                            {"name": "A", "coordinate": 0.0},
                            {"name": "B", "coordinate": 5.0},
                        ],
                        "stories": [
                            {"name": "Story 1", "height": 3.5},
                            {"name": "Story 2", "height": 3.5},
                        ],
                    },
                )
            ],
        ),
        # Turn 2: VLM concludes after inspecting result
        LLMResponse(
            content="Successfully generated the 3D structural model with 18 nodes and 12 columns. Fixed 3D supports placed at base.",
        ),
    ]

    provider = MockVLMProvider(mock_turns)
    runner = AgentRunner(provider=provider, max_turns=5)

    events = list(runner.run_stream(
        user_prompt="Please model this structural floor plan.",
        attachments=[str(sample_png)],
    ))

    # Verify user message was formatted as multimodal in messages sent to provider
    first_user_msg = [m for m in provider.last_messages if m.get("role") == "user"][0]
    assert isinstance(first_user_msg["content"], list)
    assert first_user_msg["content"][0]["text"] == "Please model this structural floor plan."
    assert first_user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    # Verify project was compiled and populated in runner
    assert runner.project is not None
    assert runner.project.metadata.title == "VLM Modeled Frame"
    assert len(runner.project.model.nodes) == 3 * 2 * 3  # 18 nodes
    assert len(runner.project.model.supports) == 6
