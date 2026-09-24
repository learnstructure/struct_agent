"""
Structured event definitions for streaming LLM orchestration.

These events allow CLIs, web sockets, and GUI frontends (e.g., Next.js, Three.js,
Streamlit) to display real-time thoughts, tool calls, tool results, and 3D state updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    """Categorization of agent streaming events."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    TEXT_CHUNK = "text_chunk"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class LLMEvent:
    """Base class for all agent orchestration events."""

    event_type: EventType
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class ThoughtEvent(LLMEvent):
    """Internal reasoning or chain-of-thought produced by the agent."""

    thought: str = ""

    def __init__(self, thought: str):
        super().__init__(event_type=EventType.THOUGHT)
        self.thought = thought


@dataclass
class ToolCallEvent(LLMEvent):
    """Notification that the agent decided to invoke an engineering tool."""

    tool_name: str = ""
    call_id: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, tool_name: str, call_id: str, arguments: Dict[str, Any]):
        super().__init__(event_type=EventType.TOOL_CALL)
        self.tool_name = tool_name
        self.call_id = call_id
        self.arguments = arguments


@dataclass
class ToolResultEvent(LLMEvent):
    """Result returned by a deterministic engineering tool function."""

    tool_name: str = ""
    call_id: str = ""
    ok: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    next_stage: Optional[str] = None

    def __init__(
        self,
        tool_name: str,
        call_id: str,
        ok: bool,
        data: Dict[str, Any],
        error_message: Optional[str] = None,
        next_stage: Optional[str] = None,
    ):
        super().__init__(event_type=EventType.TOOL_RESULT)
        self.tool_name = tool_name
        self.call_id = call_id
        self.ok = ok
        self.data = data
        self.error_message = error_message
        self.next_stage = next_stage


@dataclass
class StateUpdateEvent(LLMEvent):
    """
    Emitted whenever the underlying struct_core.Project model changes.
    Allows 3D viewports in GUIs to re-render in real time.
    """

    project: Any = None
    stage: str = "UNKNOWN"

    def __init__(self, project: Any, stage: str):
        super().__init__(event_type=EventType.STATE_UPDATE)
        self.project = project
        self.stage = stage


@dataclass
class TextChunkEvent(LLMEvent):
    """Streamed natural language tokens meant for user chat UI."""

    chunk: str = ""

    def __init__(self, chunk: str):
        super().__init__(event_type=EventType.TEXT_CHUNK)
        self.chunk = chunk


@dataclass
class ErrorEvent(LLMEvent):
    """Top-level error encountered during orchestration."""

    message: str = ""
    details: Optional[Dict[str, Any]] = None

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(event_type=EventType.ERROR)
        self.message = message
        self.details = details or {}


@dataclass
class CompleteEvent(LLMEvent):
    """Fired when the multi-turn agent finishes its objective."""

    final_message: str = ""
    total_turns: int = 0
    governing_dcr: Optional[float] = None
    passed: Optional[bool] = None

    def __init__(
        self,
        final_message: str,
        total_turns: int,
        governing_dcr: Optional[float] = None,
        passed: Optional[bool] = None,
    ):
        super().__init__(event_type=EventType.COMPLETE)
        self.final_message = final_message
        self.total_turns = total_turns
        self.governing_dcr = governing_dcr
        self.passed = passed
