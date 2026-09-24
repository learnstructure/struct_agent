"""
struct_agent.llm: Cognitive orchestration and tool calling layer for LLMs.
"""

from .events import (
    CompleteEvent,
    ErrorEvent,
    EventType,
    LLMEvent,
    StateUpdateEvent,
    TextChunkEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .prompts import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from .providers import (
    BaseLLMProvider,
    LLMResponse,
    NativeHTTPProvider,
    ToolCallRequest,
    get_provider,
)
from .runner import AgentRunner
from .schemas import ENGINEERING_TOOL_SCHEMAS, get_tool_schemas

__all__ = [
    "AgentRunner",
    "EventType",
    "LLMEvent",
    "ThoughtEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StateUpdateEvent",
    "TextChunkEvent",
    "ErrorEvent",
    "CompleteEvent",
    "DEFAULT_SYSTEM_PROMPT",
    "build_system_prompt",
    "get_tool_schemas",
    "ENGINEERING_TOOL_SCHEMAS",
    "BaseLLMProvider",
    "NativeHTTPProvider",
    "LLMResponse",
    "ToolCallRequest",
    "get_provider",
]
