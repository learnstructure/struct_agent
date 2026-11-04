"""
Abstract base class and data containers for LLM providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from ..events import LLMEvent


@dataclass
class ToolCallRequest:
    """A tool invocation requested by the language model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response returned by any LLM provider."""

    content: Optional[str] = None
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None
    usage: Dict[str, int] = field(default_factory=dict)
    # Reasoning of a thinking model, as the endpoint returned it (Ollama and OpenRouter: ``message.reasoning``).
    # It goes back with the turn's tool calls: the chat templates of gpt-oss and Qwen3 render the reasoning of
    # the current tool-call chain, and without it gpt-oss ended runs mid-workflow with an empty reply.
    reasoning: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLMProvider(ABC):
    """Abstract interface for LLM backends (OpenAI, Ollama, Gemini, Claude, etc.)."""

    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        # Most tokens one response may generate, thinking included (None: the server's default); a response
        # cut off at it ends with finish_reason 'length'
        self.max_tokens: Optional[int] = None
        # Thinking effort of a reasoning model ('low', 'medium', 'high'; None: the server's default). Through
        # Ollama's OpenAI-compatible endpoint gemma4 thinks only when it is set; qwen3.5 and minimax always think.
        self.reasoning_effort: Optional[str] = None

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        """Execute a blocking chat completion with tool calling."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Generator[LLMEvent, None, LLMResponse]:
        """Stream tokens and tool calls, yielding LLMEvents in real time."""
        raise NotImplementedError
