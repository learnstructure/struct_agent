"""
Optional LiteLLM provider for multi-model access (Gemini, Claude, GPT-4, Mistral, etc.).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional

from ..events import LLMEvent, TextChunkEvent
from .base import BaseLLMProvider, LLMResponse, ToolCallRequest


class LiteLLMProvider(BaseLLMProvider):
    """
    Adapter that leverages litellm to support 100+ LLM providers with one unified interface.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        try:
            import litellm  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "The 'litellm' package is required for LiteLLMProvider.\n"
                "Install it with: pip install litellm\n"
                "Or use the zero-dependency provider='native'."
            ) from err

        super().__init__(model=model, api_key=api_key, api_base=api_base)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        import litellm

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = litellm.completion(**kwargs)

        choice = response.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        raw_tools = getattr(msg, "tool_calls", None) or []

        parsed_tools: List[ToolCallRequest] = []
        for tc in raw_tools:
            fn = getattr(tc, "function", None) or {}
            fname = getattr(fn, "name", "")
            raw_args = getattr(fn, "arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
            else:
                args = raw_args
            call_id = getattr(tc, "id", "")
            parsed_tools.append(ToolCallRequest(id=call_id, name=fname, arguments=args))

        usage = getattr(response, "usage", {})
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        }

        return LLMResponse(
            content=content,
            tool_calls=parsed_tools,
            finish_reason=getattr(choice, "finish_reason", "stop"),
            raw_response=response,
            usage=usage_dict,
        )

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Generator[LLMEvent, None, LLMResponse]:
        resp = self.generate(messages, tools, temperature, seed=seed)
        if resp.content:
            yield TextChunkEvent(resp.content)
        return resp
