"""
Lightweight native HTTP provider for OpenAI-compatible REST endpoints.

Zero external dependencies required (uses standard library urllib and json).
Supports OpenAI, local Ollama (http://localhost:11434/v1), Groq, DeepSeek, and OpenRouter.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional

from ..events import LLMEvent, TextChunkEvent, ThoughtEvent
from .base import BaseLLMProvider, LLMResponse, ToolCallRequest


class NativeHTTPProvider(BaseLLMProvider):
    """Zero-dependency HTTP client for any OpenAI-compatible completions API."""

    DEFAULT_API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 60.0,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")
        base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_API_BASE", self.DEFAULT_API_BASE)
        super().__init__(model=model, api_key=key, api_base=base.rstrip("/"))
        self.timeout = timeout

    def _prepare_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        url = f"{self.api_base}/chat/completions"
        payload = self._prepare_payload(messages, tools, temperature, stream=False)
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed with status {exc.code}: {err_msg}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to LLM endpoint at {url}: {exc}") from exc

        choice = res_body.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        raw_tools = message.get("tool_calls") or []

        parsed_tools: List[ToolCallRequest] = []
        for tc in raw_tools:
            fn = tc.get("function", {})
            fname = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args
            parsed_tools.append(ToolCallRequest(id=tc.get("id", ""), name=fname, arguments=args))

        usage = res_body.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=parsed_tools,
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=res_body,
            usage=usage,
        )

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Generator[LLMEvent, None, LLMResponse]:
        """
        Streaming implementation. For simplicity and robustness across mockable/non-mockable
        endpoints, this delegates to generate() and yields chunk events.
        """
        resp = self.generate(messages, tools, temperature)
        if resp.content:
            yield TextChunkEvent(resp.content)
        return resp
