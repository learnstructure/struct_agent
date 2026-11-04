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
        timeout: Optional[float] = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")
        base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_API_BASE", self.DEFAULT_API_BASE)
        super().__init__(model=model, api_key=key, api_base=base.rstrip("/"))
        env_timeout_str = os.environ.get("LLM_TIMEOUT") or os.environ.get("OLLAMA_TIMEOUT")
        default_timeout = float(env_timeout_str) if env_timeout_str else 300.0
        self.timeout = timeout if timeout is not None else default_timeout

    def _prepare_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        url = f"{self.api_base}/chat/completions"
        payload = self._prepare_payload(messages, tools, temperature, seed=seed, stream=False)
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_retries = 5
        base_delay = 5.0
        res_body = None

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                err_msg = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 503) and attempt < max_retries:
                    import time
                    retry_wait = base_delay * (1.5 ** attempt)
                    try:
                        err_json = json.loads(err_msg)
                        if isinstance(err_json, list) and len(err_json) > 0:
                            err_json = err_json[0]
                        details = err_json.get("error", {}).get("details", [])
                        for d in details:
                            if "retryDelay" in d:
                                raw_delay = d["retryDelay"].rstrip("s")
                                retry_wait = max(float(raw_delay) + 1.0, retry_wait)
                    except Exception:
                        pass
                    time.sleep(retry_wait)
                    continue
                raise RuntimeError(f"LLM API request failed with status {exc.code}: {err_msg}") from exc
            except urllib.error.URLError as exc:
                if attempt < max_retries:
                    import time
                    time.sleep(base_delay)
                    continue
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
            reasoning=message.get("reasoning") or None,
        )

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Generator[LLMEvent, None, LLMResponse]:
        """
        Streaming implementation. For simplicity and robustness across mockable/non-mockable
        endpoints, this delegates to generate() and yields chunk events.
        """
        resp = self.generate(messages, tools, temperature, seed=seed)
        content = resp.content or ""
        if "<think>" in content and "</think>" in content:
            parts = content.split("</think>", 1)
            thought_part = parts[0].replace("<think>", "").strip()
            final_content = parts[1].strip()
            if thought_part:
                yield ThoughtEvent(thought_part)
            if final_content:
                yield TextChunkEvent(final_content)
        elif content:
            yield TextChunkEvent(content)
        return resp
