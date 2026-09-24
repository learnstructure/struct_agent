"""
LLM Provider registry and factory.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseLLMProvider, LLMResponse, ToolCallRequest
from .http_client import NativeHTTPProvider


def get_provider(
    provider: str = "auto",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Factory function to retrieve an instantiated LLM provider.

    Parameters
    ----------
    provider : str
        'native' (or 'openai', 'ollama'), 'litellm', or 'auto'.
    model : str
        Model identifier (e.g. 'gpt-4o', 'gemini/gemini-2.0-flash', 'claude-3-5-sonnet-20241022', 'llama3').
    api_key : str, optional
        API key (or taken from environment variables).
    api_base : str, optional
        Custom API endpoint (e.g. 'http://localhost:11434/v1' for Ollama).
    """
    p_lower = provider.lower().strip()

    if p_lower in ("native", "openai", "ollama", "groq", "deepseek"):
        return NativeHTTPProvider(model=model, api_key=api_key, api_base=api_base)

    if p_lower == "litellm":
        from .litellm_client import LiteLLMProvider
        return LiteLLMProvider(model=model, api_key=api_key, api_base=api_base)

    if p_lower == "auto":
        # Check if model requires litellm (e.g. gemini/ or claude-)
        if "/" in model or "claude" in model or "gemini" in model:
            try:
                from .litellm_client import LiteLLMProvider
                return LiteLLMProvider(model=model, api_key=api_key, api_base=api_base)
            except ImportError:
                pass
        return NativeHTTPProvider(model=model, api_key=api_key, api_base=api_base)

    raise ValueError(f"Unknown provider: {provider!r}. Choose 'native', 'litellm', or 'auto'.")


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "ToolCallRequest",
    "NativeHTTPProvider",
    "get_provider",
]
