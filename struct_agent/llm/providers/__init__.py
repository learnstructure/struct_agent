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
    timeout: Optional[float] = None,
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
    timeout : float, optional
        Request timeout in seconds.
    """
    p_lower = provider.lower().strip()

    # Ollama provider or models tagged with ollama/ or local ollama base
    if p_lower == "ollama" or model.startswith("ollama/") or (api_base and "11434" in api_base):
        clean_model = model[7:] if model.startswith("ollama/") else model
        import os
        base = api_base or os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not base.endswith("/v1"):
            base = f"{base.rstrip('/')}/v1"
        key = api_key or "ollama"
        return NativeHTTPProvider(model=clean_model, api_key=key, api_base=base, timeout=timeout)

    # Google Gemini direct support (uses litellm if installed, else native Google OpenAI-compatible endpoint)
    if p_lower == "gemini" or ("gemini" in model.lower() and p_lower in ("auto", "gemini")):
        import os
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        try:
            from .litellm_client import LiteLLMProvider
            return LiteLLMProvider(model=model, api_key=key, api_base=api_base)
        except ImportError:
            pass
        clean_model = model[7:] if model.startswith("gemini/") else model
        base = api_base or "https://generativelanguage.googleapis.com/v1beta/openai"
        return NativeHTTPProvider(model=clean_model, api_key=key, api_base=base, timeout=timeout)

    if p_lower in ("native", "openai", "groq", "deepseek"):
        return NativeHTTPProvider(model=model, api_key=api_key, api_base=api_base, timeout=timeout)

    if p_lower == "litellm":
        from .litellm_client import LiteLLMProvider
        return LiteLLMProvider(model=model, api_key=api_key, api_base=api_base)

    if p_lower == "auto":
        # Check if model requires litellm (e.g. claude-)
        if "/" in model or "claude" in model:
            try:
                from .litellm_client import LiteLLMProvider
                return LiteLLMProvider(model=model, api_key=api_key, api_base=api_base)
            except ImportError:
                pass
        return NativeHTTPProvider(model=model, api_key=api_key, api_base=api_base)

    raise ValueError(f"Unknown provider: {provider!r}. Choose 'native', 'ollama', 'gemini', 'litellm', or 'auto'.")


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "ToolCallRequest",
    "NativeHTTPProvider",
    "get_provider",
]
