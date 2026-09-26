"""
harness.model
=============

Provider-agnostic LLM abstraction layer.

Exposes a single :func:`call_model` function that the rest of the harness
uses for **every** LLM interaction.  No other module should import a provider
SDK directly — this gateway centralises configuration loading, provider
dispatch, token-usage extraction, and latency measurement.

Configuration is read from ``config.yaml`` (at the project root) or
overridden via environment variables.  The actual API key is **never**
stored in the config file — only the *name* of the environment variable
that holds it.

Provider support:
  - ``anthropic`` (default) — Anthropic Claude with native tool-calling
  - ``openai``  — planned
  - ``google``  — planned

To add a new provider, implement a private ``_call_<provider>()`` function
and register it in :data:`_PROVIDER_DISPATCH`.

Typical usage::

    from harness.model import call_model

    # Simple text completion
    resp = call_model(
        messages=[{"role": "user", "content": "Summarise this file"}],
    )
    print(resp.text)

    # With tool-calling
    tools = [{"name": "read_file", "description": "…", "input_schema": {…}}]
    resp = call_model(
        messages=[{"role": "user", "content": "Read main.py"}],
        tools=tools,
    )
    for block in resp.tool_calls:
        print(block)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Resolved path to the project root (one level above ``harness/``)."""

_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"
"""Default location for the configuration file."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the model abstraction layer.

    Attributes:
        provider:      Which LLM backend to use.
        api_key:       API key / token (resolved from the env var at load time).
        base_url:      Optional override for the API base URL
                       (useful for Azure, local proxies, vLLM, etc.).
        default_model: Model identifier to use when the caller does not
                       specify one explicitly (e.g. ``"claude-sonnet-4-20250514"``).
        temperature:   Default sampling temperature.
        max_tokens:    Default max tokens to generate.
        extra:         Provider-specific keyword arguments forwarded
                       verbatim to the underlying SDK.
    """

    provider: Provider = Provider.ANTHROPIC
    api_key: str | None = None
    base_url: str | None = None
    default_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve API key from environment if not explicitly provided."""
        if self.api_key is None:
            _load_dotenv_if_present()
            if self.provider == Provider.ANTHROPIC:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == Provider.GOOGLE:
                self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            elif self.provider == Provider.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")

        if self.base_url is None and self.provider == Provider.OPENAI:
            self.base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")



@dataclass
class TokenUsage:
    """Token usage statistics for a single model call.

    Attributes:
        prompt_tokens:     Tokens in the prompt / input.
        completion_tokens: Tokens in the completion / output.
        total_tokens:      Sum of prompt + completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    """A single tool-use request returned by the model.

    Attributes:
        id:    The tool-use block id (provider-assigned).
        name:  Name of the tool the model wants to invoke.
        input: The arguments dict the model passed to the tool.
    """

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Structured response from a single LLM call.

    Attributes:
        text:        The concatenated text content (empty if the model
                     only returned tool calls).
        tool_calls:  List of tool-use requests (empty if the model
                     responded with plain text only).
        usage:       Token usage statistics.
        model:       The model identifier that actually served the request.
        latency_ms:  Round-trip latency in milliseconds.
        stop_reason: Why the model stopped (``"end_turn"``,
                     ``"tool_use"``, ``"max_tokens"``, etc.).
        raw:         The unprocessed response object from the provider SDK,
                     for advanced introspection (not serialised to telemetry).
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: float = 0.0
    stop_reason: str = ""
    raw: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Type alias used by other modules
# ---------------------------------------------------------------------------

CallModelFn = Callable[..., ModelResponse]
"""Callable type alias for dependency-injecting :func:`call_model`."""


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def _load_dotenv_if_present() -> None:
    """Populate os.environ from a local .env file if it exists."""
    dotenv_path = _PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v


def load_config(config_path: Path | None = None) -> ModelConfig:
    """Load model configuration from a YAML file.

    The API key is resolved by reading the environment variable named in
    ``api_key_env_var`` (or default per provider: ``ANTHROPIC_API_KEY``,
    ``GOOGLE_API_KEY``, etc.).  This keeps secrets out of the config file.

    Args:
        config_path: Explicit path to the YAML config.  Falls back to
                     ``<project_root>/config.yaml``.

    Returns:
        A populated :class:`ModelConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        EnvironmentError:  If the API key env var is not set.
    """
    _load_dotenv_if_present()
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    provider_str = raw.get("model_provider", "anthropic").lower()
    provider = Provider(provider_str)

    default_env_vars = {
        Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
        Provider.GOOGLE: "GOOGLE_API_KEY",
        Provider.OPENAI: "OPENAI_API_KEY",
    }
    default_models = {
        Provider.ANTHROPIC: "claude-sonnet-4-20250514",
        Provider.GOOGLE: "gemini-3.8-flash",
        Provider.OPENAI: "gpt-4o",
    }

    # Resolve the API key from the named environment variable or provider default.
    api_key_env = raw.get("api_key_env_var") or default_env_vars.get(provider, "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    # Extra fallback for Google: also check GEMINI_API_KEY if GOOGLE_API_KEY is not set
    if not api_key and provider == Provider.GOOGLE:
        fallback_key = os.environ.get("GEMINI_API_KEY", "")
        if fallback_key:
            api_key = fallback_key
            api_key_env = "GEMINI_API_KEY"

    # Extra fallback for OpenAI: also check NVIDIA_API_KEY if OPENAI_API_KEY is not set
    if not api_key and provider == Provider.OPENAI:
        fallback_key = os.environ.get("NVIDIA_API_KEY", "")
        if fallback_key:
            api_key = fallback_key
            api_key_env = "NVIDIA_API_KEY"

    if not api_key:
        hint = f"export {api_key_env}=..." if api_key_env else "export API key"
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set.  "
            f"Export it with:  {hint}"
        )

    model_name = raw.get("model_name") or default_models.get(provider, "claude-sonnet-4-20250514")

    # Base URL: check config file, then fallback to OPENAI_BASE_URL / OPENAI_API_BASE
    base_url = raw.get("base_url")
    if not base_url and provider == Provider.OPENAI:
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    return ModelConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        default_model=model_name,
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 4096)),
    )



# ---------------------------------------------------------------------------
# Provider backends  (private — add new providers here)
# ---------------------------------------------------------------------------

def _call_anthropic(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    system: str | None,
    config: ModelConfig,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> ModelResponse:
    """Anthropic Claude backend with native tool-calling.

    Uses the ``anthropic`` SDK's ``messages.create`` endpoint.
    """
    import anthropic  # Lazy import — only loaded when this provider is selected.

    client_kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = anthropic.Anthropic(**client_kwargs)

    create_kwargs: dict[str, Any] = {
        "model": model or config.default_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system:
        create_kwargs["system"] = system
    if tools:
        create_kwargs["tools"] = tools

    t0 = time.perf_counter()
    response = client.messages.create(**create_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    # ── Parse response content blocks ──────────────────────────
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, input=block.input)
            )

    usage = TokenUsage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    return ModelResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        model=response.model,
        latency_ms=latency_ms,
        stop_reason=response.stop_reason or "",
        raw=response,
    )


def _call_openai(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    system: str | None,
    config: ModelConfig,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> ModelResponse:
    """OpenAI / OpenAI-compatible backend with tool-calling support.

    Uses the ``openai`` SDK's ``chat.completions.create`` endpoint.
    Compatible with GPT-4o, GPT-4-turbo, GPT-3.5-turbo, and any
    OpenAI-compatible base_url (Azure, local proxies, etc.).
    """
    import openai  # Lazy import — only loaded when this provider is selected.

    client_kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = openai.OpenAI(**client_kwargs)

    # Build the messages list — prepend a system message if provided
    oai_messages: list[dict[str, Any]] = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)

    create_kwargs: dict[str, Any] = {
        "model": model or config.default_model,
        "messages": oai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if config.extra:
        create_kwargs.update(config.extra)

    # Convert Anthropic-style tool defs to OpenAI function-calling format
    if tools:
        oai_tools: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                if t.get("type") == "function":
                    # Already in OpenAI format
                    oai_tools.append(t)
                elif "name" in t:
                    # Anthropic-style: {name, description, input_schema}
                    oai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema") or t.get("parameters") or {},
                        },
                    })
        if oai_tools:
            create_kwargs["tools"] = oai_tools
            create_kwargs["tool_choice"] = "auto"

    t0 = time.perf_counter()
    response = client.chat.completions.create(**create_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse response
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    choice = response.choices[0] if response.choices else None
    stop_reason = ""
    if choice:
        stop_reason = choice.finish_reason or ""
        msg = choice.message

        if msg.content:
            text_parts.append(msg.content)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                import json as _json
                try:
                    input_dict = _json.loads(tc.function.arguments or "{}")
                except Exception:
                    input_dict = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=input_dict,
                ))

    usage_data = response.usage
    usage = TokenUsage(
        prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_data, "total_tokens", 0) or 0,
    )

    return ModelResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        model=response.model or (model or config.default_model),
        latency_ms=latency_ms,
        stop_reason=stop_reason,
        raw=response,
    )


def _call_google(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    system: str | None,
    config: ModelConfig,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> ModelResponse:
    """Google Gemini backend with native tool-calling.

    Uses the ``google-generativeai`` SDK.
    """
    import google.generativeai as genai

    genai.configure(api_key=config.api_key)

    model_name = model or config.default_model
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]

    # Format tools for Gemini if provided
    gemini_tools = None
    if tools:
        fn_declarations = []
        for t in tools:
            if isinstance(t, dict):
                if "function_declarations" in t:
                    fn_declarations.extend(t["function_declarations"])
                elif "name" in t:
                    fn_declarations.append({
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                    })
                else:
                    fn_declarations.append(t)
            else:
                fn_declarations.append(t)
        if fn_declarations:
            gemini_tools = [{"function_declarations": fn_declarations}]

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    generative_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system,
        tools=gemini_tools,
        generation_config=generation_config,
    )

    # Convert messages to Gemini contents format
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        gemini_role = "model" if role in ("assistant", "model") else "user"
        content = msg.get("content", "")

        if isinstance(content, str):
            parts = [content]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "tool_result":
                        parts.append(item.get("content", ""))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
        else:
            parts = [str(content)]

        contents.append({"role": gemini_role, "parts": parts})

    t0 = time.perf_counter()
    response = generative_model.generate_content(contents)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse response content, text, and function calls (tool_use)
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    candidates = getattr(response, "candidates", []) or []
    stop_reason = ""
    if candidates:
        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            stop_reason = str(getattr(finish_reason, "name", finish_reason))

        content = getattr(candidate, "content", None)
        if content:
            for part in getattr(content, "parts", []):
                part_text = getattr(part, "text", None)
                if part_text:
                    text_parts.append(part_text)

                func_call = getattr(part, "function_call", None)
                if func_call:
                    call_name = getattr(func_call, "name", "")
                    call_args = getattr(func_call, "args", {})
                    # Convert MapComposite / protobuf map to standard dict
                    input_dict = dict(call_args) if call_args else {}
                    call_id = getattr(func_call, "id", None) or f"call_{call_name}_{int(time.time() * 1000)}"
                    tool_calls.append(ToolCall(id=call_id, name=call_name, input=input_dict))

    # Parse token usage
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta:
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_meta, "total_token_count", 0) or (prompt_tokens + completion_tokens)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    else:
        usage = TokenUsage()

    resp_model = getattr(response, "model_version", None) or model_name

    return ModelResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        model=resp_model,
        latency_ms=latency_ms,
        stop_reason=stop_reason,
        raw=response,
    )


# Registry — maps Provider enum → backend function.
_PROVIDER_DISPATCH = {
    Provider.ANTHROPIC: _call_anthropic,
    Provider.OPENAI: _call_openai,
    Provider.GOOGLE: _call_google,
}


# ---------------------------------------------------------------------------
# Singleton config cache
# ---------------------------------------------------------------------------

_cached_config: ModelConfig | None = None


def _get_config() -> ModelConfig:
    """Return (and cache) the loaded model configuration."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reset_config() -> None:
    """Clear the cached config so the next call reloads from disk.

    Useful for testing or when the config file has changed.
    """
    global _cached_config
    _cached_config = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_model(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
    system_prompt: str | None = None,
    config: ModelConfig | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Send messages to the configured LLM and return a structured response.

    This is the **only** function the rest of the harness should use for
    LLM interactions.  It handles:
      - Config loading (from ``config.yaml`` or passed explicitly)
      - Provider dispatch (Anthropic / OpenAI / Google)
      - Messages + tools → provider-native API translation
      - Token-usage extraction
      - Latency measurement

    Args:
        messages:      List of message dicts, each with ``role`` and ``content``
                       keys (e.g. ``[{"role": "user", "content": "Hello"}]``).
        tools:         Optional list of tool definitions for tool-calling.
                       Format follows the Anthropic tool schema:
                       ``{"name": …, "description": …, "input_schema": {…}}``.
        system:        Optional system-level instruction.
        system_prompt: Alias for ``system``.
        config:        Explicit :class:`ModelConfig`.  When ``None``, the config
                       is loaded from ``config.yaml`` (cached after first load).
        model:         Override the model identifier for this single call.
        temperature:   Override the sampling temperature for this call.
        max_tokens:    Override the max tokens for this call.

    Returns:
        A :class:`ModelResponse` containing the text, any tool calls,
        token usage, and latency.
    """
    cfg = config or _get_config()

    dispatch_fn = _PROVIDER_DISPATCH.get(cfg.provider)
    if dispatch_fn is None:
        raise ValueError(f"Unsupported provider: {cfg.provider!r}")

    eff_system = system or system_prompt or kwargs.get("system_prompt")

    return dispatch_fn(
        messages,
        tools=tools,
        system=eff_system,
        config=cfg,
        model=model,
        temperature=temperature if temperature is not None else cfg.temperature,
        max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
    )
