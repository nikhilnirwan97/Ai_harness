#!/usr/bin/env python3
"""
Smoke test for harness.model.call_model()
==========================================

Run with::

    .venv/bin/python tests/test_model_smoke.py

Requires ``ANTHROPIC_API_KEY`` to be set in the environment.

Tests:
  1. Simple text completion  — sends a trivial prompt, prints the response.
  2. Tool-calling            — sends a prompt with a dummy tool, prints any
                               tool-use blocks the model returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `harness` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.model import call_model, ModelResponse


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_response(resp: ModelResponse) -> None:
    print(f"  Model      : {resp.model}")
    print(f"  Latency    : {resp.latency_ms:.0f} ms")
    print(f"  Tokens     : {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out")
    print(f"  Stop reason: {resp.stop_reason}")
    if resp.text:
        print(f"  Text       : {resp.text[:300]}")
    if resp.tool_calls:
        for tc in resp.tool_calls:
            print(f"  Tool call  : {tc.name}({tc.input})")


def run_tests_for_config(config=None, label="default config") -> None:
    _divider(f"Running smoke tests ({label})")
    resp = call_model(
        messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
        max_tokens=1024,
        config=config,
    )
    _print_response(resp)
    assert resp.text.strip(), "Expected non-empty text response"
    assert resp.usage.total_tokens > 0 or resp.usage.prompt_tokens > 0, "Expected token usage"
    print("  ✅ Simple completion passed")

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'San Francisco'.",
                    }
                },
                "required": ["city"],
            },
        }
    ]
    resp = call_model(
        messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=tools,
        max_tokens=1024,
        config=config,
    )
    _print_response(resp)
    assert resp.tool_calls, "Expected at least one tool call"
    assert resp.tool_calls[0].name == "get_weather", "Expected get_weather tool call"
    print("  ✅ Tool-calling passed")

    resp = call_model(
        messages=[{"role": "user", "content": "What are you?"}],
        system="You are a pirate. Always respond in pirate speak.",
        max_tokens=1024,
        config=config,
    )
    _print_response(resp)
    assert resp.text.strip(), "Expected non-empty text response"
    print("  ✅ System prompt passed")


if __name__ == "__main__":
    import os
    from harness.model import ModelConfig, Provider

    # Check CLI arguments or environment
    provider_arg = None
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("--provider="):
            provider_arg = sys.argv[1].split("=")[1].lower()
        elif sys.argv[1] == "--provider" and len(sys.argv) > 2:
            provider_arg = sys.argv[2].lower()
        elif sys.argv[1] in ("google", "anthropic", "openai"):
            provider_arg = sys.argv[1].lower()

    try:
        if provider_arg == "google" or (not provider_arg and "GOOGLE_API_KEY" in os.environ and "ANTHROPIC_API_KEY" not in os.environ):
            cfg = ModelConfig(
                provider=Provider.GOOGLE,
                api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
                default_model="gemini-3.8-flash",
            )
            run_tests_for_config(config=cfg, label="Google Gemini")
        elif provider_arg == "anthropic":
            cfg = ModelConfig(
                provider=Provider.ANTHROPIC,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                default_model="claude-sonnet-4-20250514",
            )
            run_tests_for_config(config=cfg, label="Anthropic Claude")
        else:
            run_tests_for_config(config=None, label="config.yaml")

        _divider("All tests passed ✅")
    except Exception as e:
        print(f"\n❌ FAILED: {e}", file=sys.stderr)
        raise SystemExit(1)

