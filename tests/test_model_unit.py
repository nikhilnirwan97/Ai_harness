"""
Unit tests for harness.model
============================

Tests provider dispatch, response formatting, and configuration cascade
for both Anthropic and Google providers using mocks (no live API keys required).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.model import (
    ModelConfig,
    ModelResponse,
    Provider,
    TokenUsage,
    ToolCall,
    _call_google,
    _call_openai,
    call_model,
    load_config,
)


class TestModelGoogle(unittest.TestCase):
    """Test suite for Google Gemini implementation in model.py."""

    def test_call_google_parsing(self):
        """Verify _call_google parses text, tools, tokens, latency into ModelResponse."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "STOP"

        # Part 1: Text
        mock_part_text = MagicMock()
        mock_part_text.text = "Hello from Gemini"
        mock_part_text.function_call = None

        # Part 2: Tool call
        mock_part_func = MagicMock()
        mock_part_func.text = None
        mock_part_func.function_call.name = "read_file"
        mock_part_func.function_call.args = {"path": "main.py"}
        mock_part_func.function_call.id = "call_abc_456"

        mock_candidate.content.parts = [mock_part_text, mock_part_func]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 28
        mock_response.usage_metadata.total_token_count = 40
        mock_response.model_version = "gemini-1.5-pro-002"

        with patch("google.generativeai.GenerativeModel") as MockModel:
            instance = MockModel.return_value
            instance.generate_content.return_value = mock_response

            cfg = ModelConfig(
                provider=Provider.GOOGLE,
                api_key="test_google_key",
                default_model="gemini-1.5-pro",
            )
            resp = _call_google(
                messages=[{"role": "user", "content": "Inspect main.py"}],
                tools=[
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                system="You are a coding assistant.",
                config=cfg,
                model=None,
                temperature=0.0,
                max_tokens=1000,
            )

            # Assert internal format matches ModelResponse contract
            self.assertEqual(resp.text, "Hello from Gemini")
            self.assertEqual(len(resp.tool_calls), 1)
            self.assertEqual(resp.tool_calls[0].name, "read_file")
            self.assertEqual(resp.tool_calls[0].input, {"path": "main.py"})
            self.assertEqual(resp.tool_calls[0].id, "call_abc_456")
            self.assertEqual(resp.usage.prompt_tokens, 12)
            self.assertEqual(resp.usage.completion_tokens, 28)
            self.assertEqual(resp.usage.total_tokens, 40)
            self.assertEqual(resp.model, "gemini-1.5-pro-002")
            self.assertEqual(resp.stop_reason, "STOP")
            self.assertGreaterEqual(resp.latency_ms, 0.0)

    def test_config_cascade_google(self):
        """Test config cascade with GOOGLE_API_KEY and GEMINI_API_KEY."""
        with patch("harness.model._load_dotenv_if_present"):
            # 1. Error when key missing
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(EnvironmentError):
                    load_config()

            # 2. Key from GOOGLE_API_KEY
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "ai-key-google-999"}, clear=True):
                cfg = ModelConfig(provider=Provider.GOOGLE)
                self.assertEqual(cfg.api_key, "ai-key-google-999")

            # 3. Fallback to GEMINI_API_KEY
            with patch.dict(os.environ, {"GEMINI_API_KEY": "ai-key-gemini-888"}, clear=True):
                cfg = ModelConfig(provider=Provider.GOOGLE)
                self.assertEqual(cfg.api_key, "ai-key-gemini-888")


class TestModelOpenAI(unittest.TestCase):
    """Test suite for OpenAI and OpenAI-compatible (NVIDIA Nemotron) backends."""

    def test_call_openai_text_parsing(self):
        """Verify _call_openai parses text, tokens, model, and latency into ModelResponse."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Here is the solution."
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 75
        mock_response.usage.total_tokens = 100
        mock_response.model = "gpt-4o-2024-08-06"

        with patch("openai.OpenAI") as MockClientClass:
            mock_client = MockClientClass.return_value
            mock_client.chat.completions.create.return_value = mock_response

            cfg = ModelConfig(
                provider=Provider.OPENAI,
                api_key="test-key-123",
                default_model="gpt-4o",
            )

            resp = _call_openai(
                messages=[{"role": "user", "content": "Solve this issue."}],
                tools=None,
                system="You are a helpful coding assistant.",
                config=cfg,
                model=None,
                temperature=0.0,
                max_tokens=2048,
            )

            # Assert internal format matches ModelResponse contract like _call_anthropic
            self.assertEqual(resp.text, "Here is the solution.")
            self.assertEqual(resp.tool_calls, [])
            self.assertEqual(resp.usage.prompt_tokens, 25)
            self.assertEqual(resp.usage.completion_tokens, 75)
            self.assertEqual(resp.usage.total_tokens, 100)
            self.assertEqual(resp.model, "gpt-4o-2024-08-06")
            self.assertEqual(resp.stop_reason, "stop")
            self.assertGreaterEqual(resp.latency_ms, 0.0)

            # Check that system prompt was prepended to messages
            called_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(called_kwargs["messages"][0], {"role": "system", "content": "You are a helpful coding assistant."})
            self.assertEqual(called_kwargs["messages"][1], {"role": "user", "content": "Solve this issue."})
            self.assertEqual(called_kwargs["model"], "gpt-4o")

    def test_call_openai_tool_calls_parsing(self):
        """Verify _call_openai converts Anthropic tools to OpenAI format and parses tool_calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_xyz_123"
        mock_tool_call.function.name = "submit_plan"
        mock_tool_call.function.arguments = '{"approach": "Fix bug in text_utils.py", "files_to_modify": ["text_utils.py"]}'

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 30
        mock_response.usage.total_tokens = 80
        mock_response.model = "gpt-4o"

        with patch("openai.OpenAI") as MockClientClass:
            mock_client = MockClientClass.return_value
            mock_client.chat.completions.create.return_value = mock_response

            cfg = ModelConfig(
                provider=Provider.OPENAI,
                api_key="test-key-123",
                default_model="gpt-4o",
            )

            anthropic_tool = {
                "name": "submit_plan",
                "description": "Submit a structured implementation plan",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "approach": {"type": "string"},
                        "files_to_modify": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["approach", "files_to_modify"],
                },
            }

            resp = _call_openai(
                messages=[{"role": "user", "content": "Plan a fix"}],
                tools=[anthropic_tool],
                system=None,
                config=cfg,
                model=None,
                temperature=0.0,
                max_tokens=1000,
            )

            # Verify tool definition passed to OpenAI
            called_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertIn("tools", called_kwargs)
            expected_openai_tool = {
                "type": "function",
                "function": {
                    "name": "submit_plan",
                    "description": "Submit a structured implementation plan",
                    "parameters": anthropic_tool["input_schema"],
                },
            }
            self.assertEqual(called_kwargs["tools"], [expected_openai_tool])
            self.assertEqual(called_kwargs["tool_choice"], "auto")

            # Verify response tool_calls matches ToolCall contract
            self.assertEqual(len(resp.tool_calls), 1)
            self.assertEqual(resp.tool_calls[0].id, "call_xyz_123")
            self.assertEqual(resp.tool_calls[0].name, "submit_plan")
            self.assertEqual(resp.tool_calls[0].input, {
                "approach": "Fix bug in text_utils.py",
                "files_to_modify": ["text_utils.py"],
            })

    def test_call_openai_nvidia_nemotron_endpoint(self):
        """Verify _call_openai works against NVIDIA Nemotron endpoint with configurable base_url."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "NVIDIA Nemotron response."
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 35
        mock_response.model = "nvidia/nemotron-4-340b-instruct"

        nvidia_base_url = "https://integrate.api.nvidia.com/v1"

        with patch("openai.OpenAI") as MockClientClass:
            mock_client = MockClientClass.return_value
            mock_client.chat.completions.create.return_value = mock_response

            cfg = ModelConfig(
                provider=Provider.OPENAI,
                api_key="nvapi-test-key",
                base_url=nvidia_base_url,
                default_model="nvidia/nemotron-4-340b-instruct",
            )

            resp = _call_openai(
                messages=[{"role": "user", "content": "Hello Nemotron"}],
                tools=None,
                system="System prompt",
                config=cfg,
                model=None,
                temperature=0.2,
                max_tokens=1024,
            )

            # Verify OpenAI client was initialized with the NVIDIA base_url and api_key
            MockClientClass.assert_called_once_with(
                api_key="nvapi-test-key",
                base_url="https://integrate.api.nvidia.com/v1",
            )
            self.assertEqual(resp.text, "NVIDIA Nemotron response.")
            self.assertEqual(resp.model, "nvidia/nemotron-4-340b-instruct")

    def test_config_cascade_openai_and_nvidia(self):
        """Verify configuration resolution for OpenAI and NVIDIA Nemotron."""
        with patch("harness.model._load_dotenv_if_present"):
            # 1. Fallback to NVIDIA_API_KEY when OPENAI_API_KEY is absent
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-cascade-123"}, clear=True):
                cfg = ModelConfig(provider=Provider.OPENAI)
                self.assertEqual(cfg.api_key, "nvapi-cascade-123")

            # 2. OPENAI_BASE_URL environment variable fallback
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_BASE_URL": "https://integrate.api.nvidia.com/v1",
            }, clear=True):
                cfg = ModelConfig(provider=Provider.OPENAI)
                self.assertEqual(cfg.api_key, "sk-test")
                self.assertEqual(cfg.base_url, "https://integrate.api.nvidia.com/v1")


if __name__ == "__main__":
    unittest.main()
