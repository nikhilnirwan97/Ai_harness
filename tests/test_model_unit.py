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


if __name__ == "__main__":
    unittest.main()
