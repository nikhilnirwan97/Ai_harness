#!/usr/bin/env python3
"""
Unit and integration tests for harness.plan
===========================================

Run with::

    .venv/bin/python tests/test_plan.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.explore import RankedFile
from harness.model import ModelResponse, ToolCall, TokenUsage
from harness.plan import (
    generate_plan,
    validate_plan,
    Plan,
    PlanStep,
    PLAN_SUBMIT_TOOL,
)


class TestPlan(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Create sample files in repo
        (self.root / "auth.py").write_text("def login(): pass\n")
        (self.root / "models.py").write_text("class User: pass\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plan_structure_and_dict_compatibility(self):
        """Verify Plan conforms to dict JSON schema and object attributes."""
        plan_dict = {
            "root_cause_hypothesis": "Token validation logic ignores expiry field.",
            "files_to_modify": ["auth.py"],
            "approach": "Add expiration check in login function.",
            "risks": ["May reject valid tokens if clocks skew."],
            "test_strategy": "Run unit test with expired and valid JWT.",
        }

        plan = Plan(**plan_dict)

        # 1. Assert dictionary compatibility
        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["root_cause_hypothesis"], plan_dict["root_cause_hypothesis"])
        self.assertEqual(plan["files_to_modify"], ["auth.py"])
        self.assertEqual(plan["approach"], plan_dict["approach"])
        self.assertEqual(plan["risks"], plan_dict["risks"])
        self.assertEqual(plan["test_strategy"], plan_dict["test_strategy"])

        # 2. Assert attribute access
        self.assertEqual(plan.root_cause_hypothesis, plan_dict["root_cause_hypothesis"])
        self.assertEqual(plan.files_to_modify, ["auth.py"])

        # 3. Assert JSON serializable
        serialized = json.dumps(plan)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["files_to_modify"], ["auth.py"])

        # 4. Assert steps generated for downstream Implement phase
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].file, "auth.py")
        self.assertEqual(plan.steps[0].action, "modify")

    def test_generate_plan_tool_calling(self):
        """Test generate_plan extracts structured output from native tool call."""
        mock_response = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_submit_1",
                    name="submit_plan",
                    input={
                        "root_cause_hypothesis": "Password hash comparison is not constant time.",
                        "files_to_modify": ["auth.py"],
                        "approach": "Use hmac.compare_digest in auth.py.",
                        "risks": ["Minor performance difference."],
                        "test_strategy": "Verify login with correct and incorrect passwords.",
                    },
                )
            ],
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gemini-3.8-flash",
        )

        mock_model_fn = MagicMock(return_value=mock_response)

        ranked = [RankedFile(filepath="auth.py", score=90.0, reason="auth matches")]

        plan = generate_plan(
            issue_text="Timing attack on login password verification",
            ranked_files=ranked,
            repo_root=self.root,
            model_fn=mock_model_fn,
        )

        # Assert tool schema passed to model
        mock_model_fn.assert_called_once()
        kwargs = mock_model_fn.call_args[1]
        self.assertEqual(kwargs["tools"], [PLAN_SUBMIT_TOOL])

        # Assert structured plan output
        self.assertEqual(plan["root_cause_hypothesis"], "Password hash comparison is not constant time.")
        self.assertEqual(plan["files_to_modify"], ["auth.py"])
        self.assertEqual(plan["approach"], "Use hmac.compare_digest in auth.py.")
        self.assertEqual(plan["risks"], ["Minor performance difference."])
        self.assertEqual(plan["test_strategy"], "Verify login with correct and incorrect passwords.")

    def test_generate_plan_file_existence_validation_and_retry(self):
        """Test that invalid files in files_to_modify trigger a retry with feedback."""
        # Call 1: Model proposes a non-existent file
        resp_call_1 = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_fail",
                    name="submit_plan",
                    input={
                        "root_cause_hypothesis": "Missing helper",
                        "files_to_modify": ["non_existent_auth_helper.py"],
                        "approach": "Modify non existent file",
                        "risks": [],
                        "test_strategy": "Run tests",
                    },
                )
            ],
        )

        # Call 2 (Retry): Model corrects itself to an existing file
        resp_call_2 = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_ok",
                    name="submit_plan",
                    input={
                        "root_cause_hypothesis": "Missing helper",
                        "files_to_modify": ["auth.py"],
                        "approach": "Modify existing auth.py",
                        "risks": [],
                        "test_strategy": "Run tests",
                    },
                )
            ],
        )

        mock_model_fn = MagicMock(side_effect=[resp_call_1, resp_call_2])

        ranked = [RankedFile(filepath="auth.py", score=80.0, reason="relevant")]

        plan = generate_plan(
            issue_text="Fix auth helper",
            ranked_files=ranked,
            repo_root=self.root,
            model_fn=mock_model_fn,
        )

        # Verify model was called twice (initial + retry)
        self.assertEqual(mock_model_fn.call_count, 2)

        # Check that the retry message contained feedback on non_existent_auth_helper.py
        retry_call_args = mock_model_fn.call_args_list[1][1]
        retry_messages = retry_call_args["messages"]
        feedback_content = str(retry_messages[-1]["content"])
        self.assertIn("non_existent_auth_helper.py", feedback_content)
        self.assertIn("DO NOT exist", feedback_content)

        # Verify final plan has valid existing file
        self.assertEqual(plan["files_to_modify"], ["auth.py"])

    def test_generate_plan_previous_plan_context(self):
        """Test that previous_plan from Reflect is passed in context to update rather than start over."""
        prev_plan = {
            "root_cause_hypothesis": "Initial theory was wrong database index.",
            "files_to_modify": ["models.py"],
            "approach": "Added index on user table.",
            "risks": ["Index overhead."],
            "test_strategy": "Explain query plan.",
        }

        mock_response = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_update",
                    name="submit_plan",
                    input={
                        "root_cause_hypothesis": "Updated theory: query filter missing tenant id.",
                        "files_to_modify": ["models.py"],
                        "approach": "Update query filter in models.py.",
                        "risks": ["Tenant isolation."],
                        "test_strategy": "Run multi-tenant test suite.",
                    },
                )
            ],
        )

        mock_model_fn = MagicMock(return_value=mock_response)

        plan = generate_plan(
            issue_text="Slow query in user lookup",
            ranked_files=["models.py"],
            previous_plan=prev_plan,
            repo_root=self.root,
            model_fn=mock_model_fn,
        )

        # Check prompt contains previous plan details
        call_args = mock_model_fn.call_args[1]
        prompt = call_args["messages"][0]["content"]
        self.assertIn("Initial theory was wrong database index", prompt)
        self.assertIn("Previous Plan (Reflect Iteration)", prompt)

        self.assertEqual(plan["root_cause_hypothesis"], "Updated theory: query filter missing tenant id.")


if __name__ == "__main__":
    unittest.main()
