#!/usr/bin/env python3
"""
Unit tests for harness.reflect
==============================

Tests:
  - diagnose_failure returns {"classification": str, "updated_plan": dict, "reasoning": str}
  - Classification into "wrong_diagnosis", "bad_implementation", "unrelated_flaky"
  - Cycle number passed into the prompt with scope narrowing (cycles 1-2 vs 3-4 surgical)
  - Reflection cycle cap at 4 (configurable) prevents LLM invocation on overflow
  - Dict schema compatibility and attribute access
  - should_retry logic based on cycle cap and flaky classification
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.model import ModelResponse, ToolCall
from harness.plan import Plan
from harness.reflect import (
    DIAGNOSIS_TOOL,
    DEFAULT_MAX_REFLECT_CYCLES,
    DiagnosisResult,
    MaxCyclesExceededError,
    diagnose_failure,
    diagnose_failures,
    reflect,
    should_retry,
)
from harness.verify import TestFailure, TestResult


class TestDiagnoseFailure(unittest.TestCase):
    """Test the diagnose_failure function and its failure classifications."""

    def setUp(self):
        self.sample_plan = Plan(
            root_cause_hypothesis="Incorrect tax calculation logic for exempt items.",
            files_to_modify=["calculator/tax.py"],
            approach="Add is_tax_exempt check before multiplying rate.",
            risks=["Could misclassify zero-rated items as exempt."],
            test_strategy="Run tax calculation test suite.",
        )
        self.sample_verify_results = TestResult(
            passed=5,
            failed=1,
            failures=[
                TestFailure(
                    test_name="test_exempt_tax",
                    message="AssertionError: 0.05 != 0.0",
                    traceback="Traceback:\n  File 'test_tax.py', line 12: assert tax == 0.0",
                )
            ],
            raw_output="FAILED test_tax.py::test_exempt_tax - AssertionError: 0.05 != 0.0",
        )
        self.sample_context = {
            "diff": "--- a/calculator/tax.py\n+++ b/calculator/tax.py\n@@ -10,1 +10,1 @@\n- return price * rate\n+ return 0.05 if is_exempt else price * rate\n",
        }

    def test_return_schema_dict_compatibility(self):
        """diagnose_failure returns dict with classification, updated_plan, and reasoning."""
        mock_resp = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_diag_1",
                    name="submit_diagnosis",
                    input={
                        "classification": "bad_implementation",
                        "reasoning": "Plan correctly identified exempt check, but hardcoded 0.05 instead of 0.0.",
                        "updated_plan": {
                            "root_cause_hypothesis": "Tax calculator returned 0.05 for exempt items.",
                            "files_to_modify": ["calculator/tax.py"],
                            "approach": "Return 0.0 instead of 0.05 when is_exempt is True.",
                            "risks": ["None."],
                            "test_strategy": "Run tax tests.",
                        },
                    },
                )
            ],
        )
        mock_model_fn = MagicMock(return_value=mock_resp)

        result = diagnose_failure(
            verify_results=self.sample_verify_results,
            current_plan=self.sample_plan,
            repo_context=self.sample_context,
            cycle=1,
            max_cycles=4,
            model_fn=mock_model_fn,
        )

        # 1. Direct dict validation
        self.assertIsInstance(result, dict)
        self.assertIn("classification", result)
        self.assertIn("updated_plan", result)
        self.assertIn("reasoning", result)

        self.assertEqual(result["classification"], "bad_implementation")
        self.assertIsInstance(result["updated_plan"], dict)
        self.assertEqual(result["updated_plan"]["files_to_modify"], ["calculator/tax.py"])
        self.assertIn("hardcoded 0.05", result["reasoning"])

        # 2. Object attribute access
        self.assertEqual(result.classification, "bad_implementation")
        self.assertEqual(result.updated_plan["approach"], "Return 0.0 instead of 0.05 when is_exempt is True.")
        self.assertIn("hardcoded 0.05", result.reasoning)

    def test_classify_wrong_diagnosis(self):
        """Correctly parses and classifies 'wrong_diagnosis'."""
        mock_resp = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_diag_wrong",
                    name="submit_diagnosis",
                    input={
                        "classification": "wrong_diagnosis",
                        "reasoning": "The issue is not in tax calculation, but in cart item categorization.",
                        "updated_plan": {
                            "root_cause_hypothesis": "Cart marks all items as non-exempt before passing to calculator.",
                            "files_to_modify": ["cart/item.py"],
                            "approach": "Preserve category metadata in cart item initialization.",
                            "risks": ["Serialization changes."],
                            "test_strategy": "Test cart item creation.",
                        },
                    },
                )
            ],
        )
        mock_model_fn = MagicMock(return_value=mock_resp)

        result = diagnose_failure(
            verify_results=self.sample_verify_results,
            current_plan=self.sample_plan,
            repo_context=self.sample_context,
            cycle=1,
            model_fn=mock_model_fn,
        )

        self.assertEqual(result["classification"], "wrong_diagnosis")
        self.assertEqual(result["updated_plan"]["files_to_modify"], ["cart/item.py"])

    def test_classify_unrelated_flaky(self):
        """Correctly parses and classifies 'unrelated_flaky'."""
        mock_resp = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_diag_flaky",
                    name="submit_diagnosis",
                    input={
                        "classification": "unrelated_flaky",
                        "reasoning": "The failure was in test_remote_sync due to an external network timeout.",
                        "updated_plan": dict(self.sample_plan),
                    },
                )
            ],
        )
        mock_model_fn = MagicMock(return_value=mock_resp)

        result = diagnose_failure(
            verify_results=self.sample_verify_results,
            current_plan=self.sample_plan,
            repo_context=self.sample_context,
            cycle=1,
            model_fn=mock_model_fn,
        )

        self.assertEqual(result["classification"], "unrelated_flaky")
        self.assertIn("external network timeout", result["reasoning"])

    def test_json_text_fallback_parsing(self):
        """Parses JSON text enclosed in markdown blocks when tool call is absent."""
        mock_resp = ModelResponse(
            text='```json\n{\n  "classification": "bad_implementation",\n  "reasoning": "Off by one in loop index.",\n  "updated_plan": {\n    "root_cause_hypothesis": "Loop boundary error.",\n    "files_to_modify": ["calculator/tax.py"],\n    "approach": "Change <= to <.",\n    "risks": [],\n    "test_strategy": "Run tests."\n  }\n}\n```',
            tool_calls=[],
        )
        mock_model_fn = MagicMock(return_value=mock_resp)

        result = diagnose_failure(
            verify_results=self.sample_verify_results,
            current_plan=self.sample_plan,
            repo_context=self.sample_context,
            cycle=1,
            model_fn=mock_model_fn,
        )

        self.assertEqual(result["classification"], "bad_implementation")
        self.assertEqual(result["updated_plan"]["approach"], "Change <= to <.")
        self.assertIn("Off by one", result["reasoning"])


class TestScopeNarrowingAndCycles(unittest.TestCase):
    """Test cycle number passing and scope narrowing prompts."""

    def setUp(self):
        self.plan = Plan(
            root_cause_hypothesis="Missing null check.",
            files_to_modify=["service.py"],
            approach="Add null check.",
            risks=[],
            test_strategy="Unit test.",
        )
        self.verify = {"passed": 2, "failed": 1, "failures": [{"test_name": "test_null", "assertion_error": "NPE", "traceback_summary": "line 10"}]}

    def _get_prompt_for_cycle(self, cycle: int, max_cycles: int = 4) -> str:
        mock_resp = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="submit_diagnosis",
                    input={
                        "classification": "bad_implementation",
                        "reasoning": "Fix null handling.",
                        "updated_plan": dict(self.plan),
                    },
                )
            ],
        )
        mock_model_fn = MagicMock(return_value=mock_resp)

        diagnose_failure(
            verify_results=self.verify,
            current_plan=self.plan,
            cycle=cycle,
            max_cycles=max_cycles,
            model_fn=mock_model_fn,
        )

        call_args = mock_model_fn.call_args[1]
        return call_args["messages"][0]["content"]

    def test_cycle_1_broad_prompt(self):
        """Cycle 1 prompt contains cycle 1 label and broad/standard exploration."""
        prompt = self._get_prompt_for_cycle(cycle=1, max_cycles=4)
        self.assertIn("CYCLE 1 of 4", prompt)
        self.assertIn("Standard / Broad Attempt", prompt)

    def test_cycle_2_narrowing_prompt(self):
        """Cycle 2 prompt contains cycle 2 label and asks to narrow scope."""
        prompt = self._get_prompt_for_cycle(cycle=2, max_cycles=4)
        self.assertIn("CYCLE 2 of 4", prompt)
        self.assertIn("Narrowing Scope", prompt)

    def test_cycle_3_explicit_surgical_prompt(self):
        """Cycle 3 prompt explicitly asks model to be SURGICAL."""
        prompt = self._get_prompt_for_cycle(cycle=3, max_cycles=4)
        self.assertIn("CYCLE 3 of 4", prompt)
        self.assertIn("EXPLICIT SURGICAL SCOPE", prompt)
        self.assertIn("significantly more SURGICAL", prompt)

    def test_cycle_4_ultra_surgical_prompt(self):
        """Cycle 4 prompt explicitly asks model to be ULTRA-SURGICAL for final attempt."""
        prompt = self._get_prompt_for_cycle(cycle=4, max_cycles=4)
        self.assertIn("CYCLE 4 of 4", prompt)
        self.assertIn("ULTRA-SURGICAL / FINAL ATTEMPT", prompt)


class TestCycleCap(unittest.TestCase):
    """Test that reflect cycles are capped at 4 (configurable) without calling model."""

    def test_cycle_cap_prevents_model_invocation(self):
        """When cycle > max_cycles, call_model is NOT called and capped result is returned."""
        mock_model_fn = MagicMock()
        plan = {"root_cause_hypothesis": "test", "files_to_modify": ["a.py"], "approach": "fix", "risks": [], "test_strategy": "test"}

        result = diagnose_failure(
            verify_results={"passed": 0, "failed": 1},
            current_plan=plan,
            cycle=5,
            max_cycles=4,
            model_fn=mock_model_fn,
        )

        mock_model_fn.assert_not_called()
        self.assertEqual(result["classification"], "wrong_diagnosis")
        self.assertIn("exceeds maximum allowed cycles", result["reasoning"])
        self.assertIn("capped", result["reasoning"].lower())

    def test_configurable_cycle_cap(self):
        """Cycle cap is configurable (e.g. max_cycles=2)."""
        mock_model_fn = MagicMock()
        plan = {"root_cause_hypothesis": "test", "files_to_modify": ["a.py"], "approach": "fix", "risks": [], "test_strategy": "test"}

        # Cycle 2 succeeds within max_cycles=2
        mock_model_fn.return_value = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_2",
                    name="submit_diagnosis",
                    input={
                        "classification": "bad_implementation",
                        "reasoning": "OK",
                        "updated_plan": plan,
                    },
                )
            ],
        )
        res2 = diagnose_failure(
            verify_results={"passed": 1, "failed": 1},
            current_plan=plan,
            cycle=2,
            max_cycles=2,
            model_fn=mock_model_fn,
        )
        self.assertEqual(res2["classification"], "bad_implementation")
        self.assertEqual(mock_model_fn.call_count, 1)

        # Cycle 3 with max_cycles=2 is blocked
        mock_model_fn.reset_mock()
        res3 = diagnose_failure(
            verify_results={"passed": 1, "failed": 1},
            current_plan=plan,
            cycle=3,
            max_cycles=2,
            model_fn=mock_model_fn,
        )
        mock_model_fn.assert_not_called()
        self.assertIn("exceeds maximum allowed cycles (2)", res3["reasoning"])

    def test_raise_on_cap_flag(self):
        """When raise_on_cap=True, exceeding max_cycles raises MaxCyclesExceededError."""
        mock_model_fn = MagicMock()
        with self.assertRaises(MaxCyclesExceededError):
            diagnose_failure(
                verify_results={"passed": 0, "failed": 1},
                current_plan={},
                cycle=5,
                max_cycles=4,
                model_fn=mock_model_fn,
                raise_on_cap=True,
            )
        mock_model_fn.assert_not_called()


class TestShouldRetry(unittest.TestCase):
    """Test should_retry decision helper."""

    def test_should_retry_under_max_cycles(self):
        self.assertTrue(should_retry({"classification": "bad_implementation"}, cycle=1, max_cycles=4))
        self.assertTrue(should_retry({"classification": "wrong_diagnosis"}, cycle=3, max_cycles=4))

    def test_should_not_retry_at_or_above_max_cycles(self):
        self.assertFalse(should_retry({"classification": "bad_implementation"}, cycle=4, max_cycles=4))
        self.assertFalse(should_retry({"classification": "bad_implementation"}, cycle=5, max_cycles=4))

    def test_should_not_retry_unrelated_flaky(self):
        self.assertFalse(should_retry({"classification": "unrelated_flaky"}, cycle=1, max_cycles=4))


class TestReflectIntegration(unittest.TestCase):
    """Test reflect phase integration helper."""

    def test_reflect_wrapper(self):
        mock_resp = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call_r",
                    name="submit_diagnosis",
                    input={
                        "classification": "bad_implementation",
                        "reasoning": "Syntax error in diff.",
                        "updated_plan": {
                            "root_cause_hypothesis": "Hypothesis",
                            "files_to_modify": ["a.py"],
                            "approach": "Fix syntax",
                            "risks": [],
                            "test_strategy": "Run tests",
                        },
                    },
                )
            ],
        )
        mock_model = MagicMock(return_value=mock_resp)

        res = reflect(
            test_result=TestResult(passed=1, failed=1),
            lint_result=None,
            impl_result=MagicMock(diffs=[]),
            model_fn=mock_model,
            attempt=1,
            max_retries=4,
        )

        self.assertTrue(res.should_retry)
        self.assertEqual(res.attempt, 1)
        self.assertEqual(res.diagnosis["classification"], "bad_implementation")


if __name__ == "__main__":
    unittest.main()
