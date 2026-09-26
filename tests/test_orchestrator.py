#!/usr/bin/env python3
"""
Unit tests for harness.orchestrator
===================================

Tests:
  - Adaptive token budgeting:
    - measure_repo_size calculates file count and LOC
    - calculate_adaptive_budget scales from base config value
    - Allocates budget across phases (Explore 15%, Plan 10%, Implement 35%, Verify 15%, Reflect 25%)
    - At 90% budget consumed, stops attempting new fixes and goes straight to Finalize
      with best passing state so far — never hard-crashing.
  - State machine lifecycle:
    - Runs Explore once
    - Loops Plan → Implement → Verify → Reflect
    - Terminates successfully when tests pass
    - Terminates gracefully when retry cap is reached
  - JSONL Telemetry logging:
    - Verifies transitions logged with timestamp, phase, iteration, action, result, tokens_used.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.implement import Diff
from harness.model import ModelConfig, ModelResponse, TokenUsage, ToolCall
from harness.orchestrator import (
    BUDGET_EXHAUSTION_THRESHOLD,
    DEFAULT_BASE_TOKEN_BUDGET,
    OrchestratorConfig,
    Phase,
    RunResult,
    TokenBudget,
    calculate_adaptive_budget,
    measure_repo_size,
    run,
)
from harness.plan import Plan
from harness.verify import TestFailure, TestResult


class TestAdaptiveBudgeting(unittest.TestCase):
    """Test adaptive token budgeting calculations and scaling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create a sample project structure
        (self.root / "app.py").write_text("def main():\n    print('hello')\n    return 0\n")
        (self.root / "test_app.py").write_text("def test_main():\n    assert True\n")
        (self.root / "utils.py").write_text("def helper():\n    return 42\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_measure_repo_size(self):
        """measure_repo_size returns accurate file count and line count."""
        files, loc = measure_repo_size(self.root)
        self.assertEqual(files, 3)
        self.assertGreater(loc, 0)

    def test_calculate_adaptive_budget_allocations(self):
        """Budget scales and matches phase ratios (15%, 10%, 35%, 15%, 25%)."""
        base_budget = 40_000
        budget = calculate_adaptive_budget(self.root, base_budget=base_budget)

        self.assertIsInstance(budget, TokenBudget)
        self.assertGreaterEqual(budget.total_budget, base_budget)

        # Verify phase allocations roughly match required ratios
        total = budget.total_budget
        self.assertEqual(budget.allocated[Phase.EXPLORE], int(total * 0.15))
        self.assertEqual(budget.allocated[Phase.PLAN], int(total * 0.10))
        self.assertEqual(budget.allocated[Phase.IMPLEMENT], int(total * 0.35))
        self.assertEqual(budget.allocated[Phase.VERIFY], int(total * 0.15))
        self.assertEqual(budget.allocated[Phase.REFLECT], int(total * 0.25))

        # Sum of allocations equals total
        alloc_sum = sum(budget.allocated.values())
        self.assertEqual(alloc_sum, total)

    def test_budget_exhaustion_threshold(self):
        """is_exhausted triggers at 90% consumed."""
        budget = calculate_adaptive_budget(self.root, base_budget=10_000)
        self.assertFalse(budget.is_exhausted)

        # Consume 89%
        budget.record_usage(Phase.PLAN, int(budget.total_budget * 0.89))
        self.assertFalse(budget.is_exhausted)

        # Consume up to 90%
        budget.record_usage(Phase.IMPLEMENT, int(budget.total_budget * 0.02))
        self.assertTrue(budget.is_exhausted)


class TestOrchestratorStateMachine(unittest.TestCase):
    """Test full orchestrator execution loop, budget exhaustion, and telemetry."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.log_path = self.root / "telemetry_test.jsonl"

        # Create basic git repository setup
        (self.root / "calculator.py").write_text("def add(a, b): return a + b\n")
        (self.root / "test_calculator.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_add(self): self.assertEqual(1+1, 2)\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("harness.orchestrator.run_tests")
    @patch("harness.orchestrator.apply_diff_safely")
    @patch("harness.orchestrator.generate_diff")
    @patch("harness.orchestrator.generate_plan")
    @patch("harness.orchestrator.explore")
    def test_successful_run_first_try(
        self,
        mock_explore,
        mock_generate_plan,
        mock_generate_diff,
        mock_apply_diff,
        mock_run_tests,
    ):
        """Explore runs once, Plan → Implement → Verify succeeds, goes to Finalize."""
        # Setup mocks
        mock_explore.return_value = MagicMock(ranked_files=["calculator.py"])
        mock_generate_plan.return_value = Plan(
            root_cause_hypothesis="Good",
            files_to_modify=["calculator.py"],
            approach="None",
            risks=[],
            test_strategy="None",
        )
        mock_generate_diff.return_value = Diff(file_path="calculator.py", raw_diff_text="diff")
        mock_apply_diff.return_value = MagicMock(success=True)
        # All tests pass!
        mock_run_tests.return_value = TestResult(passed=2, failed=0, errors=0)

        cfg = OrchestratorConfig(
            log_path=self.log_path,
            max_retries=3,
            base_token_budget=50_000,
            model_fn=MagicMock(return_value=ModelResponse(text="ok", usage=TokenUsage(total_tokens=200))),
        )

        result = run("Fix calculation", self.root, cfg)

        self.assertTrue(result.success)
        self.assertEqual(result.phase_reached, Phase.FINALIZE)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(mock_explore.call_count, 1)
        self.assertEqual(mock_generate_plan.call_count, 1)
        self.assertEqual(mock_generate_diff.call_count, 1)
        self.assertEqual(mock_run_tests.call_count, 1)

        # Check telemetry JSONL log entries
        self.assertTrue(self.log_path.exists())
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(lines), 0)

        phases_logged = []
        for line in lines:
            event = json.loads(line)
            self.assertIn("timestamp", event)
            self.assertIn("phase", event)
            self.assertIn("iteration", event)
            self.assertIn("action", event)
            self.assertIn("result", event)
            self.assertIn("tokens_used", event)
            phases_logged.append(event["phase"])

        self.assertIn("explore", phases_logged)
        self.assertIn("plan", phases_logged)
        self.assertIn("implement", phases_logged)
        self.assertIn("verify", phases_logged)
        self.assertIn("finalize", phases_logged)

        # Check generated report artifacts
        self.assertIsNotNone(result.report_path)
        self.assertTrue(result.report_path.exists())
        self.assertIsNotNone(result.telemetry_path)
        self.assertTrue(result.telemetry_path.exists())
        self.assertIsNotNone(result.patch_path)
        self.assertTrue(result.patch_path.exists())

        report_txt = result.report_path.read_text(encoding="utf-8")
        self.assertIn("Issue Summary", report_txt)
        self.assertIn("Fix calculation", report_txt)
        self.assertIn("Root Cause Found", report_txt)
        self.assertIn("Changes Made", report_txt)
        self.assertIn("Test Results (Before / After)", report_txt)
        self.assertIn("Unresolved Issues", report_txt)

    @patch("harness.orchestrator.diagnose_failure")
    @patch("harness.orchestrator.run_tests")
    @patch("harness.orchestrator.apply_diff_safely")
    @patch("harness.orchestrator.generate_diff")
    @patch("harness.orchestrator.generate_plan")
    @patch("harness.orchestrator.explore")
    def test_retry_loop_until_success(
        self,
        mock_explore,
        mock_generate_plan,
        mock_generate_diff,
        mock_apply_diff,
        mock_run_tests,
        mock_diagnose,
    ):
        """Loop retries on failure and succeeds on 2nd iteration."""
        mock_explore.return_value = MagicMock(ranked_files=["calculator.py"])
        mock_generate_plan.return_value = Plan(
            root_cause_hypothesis="Good",
            files_to_modify=["calculator.py"],
            approach="None",
            risks=[],
            test_strategy="None",
        )
        mock_generate_diff.return_value = Diff(file_path="calculator.py", raw_diff_text="diff")
        mock_apply_diff.return_value = MagicMock(success=True)

        # Iteration 1 fails, Iteration 2 passes
        mock_run_tests.side_effect = [
            TestResult(passed=1, failed=1, errors=0),
            TestResult(passed=2, failed=0, errors=0),
        ]
        mock_diagnose.return_value = {
            "classification": "bad_implementation",
            "updated_plan": {"root_cause_hypothesis": "Updated", "files_to_modify": ["calculator.py"], "approach": "New approach", "risks": [], "test_strategy": "Run tests"},
            "reasoning": "Plan was good, diff was buggy",
        }

        cfg = OrchestratorConfig(
            log_path=self.log_path,
            max_retries=3,
            base_token_budget=50_000,
            model_fn=MagicMock(return_value=ModelResponse(text="ok", usage=TokenUsage(total_tokens=150))),
        )

        result = run("Fix calculation", self.root, cfg)

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(mock_explore.call_count, 1)  # Explore runs ONLY ONCE
        self.assertEqual(mock_generate_plan.call_count, 2)
        self.assertEqual(mock_diagnose.call_count, 1)

    @patch("harness.orchestrator.diagnose_failure")
    @patch("harness.orchestrator.run_tests")
    @patch("harness.orchestrator.apply_diff_safely")
    @patch("harness.orchestrator.generate_diff")
    @patch("harness.orchestrator.generate_plan")
    @patch("harness.orchestrator.explore")
    def test_budget_exhaustion_stops_and_finalizes_with_best_state(
        self,
        mock_explore,
        mock_generate_plan,
        mock_generate_diff,
        mock_apply_diff,
        mock_run_tests,
        mock_diagnose,
    ):
        """At 90% budget consumed, stops new fixes and finalizes without hard crashing."""
        def plan_side_effect(*args, **kwargs):
            if "model_fn" in kwargs and kwargs["model_fn"]:
                kwargs["model_fn"](messages=[{"role": "user", "content": "plan"}])
            return Plan(
                root_cause_hypothesis="Good",
                files_to_modify=["calculator.py"],
                approach="None",
                risks=[],
                test_strategy="None",
            )

        mock_generate_plan.side_effect = plan_side_effect
        mock_generate_diff.return_value = Diff(file_path="calculator.py", raw_diff_text="diff")
        mock_apply_diff.return_value = MagicMock(success=True)

        # Tests always fail partially (1 pass, 1 fail)
        mock_run_tests.return_value = TestResult(passed=1, failed=1, errors=0)
        mock_diagnose.return_value = {
            "classification": "bad_implementation",
            "updated_plan": {"root_cause_hypothesis": "Updated", "files_to_modify": ["calculator.py"], "approach": "New approach", "risks": [], "test_strategy": "Run tests"},
            "reasoning": "Retry",
        }

        # Model returns very high token usage exceeding 90% of base budget in iteration 1
        base_budget = 10_000
        heavy_usage_fn = MagicMock(return_value=ModelResponse(text="heavy", usage=TokenUsage(total_tokens=9_500)))

        cfg = OrchestratorConfig(
            log_path=self.log_path,
            max_retries=5,
            base_token_budget=base_budget,
            model_fn=heavy_usage_fn,
        )

        result = run("Fix calculation", self.root, cfg)

        # Does NOT hard crash! Returns RunResult gracefully with best state
        self.assertFalse(result.success)
        self.assertEqual(result.phase_reached, Phase.FINALIZE)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.budget)
        self.assertTrue(result.budget.is_exhausted)

        # Check telemetry logged budget exhaustion
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        exhaustion_logged = any("budget_exhausted" in l for l in lines)
        self.assertTrue(exhaustion_logged, "Telemetry must record budget_exhausted action")

    @patch("harness.orchestrator.diagnose_failure")
    @patch("harness.orchestrator.run_tests")
    @patch("harness.orchestrator.apply_diff_safely")
    @patch("harness.orchestrator.generate_diff")
    @patch("harness.orchestrator.generate_plan")
    @patch("harness.orchestrator.explore")
    def test_unrelated_flaky_failure_terminates_gracefully(
        self,
        mock_explore,
        mock_generate_plan,
        mock_generate_diff,
        mock_apply_diff,
        mock_run_tests,
        mock_diagnose,
    ):
        """Unrelated flaky failure is logged and stops further implement attempts."""
        mock_explore.return_value = MagicMock(ranked_files=["calculator.py"])
        mock_generate_plan.return_value = Plan(root_cause_hypothesis="Good", files_to_modify=["calculator.py"], approach="A", risks=[], test_strategy="T")
        mock_generate_diff.return_value = Diff(file_path="calculator.py", raw_diff_text="diff")
        mock_apply_diff.return_value = MagicMock(success=True)
        mock_run_tests.return_value = TestResult(passed=1, failed=1, errors=0)

        # Diagnosis says failure is unrelated/flaky
        mock_diagnose.return_value = {
            "classification": "unrelated_flaky",
            "updated_plan": {},
            "reasoning": "Remote API timeout unrelated to change",
        }

        cfg = OrchestratorConfig(
            log_path=self.log_path,
            max_retries=4,
            base_token_budget=50_000,
            model_fn=MagicMock(return_value=ModelResponse(text="ok", usage=TokenUsage(total_tokens=100))),
        )

        result = run("Fix calculation", self.root, cfg)

        self.assertEqual(result.phase_reached, Phase.FINALIZE)
        self.assertEqual(result.attempts, 1)

        # Check telemetry logged unrelated_flaky_ignored
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertTrue(any("unrelated_flaky" in l for l in lines))

    @patch("harness.orchestrator.git_rollback")
    @patch("harness.orchestrator.diagnose_failure")
    @patch("harness.orchestrator.run_tests")
    @patch("harness.orchestrator.apply_diff_safely")
    @patch("harness.orchestrator.generate_diff")
    @patch("harness.orchestrator.generate_plan")
    @patch("harness.orchestrator.explore")
    def test_best_state_restored_when_later_iterations_degrade(
        self,
        mock_explore,
        mock_generate_plan,
        mock_generate_diff,
        mock_apply_diff,
        mock_run_tests,
        mock_diagnose,
        mock_git_rollback,
    ):
        """When a later iteration degrades results, Finalize restores the best passing state."""
        mock_explore.return_value = MagicMock(ranked_files=["calculator.py"])
        plan1 = Plan(root_cause_hypothesis="Hypothesis 1", files_to_modify=["calculator.py"], approach="A1", risks=[], test_strategy="T1")
        plan2 = Plan(root_cause_hypothesis="Hypothesis 2", files_to_modify=["calculator.py"], approach="A2", risks=[], test_strategy="T2")
        mock_generate_plan.side_effect = [plan1, plan2]

        diff1 = Diff(file_path="calculator.py", raw_diff_text="diff1")
        diff2 = Diff(file_path="calculator.py", raw_diff_text="diff2")
        mock_generate_diff.side_effect = [diff1, diff2]
        mock_apply_diff.return_value = MagicMock(success=True)

        # Iteration 1: 3 passed, 1 failed (good state)
        # Iteration 2: 1 passed, 3 failed (degraded state)
        mock_run_tests.side_effect = [
            TestResult(passed=3, failed=1, errors=0),
            TestResult(passed=1, failed=3, errors=0),
        ]
        mock_diagnose.return_value = {
            "classification": "bad_implementation",
            "updated_plan": {"root_cause_hypothesis": "Updated", "files_to_modify": ["calculator.py"], "approach": "Retry"},
            "reasoning": "Still failing",
        }

        cfg = OrchestratorConfig(
            log_path=self.log_path,
            max_retries=2,
            base_token_budget=50_000,
            model_fn=MagicMock(return_value=ModelResponse(text="ok", usage=TokenUsage(total_tokens=100))),
        )

        result = run("Fix calculation", self.root, cfg)

        self.assertFalse(result.success)
        self.assertEqual(result.phase_reached, Phase.FINALIZE)
        self.assertEqual(result.attempts, 2)
        # Best state from iteration 1 should be restored
        self.assertEqual(result.test_result.passed, 3)
        self.assertEqual(result.test_result.failed, 1)
        self.assertEqual(result.plan, plan1)
        self.assertTrue(mock_git_rollback.called)

    def test_adaptive_budget_scales_for_large_repo(self):
        """Adaptive budget scales up total budget and scale_factor for larger repositories."""
        # Create 25 additional files with multiple lines
        for i in range(25):
            content = "\n".join(f"line_{j} = {j}" for j in range(100))
            (self.root / f"extra_{i}.py").write_text(content)

        files, loc = measure_repo_size(self.root)
        self.assertGreaterEqual(files, 25)
        self.assertGreaterEqual(loc, 2000)

        base_budget = 30_000
        budget = calculate_adaptive_budget(self.root, base_budget=base_budget)
        self.assertGreater(budget.scale_factor, 1.0)
        self.assertGreater(budget.total_budget, base_budget)
        self.assertEqual(sum(budget.allocated.values()), budget.total_budget)


if __name__ == "__main__":
    unittest.main()
