#!/usr/bin/env python3
"""
Unit tests for harness.telemetry and finalize_report
====================================================
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.implement import Diff
from harness.orchestrator import TokenBudget, Phase
from harness.plan import Plan
from harness.telemetry import (
    TelemetryEvent,
    TelemetryLogger,
    close_logger,
    finalize_report,
    init_logger,
    log_transition,
)
from harness.verify import TestFailure, TestResult


class TestTelemetryAndReport(unittest.TestCase):
    """Test JSONL event logging and finalize_report generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.log_path = self.output_dir / "test_events.jsonl"
        self.logger = init_logger(self.log_path)

    def tearDown(self):
        close_logger(self.logger)
        self.temp_dir.cleanup()

    def test_log_transition_and_jsonl_output(self):
        """log_transition outputs properly formatted JSONL events."""
        log_transition(
            self.logger,
            phase="plan",
            iteration=1,
            action="generate_plan",
            result="success",
            tokens_used=500,
        )
        close_logger(self.logger)

        self.assertTrue(self.log_path.exists())
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)

        event = json.loads(lines[0])
        self.assertEqual(event["phase"], "plan")
        self.assertEqual(event["iteration"], 1)
        self.assertEqual(event["action"], "generate_plan")
        self.assertEqual(event["result"], "success")
        self.assertEqual(event["tokens_used"], 500)
        self.assertIn("timestamp", event)
        self.assertIn("session_id", event)

    def test_finalize_report_generates_all_three_artifacts(self):
        """finalize_report produces report.md, telemetry.jsonl, and patch.diff."""
        # Log an event first
        log_transition(
            self.logger,
            phase="explore",
            iteration=0,
            action="start",
            result="ready",
            tokens_used=100,
        )

        plan = Plan(
            root_cause_hypothesis="Off-by-one error in loop index",
            files_to_modify=["calculator.py"],
            approach="Fix upper bound in range()",
            risks=[],
            test_strategy="Run unit tests",
        )
        diff = Diff(
            file_path="calculator.py",
            raw_diff_text="--- a/calculator.py\n+++ b/calculator.py\n@@ -1,1 +1,1 @@\n-range(10)\n+range(11)\n",
        )
        init_res = TestResult(passed=2, failed=1, errors=0)
        final_res = TestResult(passed=3, failed=0, errors=0)

        artifacts = finalize_report(
            output_dir=self.output_dir,
            task="Fix off-by-one bug in calculator",
            plan=plan,
            diffs=[diff],
            initial_test_result=init_res,
            final_test_result=final_res,
            telemetry_logger=self.logger,
            success=True,
            attempts=1,
        )

        report_md = artifacts["report"]
        telemetry_jsonl = artifacts["telemetry"]
        patch_diff = artifacts["patch"]

        self.assertTrue(report_md.exists())
        self.assertTrue(telemetry_jsonl.exists())
        self.assertTrue(patch_diff.exists())

        # Check report.md content
        content = report_md.read_text(encoding="utf-8")
        self.assertIn("## Issue Summary", content)
        self.assertIn("Fix off-by-one bug in calculator", content)

        self.assertIn("## Root Cause Found", content)
        self.assertIn("Off-by-one error in loop index", content)

        self.assertIn("## Changes Made", content)
        self.assertIn("calculator.py", content)
        self.assertIn("Fix upper bound in range()", content)

        self.assertIn("## Test Results (Before / After)", content)
        self.assertIn("Passed Tests", content)
        self.assertIn("Failed Tests", content)

        self.assertIn("## Unresolved Issues", content)
        self.assertIn("None", content)

        # Check patch.diff content
        diff_text = patch_diff.read_text(encoding="utf-8")
        self.assertIn("+range(11)", diff_text)

        # Check telemetry.jsonl content
        telemetry_lines = telemetry_jsonl.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(telemetry_lines), 1)

    def test_finalize_report_unresolved_failures(self):
        """When tests fail, report.md details unresolved failures."""
        plan = Plan(
            root_cause_hypothesis="Null pointer in auth",
            files_to_modify=["auth.py"],
            approach="Add null check",
            risks=[],
            test_strategy="Run auth tests",
        )
        fail = TestFailure(
            test_name="test_login_null",
            file="test_auth.py",
            assertion_error="AssertionError: Expected 200 got 500",
        )
        init_res = TestResult(passed=1, failed=1, errors=0)
        final_res = TestResult(passed=1, failed=1, errors=0, failures=[fail])

        budget = TokenBudget(
            total_budget=50_000,
            base_budget=50_000,
            file_count=5,
            total_loc=200,
            scale_factor=1.0,
            allocated={Phase.EXPLORE: 7500, Phase.PLAN: 5000, Phase.IMPLEMENT: 17500, Phase.VERIFY: 7500, Phase.REFLECT: 12500},
            used=46_000,
        )

        artifacts = finalize_report(
            output_dir=self.output_dir,
            task="Fix authentication null pointer",
            plan=plan,
            diffs=[],
            initial_test_result=init_res,
            final_test_result=final_res,
            success=False,
            budget=budget,
            attempts=3,
        )

        content = artifacts["report"].read_text(encoding="utf-8")
        self.assertIn("test_login_null", content)
        self.assertIn("AssertionError: Expected 200 got 500", content)
        self.assertIn("Run reached 90% token budget limit", content)

    def test_none_logger_is_safe(self):
        """Passing None as logger to log_transition, log_event, or close_logger does not raise."""
        try:
            log_transition(None, phase="explore", iteration=0, action="noop", result="ok", tokens_used=0)
            close_logger(None)
        except Exception as e:
            self.fail(f"Logger operations raised unexpected exception with None: {e}")

    def test_finalize_report_with_dict_plan_and_diffs(self):
        """finalize_report works seamlessly with dict plans and diffs."""
        plan = {
            "root_cause": "Typo in variable name",
            "files_to_modify": ["mod.py"],
            "approach": "Rename foo to bar",
        }
        diff = {
            "file_path": "mod.py",
            "raw_diff_text": "--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-foo\n+bar\n",
        }
        artifacts = finalize_report(
            output_dir=self.output_dir,
            task="Rename variable",
            plan=plan,
            diffs=[diff],
            success=True,
        )
        report = artifacts["report"].read_text(encoding="utf-8")
        self.assertIn("Typo in variable name", report)
        self.assertIn("mod.py", report)
        self.assertIn("Rename foo to bar", report)

        patch = artifacts["patch"].read_text(encoding="utf-8")
        self.assertIn("+bar", patch)


if __name__ == "__main__":
    unittest.main()
