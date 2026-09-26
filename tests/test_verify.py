#!/usr/bin/env python3
"""
Unit tests for harness.verify
==============================

Tests:
  - Auto-detection of test frameworks (pytest, unittest, jest, go test)
  - run_tests() on a small sample repo with one deliberately failing test
  - parse_test_output() for unittest text format
  - Timeout enforcement
  - test_result_to_dict() schema compliance
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.verify import (
    TestFailure,
    TestResult,
    LintResult,
    VerifyResult,
    TestFramework,
    _detect_test_framework,
    _parse_unittest_output,
    _parse_pytest_text_output,
    _parse_go_test_output,
    _parse_jest_output,
    run_tests,
    parse_test_output,
    test_result_to_dict,
    verify,
)


def _create_sample_python_repo(root: Path) -> None:
    """Create a minimal Python project with one passing and one failing test.

    Layout::

        root/
        ├── calculator.py
        └── test_calculator.py   (1 pass, 1 deliberate FAIL)
    """
    (root / "calculator.py").write_text(textwrap.dedent("""\
        def add(a, b):
            return a + b

        def subtract(a, b):
            return a + b  # BUG: should be a - b
    """))

    (root / "test_calculator.py").write_text(textwrap.dedent("""\
        import unittest
        from calculator import add, subtract

        class TestCalculator(unittest.TestCase):

            def test_add_works(self):
                self.assertEqual(add(2, 3), 5)

            def test_subtract_is_broken(self):
                # This MUST fail because subtract has a bug
                self.assertEqual(subtract(5, 3), 2)

        if __name__ == "__main__":
            unittest.main()
    """))


class TestDetectFramework(unittest.TestCase):
    """Verify that _detect_test_framework inspects the repo correctly."""

    def test_detect_unittest_from_test_files(self):
        """Detects unittest when test_*.py files exist but pytest is not available."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "test_example.py").write_text("import unittest\n")

            # Mock out pytest availability
            with patch("harness.verify.shutil.which", return_value=None):
                fw = _detect_test_framework(root)
            self.assertIn(fw, (TestFramework.PYTEST, TestFramework.UNITTEST))

    def test_detect_go_test(self):
        """Detects go test when go.mod is present."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "go.mod").write_text("module example.com/test\n\ngo 1.22\n")
            fw = _detect_test_framework(root)
            self.assertEqual(fw, TestFramework.GO_TEST)

    def test_detect_jest_from_package_json(self):
        """Detects jest when package.json lists jest in devDependencies."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "package.json").write_text(
                '{"devDependencies": {"jest": "^29.0.0"}}'
            )
            fw = _detect_test_framework(root)
            self.assertEqual(fw, TestFramework.JEST)

    def test_detect_vitest_from_package_json(self):
        """Detects vitest when package.json lists vitest."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "package.json").write_text(
                '{"devDependencies": {"vitest": "^1.0.0"}}'
            )
            fw = _detect_test_framework(root)
            self.assertEqual(fw, TestFramework.VITEST)

    def test_detect_cargo_test(self):
        """Detects cargo test when Cargo.toml is present."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Cargo.toml").write_text('[package]\nname = "test"\n')
            fw = _detect_test_framework(root)
            self.assertEqual(fw, TestFramework.CARGO_TEST)

    def test_detect_unknown_for_empty_repo(self):
        """Returns UNKNOWN for a bare directory with nothing recognisable."""
        with tempfile.TemporaryDirectory() as td:
            fw = _detect_test_framework(Path(td))
            self.assertEqual(fw, TestFramework.UNKNOWN)


class TestRunTestsWithSampleRepo(unittest.TestCase):
    """Run run_tests() on a real sample repo with a deliberate failure.

    This is the critical integration test: prove that the verify module
    correctly detects and structures a real test failure from subprocess
    output rather than approximating it.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        _create_sample_python_repo(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detects_failure_correctly(self):
        """run_tests finds 1 pass and 1 failure with structured failure info."""
        result = run_tests(self.root, timeout_s=30)

        # --- Subclass of dict and direct dictionary indexing ---
        self.assertIsInstance(result, dict, "run_tests should return a dict-compatible result")
        self.assertEqual(result["passed"], 1, "Expected exactly 1 passing test via dict indexing")
        self.assertEqual(result["failed"], 1, "Expected exactly 1 failing test via dict indexing")
        self.assertIsInstance(result["failures"], list)
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("subtract", result["failures"][0]["test_name"].lower())
        self.assertTrue(len(result["failures"][0]["assertion_error"]) > 0)
        self.assertTrue(len(result["failures"][0]["traceback_summary"]) > 0)
        self.assertIn("raw_tail", result)
        self.assertIsInstance(result["raw_tail"], str)

        # --- Core counts via attribute access ---
        self.assertEqual(result.passed, 1, "Expected exactly 1 passing test")
        self.assertEqual(result.failed, 1, "Expected exactly 1 failing test")
        self.assertTrue(result.failed > 0, "Should detect the deliberate failure")
        self.assertFalse(result.all_passed, "all_passed must be False when tests fail")

        # --- Structured failure info ---
        self.assertEqual(len(result.failures), 1, "Should capture exactly 1 failure")
        fail = result.failures[0]
        self.assertIn("subtract", fail.test_name.lower(),
                       "Failure test_name should mention 'subtract'")
        self.assertTrue(len(fail.assertion_error) > 0)
        self.assertTrue(len(fail.traceback_summary) > 0)

        # --- Raw output is captured ---
        self.assertTrue(len(result.raw_output) > 0, "raw_output should not be empty")

    def test_result_dict_schema(self):
        """test_result_to_dict returns the required JSON schema."""
        result = run_tests(self.root, timeout_s=30)
        d = test_result_to_dict(result)

        self.assertIn("passed", d)
        self.assertIn("failed", d)
        self.assertIn("failures", d)
        self.assertIn("raw_tail", d)

        self.assertIsInstance(d["passed"], int)
        self.assertIsInstance(d["failed"], int)
        self.assertIsInstance(d["failures"], list)
        self.assertIsInstance(d["raw_tail"], str)

        # Each failure has the right keys
        for f in d["failures"]:
            self.assertIn("test_name", f)
            self.assertIn("assertion_error", f)
            self.assertIn("traceback_summary", f)

    def test_all_passing_repo(self):
        """A repo with only passing tests returns all_passed=True."""
        # Overwrite the test file with only passing tests
        (self.root / "test_calculator.py").write_text(textwrap.dedent("""\
            import unittest
            from calculator import add

            class TestCalculator(unittest.TestCase):
                def test_add(self):
                    self.assertEqual(add(1, 2), 3)

            if __name__ == "__main__":
                unittest.main()
        """))

        result = run_tests(self.root, timeout_s=30)
        self.assertTrue(result.all_passed, "all_passed should be True when everything passes")
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result.failures), 0)


class TestTimeoutEnforcement(unittest.TestCase):
    """Verify the hard timeout kills runaway tests."""

    def test_timeout_kills_hanging_test(self):
        """A test that sleeps forever is killed after timeout_s."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "test_hang.py").write_text(textwrap.dedent("""\
                import time, unittest

                class TestHang(unittest.TestCase):
                    def test_infinite(self):
                        time.sleep(9999)

                if __name__ == "__main__":
                    unittest.main()
            """))

            t0 = time.perf_counter()
            result = run_tests(root, timeout_s=3)
            elapsed = time.perf_counter() - t0

            # Should complete well within 2× the timeout
            self.assertLess(elapsed, 10, "Timeout did not kill the process fast enough")

            # Should report an error
            self.assertGreater(result.errors, 0, "Timeout should count as an error")
            self.assertFalse(result.all_passed)
            self.assertIn("timeout", result.raw_output.lower())


class TestParseUnittestOutput(unittest.TestCase):
    """Unit test for the unittest text parser in isolation."""

    def test_parse_failure_output(self):
        raw = textwrap.dedent("""\
            test_add (test_calc.TestCalc.test_add) ... ok
            test_sub (test_calc.TestCalc.test_sub) ... FAIL

            ======================================================================
            FAIL: test_sub (test_calc.TestCalc.test_sub)
            ----------------------------------------------------------------------
            Traceback (most recent call last):
              File "/tmp/test_calc.py", line 10, in test_sub
                self.assertEqual(subtract(5, 3), 2)
            AssertionError: 8 != 2

            ----------------------------------------------------------------------
            Ran 2 tests in 0.001s

            FAILED (failures=1)
        """)

        result = _parse_unittest_output(raw)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.failures), 1)

        fail = result.failures[0]
        self.assertIn("test_sub", fail.test_name)
        self.assertIn("/tmp/test_calc.py", fail.file)
        self.assertEqual(fail.line, 10)
        self.assertIn("8 != 2", fail.message)

    def test_parse_all_passing(self):
        raw = textwrap.dedent("""\
            test_add (test_calc.TestCalc.test_add) ... ok
            test_mul (test_calc.TestCalc.test_mul) ... ok

            ----------------------------------------------------------------------
            Ran 2 tests in 0.001s

            OK
        """)

        result = _parse_unittest_output(raw)
        self.assertEqual(result.passed, 2)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result.failures), 0)


class TestParsePytestOutput(unittest.TestCase):
    """Unit test for the pytest text parser."""

    def test_parse_summary_line(self):
        raw = textwrap.dedent("""\
            FAILED test_app.py::test_login
            =================== 3 passed, 1 failed ===================
        """)

        result = _parse_pytest_text_output(raw)
        self.assertEqual(result.passed, 3)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("test_login", result.failures[0].test_name)


class TestParseGoTestOutput(unittest.TestCase):
    """Unit test for the go test JSON parser."""

    def test_parse_go_json(self):
        raw = (
            '{"Action":"pass","Package":"example.com/pkg","Test":"TestAdd","Elapsed":0.01}\n'
            '{"Action":"output","Package":"example.com/pkg","Test":"TestSub","Output":"    got: 8, want: 2\\n"}\n'
            '{"Action":"fail","Package":"example.com/pkg","Test":"TestSub","Elapsed":0.01}\n'
        )

        result = _parse_go_test_output(raw)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("TestSub", result.failures[0].test_name)


class TestParseJestOutput(unittest.TestCase):
    """Unit test for the jest JSON parser."""

    def test_parse_jest_json(self):
        raw = '{"numPassedTests": 5, "numFailedTests": 2, "numTotalTests": 7, "testResults": []}'
        result = _parse_jest_output(raw)
        self.assertEqual(result.passed, 5)
        self.assertEqual(result.failed, 2)


class TestVerifyIntegration(unittest.TestCase):
    """Test the top-level verify() function."""

    def test_verify_returns_structured_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _create_sample_python_repo(root)

            result = verify(root, timeout_s=30)

            self.assertIsInstance(result, VerifyResult)
            self.assertIsInstance(result.test_result, TestResult)
            self.assertFalse(result.success, "Should fail because subtract is broken")
            self.assertEqual(result.test_result.failed, 1)


if __name__ == "__main__":
    unittest.main()
