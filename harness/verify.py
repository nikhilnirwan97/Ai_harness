"""
harness.verify
==============

Phase 4 — Verify: Test execution and structured result parsing.

After changes are applied, this module runs the project's test suite (and
optionally a linter / type-checker) and parses the raw output into structured
results that the Reflect phase can reason about.

Subprocess execution is guarded with configurable timeouts to prevent
runaway test processes.

Typical usage::

    test_result = run_tests(repo_root)
    lint_result = run_linter(repo_root)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_TAIL_LIMIT = 5000
"""Maximum characters kept in raw_tail / raw_output fields."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TestFailure(dict):
    """A single test failure extracted from the test runner output.

    Accessible either as a dict or via attribute access:
      - fail["test_name"] or fail.test_name
      - fail["assertion_error"] or fail.assertion_error / fail.message
      - fail["traceback_summary"] or fail.traceback_summary / fail.traceback
    """

    __test__ = False

    def __init__(
        self,
        test_name: str = "",
        file: str | None = None,
        line: int | None = None,
        message: str = "",
        traceback: str = "",
        assertion_error: str | None = None,
        traceback_summary: str | None = None,
        **kwargs: Any,
    ) -> None:
        msg = assertion_error if assertion_error is not None else message
        tb = traceback or ""
        tb_summary = (
            traceback_summary
            if traceback_summary is not None
            else (tb[:2000] if tb else "")
        )
        super().__init__(
            test_name=test_name,
            assertion_error=msg,
            traceback_summary=tb_summary,
            **kwargs,
        )
        self.file = file
        self.line = line
        self.message = msg
        self.traceback = tb

    @property
    def test_name(self) -> str:
        return self.get("test_name", "")

    @test_name.setter
    def test_name(self, value: str) -> None:
        self["test_name"] = value

    @property
    def assertion_error(self) -> str:
        return self.get("assertion_error", self.message)

    @assertion_error.setter
    def assertion_error(self, value: str) -> None:
        self["assertion_error"] = value
        self.message = value

    @property
    def traceback_summary(self) -> str:
        return self.get("traceback_summary", self.traceback[:2000] if self.traceback else "")

    @traceback_summary.setter
    def traceback_summary(self, value: str) -> None:
        self["traceback_summary"] = value


class TestResult(dict):
    """Aggregated result of running the project's test suite.

    Subclasses dict so callers can use either dictionary indexing matching
    the required schema::

        {
            "passed": int,
            "failed": int,
            "failures": [{
                "test_name": str,
                "assertion_error": str,
                "traceback_summary": str,
            }],
            "raw_tail": str,
        }

    or object attribute access (.passed, .failed, .failures, .all_passed, .raw_output, etc.).
    """

    __test__ = False

    def __init__(
        self,
        passed: int = 0,
        failed: int = 0,
        errors: int = 0,
        skipped: int = 0,
        failures: list[TestFailure] | None = None,
        raw_output: str = "",
        duration_s: float = 0.0,
        raw_tail: str | None = None,
        **kwargs: Any,
    ) -> None:
        raw_str = raw_output or ""
        tail = (
            raw_tail
            if raw_tail is not None
            else (raw_str[-_RAW_TAIL_LIMIT:] if raw_str else "")
        )
        fail_list = [
            f if isinstance(f, TestFailure) else TestFailure(**f) if isinstance(f, dict) else f
            for f in (failures or [])
        ]
        super().__init__(
            passed=passed,
            failed=failed,
            failures=fail_list,
            raw_tail=tail,
            **kwargs,
        )
        self.errors = errors
        self.skipped = skipped
        self.raw_output = raw_str
        self.duration_s = duration_s

    @property
    def passed(self) -> int:
        return self.get("passed", 0)

    @passed.setter
    def passed(self, value: int) -> None:
        self["passed"] = value

    @property
    def failed(self) -> int:
        return self.get("failed", 0)

    @failed.setter
    def failed(self, value: int) -> None:
        self["failed"] = value

    @property
    def failures(self) -> list[TestFailure]:
        return self.get("failures", [])

    @failures.setter
    def failures(self, value: list[TestFailure]) -> None:
        self["failures"] = [
            f if isinstance(f, TestFailure) else TestFailure(**f) if isinstance(f, dict) else f
            for f in value
        ]

    @property
    def raw_tail(self) -> str:
        return self.get("raw_tail", self.raw_output[-_RAW_TAIL_LIMIT:] if self.raw_output else "")

    @raw_tail.setter
    def raw_tail(self, value: str) -> None:
        self["raw_tail"] = value

    @property
    def all_passed(self) -> bool:
        """Return True if there were no failures or errors."""
        return self.failed == 0 and self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "failures": [
                {
                    "test_name": f.test_name,
                    "assertion_error": getattr(f, "assertion_error", getattr(f, "message", "")),
                    "traceback_summary": getattr(f, "traceback_summary", getattr(f, "traceback", "")[:2000]),
                }
                for f in self.failures
            ],
            "raw_tail": self.raw_tail,
        }


@dataclass
class LintIssue:
    """A single linting or type-checking issue.

    Attributes:
        file:     Source file with the issue.
        line:     Line number.
        column:   Column number (if available).
        code:     Rule / error code (e.g. ``"E501"``, ``"mypy-error"``).
        message:  Human-readable description.
        severity: ``"error"``, ``"warning"``, or ``"info"``.
    """

    file: str
    line: int
    column: int | None = None
    code: str = ""
    message: str = ""
    severity: str = "warning"


@dataclass
class LintResult:
    """Aggregated linting / type-checking result.

    Attributes:
        issues:     List of individual lint issues.
        passed:     ``True`` if no errors were found (warnings are allowed).
        raw_output: Complete stdout+stderr from the linter.
    """

    issues: list[LintIssue] = field(default_factory=list)
    passed: bool = True
    raw_output: str = ""


@dataclass
class VerifyResult:
    """Combined output of the Verify phase.

    Attributes:
        test_result: Results from the test suite.
        lint_result: Results from the linter (``None`` if skipped).
        success:     ``True`` iff tests passed and lint passed (or was skipped).
    """

    test_result: TestResult
    lint_result: LintResult | None = None

    @property
    def success(self) -> bool:
        """Overall verification passed?"""
        lint_ok = self.lint_result is None or self.lint_result.passed
        return self.test_result.all_passed and lint_ok


# ---------------------------------------------------------------------------
# Test framework auto-detection
# ---------------------------------------------------------------------------

class TestFramework(str, Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    UNKNOWN = "unknown"


TestFramework.__test__ = False


def _detect_test_framework(repo_root: Path) -> TestFramework:
    """Inspect the repo to auto-detect the test framework.

    Detection order (first match wins):
      1. **Go**:   ``go.mod`` exists → ``go test``
      2. **Rust**:  ``Cargo.toml`` exists → ``cargo test``
      3. **Python**: Check for ``pytest.ini``, ``pyproject.toml`` [tool.pytest],
                     ``conftest.py``, or importable ``pytest``; fall back to ``unittest``.
      4. **JS/TS**: Check ``package.json`` devDependencies for jest / mocha / vitest.

    Returns:
        The detected :class:`TestFramework`.
    """
    # --- Go ---
    if (repo_root / "go.mod").exists():
        return TestFramework.GO_TEST

    # --- Rust ---
    if (repo_root / "Cargo.toml").exists():
        return TestFramework.CARGO_TEST

    # --- Python ---
    has_python_tests = (
        any(repo_root.rglob("test_*.py"))
        or any(repo_root.rglob("*_test.py"))
        or (repo_root / "tests").is_dir()
    )
    if has_python_tests:
        # Check explicit pytest markers
        if (repo_root / "pytest.ini").exists() or (repo_root / "setup.cfg").exists():
            return TestFramework.PYTEST

        pyproject = repo_root / "pyproject.toml"
        if pyproject.exists():
            try:
                text = pyproject.read_text(encoding="utf-8")
                if "[tool.pytest" in text:
                    return TestFramework.PYTEST
            except Exception:
                pass

        if (repo_root / "conftest.py").exists():
            return TestFramework.PYTEST

        # Check if pytest is importable in the repo's environment
        pytest_bin = shutil.which("pytest", path=str(repo_root / ".venv" / "bin"))
        if pytest_bin:
            return TestFramework.PYTEST

        # Also check system-level pytest
        if shutil.which("pytest"):
            return TestFramework.PYTEST

        # Fall back to stdlib unittest
        return TestFramework.UNITTEST

    # --- JavaScript / TypeScript ---
    pkg_json = repo_root / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            dev_deps = pkg.get("devDependencies", {})
            all_deps = {**pkg.get("dependencies", {}), **dev_deps}
            if "vitest" in all_deps:
                return TestFramework.VITEST
            if "jest" in all_deps:
                return TestFramework.JEST
            if "mocha" in all_deps:
                return TestFramework.MOCHA
        except Exception:
            pass

    return TestFramework.UNKNOWN


# ---------------------------------------------------------------------------
# Command builders — each returns (command_list, env_overrides)
# ---------------------------------------------------------------------------

def _build_pytest_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a pytest command with JSON report output."""
    report_path = repo_root / ".harness_pytest_report.json"
    # Prefer venv pytest, then system pytest, then python -m pytest
    venv_pytest = repo_root / ".venv" / "bin" / "pytest"
    if venv_pytest.is_file():
        base = [str(venv_pytest)]
    elif shutil.which("pytest"):
        base = ["pytest"]
    else:
        venv_python = repo_root / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.is_file() else sys.executable
        base = [python_bin, "-m", "pytest"]

    cmd = base + [
        "--tb=short",   # concise tracebacks
        "-q",           # quieter output
    ]

    # Try to use pytest-json-report if available
    try:
        check = subprocess.run(
            base + ["--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "--json-report" in check.stdout:
            cmd += [
                "--json-report",
                f"--json-report-file={report_path}",
            ]
    except Exception:
        pass

    return cmd, {}


def _build_unittest_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a unittest command."""
    venv_python = repo_root / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.is_file() else sys.executable
    return [python_bin, "-m", "unittest", "discover", "-s", str(repo_root), "-v"], {}


def _build_jest_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a jest command with JSON output."""
    npx = shutil.which("npx")
    if npx:
        return [npx, "jest", "--json", "--forceExit"], {}
    jest_bin = repo_root / "node_modules" / ".bin" / "jest"
    if jest_bin.is_file():
        return [str(jest_bin), "--json", "--forceExit"], {}
    return ["npx", "jest", "--json", "--forceExit"], {}


def _build_vitest_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a vitest command with JSON output."""
    npx = shutil.which("npx")
    if npx:
        return [npx, "vitest", "run", "--reporter=json"], {}
    return ["npx", "vitest", "run", "--reporter=json"], {}


def _build_mocha_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a mocha command with JSON output."""
    npx = shutil.which("npx")
    if npx:
        return [npx, "mocha", "--reporter", "json"], {}
    return ["npx", "mocha", "--reporter", "json"], {}


def _build_go_test_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a go test command with JSON output."""
    go_bin = shutil.which("go") or "go"
    return [go_bin, "test", "-json", "-count=1", "./..."], {}


def _build_cargo_test_cmd(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    """Build a cargo test command."""
    cargo_bin = shutil.which("cargo") or "cargo"
    return [cargo_bin, "test", "--", "--format=json", "-Z", "unstable-options"], {}


_FRAMEWORK_BUILDERS = {
    TestFramework.PYTEST: _build_pytest_cmd,
    TestFramework.UNITTEST: _build_unittest_cmd,
    TestFramework.JEST: _build_jest_cmd,
    TestFramework.VITEST: _build_vitest_cmd,
    TestFramework.MOCHA: _build_mocha_cmd,
    TestFramework.GO_TEST: _build_go_test_cmd,
    TestFramework.CARGO_TEST: _build_cargo_test_cmd,
}


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_pytest_output(raw: str, repo_root: Path) -> TestResult:
    """Parse pytest output, preferring JSON report if available."""
    report_path = repo_root / ".harness_pytest_report.json"

    # Try JSON report first
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return _parse_pytest_json_report(report, raw)
        except Exception:
            pass
        finally:
            try:
                report_path.unlink()
            except Exception:
                pass

    # Fall back to parsing text output
    return _parse_pytest_text_output(raw)


def _parse_pytest_json_report(report: dict[str, Any], raw: str) -> TestResult:
    """Parse a pytest-json-report JSON object."""
    summary = report.get("summary", {})
    tests = report.get("tests", [])

    failures: list[TestFailure] = []
    for t in tests:
        outcome = t.get("outcome", "")
        if outcome in ("failed", "error"):
            call_info = t.get("call", {})
            crash = call_info.get("crash", {})
            failures.append(TestFailure(
                test_name=t.get("nodeid", "unknown"),
                file=crash.get("path"),
                line=crash.get("lineno"),
                message=crash.get("message", ""),
                traceback=call_info.get("longrepr", ""),
            ))

    return TestResult(
        passed=summary.get("passed", 0),
        failed=summary.get("failed", 0),
        errors=summary.get("error", 0),
        skipped=summary.get("skipped", 0) + summary.get("deselected", 0),
        failures=failures,
        raw_output=raw,
        duration_s=summary.get("duration", 0.0),
    )


def _parse_pytest_text_output(raw: str) -> TestResult:
    """Parse pytest short-summary text output."""
    passed = failed = errors = skipped = 0
    failures: list[TestFailure] = []

    # Parse summary line: lines containing "=+" with passed/failed/error/skipped
    for line in raw.splitlines():
        line_clean = line.strip()
        if re.search(r"\b(?:\d+\s+passed|\d+\s+failed|\d+\s+errors?|\d+\s+skipped|\d+\s+deselected)\b", line_clean):
            m_p = re.search(r"(\d+)\s+passed\b", line_clean)
            m_f = re.search(r"(\d+)\s+failed\b", line_clean)
            m_e = re.search(r"(\d+)\s+errors?\b", line_clean)
            m_s = re.search(r"(\d+)\s+skipped\b", line_clean)
            m_d = re.search(r"(\d+)\s+deselected\b", line_clean)
            if any([m_p, m_f, m_e, m_s, m_d]):
                if m_p:
                    passed = int(m_p.group(1))
                if m_f:
                    failed = int(m_f.group(1))
                if m_e:
                    errors = int(m_e.group(1))
                if m_s:
                    skipped += int(m_s.group(1))
                if m_d:
                    skipped += int(m_d.group(1))

    # Parse FAILED lines
    fail_re = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
    for m in fail_re.finditer(raw):
        name = m.group(1)
        if not any(f.test_name == name for f in failures):
            failures.append(TestFailure(test_name=name))

    # Parse failure blocks
    failure_block_re = re.compile(
        r"_{4,}\s+(\S+)\s+_{4,}\n(.*?)(?=\n_{4,}|\n={4,}|\Z)",
        re.DOTALL,
    )
    for m in failure_block_re.finditer(raw):
        name = m.group(1)
        block = m.group(2).strip()
        # Extract assertion error
        assert_re = re.search(r"(AssertionError|assert\w*Error|Exception|Error):?\s*(.*)", block)
        msg = assert_re.group(0) if assert_re else ""
        existing = [
            f for f in failures
            if f.test_name == name or f.test_name.endswith(f"::{name}") or name in f.test_name
        ]
        if existing:
            existing[0].message = msg
            existing[0].assertion_error = msg
            existing[0].traceback = block
            existing[0].traceback_summary = block[:2000]
        else:
            failures.append(TestFailure(
                test_name=name,
                message=msg,
                traceback=block,
            ))

    return TestResult(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        failures=failures,
        raw_output=raw,
    )


def _parse_unittest_output(raw: str) -> TestResult:
    """Parse ``python -m unittest`` verbose output."""
    passed = failed = errors = skipped = 0
    failures: list[TestFailure] = []

    # Count individual test results from verbose lines:
    #   test_something (test_module.TestClass.test_something) ... ok
    #   test_something (test_module.TestClass.test_something) ... FAIL
    ok_count = len(re.findall(r"\.\.\.\s+ok\s*$", raw, re.MULTILINE))
    fail_count = len(re.findall(r"\.\.\.\s+FAIL\s*$", raw, re.MULTILINE))
    err_count = len(re.findall(r"\.\.\.\s+ERROR\s*$", raw, re.MULTILINE))
    skip_count = len(re.findall(r"\.\.\.\s+skipped\b", raw, re.MULTILINE))

    passed = ok_count
    failed = fail_count
    errors = err_count
    skipped = skip_count

    # Also parse summary line: "Ran N tests in Xs"
    ran_re = re.search(r"Ran\s+(\d+)\s+test", raw)
    total_ran = int(ran_re.group(1)) if ran_re else (passed + failed + errors)

    # Override with summary counts if present: "FAILED (failures=X, errors=Y)"
    summary_re = re.search(
        r"FAILED\s*\("
        r"(?:failures=(\d+))?"
        r"(?:,?\s*errors=(\d+))?"
        r"\)",
        raw,
    )
    if summary_re:
        if summary_re.group(1):
            failed = int(summary_re.group(1))
        if summary_re.group(2):
            errors = int(summary_re.group(2))
        passed = max(0, total_ran - failed - errors - skipped)

    # Parse FAIL / ERROR blocks
    block_re = re.compile(
        r"^={50,}\n(FAIL|ERROR):\s+(\S+).*?\n-{50,}\n(.*?)(?=\n={50,}|\nRan\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for m in block_re.finditer(raw):
        kind = m.group(1)
        name = m.group(2)
        block = m.group(3).strip()

        # Extract assertion error message
        assert_re = re.search(
            r"(AssertionError|AssertError|assert\w*Error|Exception|Error):?\s*(.*)",
            block,
        )
        msg = assert_re.group(0) if assert_re else ""

        # Extract file / line from traceback
        file_re = re.search(r'File "([^"]+)", line (\d+)', block)
        fail_file = file_re.group(1) if file_re else None
        fail_line = int(file_re.group(2)) if file_re else None

        failures.append(TestFailure(
            test_name=name,
            file=fail_file,
            line=fail_line,
            message=msg,
            traceback=block,
        ))

    return TestResult(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        failures=failures,
        raw_output=raw,
    )


def _parse_jest_output(raw: str) -> TestResult:
    """Parse Jest JSON output."""
    # Jest --json writes a JSON blob, possibly with non-JSON preamble
    json_start = raw.find("{")
    if json_start < 0:
        return _parse_generic_output(raw)

    try:
        data = json.loads(raw[json_start:])
    except json.JSONDecodeError:
        return _parse_generic_output(raw)

    passed = data.get("numPassedTests", 0)
    failed = data.get("numFailedTests", 0)
    total = data.get("numTotalTests", 0)
    skipped = total - passed - failed

    failures: list[TestFailure] = []
    for suite in data.get("testResults", []):
        for tc in suite.get("assertionResults", []) + suite.get("testResults", []):
            status = tc.get("status", "")
            if status in ("failed",):
                msgs = tc.get("failureMessages", [])
                failures.append(TestFailure(
                    test_name=tc.get("fullName") or tc.get("title", "unknown"),
                    file=suite.get("name"),
                    message=msgs[0][:500] if msgs else "",
                    traceback="\n".join(msgs),
                ))

    return TestResult(
        passed=passed,
        failed=failed,
        errors=0,
        skipped=skipped,
        failures=failures,
        raw_output=raw,
    )


def _parse_go_test_output(raw: str) -> TestResult:
    """Parse ``go test -json`` output (newline-delimited JSON events)."""
    passed = failed = errors = 0
    failures: list[TestFailure] = []
    output_buffer: dict[str, list[str]] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = ev.get("Action", "")
        test = ev.get("Test", "")
        pkg = ev.get("Package", "")
        output_text = ev.get("Output", "")

        if test and output_text:
            key = f"{pkg}/{test}"
            output_buffer.setdefault(key, []).append(output_text)

        if action == "pass" and test:
            passed += 1
        elif action == "fail" and test:
            failed += 1
            key = f"{pkg}/{test}"
            tb = "".join(output_buffer.get(key, []))
            # Extract assertion message
            msg_lines = [l for l in tb.splitlines() if "Error" in l or "FAIL" in l or "got" in l.lower()]
            failures.append(TestFailure(
                test_name=f"{pkg}.{test}",
                message=msg_lines[0] if msg_lines else "",
                traceback=tb,
            ))
        elif action == "skip" and test:
            pass  # counted implicitly

    return TestResult(
        passed=passed,
        failed=failed,
        errors=errors,
        failures=failures,
        raw_output=raw,
    )


def _parse_generic_output(raw: str) -> TestResult:
    """Best-effort parser for unknown test runner output."""
    # Try to find common patterns
    pass_re = re.findall(r"(\d+)\s+(?:pass(?:ed|ing)?|✓)", raw, re.IGNORECASE)
    fail_re = re.findall(r"(\d+)\s+(?:fail(?:ed|ing|ure)?|✗|✘)", raw, re.IGNORECASE)

    passed = int(pass_re[-1]) if pass_re else 0
    failed = int(fail_re[-1]) if fail_re else 0

    return TestResult(
        passed=passed,
        failed=failed,
        raw_output=raw,
    )


_FRAMEWORK_PARSERS = {
    TestFramework.PYTEST: lambda raw, root: _parse_pytest_output(raw, root),
    TestFramework.UNITTEST: lambda raw, root: _parse_unittest_output(raw),
    TestFramework.JEST: lambda raw, root: _parse_jest_output(raw),
    TestFramework.VITEST: lambda raw, root: _parse_jest_output(raw),  # vitest JSON is jest-compatible
    TestFramework.MOCHA: lambda raw, root: _parse_jest_output(raw),   # mocha JSON is similar
    TestFramework.GO_TEST: lambda raw, root: _parse_go_test_output(raw),
    TestFramework.CARGO_TEST: lambda raw, root: _parse_generic_output(raw),
}


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    env_overrides: dict[str, str] | None = None,
) -> tuple[str, float, bool]:
    """Run a subprocess with timeout, returning (combined_output, duration_s, timed_out).

    Args:
        cmd:            Command to run.
        cwd:            Working directory.
        timeout_s:      Hard timeout in seconds.
        env_overrides:  Extra environment variables to set.

    Returns:
        (output, duration_seconds, timed_out)
    """
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # Ensure we don't hang on interactive pagers
    env["PAGER"] = "cat"

    t0 = time.perf_counter()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        output = (e.stdout or b"").decode(errors="replace") + "\n" + (e.stderr or b"").decode(errors="replace")
        output += f"\n\n[HARNESS] Process killed after {timeout_s}s timeout.\n"
    except FileNotFoundError:
        output = f"[HARNESS] Command not found: {cmd[0]}"
    except Exception as exc:
        output = f"[HARNESS] Subprocess error: {exc}"

    duration = time.perf_counter() - t0
    return output, duration, timed_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tests(
    repo_root: Path | str,
    *,
    test_cmd: str | None = None,
    timeout_s: int = 120,
) -> TestResult:
    """Execute the project's test suite and return structured results.

    Auto-detects the test runner by inspecting the repository contents
    (pytest, unittest, jest, vitest, mocha, go test, cargo test).
    Does **not** hardcode a single framework.

    When ``test_cmd`` is ``None`` the framework is auto-detected.  When
    an explicit command string is provided it is run as-is via the shell.

    Args:
        repo_root: Absolute path to the repository root.
        test_cmd:  Explicit test command override (e.g. ``"pytest -x"``).
        timeout_s: Maximum wall-clock seconds before killing the process
                   (default 120).

    Returns:
        A :class:`TestResult` with pass/fail counts, structured failure
        details, and the raw output tail.  The return dict is compatible
        with the schema::

            {
                "passed": int,
                "failed": int,
                "failures": [{
                    "test_name": str,
                    "assertion_error": str,
                    "traceback_summary": str,
                }],
                "raw_tail": str,
            }
    """
    root = Path(repo_root).resolve()

    if test_cmd:
        # Explicit command — run via shell
        cmd_list = test_cmd.split()
        framework = _guess_framework_from_cmd(test_cmd)
        env_overrides: dict[str, str] = {}
    else:
        framework = _detect_test_framework(root)
        if framework == TestFramework.UNKNOWN:
            return TestResult(
                raw_output="[HARNESS] No test framework detected.",
            )

        builder = _FRAMEWORK_BUILDERS.get(framework)
        if builder is None:
            return TestResult(
                raw_output=f"[HARNESS] Unsupported framework: {framework}",
            )

        cmd_list, env_overrides = builder(root)

    # Execute
    raw_output, duration_s, timed_out = _run_subprocess(
        cmd_list, cwd=root, timeout_s=timeout_s, env_overrides=env_overrides,
    )

    # Parse
    parser = _FRAMEWORK_PARSERS.get(framework)
    if parser:
        result = parser(raw_output, root)
    else:
        result = _parse_generic_output(raw_output)

    result.duration_s = duration_s
    result.raw_output = raw_output[-_RAW_TAIL_LIMIT:]

    if timed_out:
        result.errors += 1
        result.failures.append(TestFailure(
            test_name="[timeout]",
            message=f"Test process killed after {timeout_s}s hard timeout.",
        ))

    return result


def _guess_framework_from_cmd(cmd: str) -> TestFramework:
    """Guess the framework from an explicit command string."""
    cmd_lower = cmd.lower()
    if "pytest" in cmd_lower:
        return TestFramework.PYTEST
    if "unittest" in cmd_lower:
        return TestFramework.UNITTEST
    if "jest" in cmd_lower:
        return TestFramework.JEST
    if "vitest" in cmd_lower:
        return TestFramework.VITEST
    if "mocha" in cmd_lower:
        return TestFramework.MOCHA
    if "go test" in cmd_lower:
        return TestFramework.GO_TEST
    if "cargo test" in cmd_lower:
        return TestFramework.CARGO_TEST
    return TestFramework.UNKNOWN


def parse_test_output(raw_output: str, framework: str | None = None) -> TestResult:
    """Parse raw test runner output into a structured :class:`TestResult`.

    Supports pytest, unittest, jest, and go test output formats.

    Args:
        raw_output: Complete stdout+stderr from the test runner.
        framework:  Optional framework hint (``"pytest"``, ``"unittest"``, etc.).
                    If ``None``, attempts auto-detection from output content.

    Returns:
        A :class:`TestResult` extracted from the output.
    """
    if framework:
        fw = TestFramework(framework) if framework in TestFramework.__members__.values() else TestFramework.UNKNOWN
    else:
        # Heuristic detection from output
        if "pytest" in raw_output or "PASSED" in raw_output or "FAILED" in raw_output:
            fw = TestFramework.PYTEST
        elif "Ran " in raw_output and "test" in raw_output:
            fw = TestFramework.UNITTEST
        elif '"numPassedTests"' in raw_output:
            fw = TestFramework.JEST
        elif '"Action"' in raw_output and '"Test"' in raw_output:
            fw = TestFramework.GO_TEST
        else:
            fw = TestFramework.UNKNOWN

    parser = _FRAMEWORK_PARSERS.get(fw)
    if parser:
        return parser(raw_output, Path.cwd())
    return _parse_generic_output(raw_output)


def run_linter(
    repo_root: Path | str,
    *,
    linter_cmd: str | None = None,
    timeout_s: int = 120,
) -> LintResult:
    """Run linting / type-checking and return structured results.

    Auto-detects the linter (ruff → flake8 → mypy) when ``linter_cmd``
    is ``None``.

    Args:
        repo_root:  Absolute path to the repository root.
        linter_cmd: Explicit linter command override.
        timeout_s:  Maximum wall-clock seconds before killing the process.

    Returns:
        A :class:`LintResult` with individual issues.
    """
    root = Path(repo_root).resolve()

    if linter_cmd:
        cmd = linter_cmd.split()
    else:
        # Auto-detect: ruff > flake8 > mypy
        if shutil.which("ruff"):
            cmd = ["ruff", "check", str(root)]
        elif shutil.which("flake8"):
            cmd = ["flake8", str(root)]
        elif shutil.which("mypy"):
            cmd = ["mypy", str(root)]
        else:
            return LintResult(passed=True, raw_output="[HARNESS] No linter detected.")

    raw, _, timed_out = _run_subprocess(cmd, cwd=root, timeout_s=timeout_s)

    issues: list[LintIssue] = []
    # Parse standard file:line:col: CODE message format
    issue_re = re.compile(r"^([^\s:]+):(\d+):(\d+):\s*(\S+)\s+(.*)", re.MULTILINE)
    for m in issue_re.finditer(raw):
        code = m.group(4)
        severity = "error" if code.startswith("E") or code.startswith("F") else "warning"
        issues.append(LintIssue(
            file=m.group(1),
            line=int(m.group(2)),
            column=int(m.group(3)),
            code=code,
            message=m.group(5),
            severity=severity,
        ))

    has_errors = any(i.severity == "error" for i in issues)
    return LintResult(issues=issues, passed=not has_errors, raw_output=raw[-_RAW_TAIL_LIMIT:])


def verify(
    repo_root: Path | str,
    *,
    test_cmd: str | None = None,
    linter_cmd: str | None = None,
    timeout_s: int = 120,
) -> VerifyResult:
    """Run the full Verify phase: tests + optional linting.

    Args:
        repo_root:  Absolute path to the repository root.
        test_cmd:   Explicit test command override.
        linter_cmd: Explicit linter command override (``None`` to skip linting).
        timeout_s:  Hard timeout for each subprocess.

    Returns:
        A :class:`VerifyResult` combining test and lint outcomes.
    """
    root = Path(repo_root).resolve()

    test_result = run_tests(root, test_cmd=test_cmd, timeout_s=timeout_s)

    lint_result: LintResult | None = None
    if linter_cmd is not None:
        lint_result = run_linter(root, linter_cmd=linter_cmd, timeout_s=timeout_s)

    return VerifyResult(test_result=test_result, lint_result=lint_result)


# ---------------------------------------------------------------------------
# Convenience: dict serialisation matching the required schema
# ---------------------------------------------------------------------------

def test_result_to_dict(result: TestResult) -> dict[str, Any]:
    """Serialize a :class:`TestResult` to the required JSON schema.

    Returns::

        {
            "passed": int,
            "failed": int,
            "failures": [{
                "test_name": str,
                "assertion_error": str,
                "traceback_summary": str,
            }],
            "raw_tail": str,
        }
    """
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "passed": result.get("passed", 0) if isinstance(result, dict) else result.passed,
        "failed": result.get("failed", 0) if isinstance(result, dict) else result.failed,
        "failures": [
            {
                "test_name": f["test_name"] if isinstance(f, dict) else f.test_name,
                "assertion_error": f.get("assertion_error", getattr(f, "message", "")) if isinstance(f, dict) else getattr(f, "assertion_error", f.message),
                "traceback_summary": f.get("traceback_summary", getattr(f, "traceback", "")[:2000]) if isinstance(f, dict) else getattr(f, "traceback_summary", f.traceback[:2000] if f.traceback else ""),
            }
            for f in (result.get("failures", []) if isinstance(result, dict) else result.failures)
        ],
        "raw_tail": result.get("raw_tail", "") if isinstance(result, dict) else getattr(result, "raw_tail", result.raw_output[-_RAW_TAIL_LIMIT:]),
    }


test_result_to_dict.__test__ = False
