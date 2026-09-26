"""
harness.telemetry
=================

Cross-cutting JSONL telemetry layer.

Every phase logs structured events (entry, exit, errors, model calls, phase
transitions) as newline-delimited JSON to a single log file.  This enables
post-hoc analysis with ``jq``, pandas, or any JSONL-aware tool.

Typical usage::

    logger = init_logger(Path("run.jsonl"))
    log_transition(
        logger,
        phase="plan",
        iteration=1,
        action="generate_plan",
        result="success",
        tokens_used=1250,
    )
    close_logger(logger)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """A single structured telemetry event.

    Attributes:
        phase:       The harness phase that emitted this event
                     (e.g. ``"explore"``, ``"plan"``, ``"model_call"``).
        timestamp:   ISO-8601 timestamp string.
        duration_ms: Wall-clock duration of the operation in milliseconds.
        data:        Arbitrary payload — token counts, file lists, scores, etc.
        error:       If the event represents a failure, a human-readable message.
        iteration:   Current loop iteration (0 for explore/initialization).
        action:      Action being performed (e.g. "start", "generate_diff", "run_tests").
        result:      Outcome description (e.g. "success", "all_passed", "tests_failed").
        tokens_used: Cumulative or step token count.
    """

    phase: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    iteration: int = 0
    action: str = ""
    result: str = ""
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert event to a flat JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "phase": self.phase,
            "iteration": self.iteration,
            "action": self.action,
            "result": self.result,
            "tokens_used": self.tokens_used,
        }
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.error is not None:
            d["error"] = self.error
        if self.data:
            # Merge any extra payload without overwriting core fields
            for k, v in self.data.items():
                if k not in d:
                    d[k] = v
                else:
                    d[f"extra_{k}"] = v
        return d


@dataclass
class TelemetryLogger:
    """Handle to an open JSONL log file plus session-level metadata.

    Attributes:
        log_path:   Absolute path to the JSONL log file.
        session_id: A unique identifier for this harness run.
        _handle:    The underlying file handle (managed internally).
    """

    log_path: Path
    session_id: str
    _handle: io.TextIOWrapper | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_logger(log_path: Path | str, *, session_id: str | None = None) -> TelemetryLogger:
    """Initialise a JSONL telemetry logger.

    Creates parent directories if necessary and opens the file at ``log_path``
    in append mode.

    Args:
        log_path:   Where to write JSONL events.
        session_id: Optional explicit session id; auto-generated if omitted.

    Returns:
        An initialised :class:`TelemetryLogger`.
    """
    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    sid = session_id or uuid.uuid4().hex[:12]
    handle = open(path, mode="a", encoding="utf-8")

    return TelemetryLogger(
        log_path=path,
        session_id=sid,
        _handle=handle,
    )


def log_event(logger: TelemetryLogger | None, event: TelemetryEvent) -> None:
    """Append a single :class:`TelemetryEvent` to the JSONL log.

    The event is serialised as a single JSON line and flushed immediately
    so that partial runs are still observable.

    Args:
        logger: An initialised :class:`TelemetryLogger` (or None to no-op).
        event:  The event to record.
    """
    if logger is None:
        return
    if logger._handle is None or logger._handle.closed:
        logger._handle = open(logger.log_path, mode="a", encoding="utf-8")

    payload = event.to_dict()
    payload["session_id"] = logger.session_id

    line = json.dumps(payload, ensure_ascii=False) + "\n"
    logger._handle.write(line)
    logger._handle.flush()


def log_transition(
    logger: TelemetryLogger | None,
    *,
    phase: str,
    iteration: int,
    action: str,
    result: str,
    tokens_used: int,
    duration_ms: float | None = None,
    error: str | None = None,
    **extra: Any,
) -> None:
    """Convenience helper to log a phase transition event conforming to the schema:

        timestamp, phase, iteration, action, result, tokens_used

    Args:
        logger:      The active telemetry logger (or None to no-op).
        phase:       Current phase name (e.g. "explore", "plan", "implement").
        iteration:   Current attempt/loop iteration count.
        action:      Specific action taken in this phase.
        result:      Outcome of the action.
        tokens_used: Total tokens consumed so far.
        duration_ms: Optional duration of this phase in milliseconds.
        error:       Optional error string if failed.
        **extra:     Arbitrary additional context to include.
    """
    if logger is None:
        return
    event = TelemetryEvent(
        phase=phase,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_ms=duration_ms,
        data=extra,
        error=error,
        iteration=iteration,
        action=action,
        result=result,
        tokens_used=tokens_used,
    )
    log_event(logger, event)


def close_logger(logger: TelemetryLogger | None) -> None:
    """Flush and close the underlying log file.

    Safe to call multiple times or with None.

    Args:
        logger: The logger to close (or None).
    """
    if logger is None:
        return
    if logger._handle is not None and not logger._handle.closed:
        try:
            logger._handle.flush()
            logger._handle.close()
        except Exception:
            pass
        finally:
            logger._handle = None


def finalize_report(
    output_dir: Path | str = Path("."),
    *,
    task: str = "",
    plan: Any = None,
    diffs: list[Any] | None = None,
    initial_test_result: Any = None,
    final_test_result: Any = None,
    telemetry_logger: TelemetryLogger | None = None,
    unresolved: str | list[str] | None = None,
    repo_root: Path | str | None = None,
    success: bool | None = None,
    budget: Any = None,
    attempts: int = 0,
    **kwargs: Any,
) -> dict[str, Path]:
    """Generate final run artifacts: report.md, telemetry.jsonl, and patch.diff.

    Artifacts generated:
      - report.md:       Issue summary, root cause found, changes made,
                         test results before/after, what's unresolved if anything.
      - telemetry.jsonl: Complete machine-readable event trace.
      - patch.diff:      The final applied diff standalone.

    Args:
        output_dir:          Directory where artifacts will be written.
        task:                The issue / task description.
        plan:                The execution plan (Plan instance or dict).
        diffs:               Diffs applied during implementation.
        initial_test_result: Baseline or iteration 1 TestResult before fixes.
        final_test_result:   Final TestResult after fixes.
        telemetry_logger:    Active TelemetryLogger instance.
        unresolved:          Explicit description or list of unresolved issues.
        repo_root:           Target repository root.
        success:             Whether the run achieved overall success.
        budget:              TokenBudget instance tracking usage.
        attempts:            Total loop iterations executed.

    Returns:
        Dict with keys "report", "telemetry", "patch" mapping to their generated Paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.md"
    telemetry_path = out_dir / "telemetry.jsonl"
    patch_path = out_dir / "patch.diff"

    # -----------------------------------------------------------------------
    # 1. telemetry.jsonl: copy or ensure machine-readable trace
    # -----------------------------------------------------------------------
    if telemetry_logger is not None:
        try:
            if telemetry_logger._handle and not telemetry_logger._handle.closed:
                telemetry_logger._handle.flush()
        except Exception:
            pass
        src = Path(telemetry_logger.log_path).resolve()
        dst = telemetry_path.resolve()
        if src.exists() and src != dst:
            shutil.copyfile(src, dst)
        elif not dst.exists():
            dst.touch()
    elif not telemetry_path.exists():
        telemetry_path.touch()

    # -----------------------------------------------------------------------
    # 2. patch.diff: standalone unified diff
    # -----------------------------------------------------------------------
    patch_content = ""
    if repo_root:
        try:
            root_p = Path(repo_root).resolve()
            # Attempt to extract diff from git against pre-run initial checkpoint
            for tag in ["checkpoint_pre_run_initial", "checkpoint_initial_clean_state", "checkpoint_initial_state"]:
                tag_check = subprocess.run(
                    ["git", "rev-parse", "--verify", tag],
                    cwd=root_p,
                    capture_output=True,
                    check=False,
                )
                if tag_check.returncode == 0:
                    res = subprocess.run(
                        ["git", "diff", tag],
                        cwd=root_p,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        patch_content = res.stdout
                        break
            if not patch_content:
                res = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=root_p,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    patch_content = res.stdout
            if not patch_content:
                res = subprocess.run(
                    ["git", "diff", "HEAD~1"],
                    cwd=root_p,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    patch_content = res.stdout
        except Exception:
            pass

    if not patch_content and diffs:
        diff_chunks = []
        for d in diffs:
            if isinstance(d, dict):
                chunk = d.get("raw_diff_text") or d.get("diff_text") or d.get("patch") or str(d)
            else:
                chunk = getattr(d, "raw_diff_text", None) or getattr(d, "diff_text", None) or str(d)
            if chunk and chunk.strip():
                diff_chunks.append(chunk.strip())
        if diff_chunks:
            patch_content = "\n\n".join(diff_chunks) + "\n"

    patch_path.write_text(patch_content, encoding="utf-8")

    # -----------------------------------------------------------------------
    # 3. report.md: markdown synthesis of run outcome
    # -----------------------------------------------------------------------
    task_summary = task.strip() if task else "No task description provided."

    # Root cause found
    root_cause = "Not specified"
    if plan:
        if hasattr(plan, "root_cause_hypothesis") and plan.root_cause_hypothesis:
            root_cause = plan.root_cause_hypothesis
        elif isinstance(plan, dict) and plan.get("root_cause_hypothesis"):
            root_cause = plan["root_cause_hypothesis"]
        elif isinstance(plan, dict) and plan.get("root_cause"):
            root_cause = plan["root_cause"]

    # Changes made
    changes_lines: list[str] = []
    files_modified: list[str] = []
    if plan:
        if hasattr(plan, "files_to_modify") and plan.files_to_modify:
            files_modified.extend(plan.files_to_modify)
        elif isinstance(plan, dict) and plan.get("files_to_modify"):
            files_modified.extend(plan["files_to_modify"])

    if diffs:
        for d in diffs:
            fp = getattr(d, "file_path", None)
            if fp and fp not in files_modified:
                files_modified.append(fp)

    if files_modified:
        changes_lines.append("### Modified Files")
        for f in files_modified:
            changes_lines.append(f"- `{f}`")
    else:
        changes_lines.append("No files modified.")

    approach_text = ""
    if plan:
        if hasattr(plan, "approach") and plan.approach:
            approach_text = plan.approach
        elif isinstance(plan, dict) and plan.get("approach"):
            approach_text = plan["approach"]

    if approach_text:
        changes_lines.append(f"\n### Approach\n{approach_text}")

    # Test results before / after
    init_p = getattr(initial_test_result, "passed", "-") if initial_test_result else "N/A"
    init_f = getattr(initial_test_result, "failed", "-") if initial_test_result else "N/A"
    init_e = getattr(initial_test_result, "errors", "-") if initial_test_result else "N/A"
    init_status = "All Passed" if getattr(initial_test_result, "all_passed", False) else ("Failed" if initial_test_result else "N/A")

    fin_p = getattr(final_test_result, "passed", "-") if final_test_result else "N/A"
    fin_f = getattr(final_test_result, "failed", "-") if final_test_result else "N/A"
    fin_e = getattr(final_test_result, "errors", "-") if final_test_result else "N/A"
    fin_status = "All Passed" if getattr(final_test_result, "all_passed", False) else ("Failed" if final_test_result else "N/A")

    test_table = f"""| Metric | Before | After |
| :--- | :--- | :--- |
| **Passed Tests** | {init_p} | {fin_p} |
| **Failed Tests** | {init_f} | {fin_f} |
| **Errors** | {init_e} | {fin_e} |
| **Overall Status** | {init_status} | {fin_status} |"""

    # Unresolved issues
    is_success = bool(success or (final_test_result and getattr(final_test_result, "all_passed", False)))
    unresolved_items: list[str] = []
    if unresolved:
        if isinstance(unresolved, list):
            unresolved_items.extend(unresolved)
        else:
            unresolved_items.append(str(unresolved))
    else:
        if is_success:
            unresolved_items.append("None — All tests passed successfully.")
        else:
            if final_test_result and hasattr(final_test_result, "failures") and final_test_result.failures:
                unresolved_items.append(f"Failing tests ({len(final_test_result.failures)} remaining):")
                for fail in final_test_result.failures:
                    tname = getattr(fail, "test_name", None) or (fail.get("test_name") if isinstance(fail, dict) else str(fail))
                    err = getattr(fail, "assertion_error", None) or (fail.get("assertion_error") if isinstance(fail, dict) else "")
                    err_msg = f": {err[:120]}..." if err else ""
                    unresolved_items.append(f"  - `{tname}`{err_msg}")
            elif final_test_result and getattr(final_test_result, "failed", 0) > 0:
                unresolved_items.append(f"{final_test_result.failed} test(s) failed in the final run.")

            if budget and getattr(budget, "is_exhausted", False):
                unresolved_items.append("Run reached 90% token budget limit; stopped attempting new fixes.")
            elif attempts and attempts >= 3 and not is_success:
                unresolved_items.append(f"Reached retry limit of {attempts} attempts without all tests passing.")

    tokens_used_str = str(getattr(budget, "used", 0))
    if budget and hasattr(budget, "total_budget"):
        tokens_used_str += f" / {budget.total_budget}"

    unresolved_md = "\n".join(f"- {u}" if not u.startswith("  -") and not u.startswith("-") else u for u in unresolved_items) if unresolved_items else "- None"

    report_content = f"""# Harness Execution Report

## Issue Summary
{task_summary}

## Root Cause Found
{root_cause}

## Changes Made
{"\n".join(changes_lines)}

## Test Results (Before / After)
{test_table}

## Unresolved Issues
{unresolved_md}

## Execution Summary
- **Outcome**: {"SUCCESS" if is_success else "FAILED"}
- **Iterations / Attempts**: {attempts}
- **Tokens Used**: {tokens_used_str}
"""
    report_path.write_text(report_content, encoding="utf-8")

    return {
        "report": report_path,
        "telemetry": telemetry_path,
        "patch": patch_path,
    }
