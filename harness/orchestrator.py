"""
harness.orchestrator
====================

The main phase loop — an adaptive state machine that drives the full
Explore → Plan → Implement → Verify → Reflect → Finalize pipeline.

State transitions:

    EXPLORE (once)  →  PLAN  →  IMPLEMENT  →  VERIFY
                                               │
                                         ┌─────┴──────┐
                                         │             │
                                    tests pass    tests fail
                                         │             │
                                         ▼             ▼
                                     FINALIZE       REFLECT
                                                       │
                                                  ┌────┴────┐
                                                  │         │
                                              retry?    give up / budget
                                                  │         │
                                                  ▼         ▼
                                              PLAN/IMPL  FINALIZE

Adaptive Token Budgeting:
  1. Measures repository size (file count & lines of code) at start.
  2. Scales the total token budget from a base configuration value.
  3. Allocates tokens across phases roughly as:
     - Explore:   15%
     - Plan:      10%
     - Implement: 35%
     - Verify:    15%
     - Reflect:   25%
  4. At 90% of budget consumed, stops attempting new fixes and transitions
     directly to Finalize with the best passing state so far — never hard-crashing.

Telemetry:
  Every phase transition is logged via :mod:`harness.telemetry` as JSONL
  recording: timestamp, phase, iteration, action, result, tokens_used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable

from harness.explore import ExploreResult, RankedFile, explore
from harness.implement import (
    ApplyResult,
    Diff,
    ImplementResult,
    apply_diff_safely,
    generate_diff,
    git_checkpoint,
    git_rollback,
)
from harness.model import CallModelFn, ModelConfig, ModelResponse, call_model
from harness.plan import Plan, generate_plan
from harness.reflect import DiagnosisResult, diagnose_failure, should_retry
from harness.telemetry import (
    TelemetryLogger,
    close_logger,
    finalize_report,
    init_logger,
    log_transition,
)
from harness.verify import TestResult, VerifyResult, run_linter, run_tests


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

DEFAULT_BASE_TOKEN_BUDGET = 50_000
BUDGET_EXHAUSTION_THRESHOLD = 0.90

PHASE_BUDGET_RATIOS: dict[Phase, float] = {}


class Phase(str, Enum):
    """Phases of the coding-agent harness."""

    EXPLORE = "explore"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REFLECT = "reflect"
    FINALIZE = "finalize"


PHASE_BUDGET_RATIOS = {
    Phase.EXPLORE: 0.15,
    Phase.PLAN: 0.10,
    Phase.IMPLEMENT: 0.35,
    Phase.VERIFY: 0.15,
    Phase.REFLECT: 0.25,
}


# ---------------------------------------------------------------------------
# Adaptive Token Budgeting
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Adaptive token budget scaled by repository size.

    Attributes:
        total_budget:         Total tokens allocated for the entire run.
        base_budget:          Base configuration budget before scaling.
        file_count:           Number of source files measured at start.
        total_loc:            Total lines of code measured at start.
        scale_factor:         Scale multiplier applied to base_budget.
        allocated:            Planned token allocation per phase.
        used:                 Total tokens consumed across all model calls.
        phase_used:           Tokens consumed partitioned by phase.
        exhaustion_threshold: Fraction of budget at which to stop (0.90).
    """

    total_budget: int
    base_budget: int
    file_count: int
    total_loc: int
    scale_factor: float
    allocated: dict[Phase, int]
    used: int = 0
    phase_used: dict[Phase, int] = field(default_factory=dict)
    exhaustion_threshold: float = BUDGET_EXHAUSTION_THRESHOLD

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.used)

    @property
    def is_exhausted(self) -> bool:
        """True when consumed tokens reach or exceed 90% of total budget."""
        return self.used >= int(self.total_budget * self.exhaustion_threshold)

    def record_usage(self, phase: Phase | str, tokens: int) -> None:
        """Record tokens consumed in a specific phase."""
        p = Phase(phase) if isinstance(phase, str) and phase in [x.value for x in Phase] else phase
        self.used += tokens
        self.phase_used[p] = self.phase_used.get(p, 0) + tokens


def measure_repo_size(repo_root: Path) -> tuple[int, int]:
    """Measure the repository size in file count and total lines of code (LOC).

    Excludes hidden files/directories and common build caches (.git, node_modules, etc.).

    Returns:
        (file_count, total_loc)
    """
    root = Path(repo_root).resolve()
    file_count = 0
    total_loc = 0

    ignored_dirs = {
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build", "target",
        ".gemini", ".idea", ".vscode"
    }

    if not root.exists():
        return 0, 0

    for path in root.rglob("*"):
        if any(part in ignored_dirs or (part.startswith(".") and part != ".env.example") for part in path.parts):
            continue
        if path.is_file():
            file_count += 1
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = sum(1 for line in f if line.strip())
                    total_loc += lines
            except Exception:
                pass

    return file_count, total_loc


def calculate_adaptive_budget(
    repo_root: Path,
    base_budget: int = DEFAULT_BASE_TOKEN_BUDGET,
) -> TokenBudget:
    """Calculate an adaptive token budget scaled by repository size (file count and LOC).

    Rough allocation:
      - Explore:   15%
      - Plan:      10%
      - Implement: 35%
      - Verify:    15%
      - Reflect:   25%
    """
    file_count, total_loc = measure_repo_size(repo_root)

    # Base scale starts at 1.0 for small repos (< 10 files, < 1000 LOC)
    scale_factor = 1.0
    if file_count > 10 or total_loc > 1000:
        file_scale = max(0.0, (file_count - 10) / 40.0)
        loc_scale = max(0.0, (total_loc - 1000) / 3000.0)
        scale_factor = min(4.0, 1.0 + 0.3 * file_scale + 0.7 * loc_scale)

    total_budget = int(base_budget * scale_factor)

    allocated = {
        Phase.EXPLORE: int(total_budget * PHASE_BUDGET_RATIOS[Phase.EXPLORE]),
        Phase.PLAN: int(total_budget * PHASE_BUDGET_RATIOS[Phase.PLAN]),
        Phase.IMPLEMENT: int(total_budget * PHASE_BUDGET_RATIOS[Phase.IMPLEMENT]),
        Phase.VERIFY: int(total_budget * PHASE_BUDGET_RATIOS[Phase.VERIFY]),
        Phase.REFLECT: int(total_budget * PHASE_BUDGET_RATIOS[Phase.REFLECT]),
    }
    remainder = total_budget - sum(allocated.values())
    if remainder:
        allocated[Phase.IMPLEMENT] += remainder

    return TokenBudget(
        total_budget=total_budget,
        base_budget=base_budget,
        file_count=file_count,
        total_loc=total_loc,
        scale_factor=round(scale_factor, 2),
        allocated=allocated,
    )


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BestState:
    """Tracks the best passing state observed across all iterations."""

    plan: Plan | None = None
    diffs: list[Diff] = field(default_factory=list)
    test_result: TestResult | None = None
    checkpoint_id: str | None = None
    passed_count: int = -1
    failed_count: int = 999999
    all_passed: bool = False

    def is_better_than(self, new_result: TestResult) -> bool:
        """Evaluate if new_result is strictly better than our current best."""
        new_passed = getattr(new_result, "passed", 0)
        new_failed = getattr(new_result, "failed", 0)

        if new_result.all_passed and not self.all_passed:
            return True
        if self.all_passed and not new_result.all_passed:
            return False

        if new_passed > self.passed_count:
            return True
        if new_passed == self.passed_count and new_failed < self.failed_count:
            return True
        return False

    def update(
        self,
        plan: Plan | None,
        diffs: list[Diff],
        test_result: TestResult,
        checkpoint_id: str | None = None,
    ) -> None:
        """Update best state with new metrics."""
        self.plan = plan
        self.diffs = list(diffs)
        self.test_result = test_result
        self.checkpoint_id = checkpoint_id
        self.passed_count = getattr(test_result, "passed", 0)
        self.failed_count = getattr(test_result, "failed", 0)
        self.all_passed = getattr(test_result, "all_passed", False)


@dataclass
class OrchestratorConfig:
    """Top-level configuration for a harness run.

    Attributes:
        model_config:      LLM provider configuration.
        max_retries:       Maximum Implement → Verify retry loops.
        test_cmd:          Explicit test command (auto-detected if ``None``).
        linter_cmd:        Explicit linter command (``None`` to skip linting).
        log_path:          Path to the JSONL telemetry log file.
        report_dir:        Optional directory for output reports (defaults to repo_root).
        base_token_budget: Base token budget to scale from (default 50,000).
        model_fn:          Optional dependency-injected model caller (for testing/mocking).
    """

    model_config: ModelConfig = field(default_factory=ModelConfig)
    max_retries: int = 3
    test_cmd: str | None = None
    linter_cmd: str | None = None
    log_path: Path = Path("harness_run.jsonl")
    report_dir: Path | str | None = None
    base_token_budget: int = DEFAULT_BASE_TOKEN_BUDGET
    model_fn: CallModelFn | None = None


@dataclass
class RunResult:
    """Final outcome of a complete harness run.

    Attributes:
        success:        ``True`` if all tests passed after implementation.
        phase_reached:  The last phase that was executed.
        plan:           The plan that was generated (``None`` if Explore failed).
        diffs:          All diffs that were applied.
        test_result:    Final test result (``None`` if Verify was never reached).
        attempts:       Total number of Implement → Verify iterations.
        error:          Human-readable error if the run failed prematurely.
        budget:         The token budget tracking object.
        report_path:    Path to the generated report.md file.
        telemetry_path: Path to the generated telemetry.jsonl file.
        patch_path:     Path to the generated patch.diff file.
    """

    success: bool = False
    phase_reached: Phase = Phase.EXPLORE
    plan: Plan | None = None
    diffs: list[Diff] = field(default_factory=list)
    test_result: TestResult | None = None
    attempts: int = 0
    error: str | None = None
    budget: TokenBudget | None = None
    report_path: Path | None = None
    telemetry_path: Path | None = None
    patch_path: Path | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    task: str,
    repo_root: Path | str,
    config: OrchestratorConfig | None = None,
) -> RunResult:
    """Execute the full coding-agent pipeline.

    Drives the state machine through all phases:
      1. **Explore** (once) — build repo map, symbol index, rank files.
      2. **Plan → Implement → Verify → Reflect** loop until:
         - Tests pass (success!), OR
         - Token budget reaches 90% exhaustion (safely stops without crash), OR
         - Retry cap hits (gives up).
      3. **Finalize** — preserves best passing state and emits final telemetry.

    Logs every phase transition via :mod:`harness.telemetry` as JSONL.

    Args:
        task:      Natural-language task description.
        repo_root: Absolute or relative path to the repository root.
        config:    Orchestrator configuration (uses defaults if ``None``).

    Returns:
        A :class:`RunResult` summarising the entire run.
    """
    cfg = config or OrchestratorConfig()
    root = Path(repo_root).resolve()

    # Ensure git ignores .jsonl files so telemetry logs are preserved across rollbacks
    try:
        gi = root / ".gitignore"
        rule = "\n*.jsonl\n"
        if gi.exists():
            txt = gi.read_text(encoding="utf-8", errors="ignore")
            if "*.jsonl" not in txt:
                gi.write_text(txt + rule, encoding="utf-8")
        else:
            gi.write_text(rule, encoding="utf-8")
    except Exception:
        pass

    # 1. Initialize Telemetry Logger
    logger: TelemetryLogger = init_logger(cfg.log_path)

    # 2. Adaptive Token Budgeting
    budget = calculate_adaptive_budget(root, base_budget=cfg.base_token_budget)

    # Model wrapper that tracks token usage per phase
    current_phase = Phase.EXPLORE

    def counting_model_fn(*args: Any, **kwargs: Any) -> ModelResponse:
        nonlocal current_phase
        base_fn = cfg.model_fn or call_model
        t0 = time.perf_counter()
        resp = base_fn(*args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate tokens consumed
        tokens = 0
        if hasattr(resp, "usage") and resp.usage:
            tokens = getattr(resp.usage, "total_tokens", 0)
            if not tokens:
                tokens = getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0)

        if not tokens:
            # Fallback estimation for mocks or models without token metadata
            prompt_chars = sum(len(m.get("content", "")) for m in kwargs.get("messages", []))
            resp_chars = len(getattr(resp, "text", ""))
            tokens = max(50, (prompt_chars + resp_chars) // 4)

        budget.record_usage(current_phase, tokens)
        return resp

    # State tracking
    best_state = BestState()
    all_diffs: list[Diff] = []
    current_plan: Plan | None = None
    previous_plan_dict: dict[str, Any] | None = None
    initial_test_result: TestResult | None = None
    last_test_result: TestResult | None = None
    phase_reached = Phase.EXPLORE
    iteration = 0
    run_error: str | None = None

    try:
        # Checkpoint initial clean state for diff extraction in finalize_report
        try:
            git_checkpoint("pre_run_initial", root)
        except Exception:
            pass

        # ===================================================================
        # PHASE 1: EXPLORE (Runs Once)
        # ===================================================================
        current_phase = Phase.EXPLORE
        phase_reached = Phase.EXPLORE
        t_phase = time.perf_counter()

        log_transition(
            logger,
            phase=Phase.EXPLORE.value,
            iteration=0,
            action="start",
            result=f"repo_size_{budget.file_count}_files_{budget.total_loc}_loc",
            tokens_used=budget.used,
            scale_factor=budget.scale_factor,
            total_budget=budget.total_budget,
        )

        explore_result: ExploreResult = explore(root, task=task)
        ranked_files = explore_result.ranked_files

        dur_ms = (time.perf_counter() - t_phase) * 1000.0
        log_transition(
            logger,
            phase=Phase.EXPLORE.value,
            iteration=0,
            action="complete",
            result=f"ranked_{len(ranked_files)}_files",
            tokens_used=budget.used,
            duration_ms=dur_ms,
        )

        # ===================================================================
        # LOOP: PLAN → IMPLEMENT → VERIFY → REFLECT
        # ===================================================================
        while iteration < cfg.max_retries:
            iteration += 1

            # --- Budget Exhaustion Check (90% threshold) ---
            if budget.is_exhausted:
                log_transition(
                    logger,
                    phase=Phase.FINALIZE.value,
                    iteration=iteration,
                    action="budget_exhausted",
                    result=f"consumed_{budget.used}_of_{budget.total_budget}_tokens_90pct_limit",
                    tokens_used=budget.used,
                )
                break

            # ---------------------------------------------------------------
            # PHASE 2: PLAN
            # ---------------------------------------------------------------
            current_phase = Phase.PLAN
            phase_reached = Phase.PLAN
            t_plan = time.perf_counter()

            log_transition(
                logger,
                phase=Phase.PLAN.value,
                iteration=iteration,
                action="start",
                result="generating_plan",
                tokens_used=budget.used,
            )

            current_plan = generate_plan(
                issue_text=task,
                ranked_files=ranked_files,
                previous_plan=previous_plan_dict,
                repo_root=root,
                model_fn=counting_model_fn,
                config=cfg.model_config,
            )

            dur_plan = (time.perf_counter() - t_plan) * 1000.0
            log_transition(
                logger,
                phase=Phase.PLAN.value,
                iteration=iteration,
                action="complete",
                result=f"plan_targets_{len(current_plan.files_to_modify)}_files",
                tokens_used=budget.used,
                duration_ms=dur_plan,
            )

            # Check budget again before implement
            if budget.is_exhausted:
                log_transition(
                    logger,
                    phase=Phase.FINALIZE.value,
                    iteration=iteration,
                    action="budget_exhausted",
                    result=f"consumed_{budget.used}_of_{budget.total_budget}_tokens",
                    tokens_used=budget.used,
                )
                break

            # ---------------------------------------------------------------
            # PHASE 3: IMPLEMENT
            # ---------------------------------------------------------------
            current_phase = Phase.IMPLEMENT
            phase_reached = Phase.IMPLEMENT
            t_impl = time.perf_counter()

            log_transition(
                logger,
                phase=Phase.IMPLEMENT.value,
                iteration=iteration,
                action="start",
                result="generating_diff",
                tokens_used=budget.used,
            )

            diff = generate_diff(
                plan=current_plan,
                ranked_files=ranked_files,
                repo_root=root,
                model_fn=counting_model_fn,
                config=cfg.model_config,
            )
            all_diffs.append(diff)

            # Checkpoint git state before applying
            checkpoint_pre = f"iter_{iteration}_pre_apply"
            try:
                git_checkpoint(checkpoint_pre, root)
            except Exception:
                pass

            apply_res = apply_diff_safely(diff.diff_text, repo_path=root)
            checkpoint_applied = f"iter_{iteration}_applied"
            if apply_res.success:
                try:
                    git_checkpoint(checkpoint_applied, root)
                except Exception:
                    pass

            dur_impl = (time.perf_counter() - t_impl) * 1000.0
            log_transition(
                logger,
                phase=Phase.IMPLEMENT.value,
                iteration=iteration,
                action="apply_diff",
                result="success" if apply_res.success else f"failed_{apply_res.error}",
                tokens_used=budget.used,
                duration_ms=dur_impl,
            )

            # ---------------------------------------------------------------
            # PHASE 4: VERIFY
            # ---------------------------------------------------------------
            current_phase = Phase.VERIFY
            phase_reached = Phase.VERIFY
            t_verify = time.perf_counter()

            log_transition(
                logger,
                phase=Phase.VERIFY.value,
                iteration=iteration,
                action="start",
                result="running_tests",
                tokens_used=budget.used,
            )

            test_result = run_tests(root, test_cmd=cfg.test_cmd, timeout_s=120)
            if initial_test_result is None:
                initial_test_result = test_result
            last_test_result = test_result

            # Check and track best passing state
            if best_state.is_better_than(test_result):
                best_state.update(
                    plan=current_plan,
                    diffs=all_diffs,
                    test_result=test_result,
                    checkpoint_id=checkpoint_applied if apply_res.success else None,
                )

            dur_verify = (time.perf_counter() - t_verify) * 1000.0
            log_transition(
                logger,
                phase=Phase.VERIFY.value,
                iteration=iteration,
                action="run_tests",
                result=f"{test_result.passed}_passed_{test_result.failed}_failed",
                tokens_used=budget.used,
                duration_ms=dur_verify,
            )

            # If all tests pass, we are DONE!
            if test_result.all_passed:
                log_transition(
                    logger,
                    phase=Phase.FINALIZE.value,
                    iteration=iteration,
                    action="tests_passed",
                    result="all_tests_passed_success",
                    tokens_used=budget.used,
                )
                break

            # ---------------------------------------------------------------
            # PHASE 5: REFLECT
            # ---------------------------------------------------------------
            current_phase = Phase.REFLECT
            phase_reached = Phase.REFLECT

            # Check if retry limit reached
            if iteration >= cfg.max_retries:
                log_transition(
                    logger,
                    phase=Phase.REFLECT.value,
                    iteration=iteration,
                    action="retry_cap_hit",
                    result=f"max_retries_{cfg.max_retries}_reached",
                    tokens_used=budget.used,
                )
                break

            # Check if budget exhausted before reflection
            if budget.is_exhausted:
                log_transition(
                    logger,
                    phase=Phase.FINALIZE.value,
                    iteration=iteration,
                    action="budget_exhausted",
                    result=f"consumed_{budget.used}_tokens_pre_reflect",
                    tokens_used=budget.used,
                )
                break

            t_reflect = time.perf_counter()
            log_transition(
                logger,
                phase=Phase.REFLECT.value,
                iteration=iteration,
                action="start",
                result=f"diagnosing_failure_cycle_{iteration}",
                tokens_used=budget.used,
            )

            diagnosis = diagnose_failure(
                verify_results=test_result,
                current_plan=current_plan,
                repo_context=diff.diff_text if diff else "",
                cycle=iteration,
                max_cycles=cfg.max_retries,
                model_fn=counting_model_fn,
                config=cfg.model_config,
            )

            dur_reflect = (time.perf_counter() - t_reflect) * 1000.0
            classification = diagnosis.get("classification", "bad_implementation")

            log_transition(
                logger,
                phase=Phase.REFLECT.value,
                iteration=iteration,
                action="complete",
                result=f"classified_{classification}",
                tokens_used=budget.used,
                duration_ms=dur_reflect,
            )

            # Unrelated / flaky test failure or should not retry
            if not should_retry(diagnosis, cycle=iteration, max_cycles=cfg.max_retries):
                action = "unrelated_flaky_ignored" if classification == "unrelated_flaky" else "retry_cap_hit"
                log_transition(
                    logger,
                    phase=Phase.FINALIZE.value,
                    iteration=iteration,
                    action=action,
                    result=f"proceeding_to_finalize_{classification}",
                    tokens_used=budget.used,
                )
                break

            # Prepare updated plan for next loop iteration
            updated_p = diagnosis.get("updated_plan")
            if updated_p:
                previous_plan_dict = updated_p

        # ===================================================================
        # PHASE 6: FINALIZE
        # ===================================================================
        current_phase = Phase.FINALIZE
        phase_reached = Phase.FINALIZE

        # Determine success
        success = best_state.all_passed or (last_test_result is not None and last_test_result.all_passed)

        # If not successful but a better past state existed, restore it
        final_plan = current_plan
        final_diffs = all_diffs
        final_test_result = last_test_result

        if not success and best_state.passed_count > 0:
            final_plan = best_state.plan
            final_diffs = best_state.diffs
            final_test_result = best_state.test_result
            if best_state.checkpoint_id:
                try:
                    git_rollback(best_state.checkpoint_id, root)
                except Exception:
                    pass

        log_transition(
            logger,
            phase=Phase.FINALIZE.value,
            iteration=iteration,
            action="finalize",
            result="success" if success else "completed_with_failures",
            tokens_used=budget.used,
            final_passed=getattr(final_test_result, "passed", 0) if final_test_result else 0,
            final_failed=getattr(final_test_result, "failed", 0) if final_test_result else 0,
        )

        # Generate final report artifacts: report.md, telemetry.jsonl, patch.diff
        report_artifacts = finalize_report(
            output_dir=cfg.report_dir or root,
            task=task,
            plan=final_plan,
            diffs=final_diffs,
            initial_test_result=initial_test_result,
            final_test_result=final_test_result,
            telemetry_logger=logger,
            repo_root=root,
            success=success,
            budget=budget,
            attempts=iteration,
        )

        return RunResult(
            success=success,
            phase_reached=Phase.FINALIZE,
            plan=final_plan,
            diffs=final_diffs,
            test_result=final_test_result,
            attempts=iteration,
            error=None,
            budget=budget,
            report_path=report_artifacts.get("report"),
            telemetry_path=report_artifacts.get("telemetry"),
            patch_path=report_artifacts.get("patch"),
        )

    except Exception as exc:
        run_error = str(exc)
        log_transition(
            logger,
            phase=current_phase.value,
            iteration=iteration,
            action="error",
            result="exception_caught",
            tokens_used=budget.used,
            error=run_error,
        )
        report_artifacts = {}
        try:
            report_artifacts = finalize_report(
                output_dir=cfg.report_dir or root,
                task=task,
                plan=current_plan,
                diffs=all_diffs,
                initial_test_result=initial_test_result,
                final_test_result=last_test_result,
                telemetry_logger=logger,
                repo_root=root,
                success=False,
                budget=budget,
                attempts=iteration,
                unresolved=f"Fatal exception: {run_error}",
            )
        except Exception:
            pass

        return RunResult(
            success=False,
            phase_reached=current_phase,
            plan=current_plan,
            diffs=all_diffs,
            test_result=last_test_result,
            attempts=iteration,
            error=run_error,
            budget=budget,
            report_path=report_artifacts.get("report"),
            telemetry_path=report_artifacts.get("telemetry"),
            patch_path=report_artifacts.get("patch"),
        )
    finally:
        close_logger(logger)
