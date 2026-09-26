"""
harness.reflect
===============

Phase 5 — Reflect: Failure diagnosis and retry logic.

When the Verify phase reports test failures or lint errors, this module
feeds the failures — along with the applied diffs and the original plan — back
to the LLM for root cause analysis and classification:
  - "wrong_diagnosis" (root cause hypothesis was wrong — go back to Plan)
  - "bad_implementation" (plan was right, code was wrong — retry Implement)
  - "unrelated_flaky" (test failure isn't related to our change — log and ignore)

Retry cycles are strictly bounded (default: 4 attempts max), and each cycle
explicitly narrows scope so cycles 3-4 demand surgical micro-fixes instead
of broad refactoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from harness.implement import ImplementResult
from harness.model import CallModelFn, ModelConfig, ModelResponse, ToolCall, call_model
from harness.plan import Plan
from harness.verify import LintResult, TestResult, VerifyResult


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

DEFAULT_MAX_REFLECT_CYCLES = 4

VALID_CLASSIFICATIONS = (
    "wrong_diagnosis",
    "bad_implementation",
    "unrelated_flaky",
)

DIAGNOSIS_TOOL = {
    "name": "submit_diagnosis",
    "description": (
        "Submit failure diagnosis and classification with an updated implementation plan."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": list(VALID_CLASSIFICATIONS),
                "description": (
                    "Classification of the failure: "
                    "'wrong_diagnosis' (plan hypothesis was wrong), "
                    "'bad_implementation' (plan was right, code was wrong), or "
                    "'unrelated_flaky' (failure unrelated to our changes)."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Clear explanation of why this classification was chosen and what went wrong.",
            },
            "updated_plan": {
                "type": "object",
                "properties": {
                    "root_cause_hypothesis": {
                        "type": "string",
                        "description": "Updated or refined root cause hypothesis.",
                    },
                    "files_to_modify": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files that need to be modified in the retry.",
                    },
                    "approach": {
                        "type": "string",
                        "description": "Updated step-by-step implementation approach, refined based on test failures.",
                    },
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identified risks or edge cases to watch out for.",
                    },
                    "test_strategy": {
                        "type": "string",
                        "description": "How the updated changes will be verified.",
                    },
                },
                "required": [
                    "root_cause_hypothesis",
                    "files_to_modify",
                    "approach",
                    "risks",
                    "test_strategy",
                ],
                "description": "The updated plan incorporating findings from the failure.",
            },
        },
        "required": ["classification", "reasoning", "updated_plan"],
    },
}

SYSTEM_PROMPT = """You are an expert principal software engineer acting as the failure reflection engine for an autonomous coding agent.
Your job is to analyze verification/test failures against the current plan and repository context, then diagnose the root cause.

Classify the failure into exactly ONE of:
1. "wrong_diagnosis":
   The initial root cause hypothesis was wrong or incomplete. The approach targeted the wrong files or misunderstandings.
   The agent must re-plan from scratch with a new hypothesis.
2. "bad_implementation":
   The root cause hypothesis and plan were sound, but the actual code change introduced syntax errors, logic bugs,
   off-by-one errors, or missed an edge case. The agent should re-implement with a refined diff.
3. "unrelated_flaky":
   The test failure is completely unrelated to the changes made (e.g. flaky network test, pre-existing broken test in
   an untouched subsystem). The agent should ignore this failure and not break other code trying to fix it.

Rules:
- You must invoke the `submit_diagnosis` tool to return your structured assessment.
- Adhere strictly to the scope guidance for the current reflection cycle number.
- For higher cycles (3 and 4), be extremely SURGICAL. Do not propose broad refactoring.
"""


# ---------------------------------------------------------------------------
# Exceptions & Data Structures
# ---------------------------------------------------------------------------

class MaxCyclesExceededError(ValueError, RuntimeError):
    """Raised when the reflection cycle count exceeds the configured maximum."""
    pass


class DiagnosisResult(dict):
    """Result of failure diagnosis conforming to the required dictionary schema::

        {
            "classification": str,
            "updated_plan": dict,
            "reasoning": str,
        }

    Also supports object attribute access for convenience and backwards-compatibility.
    """

    def __init__(
        self,
        classification: str = "bad_implementation",
        updated_plan: dict[str, Any] | None = None,
        reasoning: str = "",
        cycle: int = 1,
        root_cause: str | None = None,
        suggested_fix: str | None = None,
        confidence: float = 0.8,
        affected_files: list[str] | None = None,
        raw_response: str = "",
        **kwargs: Any,
    ) -> None:
        plan_dict = dict(updated_plan) if updated_plan is not None else {}
        super().__init__(
            classification=classification,
            updated_plan=plan_dict,
            reasoning=reasoning,
            **kwargs,
        )
        self.cycle = cycle
        self.root_cause = root_cause or reasoning
        self.suggested_fix = suggested_fix or plan_dict.get("approach", "")
        self.confidence = confidence
        self.affected_files = affected_files or plan_dict.get("files_to_modify", [])
        self.raw_response = raw_response

    @property
    def classification(self) -> str:
        return self.get("classification", "bad_implementation")

    @classification.setter
    def classification(self, value: str) -> None:
        self["classification"] = value

    @property
    def updated_plan(self) -> dict[str, Any]:
        return self.get("updated_plan", {})

    @updated_plan.setter
    def updated_plan(self, value: dict[str, Any]) -> None:
        self["updated_plan"] = value

    @property
    def reasoning(self) -> str:
        return self.get("reasoning", "")

    @reasoning.setter
    def reasoning(self, value: str) -> None:
        self["reasoning"] = value


@dataclass
class Diagnosis:
    """Legacy/dataclass analysis of why tests or lint failed."""

    root_cause: str
    suggested_fix: str
    confidence: float = 0.0
    affected_files: list[str] = field(default_factory=list)
    raw_response: str = ""
    classification: str = "bad_implementation"
    updated_plan: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def __getitem__(self, key: str) -> Any:
        if key == "classification":
            return self.classification
        elif key == "updated_plan":
            return self.updated_plan
        elif key == "reasoning":
            return self.reasoning or self.root_cause
        raise KeyError(key)


@dataclass
class ReflectResult:
    """Output of the Reflect phase."""

    diagnosis: Diagnosis | DiagnosisResult
    should_retry: bool
    attempt: int


# ---------------------------------------------------------------------------
# Scope Guidance & Prompt Formatting
# ---------------------------------------------------------------------------

def _get_cycle_scope_prompt(cycle: int, max_cycles: int) -> str:
    """Return scope-narrowing instructions tailored to the cycle number."""
    if cycle <= 1:
        return (
            f"=== REFLECTION CYCLE {cycle} of {max_cycles} (Standard / Broad Attempt) ===\n"
            "This is your initial reflection. Analyze whether the original hypothesis was flawed "
            "('wrong_diagnosis') or whether the implementation simply had a bug ('bad_implementation'). "
            "You may suggest standard adjustments to files_to_modify."
        )
    elif cycle == 2:
        return (
            f"=== REFLECTION CYCLE {cycle} of {max_cycles} (Narrowing Scope) ===\n"
            "Cycle 1 failed. You MUST narrow the scope of changes. "
            "Do not repeat the same approach. Focus specifically on edge cases, off-by-one errors, "
            "or missing validation logic within already touched files."
        )
    elif cycle == 3:
        return (
            f"=== REFLECTION CYCLE {cycle} of {max_cycles} (EXPLICIT SURGICAL SCOPE) ===\n"
            "WARNING: Cycle 3 of {max_cycles}. You MUST be significantly more SURGICAL than in cycle 1.\n"
            "- Do NOT perform broad refactoring, rewrite entire functions, or introduce new abstractions.\n"
            "- Narrow files_to_modify to the absolute minimum necessary (prefer 1 file).\n"
            "- Focus surgically on the exact assertion failure and line of code that failed.\n"
            "- Keep all other working code completely untouched."
        )
    else:
        return (
            f"=== REFLECTION CYCLE {cycle} of {max_cycles} (ULTRA-SURGICAL / FINAL ATTEMPT) ===\n"
            "URGENT: This is cycle {cycle} (FINAL ATTEMPT before limit exhaustion).\n"
            "You must be ULTRA-SURGICAL. Broad changes are strictly forbidden.\n"
            "- Propose only a minimal micro-fix (e.g. single condition check, 1-2 line correction, type fix).\n"
            "- Restrict files_to_modify strictly to the single failing file.\n"
            "- Zero risk tolerance for collateral breakage."
        )


def _format_verify_results(verify_results: Any) -> str:
    """Format test/verification results into a clean text summary for the prompt."""
    if isinstance(verify_results, VerifyResult):
        tr = verify_results.test_result
        lr = verify_results.lint_result
    elif isinstance(verify_results, TestResult) or isinstance(verify_results, dict):
        tr = verify_results
        lr = None
    else:
        return str(verify_results)

    passed = tr.get("passed", getattr(tr, "passed", 0)) if isinstance(tr, (dict, TestResult)) else 0
    failed = tr.get("failed", getattr(tr, "failed", 0)) if isinstance(tr, (dict, TestResult)) else 0
    failures = tr.get("failures", getattr(tr, "failures", [])) if isinstance(tr, (dict, TestResult)) else []
    raw_tail = tr.get("raw_tail", getattr(tr, "raw_tail", getattr(tr, "raw_output", ""))) if isinstance(tr, (dict, TestResult)) else ""

    lines = [
        f"Test Outcome: {passed} passed, {failed} failed",
    ]

    if failures:
        lines.append("\nFailures:")
        for i, f in enumerate(failures, 1):
            if isinstance(f, dict):
                name = f.get("test_name", "unknown")
                err = f.get("assertion_error", "")
                tb = f.get("traceback_summary", "")
            else:
                name = getattr(f, "test_name", "unknown")
                err = getattr(f, "assertion_error", getattr(f, "message", ""))
                tb = getattr(f, "traceback_summary", getattr(f, "traceback", ""))

            lines.append(f"  {i}. Test: {name}")
            if err:
                lines.append(f"     Assertion Error: {err}")
            if tb:
                lines.append(f"     Traceback Summary:\n{tb}")

    if lr is not None:
        issues = getattr(lr, "issues", [])
        if issues:
            lines.append("\nLint / Type-Check Issues:")
            for issue in issues[:10]:
                lines.append(f"  - {getattr(issue, 'file', '')}:{getattr(issue, 'line', '')} [{getattr(issue, 'code', '')}] {getattr(issue, 'message', '')}")

    if raw_tail:
        lines.append(f"\nRaw Runner Output Tail:\n{raw_tail[-2000:]}")

    return "\n".join(lines)


def _format_current_plan(current_plan: Any) -> str:
    """Format current implementation plan into a readable text summary."""
    if isinstance(current_plan, (dict, Plan)):
        hypothesis = current_plan.get("root_cause_hypothesis", "")
        files = current_plan.get("files_to_modify", [])
        approach = current_plan.get("approach", "")
        risks = current_plan.get("risks", [])
        strategy = current_plan.get("test_strategy", "")

        return (
            f"- Root Cause Hypothesis: {hypothesis}\n"
            f"- Files Targeted: {', '.join(files) if files else 'None'}\n"
            f"- Approach: {approach}\n"
            f"- Identified Risks: {', '.join(risks) if risks else 'None'}\n"
            f"- Test Strategy: {strategy}"
        )
    return str(current_plan)


def _format_repo_context(repo_context: Any) -> str:
    """Format repository context (diffs, files, or summary) into prompt string."""
    if isinstance(repo_context, ImplementResult):
        diff_texts = [d.diff_text for d in repo_context.diffs if hasattr(d, "diff_text")]
        return "\n".join(diff_texts) if diff_texts else "No diffs recorded."
    if isinstance(repo_context, dict):
        return json.dumps(repo_context, indent=2)
    return str(repo_context)


def _normalize_classification(raw: str) -> str:
    """Normalize classification to one of the 3 supported literals."""
    cleaned = raw.lower().strip().replace("-", "_").replace(" ", "_")
    if "wrong" in cleaned or "plan" in cleaned or "diagnosis" in cleaned:
        return "wrong_diagnosis"
    if "flaky" in cleaned or "unrelated" in cleaned or "external" in cleaned:
        return "unrelated_flaky"
    if "bad" in cleaned or "impl" in cleaned or "code" in cleaned:
        return "bad_implementation"
    return "bad_implementation"


def _extract_diagnosis_from_response(
    resp: ModelResponse,
    current_plan: Any,
) -> tuple[str, dict[str, Any], str]:
    """Extract (classification, updated_plan, reasoning) from model tool calls or text."""
    # 1. Inspect native tool calls
    if resp.tool_calls:
        for tc in resp.tool_calls:
            if isinstance(tc.input, dict):
                cls = tc.input.get("classification")
                if cls:
                    classification = _normalize_classification(str(cls))
                    reasoning = str(tc.input.get("reasoning") or "")
                    updated = tc.input.get("updated_plan")
                    if isinstance(updated, dict):
                        return classification, updated, reasoning
                    return classification, _fallback_plan(current_plan, reasoning), reasoning

    # 2. Fallback: Parse JSON from response text
    if resp.text:
        text = resp.text.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "classification" in data:
                    cls = _normalize_classification(str(data["classification"]))
                    reasoning = str(data.get("reasoning", ""))
                    updated = data.get("updated_plan")
                    if isinstance(updated, dict):
                        return cls, updated, reasoning
                    return cls, _fallback_plan(current_plan, reasoning), reasoning
            except Exception:
                pass

        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "classification" in data:
                    cls = _normalize_classification(str(data["classification"]))
                    reasoning = str(data.get("reasoning", ""))
                    updated = data.get("updated_plan")
                    if isinstance(updated, dict):
                        return cls, updated, reasoning
                    return cls, _fallback_plan(current_plan, reasoning), reasoning
            except Exception:
                pass

        # 3. Fallback: text heuristic
        lowered = text.lower()
        if "wrong_diagnosis" in lowered or "wrong diagnosis" in lowered:
            cls = "wrong_diagnosis"
        elif "unrelated_flaky" in lowered or "flaky" in lowered:
            cls = "unrelated_flaky"
        else:
            cls = "bad_implementation"

        return cls, _fallback_plan(current_plan, text[:200]), text

    return "bad_implementation", _fallback_plan(current_plan, "Model returned empty response"), "Model returned empty response."


def _fallback_plan(current_plan: Any, reasoning: str) -> dict[str, Any]:
    """Construct a sensible fallback updated_plan preserving current plan properties."""
    if isinstance(current_plan, (dict, Plan)):
        base = dict(current_plan)
        base["approach"] = f"Refined approach based on reflection: {reasoning[:200]}"
        return base

    return {
        "root_cause_hypothesis": "Failure observed during verification.",
        "files_to_modify": [],
        "approach": f"Retry with adjusted implementation: {reasoning[:200]}",
        "risks": ["Risk of regression in edge cases."],
        "test_strategy": "Re-run test suite to verify fix.",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diagnose_failure(
    verify_results: Any,
    current_plan: Any,
    repo_context: Any = "",
    cycle: int = 1,
    max_cycles: int = DEFAULT_MAX_REFLECT_CYCLES,
    *,
    model_fn: CallModelFn | None = None,
    config: ModelConfig | None = None,
    raise_on_cap: bool = False,
) -> DiagnosisResult:
    """Diagnose verification failures and classify root cause with scope narrowing.

    Calls ``call_model()`` with structured tool calling and classifies the failure
    as one of:
      - ``"wrong_diagnosis"`` (root cause hypothesis was wrong — go back to Plan)
      - ``"bad_implementation"`` (plan was right, code was wrong — retry Implement)
      - ``"unrelated_flaky"`` (test failure isn't related to our change — log and ignore)

    Total reflection cycles are capped at ``max_cycles`` (default: 4).
    The cycle number is passed into the prompt so cycles 3-4 explicitly ask
    the model to be more surgical than cycle 1's broad attempt.

    Args:
        verify_results: Structured test results from Verify (e.g. TestResult or dict).
        current_plan:   The current execution plan (Plan or dict).
        repo_context:   Repository context (diffs, modified files, repo map).
        cycle:          Current reflection cycle number (1-indexed).
        max_cycles:     Maximum reflection cycles permitted (default: 4).
        model_fn:       Optional model calling function (defaults to ``call_model``).
        config:         Optional model configuration override.
        raise_on_cap:   If ``True``, raises :class:`MaxCyclesExceededError` when
                        ``cycle > max_cycles``.

    Returns:
        A :class:`DiagnosisResult` dictionary with keys:
          - ``"classification"``: str
          - ``"updated_plan"``: dict
          - ``"reasoning"``: str
    """
    # Enforce reflection cycle cap
    if cycle > max_cycles:
        msg = f"Reflect cycle {cycle} exceeds maximum allowed cycles ({max_cycles}). Reflection capped."
        if raise_on_cap:
            raise MaxCyclesExceededError(msg)
        return DiagnosisResult(
            classification="wrong_diagnosis",
            updated_plan=dict(current_plan) if isinstance(current_plan, (dict, Plan)) else {},
            reasoning=msg,
            cycle=cycle,
        )

    scope_instructions = _get_cycle_scope_prompt(cycle, max_cycles)
    verify_summary = _format_verify_results(verify_results)
    plan_summary = _format_current_plan(current_plan)
    context_summary = _format_repo_context(repo_context)

    user_prompt = (
        f"{scope_instructions}\n\n"
        f"--- VERIFICATION RESULTS ---\n{verify_summary}\n\n"
        f"--- CURRENT PLAN ---\n{plan_summary}\n\n"
        f"--- APPLIED CHANGES / REPOSITORY CONTEXT ---\n{context_summary}\n\n"
        f"Analyze why verification failed and submit your diagnosis via `submit_diagnosis`.\n"
        f"Remember: In cycle {cycle} of {max_cycles}, tailor your scope strictly to the instructions above."
    )

    fn = model_fn or call_model
    resp = fn(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=SYSTEM_PROMPT,
        tools=[DIAGNOSIS_TOOL],
        config=config,
    )

    classification, updated_plan, reasoning = _extract_diagnosis_from_response(
        resp, current_plan
    )

    return DiagnosisResult(
        classification=classification,
        updated_plan=updated_plan,
        reasoning=reasoning,
        cycle=cycle,
        root_cause=reasoning,
        suggested_fix=updated_plan.get("approach", ""),
        confidence=0.85 if classification == "bad_implementation" else 0.75,
        affected_files=updated_plan.get("files_to_modify", []),
        raw_response=resp.text,
    )


def should_retry(
    diagnosis: Diagnosis | DiagnosisResult | dict[str, Any],
    cycle: int,
    max_cycles: int = DEFAULT_MAX_REFLECT_CYCLES,
) -> bool:
    """Decide whether the orchestrator should retry the Implement / Plan loop.

    Returns ``False`` if:
      - The current cycle reached or exceeded ``max_cycles``.
      - The failure is classified as ``"unrelated_flaky"``.
    """
    if cycle >= max_cycles:
        return False

    cls = diagnosis.get("classification") if isinstance(diagnosis, dict) else getattr(diagnosis, "classification", "")
    if cls == "unrelated_flaky":
        return False

    return True


def diagnose_failures(
    test_result: TestResult,
    lint_result: LintResult | None,
    impl_result: ImplementResult,
    model_fn: CallModelFn,
    *,
    current_plan: Any = None,
    cycle: int = 1,
    max_cycles: int = DEFAULT_MAX_REFLECT_CYCLES,
) -> DiagnosisResult:
    """Convenience adapter matching previous function signature."""
    verify_result = VerifyResult(test_result=test_result, lint_result=lint_result)
    plan = current_plan or {}
    return diagnose_failure(
        verify_results=verify_result,
        current_plan=plan,
        repo_context=impl_result,
        cycle=cycle,
        max_cycles=max_cycles,
        model_fn=model_fn,
    )


def reflect(
    test_result: TestResult,
    lint_result: LintResult | None,
    impl_result: ImplementResult,
    model_fn: CallModelFn,
    *,
    current_plan: Any = None,
    attempt: int = 1,
    max_retries: int = DEFAULT_MAX_REFLECT_CYCLES,
) -> ReflectResult:
    """Run the full Reflect phase: diagnose + decide retry."""
    diag = diagnose_failures(
        test_result=test_result,
        lint_result=lint_result,
        impl_result=impl_result,
        model_fn=model_fn,
        current_plan=current_plan,
        cycle=attempt,
        max_cycles=max_retries,
    )
    can_retry = should_retry(diag, cycle=attempt, max_cycles=max_retries)
    return ReflectResult(
        diagnosis=diag,
        should_retry=can_retry,
        attempt=attempt,
    )
