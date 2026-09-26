"""
harness.plan
============

Phase 2 — Plan: Structured JSON plan generation.

Given the task description, ranked files from the Explore phase, and optional
previous plan from the Reflect phase, this module calls the LLM using native
structured tool-calling to produce a machine-readable plan:

{
  "root_cause_hypothesis": str,
  "files_to_modify": list[str],
  "approach": str,
  "risks": list[str],
  "test_strategy": str
}

Validation:
Every file in ``files_to_modify`` is verified to exist in the repository.
If any file does not exist, the model call is automatically retried once with
corrective feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Literal

from harness.explore import ExploreResult, RankedFile
from harness.model import CallModelFn, ModelConfig, ModelResponse, ToolCall, call_model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single atomic step in the execution plan.

    Attributes:
        file:        Relative path of the target file.
        action:      What to do — create, modify, or delete.
        description: Natural-language description of the change.
        symbol:      Optional symbol name to scope the change
                     (e.g. ``"MyClass.my_method"``).
    """

    file: str
    action: Literal["create", "modify", "delete"]
    description: str
    symbol: str | None = None


class Plan(dict):
    """A validated, structured execution plan.

    Inherits from ``dict`` so it conforms directly to the required JSON schema
    as a dictionary while supporting attribute-based access and PlanStep lists
    for downstream phases.

    Schema:
        root_cause_hypothesis: str
        files_to_modify:       list[str]
        approach:              str
        risks:                 list[str]
        test_strategy:         str
    """

    def __init__(
        self,
        root_cause_hypothesis: str = "",
        files_to_modify: list[str] | None = None,
        approach: str = "",
        risks: list[str] | None = None,
        test_strategy: str = "",
        steps: list[PlanStep] | None = None,
        task: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> None:
        files = [str(f) for f in (files_to_modify or [])]
        risk_list = [str(r) for r in (risks or [])]

        data = {
            "root_cause_hypothesis": root_cause_hypothesis,
            "files_to_modify": files,
            "approach": approach,
            "risks": risk_list,
            "test_strategy": test_strategy,
            **kwargs,
        }
        super().__init__(data)

        # Object attributes
        self.root_cause_hypothesis: str = root_cause_hypothesis
        self.files_to_modify: list[str] = files
        self.approach: str = approach
        self.risks: list[str] = risk_list
        self.test_strategy: str = test_strategy
        self.task: str = task
        self.context: str = context

        # Generate atomic PlanSteps for downstream Implement phase if not provided
        if steps is not None:
            self.steps: list[PlanStep] = steps
        else:
            self.steps = [
                PlanStep(
                    file=f,
                    action="modify",
                    description=approach or f"Modify {f} according to plan",
                )
                for f in files
            ]

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'Plan' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("root_cause_hypothesis", "files_to_modify", "approach", "risks", "test_strategy"):
            self[name] = value
        super().__setattr__(name, value)


@dataclass
class PlanIssue:
    """A validation issue found in a generated plan.

    Attributes:
        step_index: Index of the problematic step (``None`` for plan-level).
        severity:   ``"error"`` or ``"warning"``.
        message:    Human-readable description of the issue.
    """

    step_index: int | None
    severity: Literal["error", "warning"]
    message: str


# ---------------------------------------------------------------------------
# Structured Output Tool Schema
# ---------------------------------------------------------------------------

PLAN_SUBMIT_TOOL = {
    "name": "submit_plan",
    "description": (
        "Submit the structured implementation plan to address the issue. "
        "You must provide all required fields conforming to the schema."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause_hypothesis": {
                "type": "string",
                "description": "Clear hypothesis explaining the root cause of the issue.",
            },
            "files_to_modify": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of repository-relative file paths that MUST be modified to fix the issue. "
                    "Every file in this list MUST already exist in the repository."
                ),
            },
            "approach": {
                "type": "string",
                "description": "Step-by-step description of the implementation strategy and code changes.",
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential regression risks, edge cases, and side effects of this change.",
            },
            "test_strategy": {
                "type": "string",
                "description": "How the solution will be verified (reproduction steps, unit tests, commands).",
            },
        },
        "required": [
            "root_cause_hypothesis",
            "files_to_modify",
            "approach",
            "risks",
            "test_strategy",
        ],
    },
}


SYSTEM_PROMPT = """You are an expert principal software engineer acting as the planning engine for an autonomous coding agent.
Your job is to analyze the issue description and repository context, then submit a concrete, structured plan.

Rules:
1. Always call the `submit_plan` tool to return your structured plan.
2. Every file path in `files_to_modify` MUST be an exact relative path that exists in the repository.
   Do NOT invent new non-existent files unless strictly required, and prefer modifying existing files.
3. Be concise, precise, and actionable. State the root cause directly and provide concrete steps in your approach.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_plan_from_response(resp: ModelResponse) -> dict[str, Any] | None:
    """Extract structured plan dictionary from model tool calls or text JSON."""
    # 1. Inspect native tool calls
    if resp.tool_calls:
        for tc in resp.tool_calls:
            if tc.name == "submit_plan" and isinstance(tc.input, dict):
                return tc.input
        # If any tool call returned dict matching expected keys
        for tc in resp.tool_calls:
            if isinstance(tc.input, dict) and "files_to_modify" in tc.input:
                return tc.input

    # 2. Fallback: Parse JSON from response text
    if resp.text:
        text = resp.text.strip()
        # Check for ```json ... ``` code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # Try to parse entire text as JSON
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    return None


def _format_context(
    issue_text: str,
    ranked_files: list[RankedFile] | list[str] | None,
    previous_plan: dict[str, Any] | Plan | None = None,
) -> str:
    """Build user message prompt incorporating issue, ranked files, and previous plan."""
    lines: list[str] = [
        "### Issue Description",
        issue_text.strip(),
        "",
        "### Ranked Relevant Files in Repository",
    ]

    if ranked_files:
        for rf in ranked_files:
            if isinstance(rf, RankedFile):
                lines.append(f"- `{rf.filepath}` (relevance score: {rf.score:.1f}; reason: {rf.reason})")
            elif isinstance(rf, str):
                lines.append(f"- `{rf}`")
            else:
                lines.append(f"- `{getattr(rf, 'filepath', str(rf))}`")
    else:
        lines.append("(No ranked files provided)")

    if previous_plan:
        lines.extend([
            "",
            "### Previous Plan (Reflect Iteration)",
            "A previous plan was attempted but tests/verification failed. "
            "Please review the previous plan below and UPDATE/REFINE it rather than starting from scratch:",
            f"- **Root Cause Hypothesis**: {previous_plan.get('root_cause_hypothesis', '')}",
            f"- **Files to Modify**: {previous_plan.get('files_to_modify', [])}",
            f"- **Approach**: {previous_plan.get('approach', '')}",
            f"- **Risks**: {previous_plan.get('risks', [])}",
            f"- **Test Strategy**: {previous_plan.get('test_strategy', '')}",
        ])

    lines.extend([
        "",
        "Please analyze the issue and submit your plan using the `submit_plan` tool.",
    ])

    return "\n".join(lines)


def _validate_files_exist(
    files: list[str],
    repo_root: Path,
) -> tuple[list[str], list[str]]:
    """Partition files into (valid_existing_files, missing_files)."""
    valid: list[str] = []
    missing: list[str] = []

    for f in files:
        clean_path = str(f).strip().lstrip("./")
        if not clean_path:
            continue
        full_path = (repo_root / clean_path).resolve()

        # Must exist and be within the repo root
        try:
            if full_path.is_file() and repo_root.resolve() in full_path.parents:
                valid.append(clean_path)
            elif (repo_root / f).is_file():
                valid.append(str(Path(f).as_posix()))
            else:
                missing.append(clean_path)
        except Exception:
            missing.append(clean_path)

    return valid, missing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plan(
    issue_text: str | Any,
    ranked_files: list[RankedFile] | list[str] | ExploreResult | Any,
    previous_plan: dict[str, Any] | Plan | None = None,
    *,
    repo_root: Path | str | None = None,
    model_fn: CallModelFn | None = None,
    config: ModelConfig | None = None,
    **kwargs: Any,
) -> Plan:
    """Generate a structured implementation plan using model tool-calling.

    Enforces schema:
    {
      "root_cause_hypothesis": str,
      "files_to_modify": list[str],
      "approach": str,
      "risks": list[str],
      "test_strategy": str
    }

    Validates that every file in ``files_to_modify`` exists in the repository.
    If non-existent files are specified, retries the model call once with
    corrective feedback.

    Args:
        issue_text:    Natural-language task or issue description.
        ranked_files:  Ranked files from Explore phase, or ExploreResult.
        previous_plan: Optional previous Plan (for retry/refinement after Reflect).
        repo_root:     Optional path to repository root (defaults to CWD or ExploreResult root).
        model_fn:      Model calling function (defaults to :func:`harness.model.call_model`).
        config:        Optional model configuration.

    Returns:
        A validated :class:`Plan` object (subclassing ``dict``).
    """
    # Handle polymorphic arguments for backwards compatibility
    # e.g., generate_plan(task, explore_result, call_model)
    if callable(previous_plan) and model_fn is None:
        model_fn = previous_plan
        previous_plan = None

    if isinstance(ranked_files, ExploreResult):
        if repo_root is None:
            repo_root = ranked_files.repo_map.root
        files_list = ranked_files.ranked_files
    else:
        files_list = ranked_files or []

    resolved_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    call_fn: CallModelFn = model_fn or call_model

    # Build prompt and messages
    user_prompt = _format_context(str(issue_text), files_list, previous_plan)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt}
    ]

    # Initial call to model with native tool-calling
    resp = call_fn(
        messages=messages,
        tools=[PLAN_SUBMIT_TOOL],
        system=SYSTEM_PROMPT,
        config=config,
        temperature=0.0,
        max_tokens=4096,
    )

    plan_dict = _extract_plan_from_response(resp) or {}

    raw_files = plan_dict.get("files_to_modify") or []
    if isinstance(raw_files, str):
        raw_files = [raw_files]

    # Validate that every file in files_to_modify actually exists in the repo
    valid_files, missing_files = _validate_files_exist(raw_files, resolved_root)

    # If non-existent files were returned, retry the model call ONCE with feedback
    if missing_files:
        # Determine available files from ranked_files or directory
        available_hints = []
        for rf in files_list[:10]:
            p = rf.filepath if isinstance(rf, RankedFile) else str(rf)
            available_hints.append(f"- `{p}`")

        feedback = (
            f"Validation Error: The following file(s) in 'files_to_modify' DO NOT exist in the repository:\n"
            + "\n".join(f"- `{mf}`" for mf in missing_files)
            + "\n\nEvery file in 'files_to_modify' must actually exist in the repository.\n"
            + "Available relevant files include:\n"
            + "\n".join(available_hints)
            + "\n\nPlease revise your plan and submit an updated plan via `submit_plan` with only existing repository files."
        )

        retry_messages = list(messages)
        # Append assistant turn
        if resp.tool_calls:
            tool_call = resp.tool_calls[0]
            retry_messages.append({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tool_call.id, "name": tool_call.name, "input": tool_call.input}
                ],
            })
            retry_messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_call.id, "content": feedback}
                ],
            })
        else:
            retry_messages.append({"role": "assistant", "content": resp.text or "Submitted plan."})
            retry_messages.append({"role": "user", "content": feedback})

        # Retry call
        retry_resp = call_fn(
            messages=retry_messages,
            tools=[PLAN_SUBMIT_TOOL],
            system=SYSTEM_PROMPT,
            config=config,
            temperature=0.0,
            max_tokens=4096,
        )

        retry_plan_dict = _extract_plan_from_response(retry_resp)
        if retry_plan_dict:
            plan_dict = retry_plan_dict
            retry_raw_files = plan_dict.get("files_to_modify") or []
            if isinstance(retry_raw_files, str):
                retry_raw_files = [retry_raw_files]
            valid_files, still_missing = _validate_files_exist(retry_raw_files, resolved_root)
            # Retain only valid existing files
            plan_dict["files_to_modify"] = valid_files

    # Final fallback normalization if model omitted required fields
    root_cause = str(plan_dict.get("root_cause_hypothesis") or f"Issue in {issue_text[:60]}")
    approach = str(plan_dict.get("approach") or "Implement targeted fix.")
    risks = plan_dict.get("risks") or []
    if isinstance(risks, str):
        risks = [risks]
    test_strategy = str(plan_dict.get("test_strategy") or "Run test suite to verify.")

    final_files = valid_files if valid_files else [str(f) for f in raw_files]

    return Plan(
        root_cause_hypothesis=root_cause,
        files_to_modify=final_files,
        approach=approach,
        risks=risks,
        test_strategy=test_strategy,
        task=str(issue_text),
        context=user_prompt,
    )


def validate_plan(plan: Plan, repo_root: Path | str | None = None) -> list[PlanIssue]:
    """Validate a plan against structural and semantic rules.

    Checks include:
      - root_cause_hypothesis and approach are non-empty.
      - files_to_modify is non-empty and paths exist in repo (if repo_root given).
      - File paths are relative and do not escape the repo root.

    Args:
        plan:      The plan to validate.
        repo_root: Optional repository root to verify file existence.

    Returns:
        A list of :class:`PlanIssue` objects.
    """
    issues: list[PlanIssue] = []

    if not plan.get("root_cause_hypothesis"):
        issues.append(
            PlanIssue(
                step_index=None,
                severity="error",
                message="Plan is missing 'root_cause_hypothesis'",
            )
        )

    if not plan.get("approach"):
        issues.append(
            PlanIssue(
                step_index=None,
                severity="error",
                message="Plan is missing 'approach'",
            )
        )

    files = plan.get("files_to_modify", [])
    if not files:
        issues.append(
            PlanIssue(
                step_index=None,
                severity="warning",
                message="Plan specifies no 'files_to_modify'",
            )
        )

    if repo_root:
        root = Path(repo_root).resolve()
        for idx, f in enumerate(files):
            p = root / f
            if not p.is_file():
                issues.append(
                    PlanIssue(
                        step_index=idx,
                        severity="error",
                        message=f"File '{f}' does not exist in repository",
                    )
                )

    return issues
