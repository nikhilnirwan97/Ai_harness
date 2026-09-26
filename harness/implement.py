"""
harness.implement
=================

Phase 3 — Implement: Diff generation and safe application in isolated git worktrees.

Key Capabilities:
1. **generate_diff(plan, ranked_files)** — calls LLM to produce a surgical unified diff
   implementing the plan (not full-file rewrites).
2. **apply_diff_safely(diff_text, repo_path)** — runs ``git apply --check`` first; only applies
   if it passes cleanly.
3. **Isolated git worktree execution** — all changes are tested inside an isolated git branch/worktree,
   never polluting the main working copy.
4. **Post-apply syntax validation** — automatically runs language-appropriate syntax checks
   (``py_compile`` for Python, ``tsc --noEmit`` for TypeScript, ``node --check`` for JavaScript,
   ``go vet / go build`` for Go, tree-sitter AST validation fallback) to catch garbage output
   immediately before reaching the test suite.
5. **Git checkpoint & rollback helpers** — ``git_checkpoint(label)`` and ``git_rollback(label)``
   for atomic rollback and state management.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Generator, Literal
import uuid

from harness.explore import ExploreResult, RankedFile
from harness.model import CallModelFn, ModelConfig, ModelResponse, call_model
from harness.plan import Plan, PlanStep


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiffHunk:
    """A single hunk inside a unified diff.

    Attributes:
        old_start: Starting line in the original file.
        old_count: Number of lines removed / context in the original.
        new_start: Starting line in the modified file.
        new_count: Number of lines added / context in the modified.
        content:   The raw hunk text (including ``+``/``-``/`` `` prefixes).
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    content: str


@dataclass
class Diff:
    """A complete unified diff for a file or set of files.

    Attributes:
        file_path:     Relative path of the primary target file.
        hunks:         Parsed hunks.
        raw_diff_text: The full unified diff as a string.
    """

    file_path: str
    hunks: list[DiffHunk] = field(default_factory=list)
    raw_diff_text: str = ""

    @property
    def diff_text(self) -> str:
        return self.raw_diff_text


@dataclass
class ApplyResult:
    """Outcome of applying a diff to the working tree.

    Attributes:
        success:     Whether the diff applied cleanly and passed syntax checks.
        file_path:   The target file(s) that were modified.
        backup_path: Path to the pre-modification backup (optional).
        error:       Human-readable error message on failure.
    """

    success: bool
    file_path: str
    backup_path: Path | None = None
    error: str | None = None


@dataclass
class ImplementResult:
    """Aggregated output of the Implement phase.

    Attributes:
        plan:          The plan that was executed.
        diffs:         Generated diffs.
        apply_results: Corresponding apply outcomes.
        all_succeeded: Convenience flag — ``True`` iff every apply succeeded.
    """

    plan: Plan
    diffs: list[Diff] = field(default_factory=list)
    apply_results: list[ApplyResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """Return ``True`` if every diff applied successfully."""
        return all(r.success for r in self.apply_results) if self.apply_results else False


# ---------------------------------------------------------------------------
# Git repository & checkpoint helpers
# ---------------------------------------------------------------------------

def _ensure_git_repo(repo_path: Path) -> None:
    """Ensure that repo_path is an initialized git repository with at least one commit."""
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init", "-b", "main"], cwd=repo_path, check=False, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo_path, check=False, capture_output=True)
        subprocess.run(["git", "config", "user.email", "agent@harness.local"], cwd=repo_path, check=False, capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=repo_path, check=False, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial repository state", "--allow-empty"], cwd=repo_path, check=False, capture_output=True)
    else:
        # Check if HEAD exists; if not (e.g. empty repo), create an initial commit
        res = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo_path, capture_output=True)
        if res.returncode != 0:
            subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo_path, check=False, capture_output=True)
            subprocess.run(["git", "config", "user.email", "agent@harness.local"], cwd=repo_path, check=False, capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=repo_path, check=False, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial repository state", "--allow-empty"], cwd=repo_path, check=False, capture_output=True)


def git_checkpoint(label: str, repo_path: Path | str | None = None) -> str:
    """Create an atomic git checkpoint with the given label.

    Tags the current working tree state as ``checkpoint_{sanitized_label}``.

    Args:
        label:     A unique label for the checkpoint (e.g. 'pre_implement').
        repo_path: Path to the git repository (defaults to CWD).

    Returns:
        The commit hash of the checkpoint.
    """
    root = Path(repo_path).resolve() if repo_path else Path.cwd().resolve()
    _ensure_git_repo(root)

    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", label)
    tag_name = f"checkpoint_{sanitized}"

    # Stage all changes and commit or update tag (excluding telemetry JSONL files)
    subprocess.run(["git", "add", "-A", "--", ":!*.jsonl"], cwd=root, capture_output=True, check=False)
    subprocess.run(
        ["git", "commit", "-m", f"checkpoint: {label}", "--allow-empty"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    subprocess.run(["git", "tag", "-f", tag_name], cwd=root, capture_output=True, check=False)

    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return res.stdout.strip()


def git_rollback(label: str, repo_path: Path | str | None = None) -> bool:
    """Roll back the repository working copy to the checkpoint associated with label.

    Discards any uncommitted modifications and resets the working tree.

    Args:
        label:     The checkpoint label previously passed to ``git_checkpoint``.
        repo_path: Path to the git repository (defaults to CWD).

    Returns:
        True if rollback succeeded, False otherwise.
    """
    root = Path(repo_path).resolve() if repo_path else Path.cwd().resolve()
    _ensure_git_repo(root)

    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", label)
    tag_name = f"checkpoint_{sanitized}"

    # Check if tag exists
    tag_check = subprocess.run(["git", "rev-parse", "--verify", tag_name], cwd=root, capture_output=True)
    target = tag_name if tag_check.returncode == 0 else "HEAD"

    res_reset = subprocess.run(["git", "reset", "--hard", target], cwd=root, capture_output=True)
    res_clean = subprocess.run(["git", "clean", "-fd", "-e", "*.jsonl"], cwd=root, capture_output=True)

    return res_reset.returncode == 0 and res_clean.returncode == 0


# ---------------------------------------------------------------------------
# Isolated Git Worktree context manager
# ---------------------------------------------------------------------------

@contextmanager
def isolated_worktree(
    repo_path: Path | str,
    branch_name: str | None = None,
) -> Generator[Path, None, None]:
    """Context manager executing code inside an isolated git worktree.

    Never touches or modifies the main working copy. Automatically cleans
    up the worktree upon exiting the context.

    Args:
        repo_path:   Root of the original git repository.
        branch_name: Name of the ephemeral branch to create.

    Yields:
        Path to the isolated worktree directory.
    """
    root = Path(repo_path).resolve()
    _ensure_git_repo(root)

    if branch_name is None:
        branch_name = f"agent-worktree-{uuid.uuid4().hex[:8]}"

    # Create temporary directory for worktree outside the repo
    temp_dir = Path(tempfile.mkdtemp(prefix="harness_wt_"))

    try:
        # Add worktree on a new branch
        add_cmd = ["git", "worktree", "add", str(temp_dir), "-b", branch_name]
        res = subprocess.run(add_cmd, cwd=root, capture_output=True, text=True)
        if res.returncode != 0:
            # If branch already exists, try without -b
            add_cmd_fallback = ["git", "worktree", "add", str(temp_dir), branch_name]
            res = subprocess.run(add_cmd_fallback, cwd=root, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Failed to create git worktree: {res.stderr}")

        yield temp_dir
    finally:
        # Clean up worktree and delete ephemeral branch
        subprocess.run(["git", "worktree", "remove", "--force", str(temp_dir)], cwd=root, capture_output=True, check=False)
        subprocess.run(["git", "branch", "-D", branch_name], cwd=root, capture_output=True, check=False)
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Post-apply Syntax Checking
# ---------------------------------------------------------------------------

def run_syntax_check(file_path: Path) -> tuple[bool, str | None]:
    """Run an appropriate syntax check based on file language.

    Checks:
      - Python (.py): ``python3 -m py_compile``
      - JavaScript (.js, .mjs, .cjs): ``node --check``
      - TypeScript (.ts, .tsx): ``tsc --noEmit --skipLibCheck``
      - Go (.go): ``go vet / go build`` (or tree-sitter Go AST error check)
      - Polyglot fallback: Tree-sitter AST validation (``tree.root_node.has_error``)

    Args:
        file_path: Absolute path to the file to check.

    Returns:
        (is_valid: bool, error_message: str | None)
    """
    if not file_path.is_file():
        return True, None

    ext = file_path.suffix.lower()

    # 1. Python (.py)
    if ext in (".py", ".pyi"):
        res = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return False, res.stderr or res.stdout

    # 2. JavaScript (.js, .mjs, .cjs)
    elif ext in (".js", ".mjs", ".cjs"):
        node_bin = shutil.which("node") or "/opt/homebrew/bin/node"
        if os.path.isfile(node_bin) and os.access(node_bin, os.X_OK):
            res = subprocess.run(
                [node_bin, "--check", str(file_path)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                return False, res.stderr or res.stdout
        else:
            return _treesitter_syntax_check(file_path, "javascript")

    # 3. TypeScript (.ts, .tsx)
    elif ext in (".ts", ".tsx"):
        tsc_bin = shutil.which("tsc")
        if tsc_bin:
            res = subprocess.run(
                [tsc_bin, "--noEmit", "--skipLibCheck", str(file_path)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                return False, res.stdout or res.stderr
        else:
            return _treesitter_syntax_check(file_path, "typescript" if ext == ".ts" else "tsx")

    # 4. Go (.go)
    elif ext == ".go":
        go_bin = shutil.which("go")
        if go_bin:
            res = subprocess.run(
                [go_bin, "vet", str(file_path)],
                cwd=file_path.parent,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                return False, res.stderr or res.stdout
        else:
            return _treesitter_syntax_check(file_path, "go")

    # 5. Rust (.rs)
    elif ext == ".rs":
        rustc_bin = shutil.which("rustc")
        if rustc_bin:
            res = subprocess.run(
                [rustc_bin, "--emit=metadata", "-o", "/dev/null", str(file_path)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                return False, res.stderr or res.stdout
        else:
            return _treesitter_syntax_check(file_path, "rust")

    return True, None


def _treesitter_syntax_check(file_path: Path, language: str) -> tuple[bool, str | None]:
    """Validate syntax using tree-sitter AST error detection."""
    try:
        from harness.explore import TreeSitterHelper
        parser = TreeSitterHelper.get_parser(language)
        if not parser:
            return True, None

        source = file_path.read_bytes()
        tree = parser.parse(source)
        if tree.root_node.has_error:
            return False, f"Syntax error detected in {file_path.name} via Tree-Sitter AST parser"
        return True, None
    except Exception:
        return True, None


# ---------------------------------------------------------------------------
# Diff Parsing Helper
# ---------------------------------------------------------------------------

def _strip_diff_prefix(path: str) -> str:
    """Remove ``a/`` or ``b/`` prefix from a diff file path without clobbering filenames."""
    for prefix in ("b/", "a/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def parse_unified_diff(diff_text: str) -> list[Diff]:
    """Parse a unified diff string into a list of Diff objects with DiffHunks.

    Uses a line-by-line state machine so that ``---``/``+++`` headers and
    ``@@`` hunks are always grouped together correctly, even when there is
    no ``diff --git`` header.
    """
    diffs: list[Diff] = []
    hunk_header_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    current_file: str | None = None
    current_hunks: list[DiffHunk] = []
    current_raw_lines: list[str] = []

    def _flush() -> None:
        """Emit the accumulated diff (if any) into *diffs*."""
        nonlocal current_file, current_hunks, current_raw_lines
        if current_file is not None:
            raw = "\n".join(current_raw_lines)
            diffs.append(Diff(file_path=current_file, hunks=current_hunks, raw_diff_text=raw))
        current_file = None
        current_hunks = []
        current_raw_lines = []

    for line in diff_text.splitlines():
        # --- New file header: "diff --git a/X b/X" or a bare "--- a/X" line.
        if line.startswith("diff --git "):
            _flush()
            # Extract target from "diff --git a/X b/X"
            parts = line.split()
            current_file = _strip_diff_prefix(parts[-1]) if len(parts) >= 4 else "unknown"
            current_raw_lines.append(line)
            continue

        if line.startswith("--- "):
            # If we haven't opened a file yet (no "diff --git" header), start one.
            if current_file is None:
                candidate = line[4:].strip()
                if candidate != "/dev/null":
                    current_file = _strip_diff_prefix(candidate)
                else:
                    current_file = "unknown"  # will be overwritten by +++
            current_raw_lines.append(line)
            continue

        if line.startswith("+++ "):
            candidate = line[4:].strip()
            if candidate != "/dev/null":
                current_file = _strip_diff_prefix(candidate)
            current_raw_lines.append(line)
            continue

        # Hunk header
        m = hunk_header_re.match(line)
        if m:
            current_raw_lines.append(line)
            current_hunks.append(
                DiffHunk(
                    old_start=int(m.group(1)),
                    old_count=int(m.group(2) or 1),
                    new_start=int(m.group(3)),
                    new_count=int(m.group(4) or 1),
                    content=line,
                )
            )
            continue

        # Ordinary diff body line (+, -, or context)
        current_raw_lines.append(line)

    _flush()
    return diffs


def _extract_target_files_from_diff(diff_text: str) -> list[str]:
    """Extract list of modified file paths from unified diff headers."""
    files: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            files.append(line[4:].strip())
    # Deduplicate while preserving order
    return list(dict.fromkeys(files))


# ---------------------------------------------------------------------------
# Public API: 2. apply_diff_safely
# ---------------------------------------------------------------------------

def apply_diff_safely(
    diff_text: str,
    repo_path: Path | str | None = None,
    *,
    skip_syntax_check: bool = False,
) -> ApplyResult:
    """Safely apply a unified diff to the repository.

    1. Runs ``git apply --check`` first.
    2. Only applies if the check passes.
    3. Runs language-appropriate syntax checks on all modified files.
    4. Automatically rolls back if syntax check fails.

    Args:
        diff_text:         Unified diff content.
        repo_path:         Target repository root directory.
        skip_syntax_check: If True, skips post-apply syntax check.

    Returns:
        An :class:`ApplyResult` indicating success or failure with error details.
    """
    root = Path(repo_path).resolve() if repo_path else Path.cwd().resolve()
    _ensure_git_repo(root)

    clean_diff = diff_text.strip() + "\n"  # git apply requires trailing newline
    if not clean_diff.strip():
        return ApplyResult(success=False, file_path="", error="Empty diff provided.")

    target_files = _extract_target_files_from_diff(clean_diff)
    primary_file = target_files[0] if target_files else ""

    # Create safety checkpoint before applying
    checkpoint_id = f"pre_apply_{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}"
    git_checkpoint(checkpoint_id, root)

    # 1. Run git apply --check first
    check_cmd = ["git", "apply", "--check", "--whitespace=nowarn", "-"]
    res_check = subprocess.run(
        check_cmd,
        input=clean_diff,
        text=True,
        cwd=root,
        capture_output=True,
    )

    if res_check.returncode != 0:
        # Check failed
        err_msg = res_check.stderr or res_check.stdout or "git apply --check failed"
        return ApplyResult(
            success=False,
            file_path=primary_file,
            error=f"git apply --check failed:\n{err_msg}",
        )

    # 2. Apply the diff cleanly
    apply_cmd = ["git", "apply", "--whitespace=nowarn", "-"]
    res_apply = subprocess.run(
        apply_cmd,
        input=clean_diff,
        text=True,
        cwd=root,
        capture_output=True,
    )

    if res_apply.returncode != 0:
        git_rollback(checkpoint_id, root)
        err_msg = res_apply.stderr or res_apply.stdout or "git apply failed"
        return ApplyResult(
            success=False,
            file_path=primary_file,
            error=f"git apply failed:\n{err_msg}",
        )

    # 3. Post-apply syntax check on every modified file
    if not skip_syntax_check:
        for f in target_files:
            target_path = root / f
            is_valid, syntax_err = run_syntax_check(target_path)
            if not is_valid:
                # Syntax error detected! Roll back immediately.
                git_rollback(checkpoint_id, root)
                return ApplyResult(
                    success=False,
                    file_path=f,
                    error=(
                        f"Post-apply syntax check failed for '{f}'.\n"
                        f"Rolled back changes automatically.\n"
                        f"Syntax error: {syntax_err}"
                    ),
                )

    return ApplyResult(success=True, file_path=primary_file, error=None)


# ---------------------------------------------------------------------------
# Public API: 1. generate_diff
# ---------------------------------------------------------------------------

DIFF_PROMPT = """You are an expert software engineer generating a minimal, surgical unified diff.
Task: Implement the provided plan precisely.

Rules:
1. Output MUST be formatted as a valid UNIFIED DIFF (standard `git diff` format):
   --- a/<filepath>
   +++ b/<filepath>
   @@ -old_start,old_count +new_start,new_count @@
2. DO NOT return full-file rewrites! Only return the specific lines being changed with surrounding context.
3. Every diff hunk MUST have accurate line counts and exact context lines.
4. Wrap your diff output in a ```diff ... ``` code block.
"""


def generate_diff(
    plan: Plan | dict[str, Any] | PlanStep | str,
    ranked_files: list[RankedFile] | list[str] | ExploreResult | str | None = None,
    model_fn: CallModelFn | None = None,
    *,
    repo_root: Path | str | None = None,
    config: ModelConfig | None = None,
) -> Diff:
    """Generate a minimal, surgical unified diff implementing the plan.

    Calls ``call_model()`` to produce standard unified diff format without
    full-file rewrites.

    Args:
        plan:         A :class:`~harness.plan.Plan` or plan dict, or PlanStep.
        ranked_files: Ranked files from Explore, or file content string.
        model_fn:     Model calling function (defaults to call_model).
        repo_root:    Path to the repository root.
        config:       Optional model configuration.

    Returns:
        A :class:`Diff` containing the parsed hunks and raw diff text.
    """
    root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    call_fn = model_fn or call_model

    # Support backwards-compatible signature: generate_diff(step, file_content, model_fn)
    if isinstance(plan, PlanStep) and isinstance(ranked_files, str):
        target_file = plan.file
        file_content = ranked_files
        approach = plan.description
        root_cause = ""
    else:
        # Extract files and approach from Plan
        if isinstance(plan, dict):
            files_to_modify = plan.get("files_to_modify", [])
            approach = plan.get("approach", "")
            root_cause = plan.get("root_cause_hypothesis", "")
        else:
            files_to_modify = getattr(plan, "files_to_modify", [])
            approach = getattr(plan, "approach", str(plan))
            root_cause = getattr(plan, "root_cause_hypothesis", "")

        target_file = files_to_modify[0] if files_to_modify else "code"
        # Read current content of target files
        contents_block: list[str] = []
        for f in files_to_modify:
            p = root / f
            if p.is_file():
                try:
                    c = p.read_text(encoding="utf-8", errors="replace")
                    # Line number annotations for high-accuracy diff generation
                    annotated = "\n".join(f"{i+1}: {line}" for i, line in enumerate(c.splitlines()))
                    contents_block.append(f"### File: {f}\n```\n{annotated}\n```")
                except Exception:
                    pass

        file_content = "\n\n".join(contents_block)

    user_message = (
        f"### Implementation Plan\n"
        f"- Root Cause Hypothesis: {root_cause}\n"
        f"- Approach: {approach}\n\n"
        f"### Existing Code\n"
        f"{file_content}\n\n"
        f"Please output the surgical unified diff implementing this plan in ```diff ... ``` format."
    )

    resp = call_fn(
        messages=[{"role": "user", "content": user_message}],
        system=DIFF_PROMPT,
        config=config,
        temperature=0.0,
        max_tokens=4096,
    )

    # Extract diff text from response
    diff_text = resp.text.strip() if resp.text else ""
    diff_match = re.search(r"```(?:diff|patch)?\s*\n(.*?)\n```", diff_text, re.DOTALL)
    if diff_match:
        diff_text = diff_match.group(1).strip()
    else:
        # Check if text itself looks like a diff
        header_match = re.search(r"(?:diff --git |--- (?:a/|[^\n]+)).*", diff_text, re.DOTALL)
        if header_match:
            diff_text = header_match.group(0).strip()

    parsed_diffs = parse_unified_diff(diff_text)
    if parsed_diffs:
        return parsed_diffs[0]

    return Diff(file_path=target_file, hunks=[], raw_diff_text=diff_text)


# ---------------------------------------------------------------------------
# Public API: 5. implement
# ---------------------------------------------------------------------------

def implement(
    plan: Plan,
    repo_root: Path | str,
    model_fn: CallModelFn | None = None,
    *,
    use_isolated_worktree: bool = True,
    config: ModelConfig | None = None,
) -> ImplementResult:
    """Run the Implement phase end-to-end.

    Executes all modifications inside an isolated git worktree without touching
    the main working copy, generating surgical unified diffs and validating
    syntax before declaring success.

    Args:
        plan:                  The validated execution plan.
        repo_root:             Absolute or relative path to the repository root.
        model_fn:              The model-calling function.
        use_isolated_worktree: If True, runs within an isolated git worktree.
        config:                Optional model configuration.

    Returns:
        An :class:`ImplementResult` with all generated diffs and apply outcomes.
    """
    root = Path(repo_root).resolve()
    _ensure_git_repo(root)

    diffs: list[Diff] = []
    apply_results: list[ApplyResult] = []

    if use_isolated_worktree:
        with isolated_worktree(root) as work_dir:
            diff = generate_diff(plan, repo_root=work_dir, model_fn=model_fn, config=config)
            diffs.append(diff)

            result = apply_diff_safely(diff.raw_diff_text, repo_path=work_dir)
            apply_results.append(result)

            # If clean, we commit to the worktree branch
            if result.success:
                subprocess.run(["git", "add", "-A"], cwd=work_dir, check=False)
                subprocess.run(["git", "commit", "-m", "Agent implementation"], cwd=work_dir, check=False)
    else:
        diff = generate_diff(plan, repo_root=root, model_fn=model_fn, config=config)
        diffs.append(diff)
        result = apply_diff_safely(diff.raw_diff_text, repo_path=root)
        apply_results.append(result)

    return ImplementResult(
        plan=plan,
        diffs=diffs,
        apply_results=apply_results,
    )
