#!/usr/bin/env python3
"""
Unit tests for harness.implement
=================================

Tests:
  - git_checkpoint / git_rollback
  - apply_diff_safely (check-then-apply, syntax validation, rollback on bad syntax)
  - isolated_worktree context manager
  - generate_diff (mocked model)
  - run_syntax_check for Python, JS, TS, and tree-sitter fallback
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.implement import (
    Diff,
    DiffHunk,
    ApplyResult,
    ImplementResult,
    apply_diff_safely,
    generate_diff,
    git_checkpoint,
    git_rollback,
    isolated_worktree,
    run_syntax_check,
    parse_unified_diff,
)
from harness.model import ModelResponse, TokenUsage
from harness.plan import Plan, PlanStep


def _init_git_repo(root: Path) -> None:
    """Initialise a minimal git repo with one committed file."""
    subprocess.run(["git", "init", "-b", "main"], cwd=root, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=root, capture_output=True, check=True)


class TestGitCheckpointRollback(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        _init_git_repo(self.root)
        (self.root / "main.py").write_text("original_content = True\n")
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.root, capture_output=True, check=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_creates_tag(self):
        """git_checkpoint creates a checkpoint tag and returns a commit hash."""
        commit_hash = git_checkpoint("step1", self.root)
        self.assertTrue(len(commit_hash) > 6)
        res = subprocess.run(
            ["git", "tag", "-l", "checkpoint_step1"], cwd=self.root, capture_output=True, text=True
        )
        self.assertIn("checkpoint_step1", res.stdout)

    def test_rollback_restores_state(self):
        """git_rollback reverts files to the checkpoint state."""
        git_checkpoint("before_change", self.root)

        # Modify and add new file
        (self.root / "main.py").write_text("modified = True\n")
        (self.root / "new_file.py").write_text("garbage\n")

        self.assertEqual((self.root / "main.py").read_text(), "modified = True\n")
        self.assertTrue((self.root / "new_file.py").exists())

        # Rollback
        success = git_rollback("before_change", self.root)
        self.assertTrue(success)
        self.assertEqual((self.root / "main.py").read_text(), "original_content = True\n")
        self.assertFalse((self.root / "new_file.py").exists())


class TestApplyDiffSafely(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        _init_git_repo(self.root)
        (self.root / "hello.py").write_text('def greet():\n    print("Hello world")\n')
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.root, capture_output=True, check=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_diff_applies_cleanly(self):
        """A well-formed diff passes git apply --check and applies successfully."""
        diff_text = (
            '--- a/hello.py\n'
            '+++ b/hello.py\n'
            '@@ -1,2 +1,2 @@\n'
            ' def greet():\n'
            '-    print("Hello world")\n'
            '+    print("Hello universe")\n'
        )
        result = apply_diff_safely(diff_text, self.root)
        self.assertTrue(result.success, f"Expected success but got error: {result.error}")
        self.assertIn("Hello universe", (self.root / "hello.py").read_text())

    def test_malformed_diff_rejected(self):
        """A diff with wrong context lines fails git apply --check."""
        bad_diff = (
            '--- a/hello.py\n'
            '+++ b/hello.py\n'
            '@@ -1,2 +1,2 @@\n'
            ' def wrong_context():\n'
            '-    print("wrong")\n'
            '+    print("also wrong")\n'
        )
        result = apply_diff_safely(bad_diff, self.root)
        self.assertFalse(result.success)
        self.assertIn("git apply --check failed", result.error)
        # File should be unchanged
        self.assertIn("Hello world", (self.root / "hello.py").read_text())

    def test_syntax_error_triggers_rollback(self):
        """If applied diff introduces a Python syntax error, it is rolled back."""
        # Diff that replaces valid Python with a syntax error
        diff_text = (
            '--- a/hello.py\n'
            '+++ b/hello.py\n'
            '@@ -1,2 +1,2 @@\n'
            '-def greet():\n'
            '-    print("Hello world")\n'
            '+def greet(\n'
            '+    print("Hello world")\n'
        )
        result = apply_diff_safely(diff_text, self.root)
        self.assertFalse(result.success)
        self.assertIn("syntax", result.error.lower())
        # File should be restored to original
        self.assertIn("def greet():", (self.root / "hello.py").read_text())

    def test_empty_diff_rejected(self):
        """Empty diff string is rejected."""
        result = apply_diff_safely("", self.root)
        self.assertFalse(result.success)
        self.assertIn("Empty diff", result.error)


class TestIsolatedWorktree(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        _init_git_repo(self.root)
        (self.root / "app.py").write_text("original = True\n")
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.root, capture_output=True, check=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_worktree_isolation(self):
        """Changes in worktree do not affect main working copy."""
        with isolated_worktree(self.root) as wt:
            self.assertTrue((wt / "app.py").exists())

            # Modify file in worktree
            (wt / "app.py").write_text("modified_in_worktree = True\n")
            self.assertIn("modified_in_worktree", (wt / "app.py").read_text())

        # Main repo should be untouched
        self.assertIn("original", (self.root / "app.py").read_text())

    def test_worktree_cleanup(self):
        """Worktree directory is cleaned up after exiting context."""
        wt_path = None
        with isolated_worktree(self.root) as wt:
            wt_path = wt
            self.assertTrue(wt_path.exists())
        self.assertFalse(wt_path.exists())


class TestSyntaxCheck(unittest.TestCase):

    def test_valid_python_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def hello():\n    return 42\n")
            f.flush()
            is_valid, err = run_syntax_check(Path(f.name))
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_invalid_python_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def broken(:\n    pass\n")
            f.flush()
            is_valid, err = run_syntax_check(Path(f.name))
        self.assertFalse(is_valid)
        self.assertIsNotNone(err)

    def test_valid_js_passes(self):
        import shutil
        if not shutil.which("node"):
            self.skipTest("node not installed")
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("function hello() { return 1; }\n")
            f.flush()
            is_valid, err = run_syntax_check(Path(f.name))
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_invalid_js_fails(self):
        import shutil
        if not shutil.which("node"):
            self.skipTest("node not installed")
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("function broken( { return 1;\n")
            f.flush()
            is_valid, err = run_syntax_check(Path(f.name))
        self.assertFalse(is_valid)
        self.assertIsNotNone(err)


class TestGenerateDiff(unittest.TestCase):

    def test_generate_diff_from_plan_mock(self):
        """generate_diff extracts a diff from model response wrapped in ```diff blocks."""
        plan = Plan(
            root_cause_hypothesis="Missing null check",
            files_to_modify=["app.py"],
            approach="Add null check in main handler",
            risks=[],
            test_strategy="Run tests",
        )

        diff_text = (
            '--- a/app.py\n'
            '+++ b/app.py\n'
            '@@ -1,2 +1,3 @@\n'
            ' def handler(req):\n'
            '+    if req is None: return None\n'
            '     return process(req)\n'
        )

        mock_resp = ModelResponse(
            text=f"Here is the diff:\n```diff\n{diff_text}\n```",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gemini-3.8-flash",
        )
        mock_fn = MagicMock(return_value=mock_resp)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app.py").write_text("def handler(req):\n    return process(req)\n")

            result = generate_diff(plan, model_fn=mock_fn, repo_root=root)

        mock_fn.assert_called_once()
        self.assertIn("app.py", result.file_path)
        self.assertIn("none", result.raw_diff_text.lower())


class TestParseDiff(unittest.TestCase):

    def test_parse_standard_diff(self):
        diff_text = (
            '--- a/hello.py\n'
            '+++ b/hello.py\n'
            '@@ -1,2 +1,2 @@\n'
            ' def greet():\n'
            '-    print("world")\n'
            '+    print("universe")\n'
        )
        diffs = parse_unified_diff(diff_text)
        self.assertEqual(len(diffs), 1)
        self.assertIn("hello.py", diffs[0].file_path)
        self.assertEqual(len(diffs[0].hunks), 1)
        self.assertEqual(diffs[0].hunks[0].old_start, 1)
        self.assertEqual(diffs[0].hunks[0].new_start, 1)


if __name__ == "__main__":
    unittest.main()
