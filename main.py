"""
Coding-Agent Harness — CLI Entrypoint
======================================

Usage::

    python main.py "Fix the login bug" --repo ./my-project --model gpt-4o
    python main.py "Add unit tests for utils.py" --max-retries 5

Uses `Typer <https://typer.tiangolo.com/>`_ for a rich CLI experience
with automatic ``--help`` generation and argument validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from harness.model import ModelConfig, Provider
from harness.orchestrator import OrchestratorConfig, run

app = typer.Typer(
    name="harness",
    help="A 6-phase autonomous coding-agent harness.",
    add_completion=False,
)


def _is_git_url(path_or_url: str) -> bool:
    """Check if the provided string is a remote git URL."""
    s = path_or_url.strip()
    return (
        s.startswith("http://")
        or s.startswith("https://")
        or s.startswith("git@")
        or s.startswith("ssh://")
        or (s.endswith(".git") and "/" in s)
    )


def _resolve_repo(repo_input: str, dest_dir: Path | None = None) -> Path:
    """Resolve repository input to a valid local Path.

    If repo_input is a remote git URL (e.g. https://github.com/org/repo.git),
    it clones the repository into dest_dir (or ./cloned_repos/<repo_name>).
    If repo_input is a local directory path, it validates and resolves it.
    """
    if _is_git_url(repo_input):
        import re
        import subprocess

        # Extract clean repository name from URL
        raw_name = repo_input.rstrip("/").split("/")[-1]
        clean_name = re.sub(r"\.git$", "", raw_name) or "cloned_repo"

        target_dir = dest_dir.resolve() if dest_dir else (Path.cwd() / "cloned_repos" / clean_name).resolve()

        if target_dir.exists() and (target_dir / ".git").exists():
            typer.echo(f"🔄 Using existing repository at: {target_dir}")
        else:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            typer.echo(f"📥 Cloning remote repository: {repo_input}")
            typer.echo(f"   Destination: {target_dir}...")
            res = subprocess.run(
                ["git", "clone", repo_input, str(target_dir)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                typer.echo(f"❌ Failed to clone repository:\n{res.stderr.strip()}", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"✅ Successfully cloned repository to: {target_dir}")

        return target_dir

    local_path = Path(repo_input).resolve()
    if not local_path.exists():
        typer.echo(f"❌ Error: Repository directory does not exist: {local_path}", err=True)
        raise typer.Exit(code=1)
    if not local_path.is_dir():
        typer.echo(f"❌ Error: Path is not a directory: {local_path}", err=True)
        raise typer.Exit(code=1)

    return local_path


def _push_changes(repo_dir: Path, branch: str | None = None, commit_msg: str | None = None) -> bool:
    """Commit applied changes and push to remote repository."""
    import subprocess
    msg = commit_msg or "fix: applied by autonomous coding-agent harness"
    try:
        subprocess.run(["git", "config", "user.name", "Agent Harness"], cwd=repo_dir, check=False)
        subprocess.run(["git", "config", "user.email", "agent@harness.local"], cwd=repo_dir, check=False)
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=False)

        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", msg], cwd=repo_dir, check=True, capture_output=True)

        if branch:
            subprocess.run(["git", "checkout", "-B", branch], cwd=repo_dir, check=False, capture_output=True)
            push_cmd = ["git", "push", "-u", "origin", branch]
        else:
            push_cmd = ["git", "push", "origin"]

        typer.echo(f"🚀 Pushing changes to remote repository...")
        res = subprocess.run(push_cmd, cwd=repo_dir, capture_output=True, text=True)
        if res.returncode == 0:
            typer.echo(f"✅ Successfully pushed fixes back to remote repository!")
            return True
        else:
            typer.echo(f"⚠️ Push failed (check repository credentials/permissions): {res.stderr.strip()}", err=True)
            return False
    except Exception as e:
        typer.echo(f"⚠️ Failed to push changes: {e}", err=True)
        return False


@app.command()
def main(
    task: str = typer.Argument(
        ...,
        help="Natural-language description of the coding task.",
    ),
    repo_arg: Optional[str] = typer.Argument(
        None,
        help="Optional path to local repository root OR remote git URL (e.g. https://github.com/user/repo.git).",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to local repository root OR remote git clone URL (e.g. https://github.com/user/repo.git).",
    ),
    dest: Optional[Path] = typer.Option(
        None,
        "--dest",
        "-d",
        help="Destination folder to clone into when repo is a git URL (defaults to ./cloned_repos/<repo_name>).",
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="Automatically commit and push verified fixes back to the remote git repository.",
    ),
    push_branch: Optional[str] = typer.Option(
        None,
        "--push-branch",
        help="Remote branch name to push to (e.g. fix/agent-patch). If omitted, pushes to current branch.",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Delete the local cloned repository folder after execution (ephemeral run, leaves no local footprint).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model identifier (defaults to config.yaml).",
    ),
    provider: Optional[Provider] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider backend (defaults to config.yaml).",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum Implement → Verify retry attempts.",
        min=0,
        max=10,
    ),
    test_cmd: Optional[str] = typer.Option(
        None,
        "--test-cmd",
        help="Explicit test command (auto-detected if omitted).",
    ),
    linter_cmd: Optional[str] = typer.Option(
        None,
        "--linter-cmd",
        help="Explicit linter command (skipped if omitted).",
    ),
    log_path: Path = typer.Option(
        Path("harness_run.jsonl"),
        "--log",
        help="Path to the JSONL telemetry log file.",
    ),
) -> None:
    """Run the coding-agent harness on a task.

    The harness will:

    1. **Explore** the repository structure and build a symbol index.
    2. **Plan** a set of changes as a structured JSON plan.
    3. **Implement** the plan by generating and applying diffs.
    4. **Verify** correctness by running the test suite.
    5. **Reflect** on failures and optionally retry.
    6. **Finalize** and report results.
    """
    from harness.model import load_config
    try:
        base_cfg = load_config()
    except Exception:
        base_cfg = ModelConfig()

    effective_provider = provider if provider is not None else base_cfg.provider
    effective_model = model if model is not None else base_cfg.default_model

    model_config = ModelConfig(
        provider=effective_provider,
        api_key=base_cfg.api_key,
        base_url=base_cfg.base_url,
        default_model=effective_model,
        temperature=base_cfg.temperature,
        max_tokens=base_cfg.max_tokens,
        extra=base_cfg.extra,
    )
    orchestrator_config = OrchestratorConfig(
        model_config=model_config,
        max_retries=max_retries,
        test_cmd=test_cmd,
        linter_cmd=linter_cmd,
        log_path=log_path,
    )

    target_input = repo or repo_arg or "."
    target_repo = _resolve_repo(target_input, dest)
    result = run(task, target_repo, orchestrator_config)

    # --- Print summary ---
    status = "✅ SUCCESS" if result.success else "❌ FAILED"
    typer.echo(f"\n{status}")
    typer.echo(f"  Phase reached : {result.phase_reached.value}")
    typer.echo(f"  Attempts      : {result.attempts}")
    if result.error:
        typer.echo(f"  Error         : {result.error}")
    if result.report_path:
        typer.echo(f"  Report        : {result.report_path}")
    if result.patch_path:
        typer.echo(f"  Patch         : {result.patch_path}")

    # --- Optional: Push back to remote repository ---
    if push:
        if result.success:
            _push_changes(target_repo, branch=push_branch)
        else:
            typer.echo("⚠️ Skipping push because verification did not succeed.")

    # --- Optional: Clean up local cloned repository (ephemeral mode) ---
    is_remote = _is_git_url(target_input)
    if cleanup and is_remote and target_repo.exists():
        import shutil
        if result.report_path and Path(result.report_path).exists():
            dest_report = Path.cwd() / "report.md"
            if Path(result.report_path).resolve() != dest_report.resolve():
                shutil.copy2(result.report_path, dest_report)
                typer.echo(f"📄 Saved report to: {dest_report}")
        if result.patch_path and Path(result.patch_path).exists():
            dest_patch = Path.cwd() / "patch.diff"
            if Path(result.patch_path).resolve() != dest_patch.resolve():
                shutil.copy2(result.patch_path, dest_patch)
                typer.echo(f"📄 Saved patch to: {dest_patch}")

        typer.echo(f"🧹 Ephemeral cleanup: removing local repository at {target_repo}")
        shutil.rmtree(target_repo, ignore_errors=True)

    raise typer.Exit(code=0 if result.success else 1)


if __name__ == "__main__":
    app()
