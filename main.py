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


@app.command()
def main(
    task: str = typer.Argument(
        ...,
        help="Natural-language description of the coding task.",
    ),
    repo: Path = typer.Option(
        Path("."),
        "--repo",
        "-r",
        help="Path to the target repository root.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="Model identifier (e.g. gpt-4o, claude-sonnet-4-20250514, gemini-2.5-pro).",
    ),
    provider: Provider = typer.Option(
        Provider.OPENAI,
        "--provider",
        "-p",
        help="LLM provider backend.",
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
    model_config = ModelConfig(
        provider=provider,
        default_model=model,
    )
    orchestrator_config = OrchestratorConfig(
        model_config=model_config,
        max_retries=max_retries,
        test_cmd=test_cmd,
        linter_cmd=linter_cmd,
        log_path=log_path,
    )

    result = run(task, repo, orchestrator_config)

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

    raise typer.Exit(code=0 if result.success else 1)


if __name__ == "__main__":
    app()
