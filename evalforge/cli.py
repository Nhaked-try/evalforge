"""
EvalForge CLI — powered by Typer.
"""

from pathlib import Path
from typing import List, Optional
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evalforge import __version__
from evalforge.evaluator import Evaluator
from evalforge.benchmarks import load_benchmark
from evalforge.judges import create_judge
from evalforge.models import create_model
from evalforge.arena import Arena
from evalforge.reporter import Reporter

app = typer.Typer(
    name="evalforge",
    help="Evaluate and compare LLMs across custom benchmarks.",
    add_completion=False,
)
console = Console()

# ── Subcommands ──────────────────────────────────────────────

benchmark_app = typer.Typer(help="Manage benchmarks")
app.add_typer(benchmark_app, name="benchmark")

results_app = typer.Typer(help="View and export results")
app.add_typer(results_app, name="results")

config_app = typer.Typer(help="Manage EvalForge configuration")
app.add_typer(config_app, name="config")


# ── Main Commands ────────────────────────────────────────────

@app.command()
def run(
    benchmark: str = typer.Option(
        ..., "--benchmark", "-b", help="Benchmark name or path to YAML"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Model identifier (e.g., gpt-4o, claude-3-opus)"
    ),
    judge: str = typer.Option(
        "llm", "--judge", "-j", help="Judge type: llm, exact, embedding, reward"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Model to use as judge (defaults to same as --model)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of prompts to evaluate"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-prompt results"),
):
    """Evaluate a single model on a benchmark."""
    console.print(Panel.fit(
        f"[bold cyan]EvalForge v{__version__}[/bold cyan]\n"
        f"Benchmark: {benchmark} | Model: {model} | Judge: {judge}",
        title="Evaluation"
    ))

    # Load benchmark
    bench = load_benchmark(benchmark)
    if limit:
        bench.prompts = bench.prompts[:limit]
    console.print(f"Loaded {len(bench.prompts)} prompts from [bold]{bench.name}[/bold]")

    # Create model and judge
    eval_model = create_model(model)
    eval_judge = create_judge(judge, model=judge_model or model)

    # Run
    evaluator = Evaluator(benchmark=bench, models=[eval_model], judge=eval_judge)
    results = evaluator.run(verbose=verbose)

    # Output
    if output:
        results.to_json(output)
        console.print(f"\n[green]Results saved to {output}[/green]")
    else:
        results.print_summary()


@app.command()
def compare(
    benchmark: str = typer.Option(
        ..., "--benchmark", "-b", help="Benchmark name or path"
    ),
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated model identifiers"
    ),
    judge: str = typer.Option(
        "llm", "--judge", "-j", help="Judge type"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Model to use as judge"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit prompts"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Compare multiple models on the same benchmark."""
    model_list = [m.strip() for m in models.split(",")]

    console.print(Panel.fit(
        f"[bold cyan]EvalForge v{__version__}[/bold cyan]\n"
        f"Benchmark: {benchmark} | Models: {', '.join(model_list)} | Judge: {judge}",
        title="Model Comparison"
    ))

    # Load
    bench = load_benchmark(benchmark)
    if limit:
        bench.prompts = bench.prompts[:limit]
    console.print(f"Loaded {len(bench.prompts)} prompts")

    # Create models and judge
    eval_models = [create_model(m) for m in model_list]
    eval_judge = create_judge(judge, model=judge_model)

    # Run
    evaluator = Evaluator(benchmark=bench, models=eval_models, judge=eval_judge)
    results = evaluator.run(verbose=verbose)

    # Output
    results.print_summary()

    if output:
        results.to_json(output)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def arena(
    benchmark: str = typer.Option(
        ..., "--benchmark", "-b", help="Benchmark name or path"
    ),
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated model identifiers"
    ),
    judge: str = typer.Option(
        "embedding", "--judge", "-j", help="Judge type for pairwise comparison"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Judge model (if LLM judge)"
    ),
    rounds: int = typer.Option(
        100, "--rounds", "-r", help="Number of pairwise battles"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for Elo ratings"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run Elo tournament: pairwise battles between models."""
    model_list = [m.strip() for m in models.split(",")]

    if len(model_list) < 2:
        console.print("[red]Need at least 2 models for arena mode[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]EvalForge Arena[/bold cyan]\n"
        f"Models: {', '.join(model_list)} | Rounds: {rounds} | Judge: {judge}",
        title="Elo Tournament"
    ))

    bench = load_benchmark(benchmark)
    eval_models = [create_model(m) for m in model_list]
    eval_judge = create_judge(judge, model=judge_model)

    arena = Arena(benchmark=bench, models=eval_models, judge=eval_judge)
    ratings = arena.run_tournament(rounds=rounds, verbose=verbose)

    # Print leaderboard
    table = Table(title="Elo Ratings")
    table.add_column("Rank", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Elo", style="green")
    table.add_column("Wins", style="yellow")
    table.add_column("Losses", style="red")

    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1]["elo"], reverse=True)
    for rank, (model_name, stats) in enumerate(sorted_ratings, 1):
        table.add_row(
            str(rank),
            model_name,
            f"{stats['elo']:.0f}",
            str(stats["wins"]),
            str(stats["losses"]),
        )

    console.print(table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(ratings, f, indent=2, default=str)
        console.print(f"\n[green]Ratings saved to {output}[/green]")


# ── Benchmark Subcommands ────────────────────────────────────

@benchmark_app.command("list")
def benchmark_list():
    """List available benchmarks."""
    from evalforge.benchmarks import list_benchmarks

    benchmarks = list_benchmarks()
    if not benchmarks:
        console.print("[yellow]No benchmarks found in configs/benchmarks/[/yellow]")
        return

    table = Table(title="Available Benchmarks")
    table.add_column("Name")
    table.add_column("Prompts")
    table.add_column("Type")
    table.add_column("Path")

    for b in benchmarks:
        table.add_row(b["name"], str(b["num_prompts"]), b["type"], b["path"])

    console.print(table)


@benchmark_app.command("create")
def benchmark_create(
    name: str = typer.Option(..., "--name", help="Benchmark name"),
    template: str = typer.Option("custom", "--template", help="Template type"),
    output: Path = typer.Option(
        Path("configs/benchmarks"), "--output", "-o", help="Output directory"
    ),
):
    """Create a new benchmark from a template."""
    # TODO: Implement template-based benchmark creation wizard
    console.print(f"[yellow]⚠ benchmark create coming in v0.4.0[/yellow]")
    console.print(f"Would create: {output}/{name}.yaml from template: {template}")


@benchmark_app.command("validate")
def benchmark_validate(
    file: Path = typer.Option(
        ..., "--file", "-f", exists=True, help="Benchmark YAML to validate"
    ),
):
    """Validate a benchmark YAML file."""
    try:
        bench = load_benchmark(str(file))
        console.print(f"[green]✓ Valid benchmark: {bench.name}[/green]")
        console.print(f"  Prompts: {len(bench.prompts)}")
        console.print(f"  Type: {bench.type}")
    except Exception as e:
        console.print(f"[red]✗ Invalid benchmark: {e}[/red]")
        raise typer.Exit(1)


# ── Results Subcommands ──────────────────────────────────────

@results_app.command("list")
def results_list():
    """List past evaluation runs."""
    from evalforge.reporter import list_runs

    runs = list_runs()
    if not runs:
        console.print("[yellow]No results found in results/[/yellow]")
        return

    table = Table(title="Recent Evaluations")
    table.add_column("Run ID")
    table.add_column("Benchmark")
    table.add_column("Models")
    table.add_column("Date")

    for r in runs[-10:]:  # Last 10
        table.add_row(r["id"], r["benchmark"], r["models"], r["date"])

    console.print(table)


@results_app.command("show")
def results_show(
    run_id: str = typer.Argument(..., help="Run ID to display"),
):
    """Show detailed results for a run."""
    # TODO: Load from SQLite cache
    console.print(f"[yellow]⚠ results show coming in v0.4.0 (run_id: {run_id})[/yellow]")


@results_app.command("export")
def results_export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    format: str = typer.Option("html", "--format", "-f", help="Export format: html, json, csv"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
):
    """Export evaluation results."""
    console.print(f"[yellow]⚠ results export coming in v0.4.0 (run_id: {run_id}, format: {format})[/yellow]")


# ── Config Subcommands ───────────────────────────────────────

@config_app.command("show")
def config_show():
    """Show current configuration."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    table = Table(title="EvalForge Configuration")
    table.add_column("Key")
    table.add_column("Value")

    keys = [
        "EVALFORGE_RESULTS_DIR",
        "EVALFORGE_CACHE_DIR",
        "EVALFORGE_MAX_WORKERS",
        "EVALFORGE_LOG_LEVEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    for key in keys:
        val = os.getenv(key, "")
        if "KEY" in key and val:
            val = val[:8] + "..." if len(val) > 8 else val
        elif not val:
            val = "[not set]"
        table.add_row(key, val)

    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key"),
    value: str = typer.Argument(..., help="Config value"),
):
    """Set a configuration value."""
    # TODO: Write to .env or config file
    console.print(f"[yellow]⚠ config set coming in v0.4.0[/yellow]")


@app.command()
def version():
    """Show version."""
    console.print(f"EvalForge v{__version__}")


if __name__ == "__main__":
    app()
