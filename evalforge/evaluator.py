"""
Core evaluator — orchestrates benchmark execution across models and judges.

TODO:
- [ ] Async concurrent model calls (currently sequential)
- [ ] Streaming results (yield per-prompt for long evals)
- [ ] SQLite caching for repeat evaluations
- [ ] Cost tracking per evaluation run
- [ ] Timeout handling for hanging model responses
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from evalforge.benchmarks.base import Benchmark, Prompt
from evalforge.models.base import BaseModel
from evalforge.judges.base import BaseJudge, Judgment

console = Console()


@dataclass
class EvalResult:
    """Result for a single prompt across all models."""

    prompt: Prompt
    model_outputs: Dict[str, str] = field(default_factory=dict)  # model_name -> output
    judgments: Dict[str, Judgment] = field(default_factory=dict)  # model_name -> judgment
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRun:
    """Complete evaluation run with multiple prompts and models."""

    benchmark_name: str
    model_names: List[str]
    judge_type: str
    results: List[EvalResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    @property
    def num_prompts(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "models": self.model_names,
            "judge": self.judge_type,
            "num_prompts": self.num_prompts,
            "elapsed_seconds": self.elapsed,
            "results": [
                {
                    "prompt_id": r.prompt.id,
                    "prompt_text": r.prompt.text,
                    "expected": r.prompt.expected,
                    "outputs": r.model_outputs,
                    "judgments": {
                        name: j.to_dict() for name, j in r.judgments.items()
                    },
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

    def to_json(self, path: str | Path):
        """Export results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print a formatted summary table."""
        # Aggregate scores per model
        model_scores: Dict[str, List[float]] = {name: [] for name in self.model_names}
        model_wins: Dict[str, int] = {name: 0 for name in self.model_names}
        total_prompts = len(self.results)

        for result in self.results:
            # Find highest-scoring model for this prompt
            best_model = None
            best_score = -float("inf")

            for model_name in self.model_names:
                if model_name in result.judgments:
                    score = result.judgments[model_name].score
                    model_scores[model_name].append(score)
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            if best_model:
                model_wins[best_model] += 1

        # Build table
        table = Table(title="Evaluation Results")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Avg Score", style="green", justify="right")
        table.add_column("Win Rate", style="yellow", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for model_name in self.model_names:
            scores = model_scores[model_name]
            if scores:
                avg = sum(scores) / len(scores)
                win_rate = (model_wins[model_name] / total_prompts * 100) if total_prompts > 0 else 0
                table.add_row(
                    model_name,
                    f"{avg:.2f}",
                    f"{win_rate:.0f}%",
                    f"{min(scores):.2f}",
                    f"{max(scores):.2f}",
                )
            else:
                table.add_row(model_name, "N/A", "N/A", "N/A", "N/A")

        console.print(table)
        console.print(f"\n[dim]Benchmark: {self.benchmark_name} | "
                      f"Prompts: {total_prompts} | "
                      f"Time: {self.elapsed:.1f}s[/dim]")


class Evaluator:
    """
    Main evaluator — runs benchmarks across models and judges results.

    Args:
        benchmark: Benchmark to evaluate against.
        models: List of model instances to evaluate.
        judge: Judge instance for scoring outputs.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        models: List[BaseModel],
        judge: BaseJudge,
    ):
        self.benchmark = benchmark
        self.models = models
        self.judge = judge

    def run(self, verbose: bool = False) -> EvalRun:
        """
        Run evaluation across all prompts and models.

        TODO: Parallelize model calls with asyncio.
        Currently runs sequentially — fine for <100 prompts but
        becomes painful at scale.
        """
        model_names = [m.name for m in self.models]
        run = EvalRun(
            benchmark_name=self.benchmark.name,
            model_names=model_names,
            judge_type=self.judge.name,
        )

        console.print(f"\n[bold]Evaluating {len(self.models)} model(s) "
                      f"on {len(self.benchmark.prompts)} prompt(s)...[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(self.benchmark.prompts))

            for prompt in self.benchmark.prompts:
                result = EvalResult(prompt=prompt)

                # Get model outputs
                for model in self.models:
                    try:
                        output = model.generate(prompt)
                        result.model_outputs[model.name] = output
                    except Exception as e:
                        console.print(f"[red]Error generating {model.name}: {e}[/red]")
                        result.model_outputs[model.name] = f"[ERROR: {e}]"
                        result.metadata[f"error_{model.name}"] = str(e)

                # Judge all outputs
                for model in self.models:
                    if model.name in result.model_outputs:
                        try:
                            judgment = self.judge.evaluate(
                                prompt=prompt,
                                response=result.model_outputs[model.name],
                            )
                            result.judgments[model.name] = judgment
                        except Exception as e:
                            console.print(f"[red]Error judging {model.name}: {e}[/red]")
                            result.judgments[model.name] = Judgment(
                                score=0.0,
                                reasoning=f"Judge error: {e}",
                            )

                run.results.append(result)

                if verbose:
                    self._print_prompt_result(result, model_names)

                progress.update(task, advance=1)

        run.completed_at = time.time()
        return run

    def _print_prompt_result(self, result: EvalResult, model_names: List[str]):
        """Print per-prompt results in verbose mode."""
        console.print(f"\n[bold]Prompt {result.prompt.id}:[/bold] {result.prompt.text[:100]}...")
        for name in model_names:
            if name in result.judgments:
                j = result.judgments[name]
                color = "green" if j.score >= 0.7 else "yellow" if j.score >= 0.4 else "red"
                console.print(f"  [{color}]{name}: {j.score:.3f}[/{color}] — {j.reasoning[:80]}...")
