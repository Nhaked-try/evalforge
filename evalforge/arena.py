"""
Elo Tournament (Arena mode) — pairwise battles between models.

Inspired by LMSys Chatbot Arena. Models compete in head-to-head
battles judged by LLM or embedding similarity.

TODO:
- [ ] Bradley-Terry model option (more statistically rigorous)
- [ ] Confidence intervals on Elo ratings
- [ ] Adaptive pairing (match models with similar ratings)
- [ ] Human-in-the-loop judging mode
"""

import random
from typing import Dict, List, Optional
import math

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from evalforge.benchmarks.base import Benchmark
from evalforge.models.base import BaseModel
from evalforge.judges.base import BaseJudge

console = Console()


class Arena:
    """
    Elo tournament arena for model comparison.

    Args:
        benchmark: Benchmark to draw prompts from.
        models: Models to compete.
        judge: Judge for pairwise comparison.
        k_factor: Elo K-factor (higher = more volatile ratings).
        initial_elo: Starting Elo rating for all models.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        models: List[BaseModel],
        judge: BaseJudge,
        k_factor: float = 32.0,
        initial_elo: float = 1200.0,
    ):
        self.benchmark = benchmark
        self.models = models
        self.judge = judge
        self.k_factor = k_factor
        self.initial_elo = initial_elo

        # Initialize ratings
        self.ratings: Dict[str, Dict] = {}
        for model in models:
            self.ratings[model.name] = {
                "elo": initial_elo,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "battles": 0,
            }

    def run_tournament(
        self,
        rounds: int = 100,
        verbose: bool = False,
    ) -> Dict[str, Dict]:
        """
        Run Elo tournament.

        Each round: randomly select two models, randomly select a prompt,
        both models respond, judge picks winner.

        Args:
            rounds: Number of pairwise battles.
            verbose: Print each battle result.

        Returns:
            Dict of model_name -> rating stats.
        """
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for tournament")

        console.print(f"\n[bold]Elo Tournament: {len(self.models)} models, {rounds} rounds[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Battling...", total=rounds)

            for round_num in range(rounds):
                # Pick two different models
                model_a, model_b = random.sample(self.models, 2)

                # Pick a random prompt
                prompt = random.choice(self.benchmark.prompts)

                # Both models respond
                try:
                    response_a = model_a.generate(prompt)
                    response_b = model_b.generate(prompt)
                except Exception as e:
                    console.print(f"[red]Generation error in round {round_num}: {e}[/red]")
                    progress.update(task, advance=1)
                    continue

                # Judge picks winner
                # TODO: Position bias mitigation — swap order and average
                try:
                    judgment_a = self.judge.evaluate(prompt, response_a)
                    judgment_b = self.judge.evaluate(prompt, response_b)
                except Exception as e:
                    console.print(f"[red]Judge error in round {round_num}: {e}[/red]")
                    progress.update(task, advance=1)
                    continue

                # Determine winner
                if judgment_a.score > judgment_b.score:
                    winner, loser = model_a.name, model_b.name
                elif judgment_b.score > judgment_a.score:
                    winner, loser = model_b.name, model_a.name
                else:
                    # Draw
                    self._update_elo_draw(model_a.name, model_b.name)
                    if verbose:
                        console.print(
                            f"  Round {round_num}: {model_a.name} vs {model_b.name} — DRAW "
                            f"({judgment_a.score:.3f} vs {judgment_b.score:.3f})"
                        )
                    progress.update(task, advance=1)
                    continue

                # Update Elo
                self._update_elo(winner, loser)

                self.ratings[winner]["wins"] += 1
                self.ratings[loser]["losses"] += 1
                self.ratings[winner]["battles"] += 1
                self.ratings[loser]["battles"] += 1

                if verbose:
                    winner_score = judgment_a.score if winner == model_a.name else judgment_b.score
                    loser_score = judgment_b.score if winner == model_a.name else judgment_a.score
                    console.print(
                        f"  Round {round_num}: {winner} beats {loser} "
                        f"({winner_score:.3f} vs {loser_score:.3f})"
                    )

                progress.update(task, advance=1)

        return self.ratings

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def _update_elo(self, winner: str, loser: str):
        """Update Elo ratings after a win/loss."""
        r_winner = self.ratings[winner]["elo"]
        r_loser = self.ratings[loser]["elo"]

        expected_winner = self._expected_score(r_winner, r_loser)

        self.ratings[winner]["elo"] = r_winner + self.k_factor * (1.0 - expected_winner)
        self.ratings[loser]["elo"] = r_loser + self.k_factor * (0.0 - (1.0 - expected_winner))

    def _update_elo_draw(self, model_a: str, model_b: str):
        """Update Elo ratings for a draw."""
        r_a = self.ratings[model_a]["elo"]
        r_b = self.ratings[model_b]["elo"]

        expected_a = self._expected_score(r_a, r_b)

        self.ratings[model_a]["elo"] = r_a + self.k_factor * (0.5 - expected_a)
        self.ratings[model_b]["elo"] = r_b + self.k_factor * (0.5 - (1.0 - expected_a))

        self.ratings[model_a]["draws"] += 1
        self.ratings[model_b]["draws"] += 1
        self.ratings[model_a]["battles"] += 1
        self.ratings[model_b]["battles"] += 1
