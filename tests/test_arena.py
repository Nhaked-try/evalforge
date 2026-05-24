"""Tests for Elo tournament arena."""

import pytest
from unittest.mock import MagicMock
from evalforge.arena import Arena
from evalforge.benchmarks.base import Prompt, Benchmark
from evalforge.judges.base import Judgment


@pytest.fixture
def arena_setup():
    benchmark = Benchmark(
        name="test",
        type="custom",
        prompts=[
            Prompt(id="p1", text="Q1"),
            Prompt(id="p2", text="Q2"),
            Prompt(id="p3", text="Q3"),
        ],
    )

    model_a = MagicMock()
    model_a.name = "model-a"
    model_a.generate.return_value = "Response A"

    model_b = MagicMock()
    model_b.name = "model-b"
    model_b.generate.return_value = "Response B"

    judge = MagicMock()
    judge.name = "mock-judge"
    # Model A always scores slightly higher
    judge.evaluate.side_effect = lambda prompt, response: Judgment(
        score=0.9 if "Response A" in response else 0.7,
        reasoning="mock",
    )

    return benchmark, [model_a, model_b], judge


class TestArena:
    def test_initial_ratings(self, arena_setup):
        benchmark, models, judge = arena_setup
        arena = Arena(benchmark, models, judge)

        for model in models:
            assert arena.ratings[model.name]["elo"] == 1200.0
            assert arena.ratings[model.name]["battles"] == 0

    def test_tournament_updates_ratings(self, arena_setup):
        benchmark, models, judge = arena_setup
        arena = Arena(benchmark, models, judge, k_factor=32.0)

        ratings = arena.run_tournament(rounds=50, verbose=False)

        # Model A should have higher Elo (always scores 0.9 vs 0.7)
        assert ratings["model-a"]["elo"] > ratings["model-b"]["elo"]

        # All models should have participated
        for model in models:
            assert ratings[model.name]["battles"] > 0

    def test_needs_two_models(self, arena_setup):
        benchmark, models, judge = arena_setup
        arena = Arena(benchmark, [models[0]], judge)

        with pytest.raises(ValueError, match="at least 2"):
            arena.run_tournament(rounds=10)
