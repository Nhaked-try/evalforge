"""Tests for metrics module."""

from evalforge.metrics import compute_win_rate_matrix, compute_bootstrap_ci
from evalforge.evaluator import EvalRun, EvalResult
from evalforge.benchmarks.base import Prompt
from evalforge.judges.base import Judgment


def test_compute_win_rate_matrix():
    run = EvalRun(
        benchmark_name="test",
        model_names=["a", "b"],
        judge_type="exact",
    )

    # Model A beats B on prompt 1, B beats A on prompt 2
    r1 = EvalResult(prompt=Prompt(id="1", text="Q1"))
    r1.judgments = {"a": Judgment(score=0.9), "b": Judgment(score=0.5)}
    run.results.append(r1)

    r2 = EvalResult(prompt=Prompt(id="2", text="Q2"))
    r2.judgments = {"a": Judgment(score=0.3), "b": Judgment(score=0.8)}
    run.results.append(r2)

    matrix = compute_win_rate_matrix(run)

    assert matrix["a"]["b"] == 0.5
    assert matrix["b"]["a"] == 0.5


def test_bootstrap_ci():
    scores = [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.82, 0.88]
    ci = compute_bootstrap_ci(scores, n_bootstrap=500)

    assert "mean" in ci
    assert "lower" in ci
    assert "upper" in ci
    assert ci["lower"] <= ci["mean"] <= ci["upper"]
