"""Tests for the Evaluator module."""

import pytest
from unittest.mock import MagicMock, patch
from evalforge.evaluator import Evaluator, EvalRun, EvalResult
from evalforge.benchmarks.base import Prompt
from evalforge.judges.base import Judgment


class TestEvalRun:
    def test_to_dict(self):
        run = EvalRun(
            benchmark_name="test",
            model_names=["model-a"],
            judge_type="exact",
        )
        d = run.to_dict()
        assert d["benchmark"] == "test"
        assert d["models"] == ["model-a"]
        assert d["num_prompts"] == 0

    def test_empty_results(self):
        run = EvalRun(benchmark_name="test", model_names=["m1"], judge_type="exact")
        run.print_summary()  # Should not crash


class TestEvaluator:
    def test_run_with_mock_models(self, sample_benchmark):
        """Test evaluator with mocked models and judge."""
        mock_model = MagicMock()
        mock_model.name = "mock-model"
        mock_model.generate.return_value = "mock response"

        mock_judge = MagicMock()
        mock_judge.name = "mock-judge"
        mock_judge.evaluate.return_value = Judgment(score=0.85, reasoning="Good")

        evaluator = Evaluator(
            benchmark=sample_benchmark,
            models=[mock_model],
            judge=mock_judge,
        )

        results = evaluator.run(verbose=False)

        assert isinstance(results, EvalRun)
        assert results.num_prompts == 3
        assert results.benchmark_name == "test-benchmark"
        assert mock_model.generate.call_count == 3
        assert mock_judge.evaluate.call_count == 3

    def test_handles_generation_error(self, sample_benchmark):
        """Evaluator should handle model errors gracefully."""
        mock_model = MagicMock()
        mock_model.name = "error-model"
        mock_model.generate.side_effect = RuntimeError("GPU OOM")

        mock_judge = MagicMock()
        mock_judge.name = "mock-judge"
        mock_judge.evaluate.return_value = Judgment(score=0.0)

        evaluator = Evaluator(
            benchmark=sample_benchmark,
            models=[mock_model],
            judge=mock_judge,
        )

        results = evaluator.run(verbose=False)

        # Should still complete (3 prompts, all errored)
        assert results.num_prompts == 3
