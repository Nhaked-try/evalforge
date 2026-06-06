"""Tests for judge implementations."""

import pytest
from unittest.mock import MagicMock, patch
from evalforge.judges.exact_match import ExactMatchJudge
from evalforge.judges.embedding_similarity import EmbeddingSimilarityJudge
from evalforge.judges.base import Judgment
from evalforge.benchmarks.base import Prompt


class TestExactMatchJudge:
    def test_exact_match(self, simple_prompt):
        judge = ExactMatchJudge()
        result = judge.evaluate(simple_prompt, "4")
        assert result.score == 1.0

    def test_no_match(self, simple_prompt):
        judge = ExactMatchJudge()
        result = judge.evaluate(simple_prompt, "5")
        assert result.score == 0.0

    def test_case_insensitive(self):
        judge = ExactMatchJudge(case_sensitive=False)
        prompt = Prompt(id="t1", text="What?", expected="Hello World")
        result = judge.evaluate(prompt, "hello world")
        assert result.score == 1.0

    def test_contains_expected(self):
        judge = ExactMatchJudge()
        prompt = Prompt(id="t1", text="Q", expected="42")
        result = judge.evaluate(prompt, "The answer is 42.")
        assert result.score > 0.5  # Expected found within response

    def test_no_expected_answer(self, empty_expected_prompt):
        judge = ExactMatchJudge()
        result = judge.evaluate(empty_expected_prompt, "anything")
        assert result.score == 0.5  # Neutral
        assert "No expected" in result.reasoning


class TestEmbeddingSimilarityJudge:
    def test_identical_text(self, simple_prompt):
        judge = EmbeddingSimilarityJudge()
        result = judge.evaluate(simple_prompt, "4")
        # Identical or nearly identical should have high score
        assert result.score > 0.8

    def test_different_text(self, simple_prompt):
        judge = EmbeddingSimilarityJudge()
        result = judge.evaluate(simple_prompt, "The capital of France is Paris.")
        # Very different from "4"
        assert result.score < 0.7

    def test_no_expected_returns_neutral(self, empty_expected_prompt):
        judge = EmbeddingSimilarityJudge()
        result = judge.evaluate(empty_expected_prompt, "anything")
        assert result.score == 0.5
