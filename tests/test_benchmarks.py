"""Tests for benchmark loading and creation."""

import pytest
from pathlib import Path
from evalforge.benchmarks import load_benchmark
from evalforge.benchmarks.coding import CodingBenchmark
from evalforge.benchmarks.reasoning import ReasoningBenchmark
from evalforge.benchmarks.safety import SafetyBenchmark


class TestBenchmarkDefaults:
    def test_coding_default(self):
        bench = CodingBenchmark.default()
        assert bench.name == "coding-basic"
        assert len(bench.prompts) == 10
        assert bench.type == "coding"

    def test_reasoning_default(self):
        bench = ReasoningBenchmark.default()
        assert bench.name == "reasoning-basic"
        assert len(bench.prompts) == 8
        assert bench.type == "reasoning"

    def test_safety_default(self):
        bench = SafetyBenchmark.default()
        assert bench.name == "safety-basic"
        assert len(bench.prompts) == 5
        assert bench.type == "safety"

    def test_prompt_has_required_fields(self):
        bench = CodingBenchmark.default()
        prompt = bench.prompts[0]
        assert prompt.id
        assert prompt.text
        assert isinstance(prompt.metadata, dict)


class TestLoadBenchmark:
    def test_load_by_name(self):
        # Should load built-in defaults
        bench = load_benchmark("coding")
        assert bench is not None
        assert len(bench.prompts) > 0

    def test_load_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_benchmark("nonexistent_benchmark_xyz")
