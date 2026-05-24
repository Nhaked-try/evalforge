"""Shared test fixtures."""

import pytest
from evalforge.benchmarks.base import Prompt, Benchmark


@pytest.fixture
def simple_prompt():
    return Prompt(
        id="test-001",
        text="What is 2+2?",
        expected="4",
    )


@pytest.fixture
def empty_expected_prompt():
    return Prompt(
        id="test-002",
        text="Write a poem about testing.",
        expected=None,
    )


@pytest.fixture
def sample_benchmark():
    prompts = [
        Prompt(id="p1", text="Question 1", expected="Answer 1"),
        Prompt(id="p2", text="Question 2", expected="Answer 2"),
        Prompt(id="p3", text="Question 3"),
    ]
    return Benchmark(
        name="test-benchmark",
        type="custom",
        prompts=prompts,
        description="Test benchmark",
    )
