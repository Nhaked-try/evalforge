#!/usr/bin/env python3
"""
Create and run a custom benchmark programmatically.

Usage:
    python examples/custom_benchmark.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evalforge.benchmarks.base import Benchmark, Prompt
from evalforge.evaluator import Evaluator
from evalforge.models import create_model
from evalforge.judges import create_judge


def main():
    print("=" * 60)
    print("EvalForge — Custom Benchmark Example")
    print("=" * 60)

    # Create a custom benchmark programmatically
    prompts = [
        Prompt(
            id="custom-001",
            text="Explain quantum computing to a 10-year-old.",
            expected=None,  # No expected answer for open-ended prompts
        ),
        Prompt(
            id="custom-002",
            text="What are three key differences between Python and Rust?",
        ),
        Prompt(
            id="custom-003",
            text="Write a haiku about machine learning.",
        ),
        Prompt(
            id="custom-004",
            text="If you had to debug a production outage at 3 AM, what's your step-by-step approach?",
        ),
        Prompt(
            id="custom-005",
            text="Explain the CAP theorem in distributed systems.",
        ),
    ]

    benchmark = Benchmark(
        name="my-custom-benchmark",
        type="custom",
        prompts=prompts,
        description="A custom benchmark created programmatically.",
    )

    print(f"Created benchmark: {benchmark.name} ({len(benchmark.prompts)} prompts)")

    # Evaluate
    model = create_model("gpt-4o-mini")
    judge = create_judge("llm", model="gpt-4o-mini")

    evaluator = Evaluator(benchmark=benchmark, models=[model], judge=judge)
    results = evaluator.run(verbose=True)

    results.print_summary()


if __name__ == "__main__":
    main()
