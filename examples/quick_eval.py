#!/usr/bin/env python3
"""
Quick evaluation — single model, single benchmark.

The simplest possible EvalForge usage. Good for smoke testing.

Usage:
    python examples/quick_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from evalforge import Evaluator
from evalforge.benchmarks import load_benchmark
from evalforge.models import create_model
from evalforge.judges import create_judge

load_dotenv()


def main():
    print("=" * 60)
    print("EvalForge — Quick Evaluation Example")
    print("=" * 60)

    # 1. Load a benchmark
    print("\n[1] Loading benchmark...")
    benchmark = load_benchmark("reasoning")
    print(f"    Loaded: {benchmark.name} ({len(benchmark.prompts)} prompts)")

    # 2. Create a model (OpenAI for quick testing)
    print("\n[2] Setting up models...")
    model = create_model("gpt-4o-mini")
    print(f"    Model: {model.name}")

    # 3. Create a judge
    print("\n[3] Setting up judge...")
    judge = create_judge("exact")
    print(f"    Judge: {judge.name}")

    # 4. Run evaluation
    print("\n[4] Running evaluation...")
    evaluator = Evaluator(
        benchmark=benchmark,
        models=[model],
        judge=judge,
    )

    results = evaluator.run(verbose=False)

    # 5. Print results
    print("\n[5] Results:")
    results.print_summary()

    # Save
    results.to_json("results/quick_eval.json")
    print("\nDone! Results saved to results/quick_eval.json")


if __name__ == "__main__":
    main()
