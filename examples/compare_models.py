#!/usr/bin/env python3
"""
Multi-model comparison — same benchmark, multiple models, side-by-side.

Usage:
    python examples/compare_models.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from evalforge import Evaluator
from evalforge.benchmarks import load_benchmark
from evalforge.models import create_model
from evalforge.judges import create_judge
from evalforge.metrics import compute_win_rate_matrix

load_dotenv()


def main():
    print("=" * 60)
    print("EvalForge — Model Comparison")
    print("=" * 60)

    # Load coding benchmark
    print("\n[1] Loading benchmark...")
    benchmark = load_benchmark("coding")
    # Only use first 5 prompts to keep it fast
    benchmark.prompts = benchmark.prompts[:5]
    print(f"    {benchmark.name}: {len(benchmark.prompts)} prompts")

    # Compare three models
    print("\n[2] Setting up models...")
    models = [
        create_model("gpt-4o"),
        create_model("gpt-4o-mini"),
    ]
    for m in models:
        print(f"    {m.name}")

    # LLM judge
    print("\n[3] Setting up LLM judge...")
    judge = create_judge("llm", model="gpt-4o")

    # Run
    print("\n[4] Running comparison...")
    evaluator = Evaluator(
        benchmark=benchmark,
        models=models,
        judge=judge,
    )

    results = evaluator.run(verbose=False)

    print("\n[5] Results:")
    results.print_summary()

    # Win rate matrix
    print("\n[6] Win Rate Matrix:")
    matrix = compute_win_rate_matrix(results)
    for model_a in results.model_names:
        for model_b in results.model_names:
            if model_a != model_b:
                rate = matrix[model_a][model_b]
                print(f"    {model_a} beats {model_b}: {rate:.1%}")

    # Save
    results.to_json("results/model_comparison.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
