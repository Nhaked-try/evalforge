#!/usr/bin/env python3
"""
Elo tournament — find out which model wins head-to-head.

Usage:
    python examples/elo_tournament.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from evalforge.arena import Arena
from evalforge.benchmarks import load_benchmark
from evalforge.models import create_model
from evalforge.judges import create_judge

load_dotenv()


def main():
    print("=" * 60)
    print("EvalForge — Elo Tournament")
    print("=" * 60)

    # Load benchmark
    benchmark = load_benchmark("reasoning")
    benchmark.prompts = benchmark.prompts[:5]  # Small set for demo
    print(f"Benchmark: {benchmark.name} ({len(benchmark.prompts)} prompts)")

    # Models to compete
    models = [
        create_model("gpt-4o-mini"),
        create_model("gpt-4o"),
    ]
    print(f"Models: {', '.join(m.name for m in models)}")

    # Judge
    judge = create_judge("llm", model="gpt-4o")
    print(f"Judge: {judge.name}")

    # Run tournament
    print(f"\nRunning 20-round tournament...")
    arena = Arena(
        benchmark=benchmark,
        models=models,
        judge=judge,
        k_factor=32.0,
        initial_elo=1200.0,
    )

    ratings = arena.run_tournament(rounds=20, verbose=True)

    # Print final standings
    print("\n" + "=" * 60)
    print("Final Standings:")
    print("-" * 60)
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1]["elo"], reverse=True)
    for rank, (name, stats) in enumerate(sorted_ratings, 1):
        print(f"  {rank}. {name}")
        print(f"     Elo: {stats['elo']:.0f} | W: {stats['wins']} | L: {stats['losses']} | D: {stats['draws']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
