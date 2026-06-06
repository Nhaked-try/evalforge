#!/usr/bin/env python3
"""
GPU-accelerated batch evaluation example.

Uses vLLM backend for fast local model evaluation.
REQUIRES: vLLM server running on localhost:8000

Usage:
    # Start vLLM server first:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.90

    # Then run:
    python examples/batch_eval_gpu.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from evalforge import Evaluator
from evalforge.benchmarks import load_benchmark
from evalforge.models import create_model
from evalforge.judges import create_judge
from evalforge.utils import format_duration

load_dotenv()


def main():
    print("=" * 60)
    print("EvalForge — GPU Batch Evaluation (vLLM)")
    print("=" * 60)

    # Check if vLLM is accessible
    import httpx
    try:
        resp = httpx.get("http://localhost:8000/health", timeout=5.0)
        print(f"vLLM server: {resp.status_code}")
    except Exception:
        print("ERROR: vLLM server not running on localhost:8000")
        print("Start it with: vllm serve <model> --gpu-memory-utilization 0.90")
        sys.exit(1)

    # Load benchmark
    benchmark = load_benchmark("coding")
    print(f"Benchmark: {benchmark.name} ({len(benchmark.prompts)} prompts)")

    # vLLM model (local, GPU-accelerated)
    model = create_model("vllm:meta-llama/Llama-3.1-8B-Instruct")
    print(f"Model: {model.name} (vLLM local)")

    # Embedding judge (also GPU-accelerated)
    judge = create_judge("embedding")
    print(f"Judge: {judge.name}")

    # Run
    print("\nRunning batch evaluation...")
    start = time.time()

    evaluator = Evaluator(benchmark=benchmark, models=[model], judge=judge)
    results = evaluator.run(verbose=False)

    elapsed = time.time() - start
    throughput = len(benchmark.prompts) / elapsed if elapsed > 0 else 0

    print(f"\nDone in {format_duration(elapsed)}")
    print(f"Throughput: {throughput:.1f} prompts/second")
    results.print_summary()


if __name__ == "__main__":
    main()
