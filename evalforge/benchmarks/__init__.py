"""Benchmark loading and management."""

from pathlib import Path
from typing import Union
import os

from evalforge.benchmarks.base import Benchmark
from evalforge.benchmarks.coding import CodingBenchmark
from evalforge.benchmarks.reasoning import ReasoningBenchmark
from evalforge.benchmarks.safety import SafetyBenchmark
from evalforge.benchmarks.custom import CustomBenchmark

BENCHMARK_REGISTRY = {
    "coding": CodingBenchmark,
    "reasoning": ReasoningBenchmark,
    "safety": SafetyBenchmark,
    "custom": CustomBenchmark,
}


def load_benchmark(name_or_path: str) -> Benchmark:
    """
    Load a benchmark by name or file path.

    Args:
        name_or_path: Either a registered benchmark name (e.g., 'coding')
                      or a path to a YAML file.

    Returns:
        A Benchmark instance.
    """
    # Check if it's a file path
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return CustomBenchmark.from_yaml(path)

    # Check registry
    config_dir = Path("configs/benchmarks")
    config_path = config_dir / f"{name_or_path}.yaml"

    if config_path.exists():
        if name_or_path in BENCHMARK_REGISTRY:
            return BENCHMARK_REGISTRY[name_or_path].from_yaml(config_path)
        return CustomBenchmark.from_yaml(config_path)

    # Try built-in defaults
    if name_or_path in BENCHMARK_REGISTRY:
        return BENCHMARK_REGISTRY[name_or_path].default()

    raise ValueError(
        f"Unknown benchmark: {name_or_path}. "
        f"Available: {list(BENCHMARK_REGISTRY.keys())}. "
        f"Or provide a path to a YAML file."
    )


def list_benchmarks() -> list[dict]:
    """List all available benchmarks."""
    benchmarks = []
    config_dir = Path("configs/benchmarks")

    if config_dir.exists():
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                bench = CustomBenchmark.from_yaml(yaml_file)
                benchmarks.append({
                    "name": bench.name,
                    "num_prompts": len(bench.prompts),
                    "type": bench.type,
                    "path": str(yaml_file),
                })
            except Exception:
                continue

    return benchmarks
