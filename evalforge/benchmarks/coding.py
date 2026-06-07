"""
Coding benchmark — programming problems with expected outputs.

Built-in default inspired by HumanEval-style problems.
"""

import yaml
from pathlib import Path
from evalforge.benchmarks.base import Benchmark, Prompt


class CodingBenchmark(Benchmark):
    """Coding problems benchmark."""

    type = "coding"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CodingBenchmark":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        prompts = [Prompt.from_dict(p) for p in data.get("prompts", [])]
        return cls(
            name=data.get("name", "coding"),
            type=data.get("type", "coding"),
            prompts=prompts,
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def default(cls) -> "CodingBenchmark":
        """Built-in default coding benchmark — 10 programming problems."""
        prompts = [
            Prompt(
                id="code-001",
                text="Write a Python function `fibonacci(n: int) -> int` that returns the nth Fibonacci number (0-indexed). Include type hints and a docstring.",
                expected="def fibonacci(n: int) -> int:\n    ...",
            ),
            Prompt(
                id="code-002",
                text="Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
            ),
            Prompt(
                id="code-003",
                text="Write a Python function `merge_sorted(arr1: List[int], arr2: List[int]) -> List[int]` that merges two sorted arrays into one sorted array. Aim for O(n) complexity.",
            ),
            Prompt(
                id="code-004",
                text="Write a Python function `flatten(nested: List[Any]) -> List[Any]` that recursively flattens a nested list of arbitrary depth.",
            ),
            Prompt(
                id="code-005",
                text="Write a Python class `LRUCache` with `get(key)` and `put(key, value)` methods, both O(1). The cache has a fixed capacity and evicts the least recently used item.",
            ),
            Prompt(
                id="code-006",
                text="Write a Python function that parses a CSV string into a list of dictionaries. Handle quoted fields and escaped commas.",
            ),
            Prompt(
                id="code-007",
                text="Write a Python decorator `@retry(max_attempts=3, backoff_factor=2)` that retries a function on exception with exponential backoff.",
            ),
            Prompt(
                id="code-008",
                text="Write a Python async function that fetches data from multiple URLs concurrently and returns results in order. Use asyncio.",
            ),
            Prompt(
                id="code-009",
                text="Write a Python function `find_duplicates(arr: List[int]) -> List[int]` that returns all duplicate values in an array. Optimize for time complexity.",
            ),
            Prompt(
                id="code-010",
                text="Write a Python function to validate a binary search tree. Given a root node, return True if the tree is a valid BST, False otherwise.",
            ),
        ]
        return cls(
            name="coding-basic",
            type="coding",
            prompts=prompts,
            description="Basic coding problems for evaluating programming ability.",
        )
