"""Custom benchmark loaded from YAML file."""

from pathlib import Path
import yaml
from evalforge.benchmarks.base import Benchmark, Prompt


class CustomBenchmark(Benchmark):
    """User-defined benchmark from YAML."""

    type = "custom"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CustomBenchmark":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        prompts = []
        for p in data.get("prompts", []):
            prompts.append(Prompt.from_dict(p))

        return cls(
            name=data.get("name", Path(path).stem),
            type=data.get("type", "custom"),
            prompts=prompts,
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def default(cls) -> "CustomBenchmark":
        """Empty custom benchmark."""
        return cls(
            name="custom-empty",
            type="custom",
            prompts=[],
            description="Empty custom benchmark — add your own prompts.",
        )
