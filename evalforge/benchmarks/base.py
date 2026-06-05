"""
Base benchmark types.

A Benchmark is a collection of Prompts with optional expected answers
and metadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Prompt:
    """A single evaluation prompt."""

    id: str
    text: str
    expected: Optional[str] = None  # Reference answer (for exact match / similarity)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "expected": self.expected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prompt":
        return cls(
            id=data["id"],
            text=data["text"],
            expected=data.get("expected"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Benchmark:
    """Base benchmark — collection of prompts."""

    name: str
    type: str
    prompts: List[Prompt]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    @abstractmethod
    def from_yaml(cls, path) -> "Benchmark":
        """Load benchmark from YAML file."""
        ...

    @classmethod
    @abstractmethod
    def default(cls) -> "Benchmark":
        """Return a default built-in benchmark."""
        ...

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Prompt:
        return self.prompts[idx]
