"""Base judge interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from evalforge.benchmarks.base import Prompt


@dataclass
class Judgment:
    """A single judgment of a model response."""

    score: float  # 0.0 to 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


class BaseJudge(ABC):
    """Abstract base for all judges."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, prompt: Prompt, response: str) -> Judgment:
        """Evaluate a response against a prompt."""
        ...
