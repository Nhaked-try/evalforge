"""Reasoning benchmark — logical deduction, problem solving."""

from pathlib import Path
import yaml
from evalforge.benchmarks.base import Benchmark, Prompt


class ReasoningBenchmark(Benchmark):
    """Logical reasoning benchmark."""

    type = "reasoning"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ReasoningBenchmark":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        prompts = [Prompt.from_dict(p) for p in data.get("prompts", [])]
        return cls(
            name=data.get("name", "reasoning"),
            type=data.get("type", "reasoning"),
            prompts=prompts,
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def default(cls) -> "ReasoningBenchmark":
        prompts = [
            Prompt(
                id="reason-001",
                text="If all cats are mammals, and all mammals are animals, can a cat be a non-animal? Explain step by step.",
                expected="No, because by transitivity, all cats are animals.",
            ),
            Prompt(
                id="reason-002",
                text="A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                expected="5 cents",
            ),
            Prompt(
                id="reason-003",
                text="You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons of water?",
            ),
            Prompt(
                id="reason-004",
                text="If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads? Show your work.",
                expected="3/8",
            ),
            Prompt(
                id="reason-005",
                text="A train leaves Station A at 60 mph. Another train leaves Station B at 80 mph. They're 280 miles apart on the same track. When will they meet?",
            ),
            Prompt(
                id="reason-006",
                text="There are 5 houses in a row, each of a different color. The owners have different nationalities, pets, drinks, and cigarettes. The Norwegian lives in the first house. The person who drinks milk lives in the middle house. Who owns the fish?",
            ),
            Prompt(
                id="reason-007",
                text="If P → Q is true and P is false, what can we conclude about Q? Explain.",
                expected="Nothing — false antecedent means the implication is true regardless of Q.",
            ),
            Prompt(
                id="reason-008",
                text="You're in a room with 3 light switches. Each controls one of 3 light bulbs in another room. You can only go to the other room once. How do you determine which switch controls which bulb?",
            ),
        ]
        return cls(
            name="reasoning-basic",
            type="reasoning",
            prompts=prompts,
            description="Logic puzzles and deductive reasoning problems.",
        )
