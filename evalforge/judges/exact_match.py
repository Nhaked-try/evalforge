"""
Exact Match judge — compares response against expected answer.

Best for: classification, multiple choice, math problems with
single correct answers.

Supports:
- Case-insensitive matching
- Whitespace normalization
- Numeric tolerance (for math answers)

TODO:
- [ ] Fuzzy matching option (Levenshtein distance)
- [ ] Regex-based matching patterns
"""

from evalforge.judges.base import BaseJudge, Judgment
from evalforge.benchmarks.base import Prompt


class ExactMatchJudge(BaseJudge):
    """Judge based on exact string matching of expected answer."""

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
        numeric_tolerance: float | None = None,
    ):
        super().__init__(name="exact-match")
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
        self.numeric_tolerance = numeric_tolerance

    def evaluate(self, prompt: Prompt, response: str) -> Judgment:
        if prompt.expected is None:
            return Judgment(
                score=0.5,
                reasoning="No expected answer provided — cannot judge.",
            )

        expected = prompt.expected
        actual = response.strip()

        if self.normalize_whitespace:
            expected = " ".join(expected.split())
            actual = " ".join(actual.split())

        if not self.case_sensitive:
            expected = expected.lower()
            actual = actual.lower()

        # Numeric comparison
        if self.numeric_tolerance is not None:
            try:
                exp_num = float(expected)
                act_num = float(actual)
                if abs(exp_num - act_num) <= self.numeric_tolerance:
                    return Judgment(score=1.0, reasoning="Numeric match within tolerance.")
            except (ValueError, TypeError):
                pass  # Fall through to string comparison

        # Exact string match
        if expected == actual:
            return Judgment(score=1.0, reasoning="Exact match.")
        elif expected in actual:
            return Judgment(score=0.7, reasoning="Expected answer found in response.")
        else:
            return Judgment(
                score=0.0,
                reasoning=f"Expected '{expected[:50]}', got '{actual[:50]}'",
            )
