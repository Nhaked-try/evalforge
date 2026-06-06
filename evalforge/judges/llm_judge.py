"""
LLM-as-Judge — uses a strong LLM to evaluate responses.

This is the most flexible judge. It can assess subjective qualities
like helpfulness, creativity, and accuracy.

Default judge model: gpt-4o (configurable).

TODO:
- [ ] Judge calibration against human labels
- [ ] Multiple judge models with agreement metrics
- [ ] Judge bias detection
- [ ] Position bias mitigation (swap response order)
"""

import os
import json
from dotenv import load_dotenv
from evalforge.judges.base import BaseJudge, Judgment
from evalforge.benchmarks.base import Prompt

load_dotenv()

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI model outputs. Your task is to rate the quality of a model's response to a given prompt.

Rate the response on a scale of 0.0 to 1.0 based on:
- Accuracy: Is the information correct?
- Completeness: Does it fully address the prompt?
- Clarity: Is it well-written and easy to understand?
- Helpfulness: Would this genuinely help the user?

Output your evaluation as a JSON object with:
{
    "score": <float 0.0-1.0>,
    "reasoning": "<brief explanation of the score>",
    "strengths": ["<what was good>"],
    "weaknesses": ["<what could be improved>"]
}
"""


class LLMJudge(BaseJudge):
    """Use an LLM to judge response quality."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        super().__init__(name=f"llm-{model_name}")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.getenv("EVALFORGE_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("EVALFORGE_JUDGE_API_KEY or OPENAI_API_KEY not set")

        self._client = OpenAI(api_key=api_key)

    def evaluate(self, prompt: Prompt, response: str) -> Judgment:
        user_prompt = f"""Prompt: {prompt.text}

Model Response:
{response}

Evaluate the response quality."""
        try:
            result = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = result.choices[0].message.content or "{}"
            parsed = json.loads(content)

            return Judgment(
                score=max(0.0, min(1.0, parsed.get("score", 0.0))),
                reasoning=parsed.get("reasoning", ""),
                metadata={
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", []),
                },
            )
        except Exception as e:
            return Judgment(
                score=0.0,
                reasoning=f"Judge error: {e}",
                metadata={"error": str(e)},
            )
