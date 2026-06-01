"""Judge factory — creates judge instances from string identifiers."""

from typing import Optional
from evalforge.judges.base import BaseJudge
from evalforge.judges.llm_judge import LLMJudge
from evalforge.judges.exact_match import ExactMatchJudge
from evalforge.judges.embedding_similarity import EmbeddingSimilarityJudge
from evalforge.judges.reward_model import RewardModelJudge


def create_judge(judge_type: str, model: Optional[str] = None) -> BaseJudge:
    """
    Create a judge instance.

    Args:
        judge_type: Type identifier: 'llm', 'exact', 'embedding', 'reward'
        model: Model identifier for LLM-based judges.

    Returns:
        A BaseJudge instance.
    """
    if judge_type == "llm":
        return LLMJudge(model_name=model or "gpt-4o")
    elif judge_type == "exact":
        return ExactMatchJudge()
    elif judge_type == "embedding":
        return EmbeddingSimilarityJudge()
    elif judge_type == "reward":
        return RewardModelJudge()
    else:
        raise ValueError(f"Unknown judge type: {judge_type}. Use: llm, exact, embedding, reward")
