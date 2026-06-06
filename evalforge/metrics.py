"""
Aggregate metrics for evaluation results.

TODO:
- [ ] Statistical significance tests (bootstrap confidence intervals)
- [ ] Cohen's d effect size
- [ ] Pairwise win rate matrix
- [ ] Calibration metrics for judges
"""

from typing import Dict, List
import numpy as np
from evalforge.evaluator import EvalRun


def compute_win_rate_matrix(run: EvalRun) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise win rates between all models.

    Returns dict[model_a][model_b] = win_rate_of_a_vs_b
    """
    model_names = run.model_names
    matrix: Dict[str, Dict[str, float]] = {
        a: {b: 0.0 for b in model_names} for a in model_names
    }

    # Count wins
    for result in run.results:
        scores = {
            name: result.judgments[name].score
            for name in model_names
            if name in result.judgments
        }

        if len(scores) < 2:
            continue

        for model_a in model_names:
            for model_b in model_names:
                if model_a == model_b:
                    continue
                if model_a not in scores or model_b not in scores:
                    continue
                if scores[model_a] > scores[model_b]:
                    matrix[model_a][model_b] += 1

    # Normalize
    num_prompts = len(run.results)
    if num_prompts > 0:
        for a in model_names:
            for b in model_names:
                if a != b:
                    matrix[a][b] /= num_prompts

    return matrix


def compute_bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for mean score.

    Args:
        scores: List of scores.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence interval (e.g., 0.95 for 95%).

    Returns:
        Dict with 'mean', 'lower', 'upper'.
    """
    if not scores:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    rng = np.random.default_rng(42)
    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))

    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100

    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(means, lower_percentile)),
        "upper": float(np.percentile(means, upper_percentile)),
    }
