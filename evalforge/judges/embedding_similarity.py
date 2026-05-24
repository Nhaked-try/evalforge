"""
Embedding similarity judge — uses cosine similarity between embeddings.

GPU-accelerated via sentence-transformers.
Falls back to CPU if no GPU available.

Best for: semantic equivalence where exact wording doesn't matter.
Useful for scoring open-ended responses against reference answers.

TODO:
- [ ] Batch embedding for multi-prompt evaluations
- [ ] Multiple embedding model options
- [ ] Cross-encoder option for more accurate scoring
"""

from evalforge.judges.base import BaseJudge, Judgment
from evalforge.benchmarks.base import Prompt


class EmbeddingSimilarityJudge(BaseJudge):
    """
    Judge based on semantic similarity of embeddings.

    Requires sentence-transformers.
    GPU recommended for datasets > 100 samples.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        super().__init__(name=f"embedding-{model_name}")
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def evaluate(self, prompt: Prompt, response: str) -> Judgment:
        if prompt.expected is None:
            return Judgment(
                score=0.5,
                reasoning="No expected answer provided — returning neutral score.",
            )

        self._load_model()

        # Embed both texts
        embeddings = self._model.encode(
            [prompt.expected, response],
            show_progress_bar=False,
        )

        # Cosine similarity
        from numpy.linalg import norm

        cos_sim = float(
            (embeddings[0] @ embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        )

        return Judgment(
            score=cos_sim,
            reasoning=f"Cosine similarity: {cos_sim:.3f}",
            metadata={"cosine_similarity": cos_sim, "model": self.model_name},
        )
