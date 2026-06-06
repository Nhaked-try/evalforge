"""
Reward Model judge — uses a pre-trained reward model (ArmoRM) to score responses.

EXPERIMENTAL — requires GPU for models > 1B parameters.

ArmoRM-Llama3-8B needs ~18GB VRAM.
Smaller reward models (OpenAssistant, UltraRM) may work on CPU.

TODO:
- [ ] Add more reward model options (UltraRM, Starling)
- [ ] Multi-dimensional scoring (helpfulness, harmlessness, honesty)
- [ ] Batch scoring for efficiency
- [ ] Model download on first use
"""

from evalforge.judges.base import BaseJudge, Judgment
from evalforge.benchmarks.base import Prompt


class RewardModelJudge(BaseJudge):
    """
    Judge using a reward model (ArmoRM by default).

    EXPERIMENTAL — requires:
    - GPU with 16GB+ VRAM (for ArmoRM-8B)
    - transformers, torch, accelerate
    """

    def __init__(
        self,
        model_name: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device: str | None = None,
    ):
        super().__init__(name=f"reward-{model_name.split('/')[-1]}")
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "torch and transformers required. Install with: "
                "pip install torch transformers"
            )

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[RewardModelJudge] Loading {self.model_name} on {self.device}...")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self._model = self._model.to(self.device)

        self._model.eval()

    def evaluate(self, prompt: Prompt, response: str) -> Judgment:
        self._load_model()

        import torch

        # Format: concatenate prompt and response
        text = f"User: {prompt.text}\n\nAssistant: {response}"

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Assume first output is scalar reward score
            raw_score = outputs.logits[0].item()

            # Normalize to 0-1 range
            # TODO: Proper calibration — current normalization is naive
            score = float(torch.sigmoid(torch.tensor(raw_score)))

        return Judgment(
            score=score,
            reasoning=f"Reward model score: {score:.3f} (raw: {raw_score:.3f})",
            metadata={
                "raw_score": raw_score,
                "model": self.model_name,
                "note": "EXPERIMENTAL — scores uncalibrated",
            },
        )
