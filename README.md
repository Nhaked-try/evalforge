# evalforge

EvalForge is a local evaluation runner for small LLM experiments. I use it to compare model outputs on the same prompts before deciding whether a model is worth fine-tuning, serving, or dropping.

## Why I built this

I was testing different small models (1B-7B parameters) for a specific task and kept switching between notebooks, manual inspection, and ad-hoc scripts. I wanted a single tool that runs the same eval prompts against multiple models, scores outputs automatically, and generates a comparison report.

## What it does

1. Takes a YAML config with eval prompts and model settings
2. Runs each prompt against each model
3. Scores outputs (BLEU, ROUGE, or custom scoring)
4. Generates a markdown report with side-by-side comparisons

## What it doesn't do

- No cloud API calls (local models only)
- No training (just evaluation)
- No fancy UI (CLI + markdown reports)
- No GPU optimization (runs models one at a time)

## Quick start

```bash
pip install -r requirements.txt
python evalforge.py --config eval_config.yaml --output report.md
```

## Philosophy

See `docs/local_eval_philosophy.md` for why I think local eval matters for indie devs.


## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.