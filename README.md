# EvalForge

HELM is too rigid. SaaS eval platforms take your data. Ad-hoc scripts break every week.

EvalForge is a local-first toolkit for evaluating and comparing LLMs. Define benchmarks in YAML, pick your judge, run tournaments, export results. Your data stays on your machine.

```bash
evalforge run --model llama-3-8b --benchmark mmlu --judge gpt4
evalforge arena --models llama-3-8b,mistral-7b --rounds 50
```

## What it does

- **Benchmark evaluation** — Run models against MMLU, HumanEval, GSM8K, or your own custom benchmarks
- **Arena comparisons** — ELO tournament ranking with configurable judges (GPT-4, Claude, local model, or rule-based)
- **Batch inference** — GPU-accelerated batch generation with vLLM backend
- **Reports** — HTML/PDF reports with confidence intervals, category breakdowns, error analysis
- **CI integration** — `evalforge check --baseline results.json --model new_model` for regression testing

## Judges

You can judge with:
- **GPT-4 / Claude** — expensive but high quality
- **Local model** — run a judge LLM on your GPU (Llama 3, Mistral, etc.)
- **Rule-based** — exact match, regex, BLEU, ROUGE
- **Human** — export to a review interface

## Custom benchmarks

```yaml
# benchmarks/my_benchmark.yaml
name: my_coding_test
samples:
  - prompt: "Write a function to reverse a linked list"
    reference: |
      def reverse_list(head): ...
    judge: gpt4
    criteria: "correctness, efficiency, readability"
```

## Why local-first?

Because eval data is often proprietary. You're testing your fine-tuned models against your domain-specific prompts. Sending that to a third-party API is a non-starter for a lot of teams.

Also: no rate limits, no API costs for rule-based judges, works offline.

## Performance

Batch evaluation with vLLM backend:
- Llama 3 8B on RTX 4090: ~800 tok/s
- Mistral 7B on A100: ~1200 tok/s

Arena requires a judge API (or a local judge model). Budget accordingly.

## Install

```bash
pip install evalforge
```

Requires Python 3.10+. GPU optional but recommended for batch inference.

## Status

Alpha. I use it for my own model comparisons. The arena ELO calculation is stable. Custom benchmark support is new-ish — report bugs.

MIT License.


## Recent Updates
- Performance improvements for batch processing
- Better error messages for common issues