# Contributing

Want to make LLM evaluation less painful? Cool.

## What I'd love help with

- More judge implementations (Anthropic, Cohere, local reward models)
- Additional built-in benchmarks (TruthfulQA, HellaSwag, ARC)
- Better ELO rating math (current implementation is basic Bradley-Terry)
- CLI UX improvements
- Documentation (I know, I know)

## Setup

```bash
git clone https://github.com/Nhaked-try/evalforge.git
pip install -e ".[dev]"
```

## Before you PR

- Run `pytest test/` — all tests should pass
- If you add a judge type, add tests in `test/test_judges.py`
- If you add a benchmark, add a sample config in `cfg/`
- Keep it local-first — no cloud-only features in the core package

## Code style

Black + ruff. That's it. No opinionated linting rules beyond that.

## Design philosophy

EvalForge exists because existing eval tools are either too rigid (HELM) or too expensive (SaaS). Keep that in mind — flexibility and local-first are non-negotiable.
