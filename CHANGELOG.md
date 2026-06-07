# Changelog

## 0.4.0 — 2025-06-05
- Arena mode: ELO tournament with configurable rounds and judges
- New judge: reward model (any HuggingFace reward model as judge)
- CLI: `evalforge arena` for head-to-head model comparison
- Fixed: batch inference hanging on empty prompts

## 0.3.0 — 2025-05-20
- Custom benchmark support via YAML configs
- New judge type: embedding similarity (fast, no API cost)
- HTML report generation with category breakdowns
- Breaking: benchmark config format changed (v2 migration guide in docs/)

## 0.2.0 — 2025-05-01
- GPU-accelerated batch evaluation via vLLM backend
- Added GSM8K and HumanEval built-in benchmarks
- `evalforge compare` command for side-by-side model comparison
- Fixed: GPT-4 judge sometimes returning empty scores

## 0.1.0 — 2025-04-10
- Initial release
- MMLU benchmark with GPT-4 and rule-based judges
- Basic CLI: `evalforge run`
- JSON export
