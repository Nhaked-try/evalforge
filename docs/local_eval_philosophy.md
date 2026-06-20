# Why Local Eval Matters

## The problem with cloud eval

Most eval tools assume you are using cloud APIs. That works for big companies with API budgets. It does not work for indie devs testing small open-source models.

## My workflow

1. Find a new model on Hugging Face
2. Download it (GGUF or HF format)
3. Run evalforge with my standard eval set
4. Compare against my current model
5. Decide: keep, fine-tune, or drop

This takes 10-30 minutes, costs nothing beyond compute, and gives me concrete numbers instead of vibes.

## What good enough means

I do not need state-of-the-art eval. I need:
- Consistent comparison (same prompts, same settings)
- Fast iteration (run eval in minutes, not hours)
- Clear output (side-by-side comparison, not a single score)

## The indie dev advantage

Big companies eval at scale. I eval at depth - small number of prompts, but I inspect every output manually. The automated scoring is a filter, not a judge.
