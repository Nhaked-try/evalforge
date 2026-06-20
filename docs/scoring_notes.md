# Scoring Notes

## Methods implemented

### Exact match
- Compare output to expected string
- Case-insensitive, whitespace-normalized
- Good for: math, factual questions
- Bad for: open-ended, creative answers

### ROUGE-L
- Longest common subsequence between output and expected
- Good for: summarization, paraphrasing
- Bad for: code, structured output

### BLEU
- N-gram overlap
- Good for: translation
- Bad for: short answers, creative writing

### Contains code
- Check if output contains a code block with expected pattern
- Uses regex to find code blocks, then checks for keywords

## What I want to add

### LLM-as-judge
Use a stronger model to judge weaker model outputs.
Problem: Requires a good judge model. Circular if you are evaling the best model you have.

### Embedding similarity
Compare output embedding to expected embedding.
Good for: semantic similarity. Bad for: factual accuracy.
