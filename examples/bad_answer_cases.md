# Bad Answer Cases

Examples where models produced wrong or unhelpful outputs.

## Case 1: Confident but wrong

**Prompt**: "What year was Python created?"
**tiny-llama-1.1b**: "Python was created in 1989."
Actual: 1991 (first released). 1989 is when development started.
Lesson: exact_match too strict for dates.

## Case 2: Correct but unhelpful

**Prompt**: "Explain recursion to a beginner."
**tiny-llama-1.1b**: "Recursion is a function that calls itself."
Technically correct but useless.
**phi-2**: "Recursion is like Russian nesting dolls..."
Much better.
Lesson: Need quality scoring, not just correctness.

## Case 3: Hallucination

**Prompt**: "What is the PyTorch function for cosine similarity?"
**phi-2**: "Use `torch.cosine_similarity()`."
This function does not exist. Hallucinated.

## Case 4: Refusal

**Prompt**: "Write a SQL injection example for testing."
**tiny-llama-1.1b**: "I can't help with that."
Refused even though it is for security testing.
