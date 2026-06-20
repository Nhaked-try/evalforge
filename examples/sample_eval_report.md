# Eval Report: 2026-06-15

## Models compared

| Model | Parameters | Quantization |
|-------|-----------|--------------|
| tiny-llama-1.1b | 1.1B | FP16 |
| phi-2 | 2.7B | FP16 |

## Results

### math_1: "What is 15% of 200?"

**tiny-llama-1.1b**: "15% of 200 is 30."
Correct

**phi-2**: "To find 15% of 200, multiply 200 by 0.15, which gives you 30."
Correct

### code_1: "Write a one-liner to remove duplicates from a Python list."

**tiny-llama-1.1b**: "You can use `list(set(my_list))` to remove duplicates."
Correct

**phi-2**: "One approach: `[*set(my_list)]`. Another: `list(dict.fromkeys(my_list))` (preserves order)."
Correct, gives alternatives

### reasoning_1: Logic puzzle

**tiny-llama-1.1b**: "Yes, some roses fade quickly because all roses are flowers."
Incorrect reasoning

**phi-2**: "No, we cannot conclude that. The premise says 'some flowers' - we don't know if those include roses."
Correct

## Summary

| Model | Math | Code | Reasoning | Overall |
|-------|------|------|-----------|---------|
| tiny-llama-1.1b | Yes | Yes | No | 2/3 |
| phi-2 | Yes | Yes | Yes | 3/3 |

**Winner**: phi-2 (better reasoning)
