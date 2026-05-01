def build_json_judge_prompt(instruction, prediction):
    return f"""
You are a strict JSON validator.

Instruction:
{instruction}

Model Output:
{prediction}

Check:
1. Is valid JSON?
2. Matches expected structure?
3. No hallucinated keys?

Return ONLY:
PASS or FAIL
"""


def build_alpaca_judge_prompt(instruction, reference, prediction):
    return f"""
You are an expert evaluator.

Instruction:
{instruction}

Reference Answer:
{reference}

Model Answer:
{prediction}

Score the model answer from 1 to 10 based on:
- correctness
- completeness
- clarity

Return ONLY a number (1-10).
"""

def build_forgetting_prompt(instruction, reference, output_a, output_b):
    return f"""
You are comparing two model outputs.

Instruction:
{instruction}

Reference Answer:
{reference}

Model A (Checkpoint 1):
{output_a}

Model B (Checkpoint 2):
{output_b}

Which is better?

Return ONLY:
A or B or TIE
"""


def build_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")

    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
"""