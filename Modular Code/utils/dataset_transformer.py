# utils/dataset_transformer.py

import random

from prompts.custom_prompts import CUSTOM_PROMPTS

def convert_to_json_task(example):
    """
    Convert Alpaca example into JSON-style task
    """

    instruction = example["instruction"]
    input_text = example.get("input", "")

    # Map to task types
    task_type = random.choice([
        "extraction",
        "classification",
        "schema",
        "repair",
        "tool"
    ])

    # Create new JSON-oriented prompt
    new_input = f"""
Instruction: {instruction}
Input: {input_text}

Convert the above into a structured JSON response.
"""

    return task_type, new_input



def sample_custom_prompt():
    """
    Randomly select a task and prompt from custom prompt pool
    """

    task_type = random.choice(list(CUSTOM_PROMPTS.keys()))
    prompt = random.choice(CUSTOM_PROMPTS[task_type])

    return task_type, prompt