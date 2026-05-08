import json
from tqdm import tqdm
from stage4.teacher import call_teacher
from stage4.prompts import (
    build_prompt,
    build_json_judge_prompt,
    build_alpaca_judge_prompt
)
from stage4.utils import is_valid_json

import re
import torch

def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def evaluate_json(model, tokenizer, config):
    path = config["evaluation"]["json_eval_path"]
    max_new_tokens = config["evaluation"]["max_new_tokens"]

    results = []

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    for ex in tqdm(data, desc="JSON Eval"):
        prompt = build_prompt(ex)
        pred = generate(model, tokenizer, prompt, max_new_tokens)

        local_valid = is_valid_json(pred)

        judge_prompt = build_json_judge_prompt(ex["instruction"], pred)
        verdict = call_teacher(judge_prompt)

        teacher_pass = 1 if "PASS" in verdict else 0

        results.append({
            "instruction": ex["instruction"],
            "prediction": pred,
            "local_valid": int(local_valid),
            "teacher_pass": teacher_pass
        })

    return results



def evaluate_alpaca(model, tokenizer, config):
    path = config["evaluation"]["alpaca_eval_path"]
    max_new_tokens = config["evaluation"]["max_new_tokens"]

    results = []

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    for ex in tqdm(data, desc="Alpaca Eval"):
        prompt = build_prompt(ex)
        pred = generate(model, tokenizer, prompt, max_new_tokens)

        judge_prompt = build_alpaca_judge_prompt(
            ex["instruction"],
            ex["output"],
            pred
        )

        score = call_teacher(judge_prompt)

        try:
            score = float(score.strip())
        except:
            score = 0

        results.append({
            "instruction": ex["instruction"],
            "prediction": pred,
            "score": score
        })

    return results

#
#   NEW CODE
#
def extract_new_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def generate_new(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 Remove prompt from output
    generated = decoded[len(prompt):].strip()

    return generated


def evaluate_new_json(model, tokenizer, config):
    path = config["evaluation"]["json_eval_path"]
    max_new_tokens = config["evaluation"]["max_new_tokens"]

    results = []

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    for ex in tqdm(data, desc="JSON Eval"):
        prompt = build_prompt(ex)
        pred = generate_new(model, tokenizer, prompt, max_new_tokens)

        # 🔥 Extract JSON before validation
        #clean_pred = extract_new_json(pred)
        #local_valid = is_valid_json(clean_pred)
        local_valid = is_valid_json(pred)

        judge_prompt = build_json_judge_prompt(ex["instruction"], pred)
        verdict = call_teacher(judge_prompt)

        # 🔥 Safer PASS check
        teacher_pass = 1 if verdict.strip().upper().startswith("PASS") else 0

        results.append({
            "instruction": ex["instruction"],
            "prediction_raw": pred,
            "prediction_clean": pred,
            "local_valid": int(local_valid),
            "teacher_pass": teacher_pass
        })

    return results


def evaluate_new_alpaca(model, tokenizer, config):
    path = config["evaluation"]["alpaca_eval_path"]
    max_new_tokens = config["evaluation"]["max_new_tokens"]

    results = []

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    for ex in tqdm(data, desc="Alpaca Eval"):
        prompt = build_prompt(ex)
        pred = generate_new(model, tokenizer, prompt, max_new_tokens)

        judge_prompt = build_alpaca_judge_prompt(
            ex["instruction"],
            ex["output"],
            pred
        )

        score_text = call_teacher(judge_prompt)

        # 🔥 Robust number extraction
        match = re.search(r"\d+(\.\d+)?", score_text)
        score = float(match.group()) if match else 0

        results.append({
            "instruction": ex["instruction"],
            "prediction": pred,
            "score": score
        })

    return results
