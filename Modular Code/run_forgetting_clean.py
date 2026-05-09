import json
import os
import torch
from tqdm import tqdm

from config.config_loader import load_config
from stage4.teacher import call_teacher
from stage_2_training.model import load_stage2_merged_model, load_stage1_model

from stage4.evaluator import generate_new

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from stage4.prompts import (
    build_prompt,
    build_forgetting_prompt
)



def main():
    print("Running Forgetting Analysis...")

    config = load_config()

    max_new_tokens = config["evaluation"]["max_new_tokens"]
    data_path = config["evaluation"]["alpaca_eval_path"]

    # Load models
    print("Loading Stage 1 model...")
    model1, tokenizer = load_stage1_model()

    print("Loading Stage 2 model...")
    model2, _ = load_stage2_merged_model()

    # Load eval data
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    results = []

    for ex in tqdm(data, desc="Forgetting Eval"):
        prompt = build_prompt(ex)

        out1 = generate_new(model1, tokenizer, prompt, max_new_tokens)
        out2 = generate_new(model2, tokenizer, prompt, max_new_tokens)

        judge_prompt = build_forgetting_prompt(
            ex["instruction"],
            ex.get("output", ""),
            out1,
            out2
        )

        verdict = call_teacher(judge_prompt)

        verdict_clean = verdict.strip().upper()

        if verdict_clean == "A":
            winner = "checkpoint1"
        elif verdict_clean == "B":
            winner = "checkpoint2"
        else:
            winner = "tie"

        results.append({
            "instruction": ex["instruction"],
            "output_stage1": out1,
            "output_stage2": out2,
            "teacher_verdict": verdict_clean,
            "winner": winner
        })

    # Aggregate
    counts = {
        "checkpoint1": sum(r["winner"] == "checkpoint1" for r in results),
        "checkpoint2": sum(r["winner"] == "checkpoint2" for r in results),
        "tie": sum(r["winner"] == "tie" for r in results),
    }

    total = max(sum(counts.values()), 1)

    report = {
        "checkpoint1_win_rate": counts["checkpoint1"] / total,
        "checkpoint2_win_rate": counts["checkpoint2"] / total,
        "tie_rate": counts["tie"] / total,

        # TRUE forgetting condition
        "forgetting_detected": counts["checkpoint1"] > counts["checkpoint2"]
    }

    path = config["evaluation"]["output_dir_new"]

    os.makedirs(path, exist_ok=True)

    with open(f"{path}/forgetting_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{path}/forgetting_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Forgetting Report:", report)


if __name__ == "__main__":
    main()
