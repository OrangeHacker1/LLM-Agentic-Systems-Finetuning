import json
import os

from config.config_loader import load_config

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import torch

from stage4.evaluator import (
    evaluate_new_json,
    evaluate_new_alpaca
)

from stage4.metrics import save_metrics

from stage4.advanced_metrics import (
    compute_exact_match,
    compute_rouge_l,
    compute_bertscore,
    compute_schema_compliance
)


def load_stage1_model():
    config = load_config()

    model_name = config["model"]["name"]
    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print("Loading Stage1 adapter...")
    model = PeftModel.from_pretrained(base, stage1_path)

    model.config.use_cache = False

    return model, tokenizer


def main():
    print("===== Phase 4 Evaluation (Stage1) =====")

    config = load_config()

    output_dir = "results/phase4_model1"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading Stage1 model...")
    model, tokenizer = load_stage1_model()

    #
    # JSON Evaluation
    #
    print("Running JSON evaluation...")
    json_results = evaluate_new_json(model, tokenizer, config)

    #
    # Alpaca Evaluation
    #
    print("Running Alpaca evaluation...")
    alpaca_results = evaluate_new_alpaca(model, tokenizer, config)

    #
    # Build references/predictions
    #
    references = []
    predictions = []

    with open(config["evaluation"]["alpaca_eval_path"], "r") as f:
        data = [json.loads(line) for line in f]

    for ex, result in zip(data, alpaca_results):
        references.append(ex["output"])
        predictions.append(result["prediction"])

    #
    # Advanced Metrics
    #
    print("Computing ROUGE-L...")
    rouge_l = compute_rouge_l(predictions, references)

    print("Computing BERTScore...")
    bertscore = compute_bertscore(predictions, references)

    print("Computing Exact Match...")
    exact_match = compute_exact_match(predictions, references)

    print("Computing Schema Compliance...")
    schema_compliance = compute_schema_compliance(json_results)

    #
    # Standard Metrics
    #
    metrics = {
        "json_local_valid_rate":
            sum(r["local_valid"] for r in json_results) / max(len(json_results), 1),

        "json_teacher_pass_rate":
            sum(r["teacher_pass"] for r in json_results) / max(len(json_results), 1),

        "alpaca_avg_score":
            sum(r["score"] for r in alpaca_results) / max(len(alpaca_results), 1),

        "rouge_l":
            rouge_l,

        "bertscore_f1":
            bertscore,

        "exact_match":
            exact_match,

        "schema_compliance":
            schema_compliance
    }

    #
    # Save metrics
    #
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    #
    # Save outputs
    #
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(alpaca_results, f, indent=2)

    print("===== Stage1 Evaluation Complete =====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()