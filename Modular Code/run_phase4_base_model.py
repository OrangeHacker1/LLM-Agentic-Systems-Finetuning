import json
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from config.config_loader import load_config

from stage4.evaluator import (
    evaluate_new_json,
    evaluate_new_alpaca
)

from stage4.advanced_metrics import (
    compute_exact_match,
    compute_rouge_l,
    compute_bertscore,
    compute_schema_compliance
)


def load_base_model():
    config = load_config()

    model_name = config["model"]["name"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False

    return model, tokenizer


def main():
    print("===== Phase 4 Evaluation (Base Model) =====")

    config = load_config()

    output_dir = "results/phase4_base"

    os.makedirs(output_dir, exist_ok=True)

    #
    # Load model
    #
    print("Loading Base Model...")
    model, tokenizer = load_base_model()

    #
    # JSON Evaluation
    #
    print("Running JSON evaluation...")
    json_results = evaluate_new_json(
        model,
        tokenizer,
        config
    )

    #
    # Alpaca Evaluation
    #
    print("Running Alpaca evaluation...")
    alpaca_results = evaluate_new_alpaca(
        model,
        tokenizer,
        config
    )

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
    rouge_l = compute_rouge_l(
        predictions,
        references
    )

    print("Computing BERTScore...")
    bertscore = compute_bertscore(
        predictions,
        references
    )

    print("Computing Exact Match...")
    exact_match = compute_exact_match(
        predictions,
        references
    )

    print("Computing Schema Compliance...")
    schema_compliance = compute_schema_compliance(
        json_results
    )

    #
    # Aggregate Metrics
    #
    metrics = {

        # JSON metrics
        "json_local_valid_rate":
            sum(r["local_valid"] for r in json_results)
            / max(len(json_results), 1),

        "json_teacher_pass_rate":
            sum(r["teacher_pass"] for r in json_results)
            / max(len(json_results), 1),

        # Alpaca judge score
        "alpaca_avg_score":
            sum(r["score"] for r in alpaca_results)
            / max(len(alpaca_results), 1),

        # Text similarity metrics
        "rouge_l":
            rouge_l,

        "bertscore_f1":
            bertscore,

        "exact_match":
            exact_match,

        # JSON structure quality
        "schema_compliance":
            schema_compliance
    }

    #
    # Save Metrics
    #
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    #
    # Save Raw Outputs
    #
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(alpaca_results, f, indent=2)

    print("===== Base Model Evaluation Complete =====")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()