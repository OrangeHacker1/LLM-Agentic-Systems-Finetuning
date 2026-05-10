import json
import os

from config.config_loader import load_config

#from stage_2_training.model import load_stage2_new_model
from stage_2_training.model import load_stage2_merged_model

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


def main():
    print("===== Phase 4 Advanced Evaluation (Stage2) =====")

    config = load_config()

    output_dir = "results/phase4_model2"

    os.makedirs(output_dir, exist_ok=True)

    #
    # Load Stage2 model
    #
    print("Loading Stage2 model...")
    model, tokenizer = load_stage2_merged_model()

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
    # Build prediction/reference lists
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

        # Similarity metrics
        "rouge_l":
            rouge_l,

        "bertscore_f1":
            bertscore,

        "exact_match":
            exact_match,

        # JSON structural metric
        "schema_compliance":
            schema_compliance
    }

    #
    # Save metrics
    #
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    #
    # Save raw outputs
    #
    with open(f"{output_dir}/json_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    with open(f"{output_dir}/alpaca_results.json", "w") as f:
        json.dump(alpaca_results, f, indent=2)

    print("===== Stage2 Advanced Evaluation Complete =====")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
