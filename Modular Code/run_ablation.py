import os
import json

from config.config_loader import load_config

from stage_2_training.model import load_trained_ablation_model

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
    print("===== Starting Ablation Study =====")

    config = load_config()

    lrs = config["stage2"]["learning_rates"]

    for lr in lrs:

        #
        # Build folder name
        #
        lr_tag = str(lr).replace("-", "").replace(".", "")

        model_path = f"./outputs/stage2_lr_{lr_tag}"

        output_dir = f"results/ablation_lr_{lr_tag}"

        #
        # Verify adapter exists
        #
        adapter_file = os.path.join(
            model_path,
            "adapter_config.json"
        )

        if not os.path.exists(adapter_file):
            print(f"Skipping {model_path} (missing adapter_config.json)")
            continue

        print(f"\n===== Evaluating LR={lr} =====")

        #
        # Load model
        #
        model, tokenizer = load_trained_ablation_model(model_path)


        print(f"Loaded ablation model from: {model_path}")

        if hasattr(model, "active_adapters"):
            print("Active adapters:", model.active_adapters)

        #
        # JSON evaluation
        #
        print("Running JSON evaluation...")

        json_results = evaluate_new_json(
            model,
            tokenizer,
            config
        )

        #
        # Alpaca evaluation
        #
        print("Running Alpaca evaluation...")

        alpaca_results = evaluate_new_alpaca(
            model,
            tokenizer,
            config
        )

        #
        # Build predictions/references
        #
        references = []
        predictions = []

        with open(config["evaluation"]["alpaca_eval_path"], "r") as f:
            data = [json.loads(line) for line in f]

        for ex, result in zip(data, alpaca_results):
            references.append(ex["output"])
            predictions.append(result["prediction"])

        #
        # Advanced metrics
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
        # Aggregate metrics
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

            # JSON structure metric
            "schema_compliance":
                schema_compliance
        }

        #
        # Create output folder
        #
        os.makedirs(output_dir, exist_ok=True)

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

        print(f"Saved results to {output_dir}")

    print("\n===== Ablation Study Complete =====")


if __name__ == "__main__":
    main()
