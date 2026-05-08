from stage_2_training.model import load_stage2_new_model
from stage4.evaluator import evaluate_new_json, evaluate_new_alpaca
from stage4.metrics import save_metrics

from config.config_loader import load_config

def main():
    print("===== Phase 4 Evaluation =====")

    config = load_config()

    print("Loading model...")
    model, tokenizer = load_stage2_new_model()

    print("Running JSON evaluation...")
    json_results =  evaluate_new_json(model, tokenizer, config)

    print("Running Alpaca evaluation...")
    alpaca_results = evaluate_new_alpaca(model, tokenizer, config)

    results = {
        "json": json_results,
        "alpaca": alpaca_results
    }

    print("Saving metrics...")
    save_metrics(results, config["evaluation"]["output_dir_new"])

    print("===== Evaluation Complete =====")


if __name__ == "__main__":
    main()