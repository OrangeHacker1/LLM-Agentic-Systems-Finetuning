import os
import yaml
from stage4.evaluator import evaluate_json, evaluate_alpaca
from stage4.metrics import save_metrics
from src.model import load_model
from config.config_loader import load_config
from stage_2_training.model import load_ablation_model


def main():
    config = load_config()

    lrs = config["stage2"]["learning_rates"]

    #lrs = [
        #2e-5,   # baseline   # Done
     #   5e-5,   # medium
      #  1e-4    # strong
    #]

    all_results = {}

    for lr in lrs:
        lr_tag = str(lr).replace("-", "").replace(".", "")
        model_path = f"./outputs/stage2_lr_{lr_tag}"

        # Error handling
        if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"⚠️ Skipping {model_path} (missing adapter_config.json)")
            continue

        print(f"\n===== Evaluating LR={lr} =====")

        model, tokenizer = load_ablation_model(model_path)

        print("Running JSON evaluation...")
        json_results = evaluate_json(model, tokenizer, config)
        print("Running Alpaca evaluation...")
        alpaca_results = evaluate_alpaca(model, tokenizer, config)

        results = {
            "json": json_results,
            "alpaca": alpaca_results
        }

        save_metrics(results, f"results/ablation_lr_{lr_tag}")

        all_results[str(lr)] = results

    print("\n===== Ablation Complete =====")

if __name__ == "__main__":
    main()