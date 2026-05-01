from stage_2_training.model import load_stage2_model
from stage_2_training.dataset import load_and_prepare_dataset
from stage_2_training.trainer import get_trainer
from config.config_loader import load_config
import copy
import os


def main():
    print("===== Phase 3: Stage 2 Ablation Training =====")

    config = load_config()

    # 🔥 Define learning rates for ablation

    learning_rates = config["stage2"]["learning_rates"]

    # Maual
    #learning_rates = [
    #    2e-5,   # baseline
    #    5e-5,   # medium
    #    1e-4    # strong
    #]

    print(f"Ablation Learning Rates: {learning_rates}")

    # Load dataset ONCE (efficient)
    print("Preparing dataset...")
    # tokenizer needed → load temp model
    temp_model, tokenizer = load_stage2_model()
    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)
    del temp_model  # free memory

    for lr in learning_rates:
        print(f"\n===== Training with LR = {lr} =====")

        # Deep copy config so runs don't interfere
        run_config = copy.deepcopy(config)

        # 🔥 Override learning rate
        run_config["training"]["learning_rate"] = lr

        # 🔥 Unique output directory
        lr_tag = str(lr).replace("-", "").replace(".", "")
        output_dir = f"./outputs/stage2_lr_{lr_tag}"
        os.makedirs(output_dir, exist_ok=True)

        run_config["training"]["output_dir"] = output_dir

        print(f"Output directory: {output_dir}")

        # Load fresh model EACH run
        print("Loading model...")
        model, tokenizer = load_stage2_model()

        print("Initializing trainer...")
        trainer = get_trainer(
            model,
            train_dataset,
            eval_dataset,
            override_config=run_config   # 🔥 NEW
        )

        print("Starting training...")
        trainer.train()

        #rint("Saving model...")
        #trainer.save_model(output_dir)
        print("Saving model...")
        output_dir = f"./outputs/stage2_lr_{lr_tag}"

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"===== Completed LR = {lr} =====")

    print("===== Ablation Training Complete =====")


if __name__ == "__main__":
    main()
