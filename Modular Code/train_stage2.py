from stage_2_training.model import load_stage1_model
from stage_2_training.dataset import load_and_prepare_dataset
from stage_2_training.trainer import get_trainer

from config.config_loader import load_config

from peft import LoraConfig, get_peft_model

import copy
import os
import torch


def main():
    print("===== Phase 3: Stage 2 Ablation Training =====")

    config = load_config()

    learning_rates = config["stage2"]["learning_rates"]

    print(f"Ablation Learning Rates: {learning_rates}")

    #
    # Load tokenizer + dataset ONCE
    #
    print("Loading tokenizer and dataset...")

    temp_model, tokenizer = load_stage1_model()

    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)

    # Free temporary model memory
    del temp_model
    torch.cuda.empty_cache()

    #
    # Run ablation for each LR
    #
    for lr in learning_rates:

        print(f"\n===== Training with LR = {lr} =====")

        #
        # Create isolated config
        #
        run_config = copy.deepcopy(config)

        # Override LR
        run_config["training"]["learning_rate"] = lr

        #
        # Unique output directory
        #
        lr_tag = str(lr).replace("-", "").replace(".", "")

        output_dir = f"./outputs/stage2_lr_{lr_tag}"

        os.makedirs(output_dir, exist_ok=True)

        run_config["training"]["output_dir"] = output_dir

        print(f"Output directory: {output_dir}")

        #
        # Load fresh Stage 1 model
        #
        print("Loading fresh Stage 1 model...")

        model, tokenizer = load_stage1_model()

        #
        # Create NEW Stage 2 LoRA adapter
        #
        print("Creating fresh Stage 2 LoRA adapter...")

        peft_cfg = config["lora"]

        lora_config = LoraConfig(
            r=peft_cfg["r"],
            lora_alpha=peft_cfg["alpha"],
            lora_dropout=peft_cfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=peft_cfg["target_modules"]
        )

        # ✅ ADD adapter properly (DO NOT wrap again)
        model.add_adapter("stage2", lora_config)

        # ✅ Activate it
        model.set_adapter("stage2")

        model.print_trainable_parameters()

        #
        # Initialize trainer
        #
        print("Initializing trainer...")

        trainer = get_trainer(
            model,
            train_dataset,
            eval_dataset,
            override_config=run_config
        )

        #
        # Train
        #
        print("Starting training...")

        trainer.train()

        #
        # Save adapter
        #
        print(f"Saving model at {output_dir}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"===== Completed LR = {lr} =====")

        #
        # Cleanup GPU memory
        #
        del model
        del trainer

        torch.cuda.empty_cache()

    print("\n===== Ablation Training Complete =====")


if __name__ == "__main__":
    main()