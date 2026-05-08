from stage_2_training.model import load_stage2_model
from stage_2_training.model import load_stage1_model
from stage_2_training.dataset import load_and_prepare_dataset
from stage_2_training.trainer import get_trainer
#from stage_2_training.evaluator import evaluate_model

# LORA CONFIG
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from config.config_loader import load_config

import os

def main():
    print("===== Phase 3: Stage 2 Training =====")

    config = load_config()

    stage2_path = config["stage2"]["checkpoint"]["stage2_path"]

    print(f"Output Directory: {stage2_path}")
    output_dir = stage2_path

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_stage1_model()

    print("Making new Lora Layer...")
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

    print("Preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)

    print("Initializing trainer...")
    trainer = get_trainer(model, train_dataset, eval_dataset)

    print("Starting training...")
    trainer.train()

    #rint("Saving model...")
    #trainer.save_model()
    print(f"Saving model at {output_dir}")

    #output_dir = stage2_path

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Save Successful: {output_dir}")

    # Phase 4
    #print("Running evaluation...")
    #evaluate_model(model, tokenizer)

    print("===== Phase 3 Complete =====")


if __name__ == "__main__":
    main()

