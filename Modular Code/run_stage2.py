from stage_2_training.model import load_stage1_merged_for_stage2_training

from peft import get_peft_model

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

    stage2_path = config["stage2"]["checkpoint"]["stage2_path_final"]

    print(f"Output Directory: {stage2_path}")
    output_dir = stage2_path

    os.makedirs(output_dir, exist_ok=True)

    print("Loading stage 1 merged model...")
    
    model, tokenizer = load_stage1_merged_for_stage2_training()

    """
    #
    # IMPORTANT:
    # Enable grads BEFORE adding adapter
    #
    model.enable_input_require_grads()
    """

    #
    # CREATE FRESH STAGE2 ADAPTER
    #
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
    """

    # ✅ ADD adapter properly (DO NOT wrap again)
    #model.add_adapter("stage2", lora_config)

    # ✅ Activate it
    #model.set_adapter("stage2")
    #
    # ADD NEW STAGE2 ADAPTER
    #
    model.add_adapter(
        lora_config,
        adapter_name="stage2"
    )


    #
    # ACTIVATE STAGE2
    #
    model.set_adapter("stage2")

    #
    # VERY IMPORTANT
    #
    model.enable_input_require_grads() 


    #
    # DISABLE CACHE
    #
    model.config.use_cache = False

    #
    # VERIFY TRAINABLE PARAMS
    #
    model.print_trainable_parameters()
    """

    #
    # Enable grads
    #
    model.enable_input_require_grads()

    #
    # IMPORTANT:
    # Create NEW PEFT wrapper
    #

    model = get_peft_model(
        model,
        lora_config
    )

    #
    # Enable grads again
    #
    model.enable_input_require_grads()

    #
    # Disable cache
    #
    model.config.use_cache = False

    #
    # Print trainable params
    #
    model.print_trainable_parameters()


    #
    # LOAD DATASET
    #
    print("Preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)

    #
    # TRAINER
    #
    print("Initializing trainer...")
    trainer = get_trainer(model, train_dataset, eval_dataset)

    #
    # TRAIN
    #
    print("Starting training...")
    trainer.train()

    
    #
    # SAVE ONLY STAGE2 ADAPTER
    #
    print(f"Saving model at {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Save Successful: {output_dir}")

    print("===== Phase 3 Complete =====")


if __name__ == "__main__":
    main()

