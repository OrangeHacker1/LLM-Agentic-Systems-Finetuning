#
#   STAGE 1 TRAINING
#

from src.model import load_model
from src.data import load_and_prepare_data
from src.trainer import get_trainer

from config.config_loader import load_config

import os

def main():

    print("===== Phase 2: Stage 1 Training =====")

    config = load_config()

    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    print(f"Output Directory: {stage1_path}")
    output_dir = stage1_path

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model()

    print("Preparing dataset...")
    train_data, eval_data = load_and_prepare_data(tokenizer)

    print("Initializing trainer...")
    trainer = get_trainer(model, tokenizer, train_data, eval_data)

    print("Starting training...")
    trainer.train()

    print("Saving adapter...")
    #model.save_pretrained("./outputs/stage1_adapter")
    #tokenizer.save_pretrained("./outputs/stage1_adapter")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Save Successful: {output_dir}")

if __name__ == "__main__":
    main()