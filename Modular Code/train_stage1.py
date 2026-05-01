#
#   STAGE 1 TRAINING
#

from src.model import load_model
from src.data import load_and_prepare_data
from src.trainer import get_trainer

def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Preparing dataset...")
    train_data, eval_data = load_and_prepare_data(tokenizer)

    print("Initializing trainer...")
    trainer = get_trainer(model, tokenizer, train_data, eval_data)

    print("Starting training...")
    trainer.train()

    print("Saving adapter...")
    model.save_pretrained("./outputs/stage1_adapter")
    tokenizer.save_pretrained("./outputs/stage1_adapter")

if __name__ == "__main__":
    main()