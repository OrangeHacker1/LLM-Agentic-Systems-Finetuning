from stage_2_training.model import load_stage2_model
from stage_2_training.dataset import load_and_prepare_dataset
from stage_2_training.trainer import get_trainer
#from stage_2_training.evaluator import evaluate_model


def main():
    print("===== Phase 3: Stage 2 Training =====")

    print("Loading model...")
    model, tokenizer = load_stage2_model()

    print("Preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)

    print("Initializing trainer...")
    trainer = get_trainer(model, train_dataset, eval_dataset)

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()

    # Phase 4
    #print("Running evaluation...")
    #evaluate_model(model, tokenizer)

    print("===== Phase 3 Complete =====")


if __name__ == "__main__":
    main()




"""
from config.config_loader import load_config
from config.env_loader import load_environment

from stage_2_training.dataset import load_json_dataset, preprocess_dataset
from stage_2_training.model import load_model
from stage_2_training.train import setup_trainer
from stage_2_training.evaluator import evaluate_model


def main():
    # 1. Load config + environment
    config = load_config()
    env = load_environment()

    stage2_cfg = config["stage2"]

    # 2. Load dataset
    dataset = load_json_dataset(stage2_cfg["data"]["path"])

    # 3. Load model
    model, tokenizer = load_model(config, stage2_cfg)

    # 4. Preprocess dataset
    dataset = preprocess_dataset(dataset, tokenizer, config)

    # 5. Setup trainer
    trainer = setup_trainer(model, tokenizer, dataset, config)

    # 6. Train
    trainer.train()

    # 7. Save
    trainer.save_model(stage2_cfg["output_dir"])

    # 8. Evaluate
    evaluate_model(model, tokenizer, config, env)


if __name__ == "__main__":
    main()"""


"""""
from config.config_loader import load_config
from config.env_loader import load_environment

from stage_2_training.dataset import load_json_dataset, preprocess_dataset
from stage_2_training.model import load_model
from stage_2_training.train import setup_trainer
from stage_2_training.evaluator import evaluate_model


def main():
    # 1. Load config + environment
    config = load_config()
    env = load_environment()

    stage2_cfg = config["stage2"]

    # 2. Load dataset
    dataset = load_json_dataset(stage2_cfg["data"]["path"])

    # 3. Load model
    model, tokenizer = load_model(config, stage2_cfg)

    # 4. Preprocess dataset
    dataset = preprocess_dataset(dataset, tokenizer, config)

    # 5. Setup trainer
    trainer = setup_trainer(model, tokenizer, dataset, config)

    # 6. Train
    trainer.train()

    # 7. Save
    trainer.save_model(stage2_cfg["output_dir"])

    # 8. Evaluate
    evaluate_model(model, tokenizer, config, env)


if __name__ == "__main__":
    main()
    """