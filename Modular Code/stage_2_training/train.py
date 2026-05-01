# stage_2_training/train.py

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


def setup_trainer(model, tokenizer, dataset, config):
    """
    Builds and returns the Hugging Face Trainer used for Stage 2 fine-tuning.

    Responsibilities:
    - Reads Stage 2 hyperparameters from config
    - Creates TrainingArguments
    - Creates a causal language modeling data collator
    - Returns a Trainer ready to run trainer.train()
    """

    training_cfg = config["stage2"]["training"]
    output_dir = config["stage2"]["output_dir"]
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],#   training_cfg.get("eval_batch_size", training_cfg["batch_size"]),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],#   training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=float(training_cfg["learning_rate"]),
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        data_seed=42,
        save_total_limit=2,

        # Help prevent freezes.
        dataloader_num_workers=0,
        dataloader_pin_memory=False

        logging_dir=f"{output_dir}/logs"
    )

    """
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    """
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator
    )

    return trainer