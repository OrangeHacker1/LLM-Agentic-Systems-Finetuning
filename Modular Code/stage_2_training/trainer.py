from transformers import Trainer, TrainingArguments, default_data_collator
from config.config_loader import load_config
import torch


def get_trainer(model, train_dataset, eval_dataset, override_config=None):

    # 🔥 Use override config if provided
    if override_config is not None:
        config = override_config
    else:
        config = load_config()

    train_cfg = config["training"]

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    print("Initializing TrainingArguments...")

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=float(train_cfg["learning_rate"]),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy="steps",   # already fixed
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        data_seed=42,
        save_total_limit=2,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator
    )

    return trainer

"""

from transformers import Trainer, TrainingArguments, default_data_collator
from config.config_loader import load_config
import torch


def get_trainer(model, train_dataset, eval_dataset):
    config = load_config()
    train_cfg = config["training"]

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    print("Initializing TrainingArguments...")

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=float(train_cfg["learning_rate"]),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy="steps",   # IMPORTANT FIX
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        data_seed=42,
        save_total_limit=2,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator
    )

    return trainer"""