# stage_2_training/dataset.py
from datasets import load_dataset
from config.config_loader import load_config


def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}


def tokenize_function(example, tokenizer, max_length):
    enc = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    labels = enc["input_ids"].copy()
    labels = [
        token if token != tokenizer.pad_token_id else -100
        for token in labels
    ]

    enc["labels"] = labels
    return enc


def load_and_prepare_dataset(tokenizer):
    config = load_config()
    data_cfg = config["dataset"]
    model_cfg = config["model"]

    print("Loading dataset...")
    ds = load_dataset(
        "json",
        data_files=data_cfg["path"]
    )["train"]

    max_samples = data_cfg.get("max_samples")
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    print("Splitting dataset...")
    split = ds.train_test_split(
        test_size=data_cfg["eval_split"],
        seed=42,
        shuffle=True
    )

    print("Formatting dataset...")
    split = split.map(format_example, load_from_cache_file=False)

    print("Tokenizing dataset...")
    split = split.map(
        lambda x: tokenize_function(
            x,
            tokenizer,
            model_cfg["max_length"]
        ),
        remove_columns=split["train"].column_names,
        load_from_cache_file=False
    )

    split.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return split["train"], split["test"]



"""
import json
from datasets import Dataset


def load_json_dataset(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_example(example):
    return f"" "
### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output_str']}
"" "


def preprocess_dataset(dataset, tokenizer, config):
    max_length = config["model"]["max_length"]

    def tokenize(example):
        text = format_example(example)

        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        labels = enc["input_ids"].copy()
        labels = [
            token if token != tokenizer.pad_token_id else -100
            for token in labels
        ]

        enc["labels"] = labels
        return enc

    dataset = dataset.map(tokenize, load_from_cache_file=False)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return dataset"""