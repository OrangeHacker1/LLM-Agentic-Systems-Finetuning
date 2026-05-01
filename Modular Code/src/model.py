import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from config.config_loader import load_config


def load_model():
    config = load_config()
    model_name = config["model"]["name"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Creating 4-bit quantization config...")
    """bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )"""
    bnb_config = None

    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    """

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #device_map="auto",
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True
        #torch_dtype=torch.float16
    )

    print("Applying LoRA...")
    peft_cfg = config["lora"]

    lora_config = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["alpha"],
        lora_dropout=peft_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=peft_cfg["target_modules"]
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer