# stage_2_training/model.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
from config.config_loader import load_config


def load_stage2_model():
    config = load_config()

    model_name = config["model"]["name"]
    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating 8-bit quantization config...")
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
    #"""

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #device_map="auto",
        device_map={"": 0},
        #dtype=torch.float16,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    """

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )"""

    print("Loading Stage 1 adapter...")
    model = PeftModel.from_pretrained(model, stage1_path)

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.config.use_cache = False

    return model, tokenizer

# ==============================
# NEW: Load Ablation Model
# ==============================

def load_ablation_model(stage2_adapter_path):
    config = load_config()

    model_name = config["model"]["name"]
    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # 🔥 Step 1: Load Stage 1 adapter
    print("Loading Stage 1 adapter...")
    model = PeftModel.from_pretrained(model, stage1_path)

    # 🔥 Step 2: Load Stage 2 adapter (ablation model)
    print(f"Loading Stage 2 adapter from {stage2_adapter_path}...")
    model = PeftModel.from_pretrained(model, stage2_adapter_path)

    model.config.use_cache = False

    return model, tokenizer

def load_model(config, stage2_cfg):
    config = load_config()
    model_name = config["model"]["name"]
    stage1_path = stage2_cfg["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token

    #tokenizer.padding_side = "right"

    print("Creating 8-bit quantization config...")
    """bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )"""
    #bnb_config = None

    #"""
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    #"""

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

    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True
    )"""

    # Load Stage 1 adapter
    model = PeftModel.from_pretrained(model, stage1_path)

    return model, tokenizer












"""""

# stage_2_training/model.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
from config.config_loader import load_config


def load_model(config, stage2_cfg):
    config = load_config()
    model_name = config["model"]["name"]
    stage1_path = stage2_cfg["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token

    #tokenizer.padding_side = "right"

    print("Creating 8-bit quantization config...")
    " ""bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )" ""
    bnb_config = None

    " ""
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    " ""

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

    " ""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True
    )" ""

    # Load Stage 1 adapter
    model = PeftModel.from_pretrained(model, stage1_path)

    return model, tokenizer"""