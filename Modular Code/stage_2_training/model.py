# stage_2_training/model.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
from config.config_loader import load_config


def load_stage1_model():
    config = load_config()

    model_name = config["model"]["name"]
    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]
    #stage2_path = config["stage2"]["checkpoint"]["stage2_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Path to model: {stage1_path}")

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

    print(f"Loading Stage 1 adapter from {stage1_path}...")
    model = PeftModel.from_pretrained(model, stage1_path)

    # 🔥 Step 2: Load Stage 2 adapter (Default)
    #print(f"Loading Stage 2 adapter from {stage2_path}...")
    #model = PeftModel.from_pretrained(model, stage2_path)

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.config.use_cache = False

    return model, tokenizer

#
#   OLD MODEL
#
def load_stage2_model():
    config = load_config()

    model_name = config["model"]["name"]
    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]
    stage2_path = config["stage2"]["checkpoint"]["stage2_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Path to model: {stage2_path}")

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

    print(f"Loading Stage 1 adapter from {stage1_path}...")
    model = PeftModel.from_pretrained(model, stage1_path)

    # 🔥 Step 2: Load Stage 2 adapter (Default)
    print(f"Loading Stage 2 adapter from {stage2_path}...")
    model = PeftModel.from_pretrained(model, stage2_path)

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.config.use_cache = False

    return model, tokenizer

#
#   This should load the lora correcctly.
#
def load_stage2_new_model():
    config = load_config()

    model_name = config["model"]["name"]
    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]
    stage2_path = config["stage2"]["checkpoint"]["stage2_path"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Path to model: {stage2_path}")

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

    print(f"Loading Stage 1 adapter from {stage1_path}...")
    model = PeftModel.from_pretrained(model, stage1_path, adapter_name="stage1")

    # 🔥 Step 2: Load Stage 2 adapter (Default)
    #print(f"Loading Stage 2 adapter from {stage2_path}...")
    
    # ✅ Load Stage 2 adapter (DO NOT wrap again)
    print(f"Loading Stage 2 adapter from {stage2_path}...")
    model.load_adapter(
        stage2_path,
        adapter_name="stage2"
    )

    # ✅ Activate Stage 2
    model.set_adapter("stage2")
    model.config.use_cache = False

    return model, tokenizer


#
#   This should load the trained model.
#
def load_stage2_merged_model():
    """
    Correct inference loader.

    Steps:
    1. Load base model
    2. Load Stage1 adapter
    3. Merge Stage1 into base model
    4. Load Stage2 adapter
    5. Activate Stage2
    """

    config = load_config()

    model_name = config["model"]["name"]

    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    stage2_path = config["stage2"]["checkpoint"]["stage2_path"]

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    #
    # Load Stage1 adapter
    #
    print(f"Loading Stage1 adapter from {stage1_path}...")

    stage1_model = PeftModel.from_pretrained(
        base_model,
        stage1_path
    )

    #
    # Merge Stage1 into base model
    #
    print("Merging Stage1 weights into base model...")

    merged_model = stage1_model.merge_and_unload()

    #
    # Load Stage2 adapter
    #
    print(f"Loading Stage2 adapter from {stage2_path}...")

    final_model = PeftModel.from_pretrained(
        merged_model,
        stage2_path
    )

    #
    # Inference settings
    #
    final_model.config.use_cache = False

    #
    # IMPORTANT:
    # Put model into inference mode
    #
    final_model.eval()

    return final_model, tokenizer

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

    #
    # Load Stage1 adapter
    #
    print(f"Loading Stage1 adapter from {stage1_path}...")

    model = PeftModel.from_pretrained(
        model,
        stage1_path,
        adapter_name="stage1"
    )

    #
    # Load Stage2 ablation adapter
    #
    print(f"Loading Stage2 adapter from {stage2_adapter_path}...")

    model.load_adapter(
        stage2_adapter_path,
        adapter_name="stage2"
    )

    #
    # Activate Stage2
    #
    model.set_adapter("stage2")

    model.config.use_cache = False

    return model, tokenizer

#
#   Merged Model for stage2 training.
#
def load_stage1_merged_for_stage2_training():
    """
    Correct Stage2 training loader.

    Pipeline:
        1. Load base model
        2. Load Stage1 adapter
        3. Merge Stage1 into base weights
        4. Return merged model
           (NO active adapters)

    This is the correct way to perform
    sequential LoRA training.
    """

    config = load_config()

    model_name = config["model"]["name"]

    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    #
    # Load Stage1 adapter
    #
    print(f"Loading Stage1 adapter from {stage1_path}...")

    stage1_model = PeftModel.from_pretrained(
        base_model,
        stage1_path
    )

    #
    # Merge Stage1 into base model
    #
    print("Merging Stage1 into base model...")

    merged_model = stage1_model.merge_and_unload()

    merged_model.config.use_cache = False

    print("Stage1 merged successfully.")

    return merged_model, tokenizer


def load_trained_ablation_model(stage2_adapter_path):
    """
    Correct ablation inference loader.

    This loader properly handles sequential LoRA training.

    Training flow:
        Base Model
            -> Stage1 LoRA training
            -> Stage2 LoRA training (trained ON TOP OF Stage1)

    Correct inference flow:
        1. Load base model
        2. Load Stage1 adapter
        3. Merge Stage1 into base weights
        4. Load ablation Stage2 adapter
        5. Return final model

    This prevents the common PEFT issue where:
        model.set_adapter("stage2")

    accidentally disables Stage1 behavior.

    Each ablation model will now truly represent:
        Base + Stage1 + Specific Stage2 LR
    """

    config = load_config()

    model_name = config["model"]["name"]

    stage1_path = config["stage2"]["checkpoint"]["stage1_path"]

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    #
    # STEP 1:
    # Load Stage1 adapter
    #
    print(f"Loading Stage1 adapter from {stage1_path}...")

    stage1_model = PeftModel.from_pretrained(
        base_model,
        stage1_path
    )

    #
    # STEP 2:
    # Merge Stage1 into base model
    #
    print("Merging Stage1 weights into base model...")

    merged_model = stage1_model.merge_and_unload()

    #
    # STEP 3:
    # Load Stage2 ablation adapter
    #
    print(f"Loading Stage2 ablation adapter from {stage2_adapter_path}...")

    final_model = PeftModel.from_pretrained(
        merged_model,
        stage2_adapter_path
    )

    final_model.config.use_cache = False

    print("Ablation model loaded successfully.")
    print("Active model = Base + Stage1 merged + Stage2 ablation")

    return final_model, tokenizer


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
