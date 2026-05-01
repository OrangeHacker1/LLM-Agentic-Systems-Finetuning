import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"


def load_checkpoint(checkpoint_name):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_4bit=True
    )

    if checkpoint_name != "base":
        path = f"./outputs/{checkpoint_name}"
        model = PeftModel.from_pretrained(model, path)

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)