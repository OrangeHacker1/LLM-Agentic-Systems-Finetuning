# JUDGE

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalGenerator:
    def __init__(self, model_path):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    def generate(self, prompt, max_new_tokens=256):
        x = self.tok(prompt, return_tensors='pt').to(self.model.device)
        y = self.model.generate(**x, max_new_tokens=max_new_tokens)
        return self.tok.decode(y[0], skip_special_tokens=True)