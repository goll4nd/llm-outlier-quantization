import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "opt-125m":  "facebook/opt-125m",
    "opt-350m":  "facebook/opt-350m",
    "tinyllama": "TinyLlama/TinyLlama_v1.1",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama3-8b": "NousResearch/Meta-Llama-3-8B",
}

def load_model(key, device="cuda", dtype=torch.float16):
    """Загружает модель и токенизатор. key — короткое имя из MODELS."""
    name = MODELS.get(key, key)
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return model, tok