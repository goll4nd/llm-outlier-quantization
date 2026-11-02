import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from models import MODELS


def load_bnb_4bit(key, quant_type="nf4", double_quant=True,
                  compute_dtype=torch.float16):
    """Загружает модель с реальным 4-битным хранением весов.

    quant_type: 'nf4' (нормальная сетка) или 'fp4' (равномерная).
    """
    name = MODELS.get(key, key)
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=cfg, device_map={"": 0}
    )
    model.eval()
    return model, tok