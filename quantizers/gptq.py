import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from models import MODELS
from evaluation.calib import get_calibration_samples


def quantize_gptq(model_key, bits=4, group_size=128, desc_act=False,
                  n_samples=128, seed=0, save_dir=None):
    """Квантует модель методом GPTQ. Возвращает путь к сохранённой модели."""
    name = MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)

    cfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,   # порядок обработки каналов по важности
        sym=False,
    )

    model = GPTQModel.load(name, cfg)
    calib = get_calibration_samples(tok, n_samples=n_samples, seed=seed)
    model.quantize(calib)

    save_dir = save_dir or f"quantized/{model_key}-gptq-w{bits}-g{group_size}-s{seed}"
    model.save(save_dir)
    tok.save_pretrained(save_dir)

    del model
    torch.cuda.empty_cache()
    return save_dir


def load_gptq(path, device="cuda"):
    model = GPTQModel.load(path, device=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    return model, tok