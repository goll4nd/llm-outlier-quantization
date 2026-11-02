import torch
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization import METHOD
from gptqmodel.utils.backend import BACKEND
from transformers import AutoTokenizer
from models import MODELS
from evaluation.calib import get_calibration_samples

# НЕ РАБОТАЕТ. Эталонное торчевое ядро AWQ в GPTQModel 7.7.0 даёт
# ppl ~2300-2900 вместо ожидаемых ~29. Подробности в README,
# раздел "Ограничения". Специализированные CUDA-ядра требуют
# CUDA Toolkit, которого нет в окружении.


def quantize_awq(model_key, bits=4, group_size=128,
                 n_samples=128, seed=0, save_dir=None):
    name = MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)

    cfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        bits=bits,
        group_size=group_size,
        sym = False,
    )

    model = GPTQModel.load(name, cfg)
    calib = get_calibration_samples(tok, n_samples=n_samples, seed=seed)
    model.quantize(calib, backend=BACKEND.AWQ_TORCH)

    save_dir = save_dir or f"quantized/{model_key}-awq-w{bits}-g{group_size}-n{n_samples}-s{seed}"
    model.save(save_dir)
    tok.save_pretrained(save_dir)

    del model
    torch.cuda.empty_cache()
    return save_dir


def load_awq(path, device="cuda"):
    model = GPTQModel.load(path, device=device, backend=BACKEND.AWQ_TORCH)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    return model, tok