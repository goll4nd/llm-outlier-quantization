"""Послойная оценка perplexity для моделей, которые не помещаются в VRAM.

Идея та же, что в llama_eval / opt_eval из репозитория GPTQ:
  1. Модель создаётся «пустой» на устройстве meta — память под веса не выделяется.
  2. Веса читаются из safetensors по одному блоку (memory-mapped, без загрузки
     всей модели в RAM).
  3. Скрытые состояния всех окон WikiText-2 лежат в одном буфере; каждый
     трансформерный блок по очереди загружается на GPU, (опционально)
     квантуется, прогоняет через себя все окна и выгружается.
  4. В конце final norm + lm_head и тот же подсчёт NLL, что в evaluation/ppl.py.

Протокол (данные, нарезка на окна 2048, float32-логиты, лосс кусками по 512,
exp(total_nll / total_tokens)) совпадает с evaluate_perplexity один в один,
поэтому числа напрямую сравнимы с результатами на малых моделях.

Поддерживаются Llama-подобные архитектуры: model.model.{embed_tokens, layers,
norm, rotary_emb} + lm_head — Llama/TinyLlama, Phi-3, Mistral, Qwen2.
"""
import glob
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from evaluation.ppl import get_wikitext2_ids

DTYPE = torch.float16          # единый dtype серии экспериментов — не менять


# --------------------------------------------------------------------------
# Чтение весов
# --------------------------------------------------------------------------
class WeightReader:
    """Отдаёт тензоры чекпойнта по имени, не загружая модель в RAM целиком."""

    def __init__(self, repo):
        self._handles, self._key2file, self._sd = {}, {}, None
        root = None
        try:
            root = snapshot_download(
                repo,
                allow_patterns=["*.json", "*.safetensors"],
                ignore_patterns=["original/*"],   # у Llama-3 там дубль весов на 16 ГБ
            )
        except Exception as e:                     # нет сети/кэша — пробуем запасной путь
            print(f"[WeightReader] snapshot_download: {e}")

        files = sorted(glob.glob(os.path.join(root, "*.safetensors"))) if root else []
        if files:
            for f in files:
                h = safe_open(f, framework="pt", device="cpu")
                self._handles[f] = h
                for k in h.keys():
                    self._key2file[k] = f
        else:
            # Только для малых моделей без safetensors: state_dict целиком в RAM.
            print("[WeightReader] safetensors не найдены, загружаю state_dict в RAM")
            m = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPE)
            self._sd = m.state_dict()
            self._key2file = dict.fromkeys(self._sd)

    def keys(self):
        return self._key2file.keys()

    def get(self, key):
        if self._sd is not None:
            return self._sd[key]
        return self._handles[self._key2file[key]].get_tensor(key)


def _materialize(module, prefix, reader, device):
    """meta-модуль -> реальные fp16-веса на device из чекпойнта."""
    module.to_empty(device=device)
    module.to(DTYPE)
    sd = {k[len(prefix):]: reader.get(k) for k in reader.keys() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    # старые конвертации Llama хранят rotary_emb.inv_freq внутри блоков — это не веса
    unexpected = [k for k in unexpected if not k.endswith("rotary_emb.inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"{prefix}: missing={missing} unexpected={unexpected}")


def _release(module):
    module.to_empty(device="meta")


# --------------------------------------------------------------------------
# Перехват входа первого блока
# --------------------------------------------------------------------------
class _Caught(Exception):
    def __init__(self, hidden, args, kwargs):
        self.hidden, self.args, self.kwargs = hidden, args, kwargs


class _Catcher(nn.Module):
    """Встаёт на место layers[0]: запоминает вход и kwargs (маска, позиции,
    position_embeddings) и обрывает forward. Так маска и RoPE строятся ровно
    тем же кодом transformers, что и при обычном прогоне модели."""

    def __init__(self, layer):
        super().__init__()
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, hidden_states, *args, **kwargs):
        raise _Caught(hidden_states, args, kwargs)


def _build_skeleton(repo, device):
    config = AutoConfig.from_pretrained(repo)
    config.use_cache = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.eval()
    base = model.model
    for attr in ("embed_tokens", "layers", "norm", "rotary_emb"):
        if not hasattr(base, attr):
            raise NotImplementedError(
                f"{type(model).__name__}: нет model.{attr} — архитектура не Llama-подобная")
    # rotary_emb хранит inv_freq как буфер, вычисляемый при создании:
    # на meta он пустой, поэтому пересоздаём модуль на реальном устройстве.
    rot_cls = type(base.rotary_emb)
    try:
        base.rotary_emb = rot_cls(config=model.config, device=device)
    except TypeError:
        base.rotary_emb = rot_cls(model.config).to(device)
    return model


# --------------------------------------------------------------------------
# Основная функция
# --------------------------------------------------------------------------
@torch.no_grad()
def layerwise_perplexity(repo, layer_fn=None, seqlen=2048, chunk=512,
                         device="cuda", acts_device="cuda", max_windows=None,
                         top_k_channels=5, verbose=True):
    """Perplexity на WikiText-2 (test) с послойной загрузкой.

    layer_fn(layer, idx) -> int | None — вызывается после загрузки блока на GPU
        и до прогона; здесь делается квантование. Возвращает число
        квантованных Linear (для статистики).
    acts_device — где хранить скрытые состояния всех окон ("cuda" или "cpu").
        Для Llama-3-8B это ~2.4 ГБ fp16.

    Возвращает dict: ppl, n_windows, n_linear, hidden_stats (по слоям: max|h|
    и top-k каналов по max|h| в residual stream).
    """
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    ids = get_wikitext2_ids(tok)
    n = ids.numel() // seqlen
    if max_windows:
        n = min(n, max_windows)

    reader = WeightReader(repo)
    model = _build_skeleton(repo, device)
    base = model.model
    hidden = model.config.hidden_size
    n_layers = len(base.layers)

    # ---- 1. эмбеддинги: вход первого блока для всех окон --------------------
    _materialize(base.embed_tokens, "model.embed_tokens.", reader, device)
    acts = torch.empty(n, seqlen, hidden, dtype=DTYPE, device=acts_device)
    layer0 = base.layers[0]
    base.layers[0] = _Catcher(layer0)
    cargs, ckw = (), None
    for j in range(n):
        window = ids[:, j * seqlen:(j + 1) * seqlen].to(device)
        try:
            model(window, use_cache=False)
        except _Caught as c:
            acts[j] = c.hidden[0].to(acts_device)
            if ckw is None:
                cargs, ckw = c.args, c.kwargs
        else:
            raise RuntimeError("Catcher не сработал")
    base.layers[0] = layer0
    _release(base.embed_tokens)
    torch.cuda.empty_cache()
    if verbose:
        print(f"  эмбеддинги: {n} окон x {seqlen}, буфер "
              f"{acts.numel() * 2 / 1e9:.2f} ГБ на {acts_device}  ({time.time()-t0:.0f} c)")

    # ---- 2. трансформерные блоки по одному ----------------------------------
    n_linear, stats = 0, []
    for i in range(n_layers):
        layer = base.layers[i]
        _materialize(layer, f"model.layers.{i}.", reader, device)
        if layer_fn is not None:
            n_linear += layer_fn(layer, i) or 0

        ch_max = torch.zeros(hidden, dtype=torch.float32, device=device)
        for j in range(n):
            out = layer(acts[j:j + 1].to(device), *cargs, **ckw)
            if isinstance(out, (tuple, list)):
                out = out[0]
            ch_max = torch.maximum(ch_max, out[0].abs().amax(dim=0).float())
            acts[j] = out[0].to(acts_device)

        if not torch.isfinite(ch_max).all():
            raise FloatingPointError(
                f"блок {i}: inf/nan в скрытых состояниях — переполнение fp16")
        top = torch.topk(ch_max, top_k_channels)
        stats.append({
            "layer": i,
            "max_abs": round(top.values[0].item(), 2),
            "top_channels": [[int(c), round(v, 2)] for c, v in
                             zip(top.indices.tolist(), top.values.tolist())],
        })
        _release(layer)
        torch.cuda.empty_cache()
        if verbose:
            print(f"  блок {i+1:>2}/{n_layers}  max|h| = {stats[-1]['max_abs']:>9.2f}"
                  f"  ({time.time()-t0:.0f} c)")

    # ---- 3. final norm + lm_head + NLL (как в evaluation/ppl.py) -------------
    _materialize(base.norm, "model.norm.", reader, device)
    if "lm_head.weight" in reader.keys():
        _materialize(model.lm_head, "lm_head.", reader, device)
    else:                                          # tied embeddings
        model.lm_head.to_empty(device=device)
        model.lm_head.to(DTYPE)
        model.lm_head.weight.copy_(reader.get("model.embed_tokens.weight"))

    total_nll, total_tokens = 0.0, 0
    for j in range(n):
        h = base.norm(acts[j:j + 1].to(device))
        logits = model.lm_head(h)
        shift_logits = logits[0, :-1, :]
        shift_labels = ids[0, j * seqlen + 1:(j + 1) * seqlen].to(device)
        for s in range(0, shift_logits.size(0), chunk):
            lg = shift_logits[s:s + chunk].float()
            lb = shift_labels[s:s + chunk]
            total_nll += F.cross_entropy(lg, lb, reduction="sum").item()
            total_tokens += lb.numel()
        del logits, shift_logits

    return {
        "ppl": math.exp(total_nll / total_tokens),
        "n_windows": n,
        "n_linear": n_linear,
        "hidden_stats": stats,
        "sec": time.time() - t0,
    }
