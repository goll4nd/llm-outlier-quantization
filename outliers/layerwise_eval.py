"""Послойная оценка (Phi-3-mini, Llama-3-8B) с SmoothQuant / вращением / LLM.int8().

Использует функции evaluation/layerwise.py без изменений (WeightReader, _materialize, _release,
_Catcher, _build_skeleton) и повторяет его протокол: окна 2048 WikiText-2 test, буфер скрытых
состояний, блоки по одному, final norm + lm_head, NLL кусками по 512 в float32.

Отличия от layerwise_perplexity:
  * второй буфер — калибровочные отрезки; он идёт через НЕквантованные блоки, и по нему на каждом
    блоке собираются max|X_j| входов линейных слоёв (до сглаживания) — один проход по диску;
  * для вращения буфер окон поворачивается Q на входе блока 0 и возвращается Q^T перед финальной
    нормой (эквивалентно повороту эмбеддингов и lm_head, проверено в verify_math.py и test_outliers.py);
  * вместо RTN каждый блок обрабатывает outliers.layerwise_api.transform_block.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from evaluation.layerwise import (DTYPE, WeightReader, _build_skeleton, _Catcher, _Caught,
                                  _materialize, _release)
from evaluation.ppl import get_wikitext2_ids

from .layerwise_api import (meanmax_from, register_amax_hooks, remove_hooks, rotate_hidden, transform_block,
                            unrotate_hidden)
from .pipeline import PipeCfg
from .rotate import make_rotation


@torch.no_grad()
def _catch(model, windows, device, acts_device):
    base = model.model
    T = windows[0].shape[1]
    acts = torch.empty(len(windows), T, model.config.hidden_size, dtype=DTYPE, device=acts_device)
    layer0 = base.layers[0]
    base.layers[0] = _Catcher(layer0)
    cargs, ckw = (), None
    try:
        for j, w in enumerate(windows):
            try:
                model(w.to(device), use_cache=False)
            except _Caught as c:
                acts[j] = c.hidden[0].to(acts_device)
                if ckw is None:
                    cargs, ckw = c.args, c.kwargs
            else:
                raise RuntimeError("Catcher не сработал")
    finally:
        base.layers[0] = layer0
    return acts, cargs, ckw


@torch.no_grad()
def _run(layer, acts, cargs, ckw, device, acts_device, track=False):
    ch_max = None
    for j in range(acts.shape[0]):
        out = layer(acts[j:j + 1].to(device), *cargs, **ckw)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if track:
            m = out[0].abs().amax(dim=0).float()
            ch_max = m if ch_max is None else torch.maximum(ch_max, m)
        acts[j] = out[0].to(acts_device)
    return ch_max


@torch.no_grad()
def layerwise_actquant(repo, cfg: PipeCfg, calib_windows=None, seqlen=2048, chunk=512, device="cuda",
                       acts_device="cuda", max_windows=None, verbose=True):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    ids = get_wikitext2_ids(tok)
    n = ids.numel() // seqlen
    if max_windows:
        n = min(n, max_windows)
    windows = [ids[:, j * seqlen:(j + 1) * seqlen] for j in range(n)]

    reader = WeightReader(repo)
    model = _build_skeleton(repo, device)
    base, config = model.model, model.config
    n_layers = len(base.layers)

    need_scales = cfg.method in ("smoothquant", "smoothrot") or (cfg.a.enabled and cfg.a.static)
    if need_scales and not calib_windows:
        raise ValueError("для сглаживания / статических шкал нужны калибровочные отрезки")

    _materialize(base.embed_tokens, "model.embed_tokens.", reader, device)
    acts, cargs, ckw = _catch(model, windows, device, acts_device)
    if need_scales:
        cacts, ccargs, cckw = _catch(model, calib_windows, device, acts_device)
    _release(base.embed_tokens)
    torch.cuda.empty_cache()

    Q = None
    if cfg.method in ("rotate", "smoothrot") and cfg.rotate_mode == "full":
        Q = make_rotation(config.hidden_size, cfg.rotate_kind, cfg.rot_seed, device=device)
        for j in range(n):
            acts[j] = rotate_hidden(acts[j].to(device), Q).to(acts_device)
    if verbose:
        print(f"  буферы готовы: {n} окон" + (f", {len(calib_windows)} калибровочных" if need_scales else "")
              + f"  ({time.time() - t0:.0f} c)")

    stats = []
    for i in range(n_layers):
        layer = base.layers[i]
        _materialize(layer, f"model.layers.{i}.", reader, device)
        scales = sm_scales = None
        if need_scales:                       # FP-блок: статистики входов и вход следующего блока
            store, mstore = {}, ({} if cfg.act_stat == "meanmax" else None)
            hs = register_amax_hooks(layer, store, mean_store=mstore)
            _run(layer, cacts, ccargs, cckw, device, acts_device)
            remove_hooks(hs)
            scales = {k: v.float().cpu() for k, v in store.items()}
            if mstore is not None:
                sm_scales = {k: v.float().cpu() for k, v in meanmax_from(mstore).items()}
        transform_block(layer, config, cfg, scales, Q, smooth_scales=sm_scales)
        ch_max = _run(layer, acts, cargs, ckw, device, acts_device, track=True)
        if not torch.isfinite(ch_max).all():
            raise FloatingPointError(f"блок {i}: inf/nan в скрытых состояниях")
        stats.append({"layer": i, "max_abs": round(ch_max.max().item(), 2)})
        _release(layer)
        torch.cuda.empty_cache()
        if verbose:
            print(f"  блок {i + 1:>2}/{n_layers}  max|h| = {stats[-1]['max_abs']:>9.2f}  ({time.time() - t0:.0f} c)")

    _materialize(base.norm, "model.norm.", reader, device)
    if "lm_head.weight" in reader.keys():
        _materialize(model.lm_head, "lm_head.", reader, device)
    else:
        model.lm_head.to_empty(device=device)
        model.lm_head.to(DTYPE)
        model.lm_head.weight.copy_(reader.get("model.embed_tokens.weight"))

    total_nll, total_tokens = 0.0, 0
    for j in range(n):
        h = acts[j:j + 1].to(device)
        if Q is not None:
            h = unrotate_hidden(h, Q)
        logits = model.lm_head(base.norm(h))
        shift_logits = logits[0, :-1, :]
        shift_labels = ids[0, j * seqlen + 1:(j + 1) * seqlen].to(device)
        for s in range(0, shift_logits.size(0), chunk):
            total_nll += F.cross_entropy(shift_logits[s:s + chunk].float(), shift_labels[s:s + chunk],
                                         reduction="sum").item()
            total_tokens += shift_labels[s:s + chunk].numel()
        del logits, shift_logits

    return {"ppl": math.exp(total_nll / total_tokens), "n_windows": n, "hidden_stats": stats,
            "sec": time.time() - t0}
