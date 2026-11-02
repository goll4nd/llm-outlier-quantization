"""Калибровочные данные и статистики входов линейных слоёв.

Статистики по каждому nn.Linear внутри блоков (вход = то, что квантуется в W8A8):
  amax[c]           max|x| по каналу (SmoothQuant act_scales);
  max_abs           max|x| по тензору;
  kurtosis          избыточный эксцесс по Фишеру по всем элементам тензора (нормаль = 0);
  ch_ratio          max_c amax / median_c amax (ранжирование слоёв, как в QFeM);
  tok_ratio_mean/max  max_t / median_t поканально-максимальной величины токена (PrefixQuant);
  argmax_pos0       доля калибровочных отрезков, где самый «громкий» токен стоит на позиции 0;
  n_ch_ge6          число каналов с |x| >= 6 хотя бы на одном токене (порог LLM.int8());
  massive           max по отрезкам числа элементов с |x| > 100 и > 1000·median|x| (Sun et al.).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import torch
import torch.nn as nn

from .arch import get_layers, layer_prefix
from .quant import QLinear


# ----------------------------------------------------------------- данные

def _wikitext_ids(tok, split: str) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return tok(text, return_tensors="pt").input_ids


def get_test_ids(tok) -> torch.Tensor:
    return _wikitext_ids(tok, "test")


def get_calib_samples(tok, dataset: str = "wikitext2", n: int = 128, seqlen: int = 512,
                      seed: int = 0, bos: bool = False, min_len: int = 16) -> list[torch.Tensor]:
    """Список тензоров [1, seqlen].

    wikitext2: случайные отрезки из train (как калибровка GPTQ в работе; отрезок начинается в середине
               текста, как и все окна оценки, кроме первого). bos=True добавляет BOS в начало.
    pile:      как в официальном SmoothQuant generate_act_scales.py — каждый документ pile-val
               токенизируется отдельно с truncation до seqlen (токенизатор сам ставит BOS).
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    if dataset == "wikitext2":
        ids = _wikitext_ids(tok, "train")
        L = seqlen - 1 if bos else seqlen
        starts = torch.randint(0, ids.shape[1] - L - 1, (n,), generator=g).tolist()
        for s in starts:
            x = ids[:, s:s + L]
            if bos:
                x = torch.cat([torch.tensor([[tok.bos_token_id]]), x], dim=1)
            out.append(x)
    elif dataset == "pile":
        from datasets import load_dataset
        ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        order = torch.randperm(len(ds), generator=g).tolist()
        for i in order:
            x = tok(ds[i]["text"], return_tensors="pt", truncation=True, max_length=seqlen).input_ids
            if x.shape[1] < min_len:
                continue
            out.append(x)
            if len(out) == n:
                break
    else:
        raise ValueError(dataset)
    return out


def block_linear_names(model) -> list[str]:
    """Логические имена линейных слоёв блоков (для QLinear — имя обёртки)."""
    names = []
    for i, blk in enumerate(get_layers(model)):
        pre = layer_prefix(model, i)
        for n, m in blk.named_modules():
            if isinstance(m, QLinear):
                names.append(f"{pre}.{n}")
            elif isinstance(m, nn.Linear) and not n.endswith(".linear"):
                names.append(f"{pre}.{n}")
    return names


def _hook_target(model, name: str) -> nn.Module:
    """Хук ставится на сам nn.Linear: у QLinear это .linear, то есть вход ПОСЛЕ онлайн-Адамара
    (и после квантователя, если он включён — для статистик держите активации в 16 битах)."""
    m = model.get_submodule(name)
    return m.linear if isinstance(m, QLinear) else m


# ----------------------------------------------------------------- статистики

class _Acc:
    def __init__(self):
        self.amax = None
        self.sum_max = None
        self.calls = 0
        self.n = 0
        self.s = [0.0, 0.0, 0.0, 0.0]
        self.tok_ratio = []
        self.pos0 = 0
        self.segments = 0
        self.ch_ge6 = []
        self.massive = 0
        self.top_tok = (0.0, -1, -1)   # (значение, позиция, id токена)


@torch.no_grad()
def collect_act_stats(model, samples: list[torch.Tensor], names: list[str] | None = None,
                      detailed: bool = True, stat: str = "max") -> tuple[dict, list[dict]]:
    """Прогон FP-модели по samples с хуками на входах линейных слоёв.
    Возвращает (поканальная статистика: полное имя -> тензор [in] на CPU, строки статистики).
    stat = "max" — max|x| по всей калибровке; "meanmax" — среднее по отрезкам поканальных max|x|."""
    if stat not in ("max", "meanmax"):
        raise ValueError(stat)
    names = names or block_linear_names(model)
    dev = next(model.parameters()).device
    accs = {n: _Acc() for n in names}
    cur_ids = {}

    def hook(name):
        def f(mod, args):
            x = args[0].detach()
            a = accs[name]
            xa = x.abs().reshape(-1, x.shape[-1])
            cm = xa.amax(dim=0).float()
            a.amax = cm if a.amax is None else torch.maximum(a.amax, cm)
            a.sum_max = cm.clone() if a.sum_max is None else a.sum_max + cm
            a.calls += 1
            if not detailed:
                return
            xf = x.double().reshape(-1)
            a.n += xf.numel()
            x2 = xf * xf
            a.s[0] += xf.sum().item(); a.s[1] += x2.sum().item()
            a.s[2] += (x2 * xf).sum().item(); a.s[3] += (x2 * x2).sum().item()
            tm = xa.amax(dim=1).float()                       # max по каналам для каждого токена
            med = tm.median().clamp(min=1e-8)
            a.tok_ratio.append((tm.max() / med).item())
            pos = int(tm.argmax())
            a.pos0 += int(pos == 0)
            a.segments += 1
            if tm[pos].item() > a.top_tok[0]:
                ids = cur_ids.get("ids")
                a.top_tok = (tm[pos].item(), pos, int(ids[0, pos]) if ids is not None else -1)
            a.ch_ge6.append(int((cm >= 6).sum()))
            flat = xa.reshape(-1)
            step = max(1, flat.numel() // 1_000_000)
            med_all = flat[::step].float().median().clamp(min=1e-8)
            a.massive = max(a.massive, int(((xa > 100) & (xa > 1000 * med_all)).sum()))
        return f

    handles = [_hook_target(model, n).register_forward_pre_hook(hook(n)) for n in names]
    try:
        for x in samples:
            cur_ids["ids"] = x
            model(input_ids=x.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if stat == "max":
        amax = {n: a.amax.cpu() for n, a in accs.items() if a.amax is not None}
    else:
        amax = {n: (a.sum_max / a.calls).cpu() for n, a in accs.items() if a.amax is not None}
    rows = []
    if detailed:
        for n, a in accs.items():
            if a.amax is None:
                continue
            mu = a.s[0] / a.n
            e2, e3, e4 = a.s[1] / a.n, a.s[2] / a.n, a.s[3] / a.n
            m2 = e2 - mu * mu
            m4 = e4 - 4 * mu * e3 + 6 * mu * mu * e2 - 3 * mu ** 4
            kurt = m4 / (m2 * m2) - 3 if m2 > 0 else float("nan")
            top = torch.topk(a.amax, k=min(5, a.amax.numel()))
            m = re.search(r"layers\.(\d+)\.(.+)$", n)
            rows.append(dict(
                name=n, layer=int(m.group(1)) if m else -1, module=m.group(2) if m else n,
                in_features=a.amax.numel(),
                max_abs=float(a.amax.max()), kurtosis=kurt,
                ch_ratio=float(a.amax.max() / a.amax.median().clamp(min=1e-8)),
                tok_ratio_mean=sum(a.tok_ratio) / len(a.tok_ratio), tok_ratio_max=max(a.tok_ratio),
                argmax_pos0=a.pos0 / max(1, a.segments),
                top_token_pos=a.top_tok[1], top_token_id=a.top_tok[2],
                n_ch_ge6_mean=sum(a.ch_ge6) / len(a.ch_ge6), n_ch_ge6_max=max(a.ch_ge6),
                massive=a.massive,
                top_channels=" ".join(f"{i}:{v:.1f}" for v, i in zip(top.values.tolist(), top.indices.tolist())),
            ))
    return amax, rows


@torch.no_grad()
def capture_inputs(model, samples: list[torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    """Сохранить входы указанных линейных слоёв (FP16, CPU): имя -> [n, T, C]. Отрезки подаются по одному."""
    dev = next(model.parameters()).device
    store = defaultdict(list)

    def hook(name):
        def f(mod, args):
            x = args[0].detach()
            # batch = 1; OPT подаёт в fc1/fc2 уже развёрнутый тензор [T, C], остальные — [1, T, C]
            store[name].append(x.reshape(-1, x.shape[-1]).to("cpu", torch.float16))
        return f

    handles = [_hook_target(model, n).register_forward_pre_hook(hook(n)) for n in names]
    try:
        for x in samples:
            model(input_ids=x.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {n: torch.stack(v) for n, v in store.items()}


def relative(d: dict, prefix: str) -> dict:
    """Ключи 'model.layers.3.self_attn.q_proj' -> 'self_attn.q_proj' для блока с данным префиксом."""
    p = prefix + "."
    return {k[len(p):]: v for k, v in d.items() if k.startswith(p)}
