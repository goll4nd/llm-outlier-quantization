"""SmoothQuant: Y = (X diag(s)^-1)(diag(s) W),  s_j = max|X_j|^a / max|W_j|^(1-a).

Две точки приложения масштаба (обе функционально точны, см. verify_math.py):
  * группы «норма -> линейные слои» (оригинальный SmoothQuant): 1/s вносится в γ (и β у LayerNorm OPT);
  * пары «линейный -> линейный» (как в OmniQuant LET / SmoothRot / QServe):
      v_proj -> o_proj  (внимание линейно по V; при GQA масштаб общий для голов группы),
      fc1 -> fc2        (OPT, ReLU положительно однородна при s > 0),
      up_proj -> down_proj (SwiGLU линейна по up).
    Пары закрывают входы o_proj/out_proj и down_proj/fc2, которые оригинальный SmoothQuant не сглаживает.

Формула и ограничения снизу (1e-5) — как в mit-han-lab/smoothquant/smooth.py.

Форма масштаба (этап 10):
  * s_floor_q = p > 0 — квантильный пол: s_j <- max(s_j, Q_p(s)). Общий множитель s не влияет ни на одно из
    используемых квантований (per-token / per-tensor активаций, per-channel / group весов), поэтому после пола
    все каналы ниже p-квантиля получают одинаковый масштаб, то есть фактически не трогаются: сглаживаются
    только выбросные каналы. p = 0 — оригинальный SmoothQuant, p = 0.5 — только каналы выше медианы,
    p = 0.99 — только верхний 1 %. Каналы, почти не активные на калибровке (мёртвые нейроны ReLU), больше не
    получают крошечного s — пол заменяет эвристику ax_floor.
  * статистика активаций (calib, stat="meanmax") — среднее по калибровочным отрезкам поканальных максимумов
    вместо общего максимума: устойчивее к редким всплескам.
Для post-LN (OPT-350M: do_layer_norm_before=False) выход нормы идёт и в остаточный поток,
поэтому группы недопустимы — разрешены только пары.
"""
from __future__ import annotations

import re

import torch

from .arch import Spec
from .quant import get_linear


ALL_TARGETS = ("attn", "mlp", "vo", "ffn")


def parse_targets(spec: str) -> set[str]:
    """'groups' -> {attn, mlp}; 'pairs' -> {vo, ffn}; 'groups+pairs' -> все; иначе список через ',' или '+'
    из attn, mlp, vo, ffn, groups, pairs (например 'ffn', 'attn,mlp,ffn', 'groups+vo')."""
    out: set[str] = set()
    for t in re.split(r"[,+]", spec.strip()):
        t = t.strip()
        if not t:
            continue
        if t == "groups":
            out |= {"attn", "mlp"}
        elif t == "pairs":
            out |= {"vo", "ffn"}
        elif t in ALL_TARGETS:
            out.add(t)
        else:
            raise ValueError(f"неизвестная точка сглаживания {t!r}; допустимо: {ALL_TARGETS}, groups, pairs")
    if not out:
        raise ValueError("пустой набор точек сглаживания")
    return out


def _s(ax: torch.Tensor, aw: torch.Tensor, alpha: float, ax_floor: float = 0.0,
       s_floor_q: float = 0.0) -> torch.Tensor:
    """ax_floor > 0: max|X_j| ограничивается снизу долей медианы по каналам. Защищает каналы, почти
    не активные на калибровке (мёртвые нейроны ReLU), от крошечных s, которые на тесте раздувают
    редкую активацию в x/s (вплоть до переполнения FP16).
    s_floor_q > 0: квантильный пол масштаба (см. описание модуля)."""
    if ax_floor > 0:
        ax = torch.maximum(ax, ax_floor * ax.median())
    s = (ax.clamp(min=1e-5).pow(alpha) / aw.clamp(min=1e-5).pow(1 - alpha)).clamp(min=1e-5)
    if s_floor_q > 0:
        s = torch.maximum(s, torch.quantile(s.float(), s_floor_q))
    return s


def group_scale(block, g, ax: torch.Tensor, alpha: float, ax_floor: float = 0.0,
                s_floor_q: float = 0.0) -> torch.Tensor:
    lins = [get_linear(block, n) for n in g.linears]
    aw = torch.stack([l.weight.detach().abs().amax(dim=0).float() for l in lins]).amax(dim=0)
    return _s(ax.float().to(aw.device), aw, alpha, ax_floor, s_floor_q)


def pair_scale(block, p, ax_down: torch.Tensor, alpha: float,
               ax_floor: float = 0.0, s_floor_q: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """(s_up — по строкам up-слоя, s_down — по входным каналам down-слоя)."""
    down = get_linear(block, p.down)
    aw = down.weight.detach().abs().amax(dim=0).float()
    ax = ax_down.float().to(aw.device)
    if p.rep > 1:
        n_in = aw.numel()
        nkv = n_in // (p.rep * p.head_dim)
        # o_proj вход: [голова h, dim d], голова h читает kv-голову h // rep (repeat_kv в HF)
        ax_g = ax.view(nkv, p.rep, p.head_dim).amax(dim=1).reshape(-1)
        aw_g = aw.view(nkv, p.rep, p.head_dim).amax(dim=1).reshape(-1)
        s_up = _s(ax_g, aw_g, alpha, ax_floor, s_floor_q)
        s_down = s_up.view(nkv, 1, p.head_dim).expand(nkv, p.rep, p.head_dim).reshape(-1)
    else:
        s_up = _s(ax, aw, alpha, ax_floor, s_floor_q)
        s_down = s_up
    return s_up, s_down


@torch.no_grad()
def apply_group(block, g, s: torch.Tensor, layernorm_with_bias: bool) -> None:
    norm = block.get_submodule(g.norm)
    norm.weight.data = (norm.weight.float() / s).to(norm.weight.dtype)
    if layernorm_with_bias and getattr(norm, "bias", None) is not None:
        norm.bias.data = (norm.bias.float() / s).to(norm.bias.dtype)
    for n in g.linears:
        l = get_linear(block, n)
        l.weight.data = (l.weight.float() * s[None, :]).to(l.weight.dtype)


@torch.no_grad()
def apply_pair(block, p, s_up: torch.Tensor, s_down: torch.Tensor) -> None:
    up, down = get_linear(block, p.up), get_linear(block, p.down)
    r0, r1 = p.up_rows if p.up_rows is not None else (0, up.weight.shape[0])
    assert r1 - r0 == s_up.numel(), (p.name, r1 - r0, s_up.numel())
    W = up.weight.data
    W[r0:r1] = (W[r0:r1].float() / s_up[:, None]).to(W.dtype)
    if up.bias is not None:
        up.bias.data[r0:r1] = (up.bias.data[r0:r1].float() / s_up).to(up.bias.dtype)
    down.weight.data = (down.weight.float() * s_down[None, :]).to(down.weight.dtype)


@torch.no_grad()
def smooth_block(block, spec: Spec, act_scales: dict[str, torch.Tensor],
                 alphas: dict[str, float], use_groups: bool = True, use_pairs: bool = False,
                 ax_floor: float = 0.0, s_floor_q: float = 0.0, true_amax: dict | None = None) -> dict:
    """Сгладить один декодерный блок.

    act_scales: относительное имя линейного слоя -> max|x| по входным каналам (FP-модель).
    alphas:     имя группы/пары ('attn', 'mlp', 'vo', 'ffn') -> alpha; None — эту точку не сглаживать.
    true_amax:  настоящие max|x| (если act_scales — другая статистика, например meanmax); по ним
                считаются возвращаемые new_amax для статических шкал.
    Возвращает новые max|x| по каналам для затронутых слоёв (для статических шкал без перекалибровки)
    и фактические масштабы.
    """
    new_amax, scales = {}, {}
    tm = true_amax if true_amax is not None else act_scales
    if use_groups:
        if not spec.pre_ln:
            raise ValueError("post-LN блок: сглаживание через норму меняет остаточный поток; "
                             "используйте только пары (--smooth-targets pairs)")
        for g in spec.groups:
            if alphas.get(g.name) is None:     # поиск решил не сглаживать
                continue
            ax = act_scales[g.linears[0]]
            s = group_scale(block, g, ax, alphas[g.name], ax_floor, s_floor_q)
            apply_group(block, g, s, spec.layernorm_with_bias)
            for n in g.linears:
                new_amax[n] = tm[n].float().to(s.device) / s
            scales[g.name] = s.cpu()
    if use_pairs:
        for p in spec.pairs:
            if alphas.get(p.name) is None:
                continue
            s_up, s_down = pair_scale(block, p, act_scales[p.down], alphas[p.name], ax_floor, s_floor_q)
            apply_pair(block, p, s_up, s_down)
            new_amax[p.down] = tm[p.down].float().to(s_down.device) / s_down
            scales[p.name] = s_up.cpu()
    for name, mod in block.named_parameters():
        if not torch.isfinite(mod).all():
            raise FloatingPointError(f"после сглаживания {name} содержит inf/nan (переполнение FP16?)")
    return {"new_amax": new_amax, "scales": scales}
