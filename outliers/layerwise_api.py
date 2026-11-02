"""Встраивание в послойную оценку (evaluation/layerwise.py) для Phi-3-mini и Llama-3-8B.

Полная модель в 8 ГБ не помещается, поэтому все функции работают с одним блоком. Схема:

  Проход 1 (калибровка, FP16, без квантования) — только для SmoothQuant и статических шкал:
      буфер = калибровочные отрезки (вместо окон WikiText-2 test), для каждого блока:
          store = {}
          hs = register_amax_hooks(block, store)
          ... прогнать буфер через блок, как обычно ...
          remove_hooks(hs); all_scales[i] = store
      torch.save(all_scales, "results/act_stats/llama3-8b_act_scales_layerwise.pt")

  Проход 2 (оценка):
      Q = make_rotation(hidden, "hadamard", seed, device="cuda")      # только для вращения
      буфер_входа_блока0 = rotate_hidden(буфер, Q)                   # вместо поворота эмбеддингов
      для каждого блока i:
          (загрузить блок на GPU, как сейчас)
          transform_block(block, model.config, cfg, all_scales.get(i), Q, alphas=None)
          (вместо прежнего RTN-квантования весов; дальше всё как было)
      перед финальной нормой: буфер = unrotate_hidden(буфер, Q)       # и norm/lm_head без изменений

  Алгебра «буфер @ Q на входе, @ Q^T перед финальной нормой» проверена в verify_math.py.
"""
from __future__ import annotations

import torch

from .arch import block_spec
from .pipeline import PipeCfg, bmm_out_cfg
from .quant import QLinear, as_qlinear, get_linear
from .rotate import rotate_block, make_rotation  # noqa: F401  (make_rotation — для вызывающего кода)
from .smooth import parse_targets, smooth_block


def register_amax_hooks(block, store: dict, names: list[str] | None = None,
                        mean_store: dict | None = None) -> list:
    """Копить max|x| по каналам входов линейных слоёв блока в store[rel_name] (float32, на устройстве).
    mean_store: если задан, копит сумму поканальных max по вызовам (одному отрезку) в mean_store[n] = [сумма, число];
    среднее — meanmax_from(mean_store)."""
    if names is None:
        names = [n for n, m in block.named_modules() if isinstance(m, torch.nn.Linear) and not n.endswith(".linear")]
    hs = []
    for n in names:
        def f(mod, args, n=n):
            x = args[0].detach()
            cm = x.abs().reshape(-1, x.shape[-1]).amax(dim=0).float()
            store[n] = cm if n not in store else torch.maximum(store[n], cm)
            if mean_store is not None:
                if n in mean_store:
                    mean_store[n][0] += cm
                    mean_store[n][1] += 1
                else:
                    mean_store[n] = [cm.clone(), 1]
        hs.append(get_linear(block, n).register_forward_pre_hook(f))
    return hs


def meanmax_from(mean_store: dict) -> dict:
    return {n: v[0] / v[1] for n, v in mean_store.items()}


def remove_hooks(hs: list) -> None:
    for h in hs:
        h.remove()


def rotate_hidden(h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return (h.to(Q.dtype) @ Q.to(h.device)).to(h.dtype)


def unrotate_hidden(h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return (h.to(Q.dtype) @ Q.to(h.device).T).to(h.dtype)


@torch.no_grad()
def transform_block(block, config, cfg: PipeCfg, act_scales: dict | None = None,
                    Q: torch.Tensor | None = None, alphas: dict | None = None,
                    smooth_scales: dict | None = None) -> list[QLinear]:
    """Сглаживание / вращение / обёртка и квантование одного блока на месте.

    act_scales — max|X_j| по относительным именам этого блока из прохода 1 (нужны для smoothquant,
    smoothrot и статических шкал). Статические шкалы выводятся из act_scales точно:
    max по тензору = max_j (max|X_j| / s_j) — перекалибровка после сглаживания не нужна.
    Для вращения статические шкалы так не выводятся — используйте динамические (как в QuaRot).
    smooth_scales — статистика для масштаба сглаживания, если она отличается от max (например meanmax);
    по умолчанию act_scales.
    """
    spec = block_spec(config)
    new_amax = {}
    if cfg.method in ("smoothquant", "smoothrot"):
        targets = {"vo", "ffn"} if cfg.method == "smoothrot" else parse_targets(cfg.smooth_targets)
        use_groups = bool(targets & {"attn", "mlp"})
        use_pairs = bool(targets & {"vo", "ffn"})
        pa = cfg.alpha if cfg.pair_alpha is None else cfg.pair_alpha
        base = {"attn": cfg.alpha, "mlp": cfg.alpha, "vo": pa, "ffn": pa}
        al = alphas or {k: (v if k in targets else None) for k, v in base.items()}
        new_amax = smooth_block(block, spec, smooth_scales if smooth_scales is not None else act_scales,
                                al, use_groups, use_pairs, ax_floor=cfg.ax_floor, s_floor_q=cfg.s_floor_q,
                                true_amax=act_scales)["new_amax"]
    if cfg.method in ("rotate", "smoothrot"):
        rotate_block(block, spec, Q, mode=cfg.rotate_mode)
    qs = []
    for n in spec.linears:
        q = as_qlinear(block, n)
        q.act_cfg = cfg.a if not any(n.endswith(s) for s in cfg.a_skip) else type(cfg.a)()
        if cfg.quant_bmm and n in spec.groups[0].linears:
            q.out_cfg = bmm_out_cfg(cfg)
        if q.act_cfg.enabled and q.act_cfg.granularity == "per_tensor" and q.act_cfg.static:
            if cfg.method in ("rotate", "smoothrot"):
                raise NotImplementedError("статические шкалы после вращения: нужен калибровочный проход по "
                                          "уже повёрнутому блоку")
            amax = new_amax.get(n, act_scales[n] if act_scales else None)
            if amax is None:
                raise ValueError(f"нет act_scales для {n}")
            q.static_scale = amax.float().max().reshape(()) * q.act_cfg.clip_ratio
        if cfg.method == "llm_int8":
            q.outlier_threshold = cfg.int8_threshold
        elif cfg.w.enabled:
            q.quantize_weight_(cfg.w)
        qs.append(q)
    return qs
