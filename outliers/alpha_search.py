"""Послойный (поблочный, по группам) поиск alpha по сетке — аналог авто-alpha Intel Neural Compressor.

Для каждой группы «норма -> линейные слои» и каждой пары «up -> down» перебирается alpha из сетки,
и выбирается значение с минимальной ошибкой выхода
    L(a) = sum_l || Q_a(X / s) · Q_w(W_l · s)^T  -  X · W_l^T ||^2 / N,
где X — сохранённые FP-входы (calib.capture_inputs), Q_a/Q_w — те же квантователи, что при оценке.

Для пар учитывается ошибка только down-слоя: при per-channel/group квантовании весов деление строк
up-слоя на s масштабирует их ошибку пропорционально, и после умножения на s в down-слое вклад тот же.
Сетка, квантователь весов per-channel и статическая/динамическая шкала активаций задаются снаружи.

Варианты (этап 6):
  loss = "mse"    — сумма квадратов ошибок (как Intel INC);
         "nopos0" — то же без позиции 0 каждого отрезка (шкала квантования по-прежнему по всем токенам):
                    у OPT самый «громкий» токен всегда на позиции 0 и доминирует в MSE;
         "tokrel" — sum_t ||err_t||^2 / ||ref_t||^2: каждый токен весит одинаково.
  identity=True — кандидат «не сглаживать» (s = 1); семейство s(alpha) тождество не содержит,
                  и при узких весах оно бывает лучше любого alpha. Тогда возвращается alpha = None.
"""
from __future__ import annotations

import torch

from .arch import Spec
from .quant import ActCfg, WCfg, quantize_act, quantize_weight, get_linear
from .smooth import group_scale, pair_scale


def _act_q(X: torch.Tensor, cfg: ActCfg) -> torch.Tensor:
    """X [n, T, C]. Статическая per-tensor шкала = max по всей калибровке (как при оценке)."""
    if not cfg.enabled:
        return X
    if cfg.granularity == "per_tensor" and cfg.static:
        return quantize_act(X, cfg, static_scale=X.abs().amax() * cfg.clip_ratio)
    if cfg.granularity == "per_tensor":
        return torch.stack([quantize_act(x, cfg) for x in X])   # одна шкала на отрезок
    return quantize_act(X, cfg)


@torch.no_grad()
def _loss(X: torch.Tensor, Ws: list[torch.Tensor], s: torch.Tensor | None, a_cfg: ActCfg, w_cfg: WCfg,
          refs: list[torch.Tensor], loss: str = "mse") -> float:
    n, T, C = X.shape
    Xs = X if s is None else X / s
    Xq = _act_q(Xs, a_cfg)
    if loss == "nopos0":
        Xq = Xq[:, 1:]
    Xq = Xq.reshape(-1, C)
    tot = 0.0
    for W, ref in zip(Ws, refs):
        Wq = quantize_weight(W if s is None else W * s[None, :], w_cfg)
        err = Xq @ Wq.T - ref
        if loss == "tokrel":
            tot += (err.pow(2).sum(1) / ref.pow(2).sum(1).clamp(min=1e-12)).mean().item()
        else:
            tot += err.pow(2).mean().item()
    return tot


@torch.no_grad()
def search_block(block, spec: Spec, inputs: dict[str, torch.Tensor], act_scales: dict[str, torch.Tensor],
                 grid: list[float], w_cfg: WCfg, a_cfg_for, use_groups=True, use_pairs=False,
                 device="cuda", ax_floor: float = 0.0, loss: str = "mse",
                 identity: bool = False, targets: set | None = None, s_floor_q: float = 0.0) -> tuple[dict[str, float | None], dict[str, list[float]]]:
    """inputs / act_scales — по относительным именам блока. a_cfg_for(rel_name) -> ActCfg.
    Возвращает (alpha по имени группы/пары, кривые потерь по сетке)."""
    best, curves = {}, {}
    jobs = []
    if use_groups:
        for g in spec.groups:
            if targets is not None and g.name not in targets:
                continue
            jobs.append((g.name, g.linears[0], [get_linear(block, n) for n in g.linears],
                         lambda a, g=g: (group_scale(block, g, act_scales[g.linears[0]], a, ax_floor, s_floor_q),) * 2))
    if use_pairs:
        for p in spec.pairs:
            if targets is not None and p.name not in targets:
                continue
            jobs.append((p.name, p.down, [get_linear(block, p.down)],
                         lambda a, p=p: pair_scale(block, p, act_scales[p.down], a, ax_floor, s_floor_q)))
    for name, in_name, lins, scale_fn in jobs:
        X = inputs[in_name].to(device=device, dtype=torch.float32)
        Ws = [l.weight.detach().to(device=device, dtype=torch.float32) for l in lins]
        Xf = (X[:, 1:] if loss == "nopos0" else X).reshape(-1, X.shape[-1])
        refs = [Xf @ W.T for W in Ws]
        a_cfg = a_cfg_for(in_name)
        losses = []
        for a in grid:
            _, s_in = scale_fn(a)            # масштаб по входным каналам читающего слоя
            losses.append(_loss(X, Ws, s_in.to(device=device, dtype=torch.float32), a_cfg, w_cfg, refs, loss))
        curves[name] = losses
        none = _loss(X, Ws, None, a_cfg, w_cfg, refs, loss)               # без сглаживания (s = 1)
        curves[name + "_none"] = [none]
        best[name] = grid[min(range(len(grid)), key=losses.__getitem__)]
        if identity and none < min(losses):
            best[name] = None
        del X, Ws, refs
    return best, curves
