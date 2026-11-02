"""Сборка конфигурации квантования для полной модели (OPT-125M/350M, TinyLlama).

Порядок шагов важен:
  1. статистики входов (max|X_j|) — на ИСХОДНОЙ FP-модели;
  2. (опц.) поиск alpha по сетке — тоже на FP-входах;
  3. сглаживание (меняет веса и нормы, функция модели та же);
  4. (опц.) вращение;
  5. обёртка всех линейных слоёв блоков в QLinear и конфиги активаций;
  6. (опц.) калибровка статических шкал — после 3–4 и до квантования весов, активации в 16 битах;
  7. квантование весов RTN.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .alpha_search import search_block
from .arch import block_spec, get_layers, layer_prefix
from .calib import capture_inputs, collect_act_stats, relative
from .quant import ActCfg, QLinear, WCfg, as_qlinear
from .rotate import rotate_model
from .smooth import parse_targets, smooth_block


@dataclass
class PipeCfg:
    method: str = "none"                 # none | smoothquant | rotate | smoothrot | llm_int8
    smooth_targets: str = "groups"       # groups | pairs | groups+pairs | список из attn, mlp, vo, ffn
    alpha: float = 0.5                   # для групп «норма -> линейные»
    pair_alpha: float | None = None      # для пар v->o, up->down; None = alpha
    ax_floor: float = 0.0                # нижняя граница max|X_j| (доля медианы); 0 = как в SmoothQuant
    s_floor_q: float = 0.0               # квантильный пол масштаба: s <- max(s, Q_p(s)); 0 = как в SmoothQuant
    act_stat: str = "max"                # статистика активаций для масштаба: max | meanmax
    alpha_mode: str = "global"           # global | search
    alpha_grid: list = field(default_factory=lambda: [round(0.05 * i, 2) for i in range(21)])
    search_n: int = 8                    # число калибровочных отрезков для поиска
    search_loss: str = "mse"             # mse | nopos0 | tokrel
    search_identity: bool = False        # разрешить «не сглаживать» (alpha = None)
    rotate_mode: str = "full"            # full | online
    rotate_kind: str = "hadamard"        # hadamard | orthogonal
    rot_seed: int = 0
    w: WCfg = field(default_factory=WCfg)
    a: ActCfg = field(default_factory=ActCfg)
    a_skip: list = field(default_factory=list)   # суффиксы модулей, чьи входы остаются в 16 битах
    quant_bmm: bool = False              # квантовать выходы q/k/v (входы BMM внимания), как SmoothQuant
    int8_threshold: float = 6.0


def bmm_out_cfg(cfg: PipeCfg) -> ActCfg:
    """Входы BMM квантуются динамически той же разрядностью и гранулярностью, что и активации
    (как quantize_output в официальном fake_quant.py; статическая шкала заменяется динамической)."""
    return ActCfg(bits=cfg.a.bits, granularity=cfg.a.granularity, static=False)


def _a_cfg_for(cfg: PipeCfg):
    def f(rel_name: str) -> ActCfg:
        if any(rel_name.endswith(s) for s in cfg.a_skip):
            return ActCfg()
        return cfg.a
    return f


@torch.no_grad()
def prepare_model(model, cfg: PipeCfg, calib: list[torch.Tensor] | None = None,
                  act_scales: dict | None = None, log=print) -> dict:
    t0 = time.time()
    spec = block_spec(model.config)
    layers = get_layers(model)
    a_cfg_for = _a_cfg_for(cfg)
    info: dict = {"alphas": {}, "curves": {}}
    dev = next(model.parameters()).device

    if cfg.method in ("smoothquant", "smoothrot"):
        targets = {"vo", "ffn"} if cfg.method == "smoothrot" else parse_targets(cfg.smooth_targets)
        use_groups = bool(targets & {"attn", "mlp"})
        use_pairs = bool(targets & {"vo", "ffn"})
        if act_scales is None:
            log(f"[calib] {cfg.act_stat} |X_j| на {len(calib)} отрезках")
            act_scales, _ = collect_act_stats(model, calib, detailed=False, stat=cfg.act_stat)
        pa = cfg.alpha if cfg.pair_alpha is None else cfg.pair_alpha
        base = {"attn": cfg.alpha, "mlp": cfg.alpha, "vo": pa, "ffn": pa}
        per_layer = [{k: (v if k in targets else None) for k, v in base.items()} for _ in layers]
        if cfg.alpha_mode == "search":
            if cfg.method == "smoothrot":
                raise NotImplementedError("поиск alpha с последующим Адамаром не реализован")
            names = []
            for i, _ in enumerate(layers):
                pre = layer_prefix(model, i)
                names += [f"{pre}.{g.linears[0]}" for g in spec.groups if g.name in targets]
                names += [f"{pre}.{p.down}" for p in spec.pairs if p.name in targets]
            log(f"[search] захват входов {len(names)} слоёв на {cfg.search_n} отрезках")
            inputs = capture_inputs(model, calib[:cfg.search_n], names)
            for i, blk in enumerate(layers):
                pre = layer_prefix(model, i)
                best, curves = search_block(blk, spec, relative(inputs, pre), relative(act_scales, pre),
                                            cfg.alpha_grid, cfg.w, a_cfg_for, use_groups, use_pairs, dev,
                                            ax_floor=cfg.ax_floor, loss=cfg.search_loss,
                                            identity=cfg.search_identity, targets=targets,
                                            s_floor_q=cfg.s_floor_q)
                per_layer[i].update(best)
                info["curves"][i] = curves
                log(f"[search] блок {i}: " + ", ".join(f"{k}={'none' if v is None else f'{v:.2f}'}"
                                                         for k, v in best.items()))
            del inputs
        for i, blk in enumerate(layers):
            pre = layer_prefix(model, i)
            smooth_block(blk, spec, relative(act_scales, pre), per_layer[i], use_groups, use_pairs,
                         ax_floor=cfg.ax_floor, s_floor_q=cfg.s_floor_q)
            info["alphas"][i] = {k: v for k, v in per_layer[i].items() if k in targets}

    if cfg.method in ("rotate", "smoothrot"):
        rotate_model(model, kind=cfg.rotate_kind, seed=cfg.rot_seed, mode=cfg.rotate_mode)

    qlins: list[QLinear] = []
    attn_in = set(spec.groups[0].linears)
    for blk in layers:
        for n in spec.linears:
            q = as_qlinear(blk, n)
            q.act_cfg = a_cfg_for(n)
            if cfg.quant_bmm and n in attn_in:
                q.out_cfg = bmm_out_cfg(cfg)
            qlins.append(q)

    if cfg.method == "llm_int8":
        for q in qlins:
            q.outlier_threshold = cfg.int8_threshold
        info["prep_time_s"] = time.time() - t0
        return info

    if cfg.a.enabled and cfg.a.granularity == "per_tensor" and cfg.a.static:
        log(f"[static] калибровка per-tensor шкал на {len(calib)} отрезках")
        for q in qlins:
            q.start_observing()
        for x in calib:
            model(input_ids=x.to(dev), use_cache=False)
        for q in qlins:
            q.stop_observing()

    if cfg.w.enabled:
        for q in qlins:
            q.quantize_weight_(cfg.w)
    info["prep_time_s"] = time.time() - t0
    return info


def int8_report(model) -> list[dict]:
    rows = []
    for n, m in model.named_modules():
        if isinstance(m, QLinear) and m.outlier_threshold is not None and m.int8_stats["calls"]:
            st = m.int8_stats
            rows.append({"name": n, "outlier_dims_max": st["outlier_dims_max"],
                         "outlier_dims_mean": st["outlier_dims_sum"] / st["calls"]})
    return rows
