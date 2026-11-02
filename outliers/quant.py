"""Fake-квантование весов и активаций и обёртка QLinear.

Соглашения (как в SmoothQuant fake_quant.py и QuaRot):
  * активации — симметрично, per-token (O1) или per-tensor (O2 динамическая / O3 статическая);
  * веса — симметрично или асимметрично, per-channel (по выходному каналу), per-tensor или группами;
  * всё считается в float32 и возвращается в исходном dtype (обычно FP16).

QLinear(x) = Linear( Q_a( H(x) ) ) с весами Q_w(W), где
  H — опциональный онлайн-Адамар (ставится rotate.py; его транспонированная копия уже внесена в W),
  Q_a — квантователь активаций (16 бит = тождество).
Отдельный режим outlier_threshold воспроизводит LLM.int8(): каналы с max|x| >= t считаются
в FP16 по исходным весам, остальное — векторное INT8 (per-token x per-output-channel).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


# ----------------------------------------------------------------- конфиги

@dataclass
class ActCfg:
    bits: int = 16
    granularity: str = "per_token"     # per_token | per_tensor
    static: bool = False               # только для per_tensor: шкала из калибровки
    clip_ratio: float = 1.0            # QuaRot: 0.9 для A4

    @property
    def enabled(self) -> bool:
        return self.bits < 16


@dataclass
class WCfg:
    bits: int = 16
    granularity: str = "per_channel"   # per_channel | per_tensor | group
    group_size: int = 128
    sym: bool = True
    mse_clip: bool = False             # поиск коэффициента отсечения по ||w - q(w)||^2.4 (как GPTQ/QuaRot --w_clip)

    @property
    def enabled(self) -> bool:
        return self.bits < 16


# ----------------------------------------------------------------- квантователи

def quantize_act(x: torch.Tensor, cfg: ActCfg, static_scale: torch.Tensor | None = None) -> torch.Tensor:
    if not cfg.enabled:
        return x
    qmax = 2 ** (cfg.bits - 1) - 1
    xf = x.float()
    if cfg.granularity == "per_token":
        amax = xf.abs().amax(dim=-1, keepdim=True) * cfg.clip_ratio
    elif cfg.granularity == "per_tensor":
        if cfg.static:
            if static_scale is None:
                raise RuntimeError("Статическая per-tensor шкала не откалибрована")
            amax = static_scale.to(xf.device).float()
        else:
            # динамическая per-tensor: одна шкала на тензор батча (при batch=1 — на окно)
            amax = xf.abs().amax() * cfg.clip_ratio
    else:
        raise ValueError(cfg.granularity)
    scale = (amax / qmax).clamp(min=1e-8)
    q = torch.clamp(torch.round(xf / scale), -qmax - 1, qmax)
    return (q * scale).to(x.dtype)


def _quant_rows(w: torch.Tensor, bits: int, sym: bool, mse_clip: bool) -> torch.Tensor:
    """Квантование каждой строки w [R, C] со своей шкалой. float32 вход/выход."""
    if sym:
        qmax = 2 ** (bits - 1) - 1

        def q_with(shrink: torch.Tensor) -> torch.Tensor:
            amax = w.abs().amax(dim=1, keepdim=True) * shrink
            scale = (amax / qmax).clamp(min=1e-8)
            return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    else:
        qmax = 2 ** bits - 1

        def q_with(shrink: torch.Tensor) -> torch.Tensor:
            wmax = w.amax(dim=1, keepdim=True).clamp(min=0) * shrink
            wmin = w.amin(dim=1, keepdim=True).clamp(max=0) * shrink
            scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
            zero = torch.round(-wmin / scale)
            return (torch.clamp(torch.round(w / scale) + zero, 0, qmax) - zero) * scale

    if not mse_clip:
        return q_with(torch.ones(1, 1, device=w.device))
    best = q_with(torch.ones(1, 1, device=w.device))
    best_err = (best - w).abs().pow(2.4).sum(dim=1, keepdim=True)
    # сетка как в QuaRot/GPTQ WeightQuantizer: grid = 100, maxshrink = 0.8 -> p = 1.00, 0.99, ..., 0.21
    # (до этапа 9 здесь была грубая сетка 0.95..0.5 из 10 точек)
    for p in [1 - i / 100 for i in range(1, 80)]:
        cand = q_with(torch.full((1, 1), p, device=w.device))
        err = (cand - w).abs().pow(2.4).sum(dim=1, keepdim=True)
        better = err < best_err
        best = torch.where(better, cand, best)
        best_err = torch.where(better, err, best_err)
    return best


@torch.no_grad()
def quantize_weight(W: torch.Tensor, cfg: WCfg) -> torch.Tensor:
    """W [out, in] -> деквантованная копия в dtype W."""
    if not cfg.enabled:
        return W
    w = W.float()
    out, inp = w.shape
    if cfg.granularity == "per_channel":
        q = _quant_rows(w, cfg.bits, cfg.sym, cfg.mse_clip)
    elif cfg.granularity == "per_tensor":
        q = _quant_rows(w.reshape(1, -1), cfg.bits, cfg.sym, cfg.mse_clip).reshape(out, inp)
    elif cfg.granularity == "group":
        g = cfg.group_size
        if inp % g:
            raise ValueError(f"in_features={inp} не делится на group_size={g}")
        q = _quant_rows(w.reshape(-1, g), cfg.bits, cfg.sym, cfg.mse_clip).reshape(out, inp)
    else:
        raise ValueError(cfg.granularity)
    return q.to(W.dtype)


# ----------------------------------------------------------------- QLinear

class QLinear(nn.Module):
    """Обёртка над nn.Linear. Исходный модуль хранится в self.linear."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.online_had = False
        self.act_cfg = ActCfg()
        self.out_cfg = ActCfg()            # квантование выхода (входы BMM внимания: q/k/v), всегда динамическое
        self.register_buffer("static_scale", None, persistent=False)
        self.observing = False             # калибровка статической шкалы: копим max|x|, не квантуем
        self._obs_max = None
        self.outlier_threshold: float | None = None   # режим LLM.int8()
        self.int8_stats = {"calls": 0, "outlier_dims_max": 0, "outlier_dims_sum": 0}

    # для кода, который обращается к .weight / .bias проекций
    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def extra_repr(self) -> str:
        s = f"online_had={self.online_had}, act={asdict(self.act_cfg)}"
        if self.out_cfg.enabled:
            s += f", out={asdict(self.out_cfg)}"
        if self.outlier_threshold is not None:
            s += f", llm_int8_threshold={self.outlier_threshold}"
        return s

    @torch.no_grad()
    def quantize_weight_(self, cfg: WCfg) -> None:
        self.linear.weight.data = quantize_weight(self.linear.weight.data, cfg)

    def start_observing(self) -> None:
        self.observing, self._obs_max = True, None

    def stop_observing(self) -> None:
        if self._obs_max is not None:
            self.static_scale = self._obs_max * self.act_cfg.clip_ratio
        self.observing, self._obs_max = False, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.online_had:
            from .hadamard import hadamard_transform
            x = hadamard_transform(x)
        if self.outlier_threshold is not None:
            return self._llm_int8(x)
        if self.observing:
            m = x.detach().abs().amax().float()
            self._obs_max = m if self._obs_max is None else torch.maximum(self._obs_max, m)
            return self.linear(x)
        x = quantize_act(x, self.act_cfg, self.static_scale)
        y = self.linear(x)
        return quantize_act(y, self.out_cfg) if self.out_cfg.enabled else y

    def _llm_int8(self, x: torch.Tensor) -> torch.Tensor:
        """LLM.int8(): смешанная декомпозиция + векторное INT8 (fake-quant).

        Веса self.linear должны оставаться в FP16 (не вызывать quantize_weight_): регулярная часть
        квантуется на лету построчно без выбросных столбцов, выбросная часть идёт в FP16.
        """
        W = self.linear.weight
        shape = x.shape
        xs = x.reshape(-1, shape[-1])
        mask = xs.abs().amax(dim=0) >= self.outlier_threshold          # выбросные скрытые измерения
        n_out = int(mask.sum())
        st = self.int8_stats
        st["calls"] += 1
        st["outlier_dims_sum"] += n_out
        st["outlier_dims_max"] = max(st["outlier_dims_max"], n_out)
        keep = (~mask).to(xs.dtype)
        x_reg = quantize_act(xs * keep, ActCfg(bits=8, granularity="per_token"))
        w_reg = quantize_weight(W * keep[None, :], WCfg(bits=8, granularity="per_channel"))
        y = x_reg @ w_reg.T
        if n_out:
            y = y + xs[:, mask] @ W[:, mask].T
        if self.linear.bias is not None:
            y = y + self.linear.bias
        return y.reshape(*shape[:-1], W.shape[0])


def as_qlinear(parent: nn.Module, name: str) -> QLinear:
    """Вернуть QLinear по относительному имени внутри parent, при необходимости обернув nn.Linear."""
    mod = parent.get_submodule(name)
    if isinstance(mod, QLinear):
        return mod
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{name}: ожидался nn.Linear, получен {type(mod)}")
    q = QLinear(mod)
    if "." in name:
        owner_name, attr = name.rsplit(".", 1)
        owner = parent.get_submodule(owner_name)
    else:
        owner, attr = parent, name
    setattr(owner, attr, q)
    return q


def get_linear(parent: nn.Module, name: str) -> nn.Linear:
    mod = parent.get_submodule(name)
    return mod.linear if isinstance(mod, QLinear) else mod
