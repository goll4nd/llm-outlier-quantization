"""QuaRot-подобные ортогональные преобразования (Ashkboos et al., NeurIPS 2024; SliceGPT, ICLR 2024).

Полный режим (pre-LN):
  1. (OPT) центрирование всех записей в остаточный поток: эмбеддинги, out_proj, fc2 — тогда
     LayerNorm(x) == RMSNorm(x)·γ + β, и LayerNorm заменяется на RMSN без параметров;
  2. γ (и β) каждой нормы вносятся в читающие её линейные слои, финальная норма — в lm_head;
  3. остаточный поток вращается Q = diag(±1)·H: читатели W -> W Q, писатели W -> Q^T W, b -> b Q,
     эмбеддинги E -> E Q, lm_head -> W Q (связанные эмбеддинги предварительно развязываются);
  4. онлайн-Адамар на входах o_proj/out_proj и down_proj/fc2 (их вход нелинеен и не может
     поглотить вращение): x -> x H на лету, W -> W H офлайн.
Онлайн-режим (годится и для post-LN OPT-350M): только шаг 4.

Отличия от официального spcl/QuaRot (сознательные упрощения, функционально эквивалентно):
  * для o_proj берётся полный Адамар по всей размерности, а не поголовный офлайн + межголовный онлайн;
  * Адамар для Q/K (нужен только для квантования KV-кэша) не ставится — KV здесь не квантуется.
Все шаги проверены в numpy (verify_math.py) и тестом test_outliers.py на крошечных HF-моделях.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .arch import Spec, get_layers, block_spec
from .hadamard import hadamard_transform, random_hadamard, random_orthogonal
from .quant import as_qlinear, get_linear


class RMSN(nn.Module):
    """RMSNorm без обучаемых параметров (замена LayerNorm OPT после центрирования и слияния)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)

    def extra_repr(self):
        return f"{self.dim}, eps={self.eps}"


def _tdtype(device) -> torch.dtype:
    return torch.float64 if torch.device(device).type == "cpu" else torch.float32


def make_rotation(dim: int, kind: str = "hadamard", seed: int = 0, device="cpu") -> torch.Tensor:
    dt = _tdtype(device)
    if kind == "hadamard":
        return random_hadamard(dim, seed=seed, dtype=dt, device=device)
    if kind == "orthogonal":
        return random_orthogonal(dim, seed=seed, dtype=dt, device=device)
    raise ValueError(kind)


@torch.no_grad()
def _set_w(lin: nn.Linear, W: torch.Tensor):
    lin.weight.data = W.to(lin.weight.dtype)


@torch.no_grad()
def fold_norm(norm: nn.Module, readers: list[nn.Linear], with_bias: bool) -> None:
    g = norm.weight.detach().float()
    b = norm.bias.detach().float() if with_bias and getattr(norm, "bias", None) is not None else None
    for l in readers:
        W = l.weight.float()
        if b is not None:
            add = W @ b.to(W.device)
            if l.bias is None:
                l.bias = nn.Parameter(add.to(l.weight.dtype))
            else:
                l.bias.data = (l.bias.float() + add).to(l.bias.dtype)
        _set_w(l, W * g.to(W.device)[None, :])


@torch.no_grad()
def _center_writer(lin: nn.Linear) -> None:
    W = lin.weight.float()
    _set_w(lin, W - W.mean(dim=0, keepdim=True))
    if lin.bias is not None:
        lin.bias.data = (lin.bias.float() - lin.bias.float().mean()).to(lin.bias.dtype)


@torch.no_grad()
def add_online_hadamard(block, names: list[str]) -> None:
    for n in names:
        q = as_qlinear(block, n)
        if q.online_had:
            continue
        W = q.linear.weight
        q.linear.weight.data = hadamard_transform(W.float(), compute_dtype=_tdtype(W.device)).to(W.dtype)
        q.online_had = True


@torch.no_grad()
def rotate_block(block, spec: Spec, Q: torch.Tensor | None, mode: str = "full",
                 online_had: bool = True) -> None:
    """mode='full': слияние норм + вращение остаточного потока Q (+ онлайн-Адамар);
    mode='online': только онлайн-Адамар (Q не нужен)."""
    if mode == "full":
        if not spec.pre_ln:
            raise NotImplementedError("post-LN: LayerNorm стоит на самом остаточном потоке, "
                                      "вращение не сохраняет функцию; используйте mode='online'")
        if spec.layernorm_with_bias:
            for n in spec.writers:
                _center_writer(get_linear(block, n))
        for norm_name, readers in spec.readers.items():
            norm = block.get_submodule(norm_name)
            fold_norm(norm, [get_linear(block, r) for r in readers], spec.layernorm_with_bias)
            if spec.layernorm_with_bias:   # OPT: LayerNorm -> RMSN
                setattr(block, norm_name, RMSN(norm.weight.numel(), norm.eps))
            else:                          # RMSNorm: γ = 1
                norm.weight.data.fill_(1.0)
        for readers in spec.readers.values():
            for r in readers:
                l = get_linear(block, r)
                Qd = Q.to(l.weight.device)
                _set_w(l, l.weight.to(Qd.dtype) @ Qd)
        for w in spec.writers:
            l = get_linear(block, w)
            Qd = Q.to(l.weight.device)
            _set_w(l, Qd.T @ l.weight.to(Qd.dtype))
            if l.bias is not None:
                l.bias.data = (l.bias.to(Qd.dtype) @ Qd).to(l.bias.dtype)
    elif mode != "online":
        raise ValueError(mode)
    if online_had:
        add_online_hadamard(block, spec.online_had)


@torch.no_grad()
def untie_lm_head(model) -> None:
    emb = model.get_input_embeddings().weight
    if model.lm_head.weight.data_ptr() == emb.data_ptr():
        model.lm_head.weight = nn.Parameter(emb.detach().clone())
    model.config.tie_word_embeddings = False


@torch.no_grad()
def rotate_embeddings(model, Q: torch.Tensor) -> None:
    if model.config.model_type == "opt":
        dec = model.model.decoder
        if dec.project_in is not None:
            raise NotImplementedError("OPT с project_in (word_embed_proj_dim != hidden) не поддержан")
        for emb in (dec.embed_tokens, dec.embed_positions):
            E = emb.weight.to(Q.dtype)
            E = E - E.mean(dim=1, keepdim=True)
            emb.weight.data = (E @ Q.to(E.device)).to(emb.weight.dtype)
    else:
        emb = model.model.embed_tokens
        emb.weight.data = (emb.weight.to(Q.dtype) @ Q.to(emb.weight.device)).to(emb.weight.dtype)


@torch.no_grad()
def rotate_head(model, Q: torch.Tensor) -> None:
    head = model.lm_head
    if model.config.model_type == "opt":
        dec = model.model.decoder
        if dec.project_out is not None:
            raise NotImplementedError("OPT с project_out не поддержан")
        ln = dec.final_layer_norm
        new = nn.Linear(head.in_features, head.out_features, bias=True,
                        device=head.weight.device, dtype=head.weight.dtype)
        new.weight.data.copy_(head.weight.data)
        new.bias.data.zero_()
        fold_norm(ln, [new], with_bias=True)
        dec.final_layer_norm = RMSN(ln.weight.numel(), ln.eps)
        model.lm_head = head = new
    else:
        norm = model.model.norm
        fold_norm(norm, [head], with_bias=False)
        norm.weight.data.fill_(1.0)
    _set_w(head, head.weight.to(Q.dtype) @ Q.to(head.weight.device))


@torch.no_grad()
def rotate_model(model, kind: str = "hadamard", seed: int = 0, mode: str = "full",
                 online_had: bool = True) -> torch.Tensor | None:
    """Повернуть полную модель на месте. Возвращает Q (или None в онлайн-режиме)."""
    spec = block_spec(model.config)
    layers = get_layers(model)
    if mode == "online":
        for blk in layers:
            rotate_block(blk, spec, None, mode="online", online_had=online_had)
        return None
    dev = model.get_input_embeddings().weight.device
    Q = make_rotation(model.config.hidden_size, kind, seed, device=dev)
    untie_lm_head(model)
    rotate_embeddings(model, Q)
    for blk in layers:
        rotate_block(blk, spec, Q, mode="full", online_had=online_had)
    rotate_head(model, Q)
    return Q
