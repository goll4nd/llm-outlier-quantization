"""Проверки корректности пакета outliers на крошечных случайных моделях (CPU, без скачивания весов).

  python test_outliers.py

Что проверяется:
  1. матрицы Адамара всех нужных порядков ортогональны (768, 3072, 5632, 2048, 1024, 8192, 14336);
  2. SmoothQuant не меняет функцию модели (OPT pre-LN, Llama с GQA, Phi-3): группы и пары;
  3. post-LN OPT: группы запрещены, пары допустимы;
  4. QuaRot-вращение не меняет функцию (полный режим; онлайн-режим для post-LN);
  5. послойная схема (буфер @ Q, transform_block, @ Q^T) совпадает с полной моделью;
  6. квантователи: 16 бит = тождество, не больше 2^b уровней, ошибка <= шаг/2, статическая шкала;
  7. на модели с искусственными выбросами SmoothQuant уменьшает ошибку W8A8 per-tensor;
  8. режим LLM.int8(): без выбросов совпадает с векторным INT8, выбросные каналы точны;
  9. поиск alpha возвращает значения из сетки, модель после него эквивалентна исходной.
"""
from __future__ import annotations

import copy
import math

import torch
from transformers import (LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM, Phi3Config, Phi3ForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM)

from outliers import ActCfg, PipeCfg, WCfg, prepare_model, quantize_act, quantize_weight
from outliers.arch import block_spec, get_layers, layer_prefix
from outliers.calib import collect_act_stats, relative
from outliers.hadamard import hadamard_matrix, hadamard_transform, is_true_hadamard
from outliers.layerwise_api import register_amax_hooks, remove_hooks, rotate_hidden, transform_block, unrotate_hidden
from outliers.quant import QLinear
from outliers.rotate import make_rotation

torch.manual_seed(0)
T, V = 24, 300
FAILS = []


def check(name, cond, extra=""):
    print(("OK  " if cond else "FAIL"), name, extra)
    if not cond:
        FAILS.append(name)


def perturb_norms(model):
    """Неединичные γ и ненулевые β — иначе слияние норм проверялось бы вхолостую."""
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm) or type(m).__name__.endswith("RMSNorm"):
            with torch.no_grad():
                m.weight.uniform_(0.5, 1.5)
                if getattr(m, "bias", None) is not None:
                    m.bias.normal_(0, 0.1)
    for m in model.modules():   # смещения линейных слоёв OPT тоже ненулевые
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            with torch.no_grad():
                m.bias.normal_(0, 0.02)
    return model


def tiny(kind, post_ln=False):
    if kind == "opt":
        cfg = OPTConfig(vocab_size=V, hidden_size=48, ffn_dim=192, num_hidden_layers=2, num_attention_heads=4,
                        max_position_embeddings=64, word_embed_proj_dim=48, do_layer_norm_before=not post_ln,
                        pad_token_id=1, bos_token_id=2, eos_token_id=2)
        m = OPTForCausalLM(cfg)
    elif kind == "llama":   # GQA: 4 головы, 2 kv; intermediate 176 = 44·4 -> Пэли 44
        cfg = LlamaConfig(vocab_size=V, hidden_size=48, intermediate_size=176, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        m = LlamaForCausalLM(cfg)
    elif kind == "phi3":
        cfg = Phi3Config(vocab_size=V, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                         original_max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        m = Phi3ForCausalLM(cfg)
    elif kind == "qwen2":   # смещения у q/k/v, GQA
        cfg = Qwen2Config(vocab_size=V, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                          tie_word_embeddings=True)
        m = Qwen2ForCausalLM(cfg)
    else:
        raise ValueError(kind)
    return perturb_norms(m).float().eval()


def logits(m, ids):
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits.float()


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


ids = torch.randint(3, V, (1, T))
calib = [torch.randint(3, V, (1, T)) for _ in range(6)]

# 1 ---------------------------------------------------------------- Адамар
for n in (48, 768, 1024, 2048, 3072, 5632):
    H = hadamard_matrix(n)
    check(f"Hadamard n={n} orthogonal", torch.allclose(H @ H.T, torch.eye(n, dtype=H.dtype), atol=1e-10),
          f"(true Hadamard: {is_true_hadamard(n)})")
for n in (8192, 14336):
    x = torch.randn(4, n, dtype=torch.float64)
    y = hadamard_transform(x, compute_dtype=torch.float64)
    check(f"Hadamard n={n} norm-preserving", torch.allclose(y.norm(dim=1), x.norm(dim=1)) and
          torch.allclose(y @ y.T, x @ x.T, atol=1e-8), f"(true Hadamard: {is_true_hadamard(n)})")

# 2-4 -------------------------------------------------------------- инвариантность преобразований
cases = [
    ("opt", False, PipeCfg(method="smoothquant", smooth_targets="groups+pairs", alpha=0.6, pair_alpha=0.4)),
    ("llama", False, PipeCfg(method="smoothquant", smooth_targets="groups+pairs", alpha=0.7, pair_alpha=0.3)),
    ("phi3", False, PipeCfg(method="smoothquant", smooth_targets="groups+pairs", alpha=0.5)),
    ("opt", True, PipeCfg(method="smoothquant", smooth_targets="pairs", alpha=0.5)),
    ("opt", True, PipeCfg(method="smoothquant", smooth_targets="pairs", alpha=0.9, ax_floor=0.05)),
    ("qwen2", False, PipeCfg(method="smoothquant", smooth_targets="groups+pairs", alpha=0.8)),
    ("qwen2", False, PipeCfg(method="rotate")),
    ("llama", False, PipeCfg(method="smoothquant", smooth_targets="attn,ffn", alpha=0.8, s_floor_q=0.9)),
    ("opt", False, PipeCfg(method="smoothquant", smooth_targets="mlp,vo", alpha=0.6, s_floor_q=0.5,
                           act_stat="meanmax")),
    ("phi3", False, PipeCfg(method="smoothquant", smooth_targets="ffn", alpha=0.8, s_floor_q=0.99,
                            act_stat="meanmax")),
    ("llama", False, PipeCfg(method="smoothquant", smooth_targets="groups+pairs", alpha=0.85, ax_floor=0.05)),
    ("opt", False, PipeCfg(method="rotate")),
    ("opt", False, PipeCfg(method="rotate", rotate_kind="orthogonal")),
    ("llama", False, PipeCfg(method="rotate")),
    ("phi3", False, PipeCfg(method="rotate")),
    ("opt", True, PipeCfg(method="rotate", rotate_mode="online")),
    ("llama", False, PipeCfg(method="smoothrot", alpha=0.5)),
    ("opt", False, PipeCfg(method="smoothrot", alpha=0.5)),
]
for kind, post, cfg in cases:
    m = tiny(kind, post)
    ref = logits(m, ids)
    prepare_model(m, cfg, calib, log=lambda *a: None)       # все биты = 16
    tag = f"{kind}{' post-LN' if post else ''} {cfg.method} {cfg.smooth_targets if cfg.method == 'smoothquant' else cfg.rotate_mode + '/' + cfg.rotate_kind}"
    r = rel(logits(m, ids), ref)
    check(f"invariance: {tag}", r < 1e-4, f"rel={r:.2e}")

m = tiny("opt", post_ln=True)
try:
    prepare_model(m, PipeCfg(method="smoothquant", smooth_targets="groups"), calib, log=lambda *a: None)
    check("post-LN OPT: groups rejected", False)
except ValueError:
    check("post-LN OPT: groups rejected", True)
m = tiny("opt", post_ln=True)
try:
    prepare_model(m, PipeCfg(method="rotate", rotate_mode="full"), calib, log=lambda *a: None)
    check("post-LN OPT: full rotation rejected", False)
except NotImplementedError:
    check("post-LN OPT: full rotation rejected", True)

# 5 ---------------------------------------------------------------- послойная схема
for kind in ("llama", "phi3"):
    for method in ("rotate", "smoothquant"):
        m = tiny(kind)
        ref = logits(m, ids)
        cfg = PipeCfg(method=method, smooth_targets="groups+pairs", alpha=0.6)
        layers = get_layers(m)
        captured = {}
        h0 = m.model.layers[0].register_forward_pre_hook(
            lambda mod, args, kw: captured.update(args=args, kw=kw), with_kwargs=True)
        logits(m, ids)
        h0.remove()
        # проход 1: max|X_j| по блокам (на тех же данных — для проверки достаточно)
        scales = {}
        for i, blk in enumerate(layers):
            scales[i] = {}
        hs_all = []
        for i, blk in enumerate(layers):
            hs_all += register_amax_hooks(blk, scales[i])
        for x in calib:
            logits(m, x)
        remove_hooks(hs_all)
        Q = make_rotation(m.config.hidden_size, "hadamard", 0) if method == "rotate" else None
        with torch.no_grad():
            h = captured["args"][0] if captured["args"] else captured["kw"]["hidden_states"]
            kw = {k: v for k, v in captured["kw"].items() if k != "hidden_states"}
            if Q is not None:
                h = rotate_hidden(h, Q)
            for i, blk in enumerate(layers):
                transform_block(blk, m.config, cfg, scales[i], Q)
                out = blk(h, **kw)
                h = out[0] if isinstance(out, tuple) else out
            if Q is not None:
                h = unrotate_hidden(h, Q)
            lw = m.lm_head(m.model.norm(h)).float()
        r = rel(lw, ref)
        check(f"layerwise scheme: {kind} {method}", r < 1e-4, f"rel={r:.2e}")

# 5b --------------------------------------------------------------- точки и форма масштаба
from outliers.smooth import _s, parse_targets
check("parse_targets", parse_targets("groups") == {"attn", "mlp"} and parse_targets("pairs") == {"vo", "ffn"}
      and parse_targets("groups+pairs") == {"attn", "mlp", "vo", "ffn"} and parse_targets("attn, ffn") == {"attn", "ffn"})
try:
    parse_targets("down")
    check("parse_targets rejects unknown", False)
except ValueError:
    check("parse_targets rejects unknown", True)
ax = torch.rand(200) + 0.05
ax[[3, 50]] = torch.tensor([80.0, 40.0])
ax[7] = 1e-6                                              # «мёртвый» канал
aw = torch.rand(200) + 0.1
s0 = _s(ax, aw, 0.8)
s9 = _s(ax, aw, 0.8, s_floor_q=0.9)
q9 = torch.quantile(s0, 0.9)
check("quantile floor: channels below Q_p share one scale, top channels unchanged",
      torch.allclose(s9[s0 <= q9], q9.expand(int((s0 <= q9).sum()))) and torch.equal(s9[s0 > q9], s0[s0 > q9]))
check("quantile floor lifts dead channel", s0[7] < 1e-3 and s9[7] >= q9)
# только выбранные точки: при targets='ffn' нормы и v/o не меняются
m = tiny("llama")
before = {n: p_.detach().clone() for n, p_ in m.named_parameters()}
prepare_model(m, PipeCfg(method="smoothquant", smooth_targets="ffn", alpha=0.8), calib, log=lambda *a: None)
changed = {n.replace(".linear.", ".") for n, p_ in m.named_parameters()
           if n.replace(".linear.", ".") in before and not torch.equal(p_, before[n.replace(".linear.", ".")])}
check("targets='ffn' changes only up_proj / down_proj",
      changed and all(("up_proj" in n or "down_proj" in n) for n in changed), str(sorted(changed))[:120])

# 6 ---------------------------------------------------------------- квантователи
x = torch.randn(5, 64) * torch.logspace(-1, 1, 64)
check("act 16 bit identity", torch.equal(quantize_act(x, ActCfg(bits=16)), x))
for b in (4, 8):
    q = quantize_act(x, ActCfg(bits=b, granularity="per_token"))
    lv = max(len(torch.unique(r)) for r in q)
    step = (x.abs().amax(-1, keepdim=True) / (2 ** (b - 1) - 1))
    check(f"act per-token {b} bit: <= 2^b levels, err <= step/2",
          lv <= 2 ** b and bool(((q - x).abs() <= step / 2 + 1e-6).all()))
w = torch.randn(32, 256)
for gran in ("per_channel", "per_tensor", "group"):
    for sym in (True, False):
        qw = quantize_weight(w, WCfg(bits=4, granularity=gran, group_size=64, sym=sym))
        rows = qw.reshape(-1, 64) if gran == "group" else (qw.reshape(1, -1) if gran == "per_tensor" else qw)
        lv = max(len(torch.unique(r)) for r in rows)
        check(f"weight 4 bit {gran} {'sym' if sym else 'asym'}: <= 16 levels", lv <= 16, f"(max {lv})")
e_plain = (quantize_weight(w, WCfg(bits=3)) - w).pow(2).sum()
e_clip = (quantize_weight(w, WCfg(bits=3, mse_clip=True)) - w).pow(2).sum()
check("weight MSE clip does not increase error", e_clip <= e_plain * 1.001, f"{e_clip:.3f} vs {e_plain:.3f}")

# 7 ---------------------------------------------------------------- эффект SmoothQuant на выбросах
# Детерминированная проверка на уровне одного линейного слоя (свой генератор, не зависит от порядка тестов):
# W8A8 со статической per-tensor шкалой, вход с выбросами двух видов.
from outliers.smooth import _s as _scale
g = torch.Generator().manual_seed(7)
Wt = torch.randn(96, 128, generator=g) * 0.05
X_ch = torch.randn(256, 128, generator=g)
X_ch[:, [5, 77]] *= 50.0                               # канальные выбросы (как вход q/k/v)
X_tok = torch.randn(256, 128, generator=g) * 0.1
X_tok[[0, 131], 40] = 150.0                            # токенные всплески в одном канале (как вход down_proj)
a8 = ActCfg(bits=8, granularity="per_tensor", static=True)
for name_, X in (("channel outliers", X_ch), ("token spikes in one channel", X_tok)):
    ref_ = X @ Wt.T
    def err_(Xs, Ws):
        Xq = quantize_act(Xs, a8, static_scale=Xs.abs().amax())
        return ((Xq @ quantize_weight(Ws, WCfg(bits=8)).T - ref_).pow(2).mean() / ref_.pow(2).mean()).item()
    e_naive = err_(X, Wt)
    sc = _scale(X.abs().amax(0), Wt.abs().amax(0), 0.5)
    e_sq = err_(X / sc, Wt * sc)
    sc9 = _scale(X.abs().amax(0), Wt.abs().amax(0), 0.5, s_floor_q=0.9)
    e_q9 = err_(X / sc9, Wt * sc9)
    check(f"SmoothQuant reduces static W8A8 error, layer level ({name_})", e_sq < 0.5 * e_naive and e_q9 < 0.5 * e_naive,
          f"naive={e_naive:.2e} sq={e_sq:.2e} sq+floor0.9={e_q9:.2e}")


def with_outliers(kind, seed=21):
    torch.manual_seed(seed)
    m = tiny(kind)
    with torch.no_grad():
        for blk in get_layers(m):
            spec = block_spec(m.config)
            for g_ in spec.groups:
                norm = blk.get_submodule(g_.norm)
                norm.weight[[3, 17]] *= 60.0        # два «выбросных канала»
    return m

# На уровне крошечной случайной модели эффект зависит от инициализации — выводится для сведения, не проверяется.
for kind in ("opt", "llama"):
    res_ = {"naive": [], "smoothquant": []}
    for seed in (11, 12, 13):
        base = with_outliers(kind, seed)
        ref = logits(base, ids)
        for name, method in (("naive", "none"), ("smoothquant", "smoothquant")):
            m = copy.deepcopy(base)
            prepare_model(m, PipeCfg(method=method, alpha=0.5, w=WCfg(bits=8, granularity="per_channel"),
                                     a=ActCfg(bits=8, granularity="per_tensor", static=True)), calib, log=lambda *a: None)
            res_[name].append(rel(logits(m, ids), ref))
    print(f"INFO tiny {kind}, W8A8 static, 3 seeds: naive={[f'{v:.2e}' for v in res_['naive']]} "
          f"sq={[f'{v:.2e}' for v in res_['smoothquant']]}")
torch.manual_seed(0)

# 8 ---------------------------------------------------------------- LLM.int8()
lin = torch.nn.Linear(64, 16)
q = QLinear(copy.deepcopy(lin))
q.outlier_threshold = 1e9
xx = torch.randn(7, 64)
y_int8 = q(xx)
y_vec = quantize_act(xx, ActCfg(bits=8)) @ quantize_weight(lin.weight.data, WCfg(bits=8)).T + lin.bias
check("LLM.int8 without outliers == vector-wise INT8", torch.allclose(y_int8, y_vec, atol=1e-5))
xx[:, 5] *= 100
q.outlier_threshold = 6.0
y = q(xx)
check("LLM.int8 outlier dimension is detected", q.int8_stats["outlier_dims_max"] >= 1)
check("LLM.int8 close to FP with outlier", rel(y, lin(xx).detach()) < 2e-2, f"rel={rel(y, lin(xx).detach()):.2e}")

# 8b --------------------------------------------------------------- захват входов: форма [n, T, C]
from outliers.calib import capture_inputs
for kind in ("opt", "llama"):
    m = tiny(kind)
    spec = block_spec(m.config)
    names = [f"{layer_prefix(m, 0)}.{n}" for n in spec.linears]
    cap = capture_inputs(m, calib[:3], names)
    shapes_ok = all(v.shape[:2] == (3, T) for v in cap.values())
    check(f"capture_inputs shapes [n, T, C] ({kind}, incl. fc1/fc2)", shapes_ok,
          str({k.rsplit('.', 1)[-1]: tuple(v.shape) for k, v in cap.items()}))

# 9 ---------------------------------------------------------------- поиск alpha
for kind in ("opt", "llama"):
    m = with_outliers(kind)
    ref = logits(m, ids)
    cfg = PipeCfg(method="smoothquant", smooth_targets="groups+pairs", alpha_mode="search",
                  alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0], search_n=4,
                  search_loss="tokrel" if kind == "llama" else "nopos0", search_identity=True,
                  w=WCfg(bits=16), a=ActCfg(bits=8, granularity="per_tensor", static=True))
    # веса 16 бит: после квантования активаций модель уже не эквивалентна, поэтому проверяем
    # инвариантность отдельно — на копии с отключёнными квантователями
    info = prepare_model(m, cfg, calib, log=lambda *a: None)
    ok_grid = all(v is None or v in cfg.alpha_grid for d in info["alphas"].values() for v in d.values())
    for mod in m.modules():
        if isinstance(mod, QLinear):
            mod.act_cfg = ActCfg()
    r = rel(logits(m, ids), ref)
    check(f"alpha search ({kind}): alphas in grid, model invariant", ok_grid and r < 1e-4,
          f"alphas={info['alphas'][0]} rel={r:.2e}")

print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if not FAILS else f"\nПРОВАЛЕНО: {len(FAILS)}: {FAILS}")
raise SystemExit(1 if FAILS else 0)
