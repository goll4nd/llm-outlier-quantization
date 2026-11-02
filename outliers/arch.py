"""Описание архитектур: какие нормы питают какие линейные слои, где пары для сглаживания,
кто пишет в остаточный поток. Всё задаётся относительно ОДНОГО декодерного блока, чтобы
те же функции работали и для полной модели, и в послойной оценке (Phi-3, Llama-3-8B).

Поддерживаются: opt (pre-LN и post-LN), llama / mistral / qwen2 (раздельные q/k/v и gate/up),
phi3 (слитые qkv_proj и gate_up_proj).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

LLAMA_LIKE = {"llama", "mistral", "qwen2", "qwen3"}


@dataclass
class Group:
    """Норма -> линейные слои, читающие её выход (сглаживание через норму)."""
    name: str
    norm: str              # имя модуля нормы внутри блока
    linears: list[str]


@dataclass
class Pair:
    """Линейный слой -> следующий линейный слой (сглаживание через веса, без нормы).

    up_rows: (start, stop) — какие строки up-слоя соответствуют входным каналам down-слоя
             (нужно для слитых qkv_proj / gate_up_proj у Phi-3). None — все строки.
    rep:     сколько голов запросов делит одну kv-голову (GQA); масштаб по входу down
             повторяется rep раз блоками по head_dim. Для FFN rep = 1, head_dim = None.
    """
    name: str
    up: str
    down: str
    up_rows: tuple[int, int] | None = None
    rep: int = 1
    head_dim: int | None = None


@dataclass
class Spec:
    arch: str
    pre_ln: bool
    groups: list[Group]
    pairs: list[Pair]
    readers: dict[str, list[str]]   # норма -> читатели (для слияния γ/β при вращении)
    writers: list[str]              # пишут в остаточный поток
    online_had: list[str]           # входы, получаемые нелинейно: Адамар ставится онлайн
    layernorm_with_bias: bool       # LayerNorm (OPT) или RMSNorm
    linears: list[str] = field(default_factory=list)


def arch_of(config) -> str:
    t = config.model_type
    if t == "opt":
        return "opt"
    if t in LLAMA_LIKE:
        return "llama"
    if t == "phi3":
        return "phi3"
    raise NotImplementedError(f"Архитектура {t} не поддерживается")


def get_layers(model) -> nn.ModuleList:
    if model.config.model_type == "opt":
        return model.model.decoder.layers
    return model.model.layers


def _attn_dims(config):
    nh = config.num_attention_heads
    nkv = getattr(config, "num_key_value_heads", None) or nh
    hidden = config.hidden_size
    hd = getattr(config, "head_dim", None) or hidden // nh
    return nh, nkv, hd


def block_spec(config) -> Spec:
    arch = arch_of(config)
    nh, nkv, hd = _attn_dims(config)
    if arch == "opt":
        if config.activation_function != "relu":
            raise NotImplementedError("пара fc1->fc2 корректна только для ReLU")
        lin = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"]
        return Spec(
            arch="opt", pre_ln=bool(config.do_layer_norm_before),
            groups=[Group("attn", "self_attn_layer_norm", lin[:3]), Group("mlp", "final_layer_norm", ["fc1"])],
            pairs=[Pair("vo", "self_attn.v_proj", "self_attn.out_proj", rep=1, head_dim=hd),
                   Pair("ffn", "fc1", "fc2")],
            readers={"self_attn_layer_norm": lin[:3], "final_layer_norm": ["fc1"]},
            writers=["self_attn.out_proj", "fc2"],
            online_had=["self_attn.out_proj", "fc2"],
            layernorm_with_bias=True, linears=lin)
    if arch == "llama":
        lin = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        return Spec(
            arch="llama", pre_ln=True,
            groups=[Group("attn", "input_layernorm", lin[:3]),
                    Group("mlp", "post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"])],
            pairs=[Pair("vo", "self_attn.v_proj", "self_attn.o_proj", rep=nh // nkv, head_dim=hd),
                   Pair("ffn", "mlp.up_proj", "mlp.down_proj")],
            readers={"input_layernorm": lin[:3], "post_attention_layernorm": ["mlp.gate_proj", "mlp.up_proj"]},
            writers=["self_attn.o_proj", "mlp.down_proj"],
            online_had=["self_attn.o_proj", "mlp.down_proj"],
            layernorm_with_bias=False, linears=lin)
    if arch == "phi3":
        lin = ["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]
        inter = config.intermediate_size
        v0 = nh * hd + nkv * hd
        return Spec(
            arch="phi3", pre_ln=True,
            groups=[Group("attn", "input_layernorm", ["self_attn.qkv_proj"]),
                    Group("mlp", "post_attention_layernorm", ["mlp.gate_up_proj"])],
            # Phi3MLP: gate, up = gate_up_proj(x).chunk(2) -> up — вторая половина строк
            pairs=[Pair("vo", "self_attn.qkv_proj", "self_attn.o_proj", up_rows=(v0, v0 + nkv * hd),
                        rep=nh // nkv, head_dim=hd),
                   Pair("ffn", "mlp.gate_up_proj", "mlp.down_proj", up_rows=(inter, 2 * inter))],
            readers={"input_layernorm": ["self_attn.qkv_proj"], "post_attention_layernorm": ["mlp.gate_up_proj"]},
            writers=["self_attn.o_proj", "mlp.down_proj"],
            online_had=["self_attn.o_proj", "mlp.down_proj"],
            layernorm_with_bias=False, linears=lin)
    raise AssertionError


def layer_prefix(model, i: int) -> str:
    return ("model.decoder.layers." if model.config.model_type == "opt" else "model.layers.") + str(i)
