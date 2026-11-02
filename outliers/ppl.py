"""Perplexity на WikiText-2 по протоколу работы (README, «Протокол оценки»):
весь test-сплит склеивается через "\\n\\n" и токенизируется целиком; непересекающиеся окна по 2048,
хвост отбрасывается; сумма NLL по всем предсказанным токенам; логиты в float32 кусками по 512.

Это копия протокола для самостоятельности модуля. Если удобнее, замените вызов на свою
evaluation/ppl.py — первой проверкой должно быть совпадение FP16 (OPT-125M 27.656, TinyLlama 7.708).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_ppl(model, ids: torch.Tensor, seqlen: int = 2048, chunk: int = 512,
             max_windows: int | None = None) -> float:
    dev = next(model.parameters()).device
    n = ids.shape[1] // seqlen
    if max_windows:
        n = min(n, max_windows)
    base = model.model                       # OPTModel / LlamaModel / Phi3Model: финальная норма внутри
    head = model.lm_head
    nll, cnt = 0.0, 0
    for i in range(n):
        x = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        h = base(input_ids=x, use_cache=False).last_hidden_state
        if not torch.isfinite(h).all():
            raise FloatingPointError(f"окно {i}: inf/nan в скрытых состояниях")
        for a in range(0, seqlen - 1, chunk):
            b = min(a + chunk, seqlen - 1)
            logits = head(h[:, a:b]).float()
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, a + 1:b + 1].reshape(-1),
                                   reduction="sum").item()
            cnt += b - a
    return math.exp(nll / cnt)
