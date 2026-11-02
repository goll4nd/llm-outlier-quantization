"""Сверка собственной реализации LLM.int8() (fake-quant, outliers/quant.py) с bitsandbytes.

  python run_bnb_int8.py --model opt-125m --threshold 6     # LLM.int8() со смешанной декомпозицией
  python run_bnb_int8.py --model opt-125m --threshold 0     # векторный INT8 без декомпозиции

bitsandbytes квантует все nn.Linear, кроме lm_head (как и наша реализация внутри блоков). Отличие: у OPT-350M
есть project_in / project_out (512 <-> 1024) вне блоков — bitsandbytes квантует и их, наша реализация нет.
Протокол perplexity тот же (outliers/ppl.py). Результат дописывается в results/actquant.csv с method=bnb_int8.
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from outliers import MODELS
from outliers.calib import get_test_ids
from outliers.ppl import eval_ppl
from run_actquant import append_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--threshold", type=float, default=6.0, help="0 = без смешанной декомпозиции")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--out", default="results/actquant.csv")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    name = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=a.threshold)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg, dtype=torch.float16,
                                                 device_map={"": 0}).eval()
    n8 = sum(1 for m in model.modules() if type(m).__name__ == "Linear8bitLt")
    t0 = time.time()
    ppl = eval_ppl(model, get_test_ids(tok), max_windows=a.max_windows)
    print(f"{a.model} bnb_int8 threshold={a.threshold}: ppl = {ppl:.4f}  (Linear8bitLt: {n8}, eval {time.time() - t0:.0f} s)")
    if a.max_windows:
        return
    append_row(a.out, dict(model=a.model, method="bnb_int8", tag=a.tag, w_bits=8, w_gran="per_channel",
                           a_bits=8, a_gran="per_token", a_static=False, quant_bmm=False,
                           calib=f"llm_int8_threshold={a.threshold}", seed=0, ppl=round(ppl, 4),
                           eval_s=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
