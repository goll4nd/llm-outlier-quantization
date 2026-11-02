"""Оценка уже сохранённой GPTQ-модели в отдельном процессе, строго в FP16.

Зачем: (1) после квантования в том же процессе остаётся занятая память,
(2) GPTQModel по умолчанию загружает модель в «родном» dtype (у Llama-3 это
BF16), а протокол требует FP16. Здесь dtype задаётся явно и проверяется.

Пример:
    ls quantized/
    python eval_gptq_saved.py --path quantized/<папка> --model llama3-8b \
        --group-size 128 --desc-act 1 --seed 0
"""
import argparse
import csv
import os
import time

import torch
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

from evaluation.ppl import evaluate_perplexity

RESULTS = "results/gptq.csv"
FIELDS = ["model", "method", "bits", "group_size", "desc_act", "seed", "ppl",
          "weights_gb", "quant_sec", "eval_sec"]


def load_fp16(path):
    try:
        model = GPTQModel.load(path, device="cuda", dtype=torch.float16)
    except TypeError:                      # старое имя аргумента
        model = GPTQModel.load(path, device="cuda", torch_dtype=torch.float16)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="папка с квантованной моделью")
    p.add_argument("--model", required=True, help="ключ модели для CSV")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--desc-act", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quant-sec", default="", help="время квантования из лога, если известно")
    p.add_argument("--no-save", action="store_true", help="только напечатать, в CSV не писать")
    args = p.parse_args()

    torch.cuda.reset_peak_memory_stats()
    model = load_fp16(args.path)
    weights_gb = torch.cuda.memory_allocated() / 1e9

    hf = getattr(model, "model", model)
    dtype = hf.get_input_embeddings().weight.dtype
    print(f"dtype эмбеддингов: {dtype}   веса на GPU: {weights_gb:.3f} ГБ")
    if dtype != torch.float16:
        raise SystemExit("Модель загрузилась не в FP16 — оценка не соответствует протоколу")

    tok = AutoTokenizer.from_pretrained(args.path, use_fast=True)
    t0 = time.time()
    ppl = evaluate_perplexity(model, tok, verbose=True)
    eval_sec = time.time() - t0
    print(f"\nppl = {ppl:.4f}   оценка {eval_sec:.0f} c   "
          f"пик VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} ГБ")

    if args.no_save:
        return
    row = {
        "model": args.model, "method": "gptq", "bits": args.bits,
        "group_size": args.group_size, "desc_act": bool(args.desc_act),
        "seed": args.seed, "ppl": round(ppl, 4),
        "weights_gb": round(weights_gb, 3), "quant_sec": args.quant_sec,
        "eval_sec": round(eval_sec, 1),
    }
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print("Строка добавлена в", RESULTS)


if __name__ == "__main__":
    main()
