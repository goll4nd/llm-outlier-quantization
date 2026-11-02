"""RTN-квантование и perplexity для крупных моделей (послойно, 8 ГБ VRAM).

Одна конфигурация на один запуск процесса: так замер VRAM не загрязняется
остатками предыдущих прогонов (см. README, раздел про замеры памяти).

Примеры:
    python run_large.py --model tinyllama --bits 16              # проверка методики
    python run_large.py --model llama3-8b --bits 4 --group-size 128
    python run_large.py --model phi3-mini --bits 4 --group-size 128 --max-windows 5   # smoke-test
"""
import argparse
import csv
import json
import os

import torch
import torch.nn as nn

from evaluation.layerwise import layerwise_perplexity
from models import MODELS
from quantizers.rtn import avg_bits_per_weight, rtn_quantize_tensor

REPOS = {
    **MODELS,
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",   # базовой версии Phi-3 не выпускали
    # неофициальная копия meta-llama/Meta-Llama-3-8B без gating (лицензия Llama 3 та же);
    # подлинность проверяется по FP16 ppl ≈ 6.14 на WikiText-2, окно 2048
    "llama3-8b": "NousResearch/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",         # блоки той же формы, что у Llama-3-8B
}

RESULTS = "results/large.csv"
FIELDS = ["model", "method", "bits", "group_size", "sym", "ppl", "avg_bits",
          "n_linear", "n_windows", "max_abs_hidden", "peak_vram_gb", "total_sec"]


def rtn_layer_fn(bits, group_size, sym):
    """Квантует все nn.Linear внутри трансформерного блока — тот же набор слоёв,
    что quantize_model_rtn на малых моделях (эмбеддинги и lm_head не трогаем)."""
    def fn(layer, idx):
        n = 0
        for _, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                m.weight.data = rtn_quantize_tensor(m.weight.data, bits, group_size, sym)
                n += 1
        return n
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help=f"одно из {list(REPOS)}")
    p.add_argument("--bits", type=int, default=4, help="16 = без квантования")
    p.add_argument("--group-size", type=int, default=128, help="-1 per-channel, 0 per-tensor")
    p.add_argument("--sym", type=int, default=0)
    p.add_argument("--acts-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-windows", type=int, default=None, help="для быстрой проверки; в CSV не пишется")
    args = p.parse_args()

    repo = REPOS.get(args.model, args.model)
    sym = bool(args.sym)
    fp16 = args.bits >= 16
    label = "FP16" if fp16 else f"RTN W{args.bits} g={args.group_size} {'sym' if sym else 'asym'}"
    print(f"=== {args.model} ({repo}) | {label} ===")

    torch.cuda.reset_peak_memory_stats()
    res = layerwise_perplexity(
        repo,
        layer_fn=None if fp16 else rtn_layer_fn(args.bits, args.group_size, sym),
        acts_device=args.acts_device,
        max_windows=args.max_windows,
    )
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nppl = {res['ppl']:.4f}   окон: {res['n_windows']}   "
          f"пик VRAM: {peak:.2f} ГБ   время: {res['sec']:.0f} c")

    if args.max_windows:
        print("(--max-windows задан: результат не сохраняется)")
        return

    row = {
        "model": args.model,
        "method": "fp16" if fp16 else "rtn",
        "bits": args.bits,
        "group_size": 0 if fp16 else args.group_size,
        "sym": sym,
        "ppl": round(res["ppl"], 4),
        "avg_bits": 16 if fp16 else round(avg_bits_per_weight(args.bits, args.group_size, sym), 3),
        "n_linear": res["n_linear"],
        "n_windows": res["n_windows"],
        "max_abs_hidden": max(s["max_abs"] for s in res["hidden_stats"]),
        "peak_vram_gb": round(peak, 2),
        "total_sec": round(res["sec"], 1),
    }
    os.makedirs("results/hidden_stats", exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

    tag = "fp16" if fp16 else f"w{args.bits}_g{args.group_size}_{'sym' if sym else 'asym'}"
    with open(f"results/hidden_stats/{args.model}_{tag}.json", "w") as f:
        json.dump(res["hidden_stats"], f, indent=1)


if __name__ == "__main__":
    main()
