import argparse, csv, os, time, torch
from pathlib import Path
from evaluation.ppl import evaluate_perplexity
from quantizers.awq import quantize_awq, load_awq

# НЕ РАБОТАЕТ. Эталонное торчевое ядро AWQ в GPTQModel 7.7.0 даёт
# ppl ~2300-2900 вместо ожидаемых ~29. Подробности в README,
# раздел "Ограничения". Специализированные CUDA-ядра требуют
# CUDA Toolkit, которого нет в окружении.


RESULTS = "results/awq.csv"
FIELDS = ["model", "method", "bits", "group_size", "n_samples", "seed",
          "ppl", "weights_gb", "quant_sec", "eval_sec"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="opt-125m")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    path = quantize_awq(args.model, args.bits, args.group_size,
                        n_samples=args.n_samples, seed=args.seed)
    quant_sec = time.time() - t0

    torch.cuda.empty_cache()
    model, tok = load_awq(path)
    weights_gb = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1e9

    t1 = time.time()
    ppl = evaluate_perplexity(model, tok, verbose=False)
    eval_sec = time.time() - t1

    row = {
        "model": args.model, "method": "awq", "bits": args.bits,
        "group_size": args.group_size, "n_samples": args.n_samples,
        "seed": args.seed, "ppl": round(ppl, 4),
        "weights_gb": round(weights_gb, 3),
        "quant_sec": round(quant_sec, 1), "eval_sec": round(eval_sec, 1),
    }

    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"\nppl = {row['ppl']}  веса = {row['weights_gb']} ГБ  "
          f"квантование {row['quant_sec']} c")


if __name__ == "__main__":
    main()