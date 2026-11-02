import argparse, csv, os, time, torch
from evaluation.ppl import evaluate_perplexity
from quantizers.gptq import quantize_gptq, load_gptq
from pathlib import Path

RESULTS = "results/gptq.csv"
FIELDS = ["model", "method", "bits", "group_size", "desc_act", "seed",
          "ppl", "weights_gb", "quant_sec", "eval_sec"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="opt-125m")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--desc-act", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    path = quantize_gptq(args.model, args.bits, args.group_size,
                         bool(args.desc_act), seed=args.seed)
    quant_sec = time.time() - t0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_gptq(path)
    weights_gb = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1e9

    t1 = time.time()
    ppl = evaluate_perplexity(model, tok, verbose=False)
    eval_sec = time.time() - t1

    row = {
        "model": args.model, "method": "gptq", "bits": args.bits,
        "group_size": args.group_size, "desc_act": bool(args.desc_act),
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