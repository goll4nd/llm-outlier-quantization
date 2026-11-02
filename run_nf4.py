import argparse, csv, os, time, torch
from evaluation.ppl import evaluate_perplexity
from quantizers.bnb_nf4 import load_bnb_4bit

RESULTS = "results/nf4.csv"
FIELDS = ["model", "method", "quant_type", "double_quant", "ppl",
          "weights_gb", "peak_vram_gb", "eval_sec"]


def run(model_key, quant_type, double_quant):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_bnb_4bit(model_key, quant_type, double_quant)

    weights_gb = torch.cuda.memory_allocated() / 1e9

    t0 = time.time()
    ppl = evaluate_perplexity(model, tok, verbose=False)
    elapsed = time.time() - t0

    row = {
        "model": model_key, "method": "bnb",
        "quant_type": quant_type, "double_quant": double_quant,
        "ppl": round(ppl, 4),
        "weights_gb": round(weights_gb, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "eval_sec": round(elapsed, 1),
    }
    del model
    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="opt-125m")
    p.add_argument("--quant-type", default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--double-quant", type=int, default=1)
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        row = run(args.model, args.quant_type, bool(args.double_quant))
        print(f"ppl = {row['ppl']}  веса = {row['weights_gb']} ГБ")
        w.writerow(row)


if __name__ == "__main__":
    main()