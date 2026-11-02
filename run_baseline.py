import argparse, csv, os, time, torch
from models import load_model
from evaluation.ppl import evaluate_perplexity
from quantizers.rtn import quantize_model_rtn, avg_bits_per_weight

RESULTS = "results/baseline.csv"
FIELDS = ["model", "bits", "group_size", "sym", "ppl", "delta_ppl",
          "avg_bits", "n_layers", "peak_vram_gb", "eval_sec"]


def run(model_key, bits, group_size, sym, ppl_fp16=None):
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(model_key)

    if bits < 16:
        stats = quantize_model_rtn(model, bits, group_size, sym)
    else:
        stats = {"layers": 0}

    t0 = time.time()
    ppl = evaluate_perplexity(model, tok, verbose=False)
    elapsed = time.time() - t0

    row = {
        "model": model_key, "bits": bits, "group_size": group_size,
        "sym": sym, "ppl": round(ppl, 4),
        "delta_ppl": round(ppl - ppl_fp16, 4) if ppl_fp16 else "",
        "avg_bits": round(avg_bits_per_weight(bits, group_size, sym), 3)
                    if bits < 16 else 16,
        "n_layers": stats["layers"],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "eval_sec": round(elapsed, 1),
    }
    del model
    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="opt-125m")
    args = p.parse_args()

    configs = [
        (16, 0, False),        # baseline
        (8, 128, False),
        (4, -1, False),        # per-channel
        (4, 128, False),
        (4, 128, True),        # симметричная
        (4, 64, False),
        (3, 128, False),
        (2, 128, False),
    ]

    os.makedirs("results", exist_ok=True)
    new_file = not os.path.exists(RESULTS)
    ppl_fp16 = None

    with open(RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for bits, gs, sym in configs:
            label = "FP16" if bits == 16 else f"W{bits} g={gs} {'sym' if sym else 'asym'}"
            print(f"\n=== {args.model} | {label} ===")
            row = run(args.model, bits, gs, sym, ppl_fp16)
            if bits == 16:
                ppl_fp16 = row["ppl"]
            print(f"ppl = {row['ppl']}  (Δ = {row['delta_ppl']})")
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()