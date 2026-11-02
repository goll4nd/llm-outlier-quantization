"""Статистики выбросов на входах линейных слоёв + act_scales для SmoothQuant.

  python collect_act_stats.py --model opt-125m
  python collect_act_stats.py --model tinyllama --calib-bos          # абляция: BOS на позиции 0
  python collect_act_stats.py --model tinyllama --transform rotate   # те же статистики после QuaRot-вращения
  python collect_act_stats.py --model tinyllama --transform smoothquant --alpha 0.5

Пишет results/act_stats/<model>[_<transform>][_bos].csv (по слою на строку) и
<...>_act_scales.pt (max|X_j| по каналам; для исходной модели его можно передать в run_actquant.py --act-scales).
"""
from __future__ import annotations

import argparse
import csv
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from outliers import MODELS, PipeCfg, prepare_model
from outliers.calib import collect_act_stats, get_calib_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--calib", default="wikitext2", choices=["wikitext2", "pile"])
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--len", type=int, default=512)
    p.add_argument("--calib-bos", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transform", default="none", choices=["none", "smoothquant", "rotate"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rotate-mode", default="full", choices=["full", "online"])
    p.add_argument("--outdir", default="results/act_stats")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.float16).to(dev).eval()
    samples = get_calib_samples(tok, a.calib, a.n, a.len, a.seed, a.calib_bos)
    if a.transform != "none":
        # отдельная выборка для подбора преобразования, чтобы не мерить на тех же данных
        fit = get_calib_samples(tok, a.calib, a.n, a.len, a.seed + 1000, a.calib_bos)
        prepare_model(model, PipeCfg(method=a.transform, alpha=a.alpha, rotate_mode=a.rotate_mode), fit)

    amax, rows = collect_act_stats(model, samples)
    os.makedirs(a.outdir, exist_ok=True)
    stem = f"{a.outdir}/{a.model}" + (f"_{a.transform}" if a.transform != "none" else "") + \
           ("_bos" if a.calib_bos else "") + (f"_{a.calib}" if a.calib != "wikitext2" else "")
    torch.save(amax, stem + "_act_scales.pt")
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'слой':<34}{'max|x|':>9}{'эксцесс':>10}{'ch_ratio':>10}{'tok_ratio':>10}{'pos0':>6}{'≥6':>5}{'massive':>8}")
    for r in sorted(rows, key=lambda r: -r["max_abs"])[:15]:
        print(f"{r['name'][-34:]:<34}{r['max_abs']:>9.1f}{r['kurtosis']:>10.1f}{r['ch_ratio']:>10.1f}"
              f"{r['tok_ratio_mean']:>10.1f}{r['argmax_pos0']:>6.2f}{r['n_ch_ge6_max']:>5}{r['massive']:>8}")
    print(f"-> {stem}.csv, {stem}_act_scales.pt")


if __name__ == "__main__":
    main()
