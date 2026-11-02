"""Прогон W/A-квантования с борьбой с выбросами на полной модели (OPT-125M/350M, TinyLlama).

Примеры:
  # FP16-контроль протокола (должно совпасть с README: 27.656)
  python run_actquant.py --model opt-125m --w-bits 16 --a-bits 16

  # наивный W8A8: per-token (O1), per-tensor динамический (O2), per-tensor статический (O3)
  python run_actquant.py --model opt-125m --a-gran per_token
  python run_actquant.py --model opt-125m --a-gran per_tensor
  python run_actquant.py --model opt-125m --a-gran per_tensor --a-static

  # SmoothQuant, глобальный alpha; то же с послойным поиском alpha
  python run_actquant.py --model opt-125m --method smoothquant --alpha 0.5 --a-gran per_tensor --a-static
  python run_actquant.py --model opt-125m --method smoothquant --alpha-mode search --a-gran per_tensor --a-static

  # QuaRot-RTN W4A4 (веса per-channel sym + MSE-отсечение, активации per-token с отсечением 0.9)
  python run_actquant.py --model tinyllama --method rotate --w-bits 4 --w-mse-clip --a-bits 4 --a-clip 0.9

  # LLM.int8() (fake-quant): смешанная декомпозиция с порогом 6
  python run_actquant.py --model opt-125m --method llm_int8

Каждая конфигурация — отдельным процессом (см. README: VRAM не освобождается полностью).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from outliers import MODELS, ActCfg, PipeCfg, WCfg, prepare_model
from outliers.calib import get_calib_samples, get_test_ids
from outliers.pipeline import int8_report
from outliers.ppl import eval_ppl


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--method", default="none", choices=["none", "smoothquant", "rotate", "smoothrot", "llm_int8"])
    # веса
    p.add_argument("--w-bits", type=int, default=8)
    p.add_argument("--w-gran", default="per_channel", choices=["per_channel", "per_tensor", "group"])
    p.add_argument("--w-group", type=int, default=128)
    p.add_argument("--w-asym", action="store_true")
    p.add_argument("--w-mse-clip", action="store_true")
    # активации
    p.add_argument("--a-bits", type=int, default=8)
    p.add_argument("--a-gran", default="per_token", choices=["per_token", "per_tensor"])
    p.add_argument("--a-static", action="store_true")
    p.add_argument("--a-clip", type=float, default=1.0)
    p.add_argument("--quant-bmm", action="store_true", help="квантовать выходы q/k/v (входы BMM), как SmoothQuant")
    p.add_argument("--a-skip", default="", help="через запятую: суффиксы модулей с входом в 16 битах, напр. down_proj")
    # SmoothQuant
    p.add_argument("--smooth-targets", default="groups",
                   help="groups | pairs | groups+pairs | список из attn, mlp, vo, ffn через запятую")
    p.add_argument("--s-floor-q", type=float, default=0.0,
                   help="квантильный пол масштаба: s <- max(s, Q_p(s)); 0 = оригинальный SmoothQuant")
    p.add_argument("--act-stat", default="max", choices=["max", "meanmax"],
                   help="статистика активаций для масштаба сглаживания")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--pair-alpha", type=float, default=None)
    p.add_argument("--ax-floor", type=float, default=0.0,
                   help="нижняя граница max|X_j| как доля медианы по каналам (0 = оригинальный SmoothQuant)")
    p.add_argument("--alpha-mode", default="global", choices=["global", "search"])
    p.add_argument("--alpha-grid", default="0:1:0.05", help="start:stop:step")
    p.add_argument("--search-n", type=int, default=8)
    p.add_argument("--search-loss", default="mse", choices=["mse", "nopos0", "tokrel"])
    p.add_argument("--search-identity", action="store_true", help="разрешить слою не сглаживаться")
    p.add_argument("--act-scales", default=None, help="готовый .pt с max|X_j| (collect_act_stats.py)")
    # вращение
    p.add_argument("--rotate-mode", default="full", choices=["full", "online"])
    p.add_argument("--rotate-kind", default="hadamard", choices=["hadamard", "orthogonal"])
    p.add_argument("--rot-seed", type=int, default=0)
    # LLM.int8()
    p.add_argument("--int8-threshold", type=float, default=6.0)
    # калибровка и оценка
    p.add_argument("--calib", default="wikitext2", choices=["wikitext2", "pile"])
    p.add_argument("--calib-n", type=int, default=128)
    p.add_argument("--calib-len", type=int, default=512)
    p.add_argument("--calib-bos", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=None, help="быстрая проверка; в CSV не пишется")
    p.add_argument("--out", default="results/actquant.csv")
    p.add_argument("--tag", default="")
    return p.parse_args()


def append_row(path: str, row: dict) -> None:
    """Дописать строку; если появились новые колонки, переписать файл с объединённым заголовком."""
    rows, fields = [], list(row)
    if os.path.exists(path):
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            old = r.fieldnames or []
            rows = list(r)
        fields = old + [k for k in row if k not in old]
        if fields == old:
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            return
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(dev).eval()

    a0, a1, st = (float(v) for v in args.alpha_grid.split(":"))
    grid = [round(a0 + st * i, 4) for i in range(int(round((a1 - a0) / st)) + 1)]
    cfg = PipeCfg(
        method=args.method, smooth_targets=args.smooth_targets, alpha=args.alpha, pair_alpha=args.pair_alpha,
        ax_floor=args.ax_floor, s_floor_q=args.s_floor_q, act_stat=args.act_stat,
        alpha_mode=args.alpha_mode, alpha_grid=grid, search_n=args.search_n,
        search_loss=args.search_loss, search_identity=args.search_identity,
        rotate_mode=args.rotate_mode, rotate_kind=args.rotate_kind, rot_seed=args.rot_seed,
        w=WCfg(bits=args.w_bits, granularity=args.w_gran, group_size=args.w_group, sym=not args.w_asym,
               mse_clip=args.w_mse_clip),
        a=ActCfg(bits=args.a_bits, granularity=args.a_gran, static=args.a_static, clip_ratio=args.a_clip),
        a_skip=[s for s in args.a_skip.split(",") if s], int8_threshold=args.int8_threshold,
        quant_bmm=args.quant_bmm)

    need_calib = (args.method in ("smoothquant", "smoothrot") and args.act_scales is None) or \
                 (args.a_static and args.a_bits < 16) or args.alpha_mode == "search"
    calib = get_calib_samples(tok, args.calib, args.calib_n, args.calib_len, args.seed, args.calib_bos) \
        if need_calib else None
    act_scales = torch.load(args.act_scales) if args.act_scales else None

    info = prepare_model(model, cfg, calib, act_scales)
    ids = get_test_ids(tok)
    t0 = time.time()
    ppl = eval_ppl(model, ids, max_windows=args.max_windows)
    t_eval = time.time() - t0
    print(f"{args.model} {args.method} W{args.w_bits}A{args.a_bits} {args.a_gran}"
          f"{' static' if args.a_static else ''}: ppl = {ppl:.4f}  (prep {info['prep_time_s']:.0f} s, eval {t_eval:.0f} s)")

    if args.method == "llm_int8":
        rep = int8_report(model)
        tot = sum(r["outlier_dims_max"] for r in rep)
        print(f"LLM.int8(): max число выбросных измерений по слоям = "
              f"{max((r['outlier_dims_max'] for r in rep), default=0)}, сумма максимумов = {tot}")
        info["int8"] = rep

    if args.max_windows:
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    row = dict(model=args.model, method=args.method, tag=args.tag,
               w_bits=args.w_bits, w_gran=args.w_gran, w_group=args.w_group if args.w_gran == "group" else "",
               w_sym=not args.w_asym, w_mse_clip=args.w_mse_clip,
               a_bits=args.a_bits, a_gran=args.a_gran, a_static=args.a_static, a_clip=args.a_clip,
               a_skip=args.a_skip, quant_bmm=args.quant_bmm, smooth_targets=args.smooth_targets if args.method == "smoothquant" else "",
               alpha=args.alpha if args.method in ("smoothquant", "smoothrot") else "",
               pair_alpha=args.pair_alpha if args.pair_alpha is not None else "",
               ax_floor=args.ax_floor if args.method in ("smoothquant", "smoothrot") else "",
               s_floor_q=args.s_floor_q if args.method in ("smoothquant", "smoothrot") else "",
               act_stat=args.act_stat if args.method in ("smoothquant", "smoothrot") else "",
               alpha_mode=(args.alpha_mode + (f":{args.search_loss}" + ("+id" if args.search_identity else "")
                           if args.alpha_mode == "search" else "")) if args.method == "smoothquant" else "",
               rotate=f"{args.rotate_mode}/{args.rotate_kind}" if args.method in ("rotate", "smoothrot") else "",
               calib=f"{args.calib}:{args.calib_n}x{args.calib_len}{'+bos' if args.calib_bos else ''}" if need_calib else "",
               seed=args.seed, ppl=round(ppl, 4), prep_s=round(info["prep_time_s"], 1), eval_s=round(t_eval, 1))
    append_row(args.out, row)
    if info.get("alphas") or info.get("int8"):
        stem = os.path.splitext(args.out)[0]
        os.makedirs(stem + "_details", exist_ok=True)
        fn = f"{stem}_details/{args.model}_{args.method}_W{args.w_bits}A{args.a_bits}_{args.a_gran}" \
             f"{'_static' if args.a_static else ''}_{args.alpha_mode}" \
             f"{('_' + args.search_loss + ('_id' if args.search_identity else '')) if args.alpha_mode == 'search' else ''}_s{args.seed}{('_' + args.tag) if args.tag else ''}.json"
        with open(fn, "w") as f:
            json.dump({"args": vars(args), "ppl": ppl, **{k: v for k, v in info.items()}}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
