"""W/A-квантование с борьбой с выбросами для крупных моделей (послойно, 8 ГБ VRAM).

Флаги те же, что у run_actquant.py. Проверка на TinyLlama (помещается целиком — сверяем с полной оценкой):
  python run_large_actquant.py --model tinyllama --w-bits 16 --a-bits 16                    # 7.7077
  python run_large_actquant.py --model tinyllama --method rotate --w-bits 16 --a-bits 16    # 7.7076
  python run_large_actquant.py --model tinyllama --method smoothquant --smooth-targets groups+pairs \
      --alpha 0.85 --a-gran per_tensor --a-static                                            # 7.8972 (сид 0)
  python run_large_actquant.py --model tinyllama --method rotate --w-bits 4 --w-mse-clip --a-bits 4 --a-clip 0.9  # 11.9846

Llama-3-8B, сравнение с CushionCache (SmoothQuant, alpha 0.8) и QuaRot-RTN:
  python run_large_actquant.py --model llama3-8b --a-gran per_token
  python run_large_actquant.py --model llama3-8b --method smoothquant --alpha 0.8 --a-gran per_tensor --a-static
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from outliers import ActCfg, PipeCfg, WCfg
from outliers.calib import get_calib_samples
from outliers.layerwise_eval import layerwise_actquant
from run_actquant import append_row
from run_large import REPOS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="none", choices=["none", "smoothquant", "rotate", "smoothrot", "llm_int8"])
    p.add_argument("--w-bits", type=int, default=8)
    p.add_argument("--w-gran", default="per_channel", choices=["per_channel", "per_tensor", "group"])
    p.add_argument("--w-group", type=int, default=128)
    p.add_argument("--w-asym", action="store_true")
    p.add_argument("--w-mse-clip", action="store_true")
    p.add_argument("--a-bits", type=int, default=8)
    p.add_argument("--a-gran", default="per_token", choices=["per_token", "per_tensor"])
    p.add_argument("--a-static", action="store_true")
    p.add_argument("--a-clip", type=float, default=1.0)
    p.add_argument("--a-skip", default="")
    p.add_argument("--quant-bmm", action="store_true")
    p.add_argument("--smooth-targets", default="groups",
                   help="groups | pairs | groups+pairs | список из attn, mlp, vo, ffn через запятую")
    p.add_argument("--s-floor-q", type=float, default=0.0,
                   help="квантильный пол масштаба: s <- max(s, Q_p(s)); 0 = оригинальный SmoothQuant")
    p.add_argument("--act-stat", default="max", choices=["max", "meanmax"],
                   help="статистика активаций для масштаба сглаживания")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--pair-alpha", type=float, default=None)
    p.add_argument("--ax-floor", type=float, default=0.0)
    p.add_argument("--rotate-kind", default="hadamard", choices=["hadamard", "orthogonal"])
    p.add_argument("--rot-seed", type=int, default=0)
    p.add_argument("--int8-threshold", type=float, default=6.0)
    p.add_argument("--calib", default="wikitext2", choices=["wikitext2", "pile"])
    p.add_argument("--calib-n", type=int, default=128)
    p.add_argument("--calib-len", type=int, default=512)
    p.add_argument("--calib-bos", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--acts-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--out", default="results/actquant_large.csv")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    repo = REPOS.get(args.model, args.model)
    cfg = PipeCfg(
        method=args.method, smooth_targets=args.smooth_targets, alpha=args.alpha, pair_alpha=args.pair_alpha,
        ax_floor=args.ax_floor, s_floor_q=args.s_floor_q, act_stat=args.act_stat, rotate_mode="full", rotate_kind=args.rotate_kind, rot_seed=args.rot_seed,
        w=WCfg(bits=args.w_bits, granularity=args.w_gran, group_size=args.w_group, sym=not args.w_asym,
               mse_clip=args.w_mse_clip),
        a=ActCfg(bits=args.a_bits, granularity=args.a_gran, static=args.a_static, clip_ratio=args.a_clip),
        a_skip=[s for s in args.a_skip.split(",") if s], int8_threshold=args.int8_threshold,
        quant_bmm=args.quant_bmm)
    need_calib = args.method in ("smoothquant", "smoothrot") or (args.a_static and args.a_bits < 16)
    calib = None
    if need_calib:
        tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
        # буфер должен быть прямоугольным: для pile берём только документы не короче calib_len
        calib = get_calib_samples(tok, args.calib, args.calib_n, args.calib_len, args.seed, args.calib_bos,
                                  min_len=args.calib_len)

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    res = layerwise_actquant(repo, cfg, calib, acts_device=args.acts_device, max_windows=args.max_windows)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{args.model} {args.method} W{args.w_bits}A{args.a_bits} {args.a_gran}"
          f"{' static' if args.a_static else ''}: ppl = {res['ppl']:.4f}  "
          f"(пик VRAM {peak:.2f} ГБ, {res['sec']:.0f} c)")
    if args.max_windows:
        return
    row = dict(model=args.model, method=args.method, tag=args.tag,
               w_bits=args.w_bits, w_gran=args.w_gran, w_group=args.w_group if args.w_gran == "group" else "",
               w_sym=not args.w_asym, w_mse_clip=args.w_mse_clip,
               a_bits=args.a_bits, a_gran=args.a_gran, a_static=args.a_static, a_clip=args.a_clip,
               a_skip=args.a_skip, quant_bmm=args.quant_bmm,
               smooth_targets=args.smooth_targets if args.method == "smoothquant" else "",
               alpha=args.alpha if args.method in ("smoothquant", "smoothrot") else "",
               pair_alpha=args.pair_alpha if args.pair_alpha is not None else "",
               ax_floor=args.ax_floor if args.method in ("smoothquant", "smoothrot") else "",
               s_floor_q=args.s_floor_q if args.method in ("smoothquant", "smoothrot") else "",
               act_stat=args.act_stat if args.method in ("smoothquant", "smoothrot") else "",
               rotate=f"full/{args.rotate_kind}" if args.method in ("rotate", "smoothrot") else "",
               calib=f"{args.calib}:{args.calib_n}x{args.calib_len}{'+bos' if args.calib_bos else ''}" if need_calib else "",
               seed=args.seed, ppl=round(res["ppl"], 4), peak_vram_gb=round(peak, 2), total_s=round(res["sec"], 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    append_row(args.out, row)
    os.makedirs("results/hidden_stats", exist_ok=True)
    with open(f"results/hidden_stats/{args.model}_actquant_{args.method}_W{args.w_bits}A{args.a_bits}"
              f"_{args.a_gran}{'_static' if args.a_static else ''}{('_' + args.tag) if args.tag else ''}.json", "w") as f:
        json.dump(res["hidden_stats"], f, indent=1)


if __name__ == "__main__":
    main()
