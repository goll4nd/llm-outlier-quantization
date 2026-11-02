"""Графики к отчёту о борьбе с выбросами в активациях (README_outliers.md).

  python make_outlier_report.py            # из корня проекта: читает results/, пишет results/outliers_*.png

Источники: results/actquant.csv (полная оценка), results/actquant_large.csv (послойная),
results/act_stats/*.csv (статистики входов Linear), results/actquant_details/*search_mse_id*.json
(послойный поиск alpha). Цвета — первые слоты проверенной категориальной палитры; каждое значение
на графиках продублировано в таблицах README.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FP16 = {"opt-125m": 27.656, "opt-350m": 22.002, "tinyllama": 7.7077, "phi3-mini": 6.0077, "llama3-8b": 6.1358}
NAMES = {"opt-125m": "OPT-125M", "opt-350m": "OPT-350M", "tinyllama": "TinyLlama-1.1B",
         "phi3-mini": "Phi-3-mini-3.8B", "llama3-8b": "Llama-3-8B"}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # слоты 1–4 палитры (светлая тема)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def b(x):
    return str(x).strip().lower() == "true"


def load(root):
    small = pd.read_csv(os.path.join(root, "results/actquant.csv"))
    large = pd.read_csv(os.path.join(root, "results/actquant_large.csv"))
    d = pd.concat([small, large], ignore_index=True)
    for c in ("a_static", "w_mse_clip", "quant_bmm"):
        d[c] = d[c].map(b) if c in d else False
    for c in ("a_skip", "smooth_targets", "alpha_mode", "tag", "rotate"):
        d[c] = d[c].fillna("") if c in d else ""
    return d


def delta(model, ppl):
    return 100 * (ppl / FP16[model] - 1)


def bar_label(ax, rect, text):
    ax.annotate(text, (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)


# ------------------------------------------------------------------ 1. статистики входов
def fig_stats(root, out):
    st = lambda m: pd.read_csv(os.path.join(root, f"results/act_stats/{m}.csv"))
    tl, tr = st("tinyllama"), st("tinyllama_rotate")
    o1, o3 = st("opt-125m"), st("opt-350m")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    sel = lambda df, mod: df[df.module == mod].sort_values("layer")
    for df, mod, lab, c in ((tl, "self_attn.q_proj", "вход q/k/v (канальные выбросы)", C[0]),
                            (tl, "mlp.down_proj", "вход down_proj (токенные всплески)", C[1]),
                            (tr, "mlp.down_proj", "вход down_proj после вращения", C[2])):
        s = sel(df, mod)
        ax.plot(s.layer, s.max_abs, color=c, lw=1.6, marker="o", ms=4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("блок")
    ax.set_ylabel("max |x| на калибровке")
    ax.set_title("TinyLlama: где живут выбросы")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax = axs[1]
    for df, lab, c in ((o1, "OPT-125M (pre-LN)", C[0]), (o3, "OPT-350M (post-LN)", C[1])):
        s = sel(df, "self_attn.q_proj")
        ax.plot(s.layer, s.max_abs, color=c, lw=1.6, marker="o", ms=4, label=lab)
    ax.axhline(np.sqrt(1023), color=INK2, lw=1, ls="--")
    ax.annotate("√(d−1) ≈ 32 — предел выхода LayerNorm при d = 1024, γ ≈ 1", (0.3, 34.3), fontsize=8, color=INK2)
    ax.set_xlabel("блок")
    ax.set_ylabel("max |x| на входе q/k/v")
    ax.set_ylim(0, 37)
    ax.set_title("OPT: post-LN ограничивает выбросы")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 2. главный результат: W8A8 O3
def o3_rows(d):
    return d[(d.w_bits == 8) & (d.a_bits == 8) & (d.a_gran == "per_tensor") & d.a_static
             & (d.w_gran == "per_channel") & (d.a_skip == "") & (~d.quant_bmm)]


def best_alpha(df):
    if df.empty:
        return np.nan, np.nan
    g = df.groupby("alpha")["ppl"].mean()
    return g.min(), g.idxmin()


def table_o3(d):
    r = o3_rows(d)
    rows = []
    for m in ("opt-125m", "tinyllama", "phi3-mini", "llama3-8b"):
        x = r[r.model == m]
        naive = x[x.method == "none"].ppl.mean()
        sq, a_sq = best_alpha(x[(x.method == "smoothquant") & (x.smooth_targets == "groups") & (x.alpha_mode.isin(["global", ""]))])
        gp, a_gp = best_alpha(x[(x.method == "smoothquant") & (x.smooth_targets == "groups+pairs") & (x.alpha_mode.isin(["global", ""]))])
        rows.append(dict(model=m, naive=naive, sq=sq, a_sq=a_sq, gp=gp, a_gp=a_gp))
    return pd.DataFrame(rows)


def fig_o3(d, out):
    t = table_o3(d)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(t))
    w = 0.26
    for k, (col, lab) in enumerate((("naive", "наивный W8A8"), ("sq", "SmoothQuant (нормы → q/k/v, gate/up)"),
                                   ("gp", "SmoothQuant + пары v→o, up→down"))):
        vals = [max(delta(m, v), 0.05) for m, v in zip(t.model, t[col])]
        rects = ax.bar(x + (k - 1) * w, vals, w, color=C[k], edgecolor=SURFACE, linewidth=1.5, label=lab)
        if col == "gp":
            for r_, v in zip(rects, vals):
                bar_label(ax, r_, f"+{v:.1f}%")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 4e4)
    ax.set_xticks(x, [NAMES[m] for m in t.model])
    ax.set_ylabel("рост perplexity к FP16, % (лог. шкала)")
    ax.set_title("Статический per-tensor W8A8 (O3): пары возвращают GLU-модели к FP16")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return t


# ------------------------------------------------------------------ 3. кривые по alpha
def fig_alpha(d, out):
    r = o3_rows(d)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    x = r[(r.model == "opt-125m") & (r.method == "smoothquant") & (r.smooth_targets == "groups") & (r.alpha_mode.isin(["global", ""]))]
    g = x.groupby("alpha")["ppl"].agg(["mean", "std"]).reset_index()
    ax.errorbar(g.alpha, g["mean"], yerr=g["std"], color=C[0], lw=1.6, marker="o", ms=5, capsize=3,
                label="SmoothQuant, 3 сида калибровки")
    naive = r[(r.model == "opt-125m") & (r.method == "none")].ppl.mean()
    ax.axhline(naive, color=C[1], lw=1.2, ls="--")
    ax.annotate(f"наивный {naive:.2f}", (0.3, naive + 0.04), fontsize=8, color=INK2)
    ax.axhline(FP16["opt-125m"], color=INK2, lw=1, ls=":")
    ax.annotate(f"FP16 {FP16['opt-125m']:.2f}", (0.3, FP16["opt-125m"] + 0.04), fontsize=8, color=INK2)
    ax.set_xlabel("α")
    ax.set_ylabel("perplexity")
    ax.set_title("OPT-125M, W8A8 O3")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax = axs[1]
    x = r[(r.model == "llama3-8b") & (r.method == "smoothquant") & (r.smooth_targets == "groups+pairs") & (r.alpha_mode.isin(["global", ""]))]
    g = x.groupby("alpha")["ppl"].agg(["mean", "std"]).reset_index()
    ax.errorbar(g.alpha, g["mean"], yerr=g["std"].fillna(0), color=C[2], lw=1.6, marker="o", ms=5, capsize=3,
                label="SmoothQuant + пары")
    ax.axhline(FP16["llama3-8b"], color=INK2, lw=1, ls=":")
    ax.annotate(f"FP16 {FP16['llama3-8b']:.2f}", (0.6, FP16["llama3-8b"] + 0.05), fontsize=8, color=INK2)
    ax.set_xlabel("α")
    ax.set_ylabel("perplexity")
    ax.set_title("Llama-3-8B, W8A8 O3")
    ax.annotate("вне шкалы: наивный 64–96,\nSmoothQuant без пар 115", (0.72, 7.6), fontsize=8, color=INK2)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 4. alpha, выбранный поиском, по режимам
def fig_alpha_regime(root, out):
    regimes = [("W8A8_per_tensor_static", "W8A8 O3"), ("W6A6_per_token", "W6A6"), ("W4A8_per_token", "W4A8 g128")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    res = {}
    for ax, grp, title in ((axs[0], "attn", "норма → q/k/v"), (axs[1], "mlp", "норма → fc1 / gate, up")):
        for k, m in enumerate(("opt-125m", "tinyllama")):
            ys = []
            for key, _ in regimes:
                f = glob.glob(os.path.join(root, f"results/actquant_details/{m}_smoothquant_{key}_search_mse_id_s0.json"))
                if not f:
                    ys.append(np.nan)
                    continue
                al = json.load(open(f[0]))["alphas"]
                v = [al[i][grp] for i in al if al[i].get(grp) is not None]
                ys.append(float(np.mean(v)) if v else np.nan)
            res[(m, grp)] = ys
            ax.plot(range(len(regimes)), ys, color=C[k], lw=1.6, marker="o", ms=6, label=NAMES[m])
        ax.set_xticks(range(len(regimes)), [r[1] for r in regimes])
        ax.set_title(title)
        if grp == "mlp" and np.isnan(res[("tinyllama", "mlp")][2]):
            ax.annotate("TinyLlama, W4A8: поиск выбрал\n«не сглаживать» во всех блоках", (1.55, 0.3),
                        fontsize=8, color=INK2)
        ax.set_ylim(0.2, 1.05)
    axs[0].set_ylabel("средний по слоям α, выбранный поиском")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Оптимальный α убывает с сужением весов — одинаково у ReLU- и SwiGLU-модели",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


# ------------------------------------------------------------------ 5. W4A4
def fig_w4a4(d, out):
    r = d[(d.w_bits == 4) & (d.a_bits == 4) & (d.tag == "clip80")]
    models = ("opt-125m", "tinyllama", "phi3-mini", "llama3-8b")
    series = (("rotate", "full/orthogonal", "случайная ортогональная"), ("rotate", "full/hadamard", "Адамар (QuaRot-RTN)"),
              ("smoothrot", "full/hadamard", "SmoothRot"))
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(models))
    w = 0.26
    table = {}
    for k, (meth, rot, lab) in enumerate(series):
        vals = []
        for m in models:
            v = r[(r.model == m) & (r.method == meth) & (r.rotate == rot)].ppl
            table[(m, lab)] = v.mean() if len(v) else np.nan
            vals.append(delta(m, v.mean()) if len(v) else 0)
        rects = ax.bar(x + (k - 1) * w, vals, w, color=C[k], edgecolor=SURFACE, linewidth=1.5, label=lab)
        for r_, v in zip(rects, vals):
            if v > 0:
                bar_label(ax, r_, f"{v:.0f}%")
    ax.set_xticks(x, [NAMES[m] for m in models])
    ax.set_ylabel("рост perplexity к FP16, %")
    ax.set_title("W4A4 (KV16): вращение необходимо; без него — сотни и тысячи perplexity")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out", default=None, help="каталог для PNG (по умолчанию <root>/results)")
    a = p.parse_args()
    out = a.out or os.path.join(a.root, "results")
    os.makedirs(out, exist_ok=True)
    d = load(a.root)
    fig_stats(a.root, os.path.join(out, "outliers_act_stats.png"))
    t = fig_o3(d, os.path.join(out, "outliers_w8a8_o3.png"))
    fig_alpha(d, os.path.join(out, "outliers_alpha_curves.png"))
    res = fig_alpha_regime(a.root, os.path.join(out, "outliers_alpha_by_regime.png"))
    w4 = fig_w4a4(d, os.path.join(out, "outliers_w4a4.png"))
    print(t.round(3).to_string())
    print({k: [round(v, 2) for v in vs] for k, vs in res.items()})
    print({k: round(v, 3) for k, v in w4.items()})


if __name__ == "__main__":
    main()
