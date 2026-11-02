import os

import matplotlib.pyplot as plt
import pandas as pd

PANELS = ["opt-125m", "tinyllama", "phi3-mini", "llama3-8b"]   # панели quality_vs_bits
PARAMS_B = {                                                   # млрд параметров
    "opt-125m": 0.125, "opt-350m": 0.331, "tinyllama": 1.1,
    "phi3-mini": 3.8, "llama3-8b": 8.0,
}
ORDER = list(PARAMS_B)                                         # порядок моделей в таблицах

# --- RTN: малые модели (полный прогон) + крупные (послойный прогон) ---
rtn = pd.read_csv("results/baseline.csv")
rtn = rtn.drop_duplicates(subset=["model", "bits", "group_size", "sym"], keep="last")
if os.path.exists("results/large.csv"):
    large = pd.read_csv("results/large.csv")
    # tinyllama в large.csv — только проверка послойной оценки, основные числа из baseline.csv
    large = large[~large.model.isin(rtn.model.unique())]
    large = large.drop_duplicates(subset=["model", "bits", "group_size", "sym"], keep="last")
    rtn = pd.concat([rtn, large[["model", "bits", "group_size", "sym", "ppl", "avg_bits"]]],
                    ignore_index=True)
fp16 = rtn[rtn.bits == 16].set_index("model")["ppl"]
rtn["label"] = rtn.apply(
    lambda r: "FP16" if r.bits == 16
    else f"RTN W{r.bits} {'per-channel' if r.group_size == -1 else f'g={r.group_size}'} "
         f"{'sym' if r.sym else 'asym'}", axis=1)

# --- bitsandbytes ---
bnb = pd.read_csv("results/nf4.csv")
bnb = bnb.drop_duplicates(subset=["model", "quant_type", "double_quant"], keep="last")
bnb["avg_bits"] = 4 + bnb.double_quant.map({True: 0.127, False: 0.5})
bnb["label"] = bnb.quant_type.str.upper() + bnb.double_quant.map({True: "+dq", False: ""})

# --- GPTQ: усредняем по сидам ---
gptq = pd.read_csv("results/gptq.csv")
gptq = gptq.drop_duplicates(
    subset=["model", "bits", "group_size", "desc_act", "seed"], keep="last")
agg = gptq.groupby(["model", "bits", "group_size", "desc_act"], as_index=False).agg(
    ppl=("ppl", "mean"), ppl_std=("ppl", "std"), n_seeds=("seed", "count"))
agg["ppl_std"] = agg.ppl_std.fillna(0)
agg["avg_bits"] = agg.bits + 32 / agg.group_size
agg["label"] = agg.apply(
    lambda r: f"GPTQ g={r.group_size}" + (" desc_act" if r.desc_act else ""), axis=1)

for df in (rtn, bnb, agg):
    df["rel_deg_pct"] = (df.ppl / df.model.map(fp16) - 1) * 100

STYLE = {
    "NF4+dq": ("tab:green", "*", 200),
    "NF4":    ("tab:olive", "*", 170),
    "FP4+dq": ("tab:red",   "*", 170),
}

# ===================== 1. Качество vs бит на вес (2x2) =====================
panels = [m for m in PANELS if m in fp16.index]
ncols = 2
nrows = (len(panels) + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows), squeeze=False)

for ax, m in zip(axes.flat, panels):
    r = rtn[(rtn.model == m) & rtn.bits.between(3, 8) & (~rtn.sym)].sort_values("avg_bits")
    ax.plot(r.avg_bits, r.rel_deg_pct, "o-", color="tab:blue",
            lw=1.5, label="RTN (asym)", zorder=2)

    for _, row in bnb[bnb.model == m].iterrows():
        color, marker, size = STYLE.get(row.label, ("gray", "*", 150))
        ax.scatter(row.avg_bits, row.rel_deg_pct, s=size, marker=marker,
                   color=color, zorder=4, edgecolors="white", linewidths=.5,
                   label=f"bnb {row.label}")

    for _, row in agg[agg.model == m].sort_values("avg_bits").iterrows():
        color = "tab:purple" if row.desc_act else "tab:pink"
        ax.errorbar(row.avg_bits, row.rel_deg_pct,
                    yerr=row.ppl_std / fp16[m] * 100,
                    fmt="D", ms=9, capsize=4, color=color, zorder=5,
                    label=row.label)

    ax.axhline(0, color="gray", ls="--", lw=.8)
    ax.set_xlim(3.0, 5.2)
    ax.set_xlabel("Среднее бит на вес")
    ax.set_ylabel("Деградация perplexity, %")
    ax.set_title(f"{m}  (FP16 ppl = {fp16[m]:.2f})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="upper right")

for ax in list(axes.flat)[len(panels):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/quality_vs_bits.png", dpi=150)
print("Сохранено: results/quality_vs_bits.png")


# ================== 2. Деградация W4 vs размер модели ==================
def pick(df, group_size, mask=None):
    sel = (df.bits == 4) & (df.group_size == group_size)
    if mask is not None:
        sel &= mask
    out = df[sel].copy()
    out["params_b"] = out.model.map(PARAMS_B)
    return out.dropna(subset=["params_b"]).sort_values("params_b")


fig, ax = plt.subplots(figsize=(9, 6))
r128 = pick(rtn, 128, ~rtn.sym)
ax.plot(r128.params_b, r128.rel_deg_pct, "o-", color="tab:blue", label="RTN W4 g128 asym")
for _, row in r128.iterrows():
    ax.annotate(row.model, (row.params_b, row.rel_deg_pct),
                textcoords="offset points", xytext=(6, 6), fontsize=9)
rpc = pick(rtn, -1, ~rtn.sym)
ax.plot(rpc.params_b, rpc.rel_deg_pct, "s--", color="tab:cyan", label="RTN W4 per-channel")
g = pick(agg, 128, agg.desc_act)
if len(g):
    ax.plot(g.params_b, g.rel_deg_pct, "D-", color="tab:purple",
            label="GPTQ W4 g128 desc_act")
ax.set_xscale("log")
ticks = sorted(set(r128.params_b) | set(rpc.params_b))
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
ax.set_xlim(min(ticks) / 1.4, max(ticks) * 1.6)
ax.set_ylim(bottom=0)
ax.set_xlabel("Параметров, млрд (лог. шкала)")
ax.set_ylabel("Деградация perplexity, %")
ax.set_title("Деградация от 4-битного квантования в зависимости от размера модели")
ax.grid(alpha=.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig("results/degradation_vs_size.png", dpi=150)
print("Сохранено: results/degradation_vs_size.png")

# ============ 3. RTN: деградация по конфигурациям x моделям (для README) ============
q = rtn[rtn.bits < 16]
pivot = q.pivot_table(index="label", columns="model", values="rel_deg_pct")
pivot = pivot[[m for m in ORDER if m in pivot.columns]]
pivot = pivot.loc[sorted(pivot.index, key=lambda s: (-int(s.split()[1][1:]), s))]
pivot.round(2).to_csv("results/rtn_by_model.csv")
print("\nRTN, деградация % (results/rtn_by_model.csv):")
print(pivot.round(1).to_string())

# ======================= сводная таблица (все модели) =======================
cols = ["model", "label", "avg_bits", "ppl", "rel_deg_pct"]
summary = pd.concat([rtn[cols], bnb[cols], agg[cols + ["ppl_std", "n_seeds"]]])
summary["model"] = pd.Categorical(summary.model, categories=ORDER, ordered=True)
summary = summary.sort_values(["model", "rel_deg_pct"]).round(
    {"avg_bits": 3, "ppl": 4, "rel_deg_pct": 2, "ppl_std": 4})
summary.to_csv("results/summary.csv", index=False)
print("\n" + summary.to_string(index=False))
