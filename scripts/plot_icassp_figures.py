import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(r"D:\BI\DDI\figures")
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none",
})
navy, teal, coral, gold, gray = "#173F5F", "#2A9D8F", "#E76F51", "#E9C46A", "#6B7280"

# Contract: show breadth of baselines, temporal replication, and feature ablations.
methods = ["Recency–pop.", "ExtraTrees", "HGB", "FT-Trans.", "NEST-DDI", "GraphSAGE", "NEST+Graph"]
rec = [0.89444, 0.90192, 0.90437, 0.90448, 0.90526, 0.90219, np.nan]
first = [0.13253, 0.13776, 0.14570, 0.14387, 0.14781, 0.15039, 0.16407]
folds = ["24Q3→Q4", "24Q4→Q1", "25Q1→Q2", "25Q2→Q3"]
hgb = [0.88710, 0.89963, 0.90437, 0.91174]
nest = [0.88802, 0.90015, 0.90526, 0.91285]
abl_names = ["Static only", "History only", "No decay", "No drug analogues", "Full NEST-DDI"]
abl = [0.78144, 0.90402, 0.88961, 0.90451, 0.90526]

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.85), gridspec_kw={"wspace": 0.48})
fig.patch.set_facecolor("white")

 # A: temporal replication
ax[0].plot(folds, hgb, marker="o", lw=2, color=gray, label="HGB")
ax[0].plot(folds, nest, marker="o", lw=2.2, color=teal, label="NEST-DDI")
ax[0].fill_between(np.arange(len(folds)), hgb, nest, color=teal, alpha=0.12, interpolate=True)
for i, (a, b) in enumerate(zip(hgb, nest)):
    dx = 10 if i == 0 else 0
    dy = 10 if i in (0, 3) else 8
    ax[0].annotate(f"+{b-a:.4f}", (i, b), textcoords="offset points", xytext=(dx, dy), ha="center", fontsize=6.5, color=navy,
                   bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
ax[0].set_ylim(0.882, 0.916); ax[0].set_ylabel("AUPR"); ax[0].set_title("(a) Temporal replication", loc="left", fontweight="bold")
ax[0].grid(axis="y", color="#E5E7EB", linewidth=0.6); ax[0].set_axisbelow(True); ax[0].legend(fontsize=7, loc="lower right")
ax[0].tick_params(axis="x", rotation=22, pad=2)

# C: ablation
colors = [gray, gray, gray, gray, teal]
ax[1].barh(np.arange(len(abl_names)), abl, color=colors, height=0.62)
ax[1].axvline(abl[-1], color=teal, lw=1.2, ls="--", alpha=0.8)
ax[1].set_yticks(np.arange(len(abl_names)), ["Static", "History", "No decay", "No analogues", "Full NEST"]); ax[1].invert_yaxis(); ax[1].set_xlim(0.76, 0.91)
ax[1].set_xlabel("Recurrence AUPR"); ax[1].set_title("(b) Feature ablation", loc="left", fontweight="bold")
ax[1].grid(axis="x", color="#E5E7EB", linewidth=0.6); ax[1].set_axisbelow(True)
for i, v in enumerate(abl): ax[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
ax[1].text(abl[-1] - 0.001, -0.52, "full model", ha="right", va="bottom", fontsize=6.2, color=teal)

for a in ax:
    a.tick_params(labelsize=7, length=2)
    a.spines["left"].set_color("#9CA3AF"); a.spines["bottom"].set_color("#9CA3AF")
fig.savefig(OUT / "icassp_experiment_panels.svg", bbox_inches="tight")
fig.savefig(OUT / "icassp_experiment_panels.pdf", bbox_inches="tight")
fig.savefig(OUT / "icassp_experiment_panels.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT / "icassp_experiment_panels.tiff", dpi=600, bbox_inches="tight")
print(OUT / "icassp_experiment_panels.svg")

# Independent exports for manuscript Fig. 2 and Fig. 3.
fa, aa = plt.subplots(figsize=(3.45, 2.75))
aa.plot(folds, hgb, marker="o", lw=2, color=gray, label="HGB")
aa.plot(folds, nest, marker="o", lw=2.2, color=teal, label="NEST-DDI")
aa.fill_between(np.arange(len(folds)), hgb, nest, color=teal, alpha=0.12)
for i, (u, v) in enumerate(zip(hgb, nest)):
    aa.annotate(f"+{v-u:.4f}", (i, v), textcoords="offset points", xytext=(8 if i == 0 else 0, 9), ha="center", fontsize=7, color=navy)
aa.set_ylim(0.882, 0.916); aa.set_ylabel("AUPR"); aa.set_title("Temporal replication", loc="left", fontweight="bold")
aa.grid(axis="y", color="#E5E7EB", linewidth=0.6); aa.legend(fontsize=7, loc="lower right"); aa.tick_params(axis="x", rotation=20, labelsize=7)
for ext, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})]: fa.savefig(OUT / f"icassp_fig2_temporal.{ext}", bbox_inches="tight", **kwargs)

fb, bb = plt.subplots(figsize=(3.45, 2.75))
bb.barh(np.arange(len(abl_names)), abl, color=colors, height=0.62); bb.axvline(abl[-1], color=teal, lw=1.2, ls="--")
bb.set_yticks(np.arange(len(abl_names)), ["Static", "History", "No decay", "No analogues", "Full NEST"]); bb.invert_yaxis(); bb.set_xlim(0.76, 0.91)
bb.set_xlabel("Recurrence AUPR"); bb.set_title("Feature ablation", loc="left", fontweight="bold"); bb.grid(axis="x", color="#E5E7EB", linewidth=0.6)
for i, v in enumerate(abl): bb.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
for ext, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})]: fb.savefig(OUT / f"icassp_fig3_ablation.{ext}", bbox_inches="tight", **kwargs)
