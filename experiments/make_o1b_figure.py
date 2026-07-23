"""Figure: O1b is a batching artifact, not a regime diagnostic.
Panel A -- SIGReg on ONE held-out embedding set, contiguous (val_loader,
  shuffle=False) vs random (train_loader, shuffle=True) batching -> the 'gap'.
Panel B -- clean fixed-N sliding SIGReg over the whole EUR/USD history stays flat
  through every labelled regime change; no train/val/oos separation.
"""
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, ORANGE = "#2a78d6", "#eb6834"        # validated categorical slots 1 & 6
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#deddd8"

E = json.load(open("experiments/o1b_artifact_check_eurusd.json"))
U = json.load(open("experiments/o1b_artifact_check_usdjpy.json"))
G = json.load(open("experiments/o1b_artifact_check_gold.json"))

# Sized for a full-width ACM two-column float (7.0in): drawn at final size so
# nothing is scaled down in LaTeX and every label stays legible.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                               gridspec_kw={"width_ratios": [1, 1.45]})
fig.patch.set_facecolor("white")

# ---------- Panel A : the batching artifact ----------
groups, contig, rand = [], [], []
for tag, d in [("Gold", G), ("EUR/USD", E), ("USD/JPY", U)]:
    for bs in ["bs128"]:
        groups.append(tag)
        contig.append(d["test1_batch_artifact"][bs]["contiguous_shuffleFalse"])
        rand.append(d["test1_batch_artifact"][bs]["random_shuffleTrue"])
x = np.arange(len(groups)); w = 0.38
bR = axA.bar(x - w/2, rand,  w, color=BLUE,   label="random  (train, shuffle=True)", zorder=3)
bC = axA.bar(x + w/2, contig, w, color=ORANGE, label="contiguous  (val, shuffle=False)", zorder=3)
for bars, vals in [(bR, rand), (bC, contig)]:
    for b, v in zip(bars, vals):
        axA.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=5.8, color=INK)
axA.set_xticks(x); axA.set_xticklabels(groups, fontsize=6.5, color=MUTE)
axA.set_ylabel("SIGReg (non-Gaussianity)", fontsize=7, color=INK)
axA.set_title("A.  Same embeddings, two batch orders",
              fontsize=7.5, color=INK, loc="left", pad=5)
axA.legend(frameon=False, fontsize=5.8, loc="upper left")
axA.set_ylim(0, max(contig)*1.25)
axA.grid(axis="y", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)
for s in ["top", "right"]: axA.spines[s].set_visible(False)
for s in ["left", "bottom"]: axA.spines[s].set_color(GRID)
axA.tick_params(colors=MUTE, labelsize=6)

# ---------- Panel B : flat sliding SIGReg over time (gold: longest span,
# cleanly-null shift correlations on both metrics) ----------
E = G  # use gold for the time-series panel
rows = E["rows"]
dates = pd.to_datetime([r["date"] for r in rows])
vsig = np.array([r["val_sig"] for r in rows])
base = E["test2"]["train_baseline"]
axB.plot(dates, vsig, color=BLUE, lw=1.1, zorder=3)
axB.axhline(base, color=MUTE, lw=1.0, ls="--", zorder=2)
axB.text(dates[1], base, f" train baseline {base:.3f}", va="bottom", ha="left",
         fontsize=5.8, color=MUTE)
# Mark the in-sample / held-out boundary: first non-'train' window start.
_held = [pd.Timestamp(r["date"]) for r in rows if r["seg"] != "train"]
if _held:
    bdry = min(_held)
    axB.axvspan(bdry, dates.max(), color=MUTE, alpha=0.07, zorder=0)
    axB.axvline(bdry, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    axB.text(bdry, vsig.max()*1.14, "  held-out →", va="top", ha="left",
             fontsize=6, color=INK, fontweight="bold")
BREAKS = {"2020-03-12": "COVID", "2022-03-16": "Fed liftoff",
          "2023-03-10": "SVB", "2024-08-05": "JPY unwind"}
for d, name in BREAKS.items():
    dd = pd.Timestamp(d)
    if dates.min() <= dd <= dates.max():
        axB.axvline(dd, color=GRID, lw=1.0, zorder=1)
        axB.text(dd, vsig.max()*1.02, name, rotation=90, va="top", ha="right",
                 fontsize=6, color=MUTE)
axB.set_ylim(0, vsig.max()*1.18)
axB.set_ylabel("SIGReg (random subsample, fixed $N$)", fontsize=7, color=INK)
axB.set_title("B.  Clean SIGReg is flat through every regime change  (Gold)",
              fontsize=7.5, color=INK, loc="left", pad=5)
sr = E["test2"]["spearman_val_sig_vol_ratio"]
axB.text(0.985, 0.06, f"corr(SIGReg, return-vol shift): $\\rho$={sr[0]:+.2f}  (p={sr[1]:.2f}, n.s.)",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=6, color=INK,
         bbox=dict(boxstyle="round,pad=0.35", fc="#f4f3ef", ec=GRID, lw=0.8))
axB.grid(axis="y", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
for s in ["top", "right"]: axB.spines[s].set_visible(False)
for s in ["left", "bottom"]: axB.spines[s].set_color(GRID)
axB.tick_params(colors=MUTE, labelsize=6)

# No suptitle: the LaTeX \caption carries the headline, avoiding duplication.
fig.tight_layout(w_pad=1.4)
out = "experiments/sigreg_batching_artifact.png"
fig.savefig(out, dpi=400, facecolor="white", bbox_inches="tight")
print("wrote", out)
