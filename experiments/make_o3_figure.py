"""Figure: the surprise signal S_t fails all three O3 tests.
Panel A -- median surprise ratio on extreme-return bars (95% bootstrap CI);
  violation-of-expectation predicts >1, observed is at or below 1.
Panel B -- mean surprise by implied-vol decile (gold GVZ, EUR/USD VIX-as-EVZ);
  flat, rho ~ 0.
Panel C -- C3: the one surviving effect, rho(S, realized vol), pooled vs
  recomputed within volatility terciles; most of it is regime drift.

Inputs: experiments/o3_stratified_{inst}.json + o3_arrays_{inst}.npz
        (produced by experiments/o3_stratified.py all --dump-arrays)
"""
import json

import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"        # validated categorical slots 1 & 6
GREEN = "#3f9e6b"
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#deddd8"

ASSETS = [("gold", "Gold"), ("eurusd", "EUR/USD"), ("usdjpy", "USD/JPY")]
J = {k: json.load(open(f"experiments/o3_stratified_{k}.json")) for k, _ in ASSETS}
A = {k: np.load(f"experiments/o3_arrays_{k}.npz") for k, _ in ASSETS}

# Sized for a full-width ACM two-column float (7.0in): drawn at final
# size so nothing is scaled down in LaTeX and the labels stay legible.
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.45))
fig.patch.set_facecolor("white")


def _clean(ax):
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=6)


# ---------- Panel A : no violation-of-expectation on extreme returns ----------
x = np.arange(len(ASSETS))
ratios = [J[k]["extreme_ratio_pooled"] for k, _ in ASSETS]
cis = np.array([J[k]["extreme_ratio_ci"] for k, _ in ASSETS])
err = np.vstack([np.array(ratios) - cis[:, 0], cis[:, 1] - np.array(ratios)])
bars = axA.bar(x, ratios, 0.55, color=BLUE, zorder=3)
axA.errorbar(x, ratios, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9,
             capsize=3, zorder=4)
axA.axhline(1.0, color=ORANGE, lw=1.5, ls="--", zorder=2)
axA.text(-0.45, 1.02, "no elevation", ha="left", va="bottom",
         fontsize=6, color=ORANGE)
for b, v, n in zip(bars, ratios, [J[k]["n_extreme"] for k, _ in ASSETS]):
    axA.text(b.get_x() + b.get_width() / 2, 0.045, f"{v:.2f}\n$n$={n}",
             ha="center", va="bottom", fontsize=6, color="white")
axA.set_xticks(x)
axA.set_xticklabels([n for _, n in ASSETS], fontsize=6.5, color=MUTE)
axA.set_ylabel("median $S_t$ ratio  (extreme / background)", fontsize=7, color=INK)
axA.set_ylim(0, 1.45)
axA.set_title("A.  No spike on extreme-return bars",
              fontsize=7.5, color=INK, loc="left", pad=5)
_clean(axA)


# ---------- Panel B : surprise does not track implied volatility ----------
# USD/JPY has no implied-vol series in the macro set, so it is excluded (see text).
_lo, _hi = 1.0, 1.0
for (k, name), col in [(("gold", "Gold"), BLUE), (("eurusd", "EUR/USD"), ORANGE)]:
    S, IV = A[k]["S"], A[k]["IV"]
    q = np.quantile(IV, np.linspace(0, 1, 11))
    q[0], q[-1] = -np.inf, np.inf
    b = np.digitize(IV, q[1:-1])
    means = np.array([np.median(S[b == i]) for i in range(10)])
    means = means / np.median(S)                       # normalize: 1 = overall median
    _lo, _hi = min(_lo, means.min()), max(_hi, means.max())
    rho = J[k]["rho_S_IV_pooled"]
    axB.plot(np.arange(1, 11), means, "o-", color=col, lw=1.4, ms=3.0, zorder=3,
             label=f"{name}   $\\rho$={rho:+.3f}")
axB.axhline(1.0, color=MUTE, lw=1.0, ls=":", zorder=2)
axB.set_xticks(range(1, 11))
axB.set_xlabel("implied-volatility decile  (low $\\rightarrow$ high)",
               fontsize=7, color=INK)
axB.set_ylabel("median $S_t$  (/ overall median)", fontsize=7, color=INK)
axB.set_ylim(_lo - 0.06, _hi + 0.14)
axB.legend(frameon=False, fontsize=6.2, loc="upper left")
axB.set_title("B.  Flat in implied volatility",
              fontsize=7.5, color=INK, loc="left", pad=5)
_clean(axB)


# ---------- Panel C : C3 shrinks the one surviving effect ----------
w = 0.36
pooled = [J[k]["rho_S_RV_pooled"] for k, _ in ASSETS]
within = [J[k]["rho_S_RV_within_strata"] for k, _ in ASSETS]
bP = axC.bar(x - w / 2, pooled, w, color=BLUE, zorder=3, label="pooled")
bW = axC.bar(x + w / 2, within, w, color=GREEN, zorder=3,
             label="within tercile")
for bars, vals in [(bP, pooled), (bW, within)]:
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=6, color=INK)
axC.set_xticks(x)
axC.set_xticklabels([n for _, n in ASSETS], fontsize=6.5, color=MUTE)
axC.set_ylabel("Spearman $\\rho$($S_t$, realized vol)", fontsize=7, color=INK)
axC.set_ylim(0, 0.47)
axC.legend(frameon=False, fontsize=6.2, loc="upper left")
axC.set_title("C.  C3 removes most of the coupling",
              fontsize=7.5, color=INK, loc="left", pad=5)
_clean(axC)

fig.tight_layout(w_pad=1.6)
out = "experiments/surprise_null.png"
fig.savefig(out, dpi=400, facecolor="white", bbox_inches="tight")
print("wrote", out)
