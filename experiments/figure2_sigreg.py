"""
experiments/figure2_sigreg.py - Figure 2: SIGReg Training Curves

Three-panel figure showing SIGReg loss over 200 training epochs:
  Panel A (Gold):    train_sigreg + val_sigreg both converging
  Panel B (EUR/USD): train_sigreg converging, val_sigreg diverging (^EVZ artefact)
  Panel C (USD/JPY): train_sigreg converging, val_sigreg diverging (short train window)

Run from project root:
    python experiments/figure2_sigreg.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data ──────────────────────────────────────────────────────────────
with open("./experiments/checkpoints/loss_history.json") as f:
    gold = json.load(f)

with open("./experiments/checkpoints/eurusd/loss_history.json") as f:
    eur = json.load(f)

with open("./experiments/checkpoints/usdjpy/loss_history.json") as f:
    jpy = json.load(f)

epochs = np.arange(1, 201)

gold_train = np.array(gold["train_sigreg"])
gold_val   = np.array(gold["val_sigreg"])
eur_train  = np.array(eur["train_sigreg"])
eur_val    = np.array(eur["val_sigreg"])
jpy_train  = np.array(jpy["train_sigreg"])
jpy_val    = np.array(jpy["val_sigreg"])

# ── Smooth for readability (5-epoch rolling mean) ──────────────────────────
def smooth(x, w=5):
    kernel = np.ones(w) / w
    padded = np.pad(x, (w//2, w//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]

gold_train_s = smooth(gold_train)
gold_val_s   = smooth(gold_val)
eur_train_s  = smooth(eur_train)
eur_val_s    = smooth(eur_val)
jpy_train_s  = smooth(jpy_train)
jpy_val_s    = smooth(jpy_val)

# ── Style ──────────────────────────────────────────────────────────────────
GOLD_COLOR = "#C5903A"
EUR_COLOR  = "#2A5DB0"
JPY_COLOR  = "#2E8B57"
ALPHA_RAW  = 0.18
LW_SMOOTH  = 1.8
LW_RAW     = 0.7

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14, wspace=0.36)

# ── Panel A: Gold ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(epochs, gold_train,   color=GOLD_COLOR, alpha=ALPHA_RAW, lw=LW_RAW)
ax.plot(epochs, gold_val,     color=GOLD_COLOR, alpha=ALPHA_RAW, lw=LW_RAW, linestyle="--")
ax.plot(epochs, gold_train_s, color=GOLD_COLOR, lw=LW_SMOOTH, label="Train")
ax.plot(epochs, gold_val_s,   color=GOLD_COLOR, lw=LW_SMOOTH, label="Val", linestyle="--")

ax.set_title("(A)  XAU/USD — Gold", fontsize=11, fontweight="bold", pad=6)
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("SIGReg Loss", fontsize=10)
ax.legend(fontsize=9, framealpha=0.85, loc="upper right")
ax.set_xlim(1, 200)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=9)

ax.annotate("Both converge",
            xy=(180, gold_val_s[-1]),
            xytext=(115, gold_val_s[-1] + 0.025),
            fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))

# ── Panel B: EUR/USD ───────────────────────────────────────────────────────
ax = axes[1]
ax.plot(epochs, eur_train,   color=EUR_COLOR, alpha=ALPHA_RAW, lw=LW_RAW)
ax.plot(epochs, eur_val,     color=EUR_COLOR, alpha=ALPHA_RAW, lw=LW_RAW, linestyle="--")
ax.plot(epochs, eur_train_s, color=EUR_COLOR, lw=LW_SMOOTH, label="Train")
ax.plot(epochs, eur_val_s,   color=EUR_COLOR, lw=LW_SMOOTH, label="Val", linestyle="--")

ax.set_title("(B)  EUR/USD — First run", fontsize=11, fontweight="bold", pad=6)
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("SIGReg Loss", fontsize=10)
ax.legend(fontsize=9, framealpha=0.85, loc="upper right")
ax.set_xlim(1, 200)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=9)

eur_div_epoch = int(np.argmin(eur_val_s)) + 1
ax.axvline(x=eur_div_epoch, color="red", linestyle=":", lw=1.1, alpha=0.7)
ax.annotate(f"Val diverges\n(epoch {eur_div_epoch})",
            xy=(eur_div_epoch, eur_val_s[eur_div_epoch - 1]),
            xytext=(eur_div_epoch + 18, eur_val_s[eur_div_epoch - 1] - 0.012),
            fontsize=8, color="#cc0000",
            arrowprops=dict(arrowstyle="-", color="#cc0000", lw=0.8))

# ── Panel C: USD/JPY ───────────────────────────────────────────────────────
ax = axes[2]
ax.plot(epochs, jpy_train,   color=JPY_COLOR, alpha=ALPHA_RAW, lw=LW_RAW)
ax.plot(epochs, jpy_val,     color=JPY_COLOR, alpha=ALPHA_RAW, lw=LW_RAW, linestyle="--")
ax.plot(epochs, jpy_train_s, color=JPY_COLOR, lw=LW_SMOOTH, label="Train")
ax.plot(epochs, jpy_val_s,   color=JPY_COLOR, lw=LW_SMOOTH, label="Val", linestyle="--")

ax.set_title("(C)  USD/JPY — Carry unwind", fontsize=11, fontweight="bold", pad=6)
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("SIGReg Loss", fontsize=10)
ax.legend(fontsize=9, framealpha=0.85, loc="upper right")
ax.set_xlim(1, 200)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=9)

jpy_div_epoch = int(np.argmin(jpy_val_s)) + 1
ax.axvline(x=jpy_div_epoch, color="red", linestyle=":", lw=1.1, alpha=0.7)
ax.annotate(f"Val diverges\n(epoch {jpy_div_epoch})",
            xy=(jpy_div_epoch, jpy_val_s[jpy_div_epoch - 1]),
            xytext=(jpy_div_epoch + 18, jpy_val_s[jpy_div_epoch - 1] - 0.010),
            fontsize=8, color="#cc0000",
            arrowprops=dict(arrowstyle="-", color="#cc0000", lw=0.8))

# ── Save ───────────────────────────────────────────────────────────────────
out     = "./experiments/figure2_sigreg.pdf"
out_png = "./experiments/figure2_sigreg.png"
fig.savefig(out,     dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"Saved: {out_png}")

print("\nKey stats:")
print(f"  Gold     train_sigreg:  E1={gold_train[0]:.4f} -> E200={gold_train[-1]:.4f}")
print(f"  Gold     val_sigreg:    E1={gold_val[0]:.4f}  -> E200={gold_val[-1]:.4f}")
print(f"  EUR/USD  train_sigreg:  E1={eur_train[0]:.4f} -> E200={eur_train[-1]:.4f}")
print(f"  EUR/USD  val_sigreg:    E1={eur_val[0]:.4f}  -> E200={eur_val[-1]:.4f}  (diverges epoch {eur_div_epoch})")
print(f"  USD/JPY  train_sigreg:  E1={jpy_train[0]:.4f} -> E200={jpy_train[-1]:.4f}")
print(f"  USD/JPY  val_sigreg:    E1={jpy_val[0]:.4f}  -> E200={jpy_val[-1]:.4f}  (diverges epoch {jpy_div_epoch})")
