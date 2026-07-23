"""
experiments/make_figure1.py  —  MWM architecture diagram (Figure 1)

Generates a clean horizontal-flow architecture diagram matching the paper.

Run from project root:
    python experiments/make_figure1.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ────────────────────────────────────────────────────────
BG      = "white"
C_IN    = "#D6E8F7"   # input   – sky blue
C_ENC   = "#FFF0C0"   # encoder – amber
C_LATENT= "#D9F0D9"   # latent  – sage green
C_HIST  = "#E8F5E8"   # history – pale green
C_PRED  = "#EDE0F7"   # predictor – lavender
C_ACT   = "#F2F2F2"   # action  – light grey
C_LOSS  = "#FDE8E8"   # loss/surprise – blush
C_EDGE  = "#4A4A4A"
C_SIG   = "#CC3333"   # SIGReg accent

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis("off")
fig.patch.set_facecolor(BG)


# ── Helper: rounded box ────────────────────────────────────────────────────
def rbox(x, y, w, h, label, fc, fs=8.0, lw=1.1, tc="#1A1A1A", pad=0.10):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=C_EDGE,
        linewidth=lw, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fs, color=tc,
        multialignment="center", zorder=3,
    )


# ── Helper: plain arrow ────────────────────────────────────────────────────
def arr(x1, y1, x2, y2, col=C_EDGE, lw=1.15, style="->", rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=col, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=4,
    )


def hline(x1, x2, y, col="#AAAAAA", lw=0.9, ls="--"):
    ax.plot([x1, x2], [y, y], color=col, lw=lw, ls=ls, zorder=1)


# ══════════════════════════════════════════════════════════════════════════
# BLOCK 1 — Input window
# ══════════════════════════════════════════════════════════════════════════
rbox(0.08, 1.05, 1.00, 2.10,
     "Market\nWindow\n$x_t$\n"
     r"$48{\times}52$",
     C_IN, fs=8.5)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 2 — Patch embedding stack (visual stagger effect)
# ══════════════════════════════════════════════════════════════════════════
for i in range(5):
    a = 0.90 - i * 0.07
    rect = FancyBboxPatch(
        (1.30 + i * 0.07, 1.08 + i * 0.04), 0.65, 2.00,
        boxstyle="round,pad=0.05",
        facecolor="#FFE8A0", edgecolor="#BBAA60",
        alpha=a, linewidth=0.7, zorder=2 + i,
    )
    ax.add_patch(rect)
ax.text(1.68, 1.4, "Patch\nEmbed\n+CLS+PE",
        ha="center", va="center", fontsize=7.8, zorder=8,
        multialignment="center", color="#333")


# ══════════════════════════════════════════════════════════════════════════
# BLOCK 3 — Pre-LN ViT encoder (4 stacked layers)
# ══════════════════════════════════════════════════════════════════════════
layer_shade = ["#D0E8D0", "#C4E2C4", "#B8DCB8", "#ACDAAC"]
for i, fc in enumerate(layer_shade):
    rbox(2.35, 1.10 + i * 0.50, 1.00, 0.42,
         f"MHA + FFN  ({'Pre-LN'})" if i == 0 else "MHA + FFN",
         fc, fs=6.5, lw=0.8)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 4 — Projection head
# ══════════════════════════════════════════════════════════════════════════
rbox(3.65, 1.40, 0.90, 1.40,
     "Proj Head\n128→256\n→BN1d→128",
     C_ENC, fs=7.8)

# SIGReg annotation — ellipse to represent batch Gaussian
from matplotlib.patches import Ellipse
ell = Ellipse((4.10, 0.50), width=0.70, height=0.30,
              facecolor="#FFD0D0", edgecolor=C_SIG, linewidth=1.0, zorder=2)
ax.add_patch(ell)
ax.text(4.10, 0.50, r"$\mathcal{N}$-batch", ha="center", va="center",
        fontsize=7, color=C_SIG, zorder=3)
ax.text(4.10, 0.16, "SIGReg ($\\lambda=0.1$)",
        ha="center", fontsize=7.5, color=C_SIG, weight="bold")
arr(4.10, 0.65, 4.10, 1.40, col=C_SIG, lw=1.0)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 5 — z_t latent vector
# ══════════════════════════════════════════════════════════════════════════
rbox(4.80, 1.55, 0.78, 1.10,
     "$z_t$\n$\\in\\mathbb{R}^{128}$",
     C_LATENT, fs=9.5)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 6 — History buffer z_{t-H:t}  (staggered echo)
# ══════════════════════════════════════════════════════════════════════════
for i in range(3):
    alpha = 0.40 + i * 0.22
    rect = FancyBboxPatch(
        (5.82 + i * 0.10, 1.52 + i * 0.07), 0.62, 1.05,
        boxstyle="round,pad=0.05",
        facecolor="#D9F0D9", edgecolor="#88AA88",
        alpha=alpha, linewidth=0.8, zorder=2 + i,
    )
    ax.add_patch(rect)
ax.text(6.17, 2.05, "$z_{t-H:t}$\n$H=3$",
        ha="center", va="center", fontsize=8.0, zorder=6,
        multialignment="center", color="#2A4A2A")

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 7 — Causal Predictor (6 stacked layers, AdaLN)
# ══════════════════════════════════════════════════════════════════════════
pred_shades = ["#E8D8F5", "#E2D0F0", "#DCC8EB",
               "#D6C0E6", "#D0B8E1", "#CAB0DC"]
for i, fc in enumerate(pred_shades):
    rbox(7.00, 1.10 + i * 0.34, 1.10, 0.30,
         "Attn + AdaLN", fc, fs=6.4, lw=0.7)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 8 — Action vector (bottom)
# ══════════════════════════════════════════════════════════════════════════
rbox(7.00, 0.10, 1.10, 0.55,
     "$a_t \\in \\mathbb{R}^5$  (macro action)",
     C_ACT, fs=7.5)
arr(7.55, 0.65, 7.55, 1.10, col="#777777", lw=1.0)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 9 — ẑ_{t+1} output
# ══════════════════════════════════════════════════════════════════════════
rbox(9.10, 1.55, 0.78, 1.10,
     "$\\hat{z}_{t+1}$",
     C_PRED, fs=11)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 10 — Surprise signal (inference)
# ══════════════════════════════════════════════════════════════════════════
rbox(9.10, 0.10, 2.70, 0.68,
     r"$S_t = \|\hat{z}_{t+1} - z_{t+1}\|^2$   $\leftarrow$ surprise signal",
     C_LOSS, fs=8.0)

# ══════════════════════════════════════════════════════════════════════════
# BLOCK 11 — Training objective banner (top)
# ══════════════════════════════════════════════════════════════════════════
rbox(1.08, 3.62, 9.70, 0.55,
     r"Training:  $\mathcal{L} = \mathrm{MSE}(\hat{z}_{t+1},\,z_{t+1})$"
     "  [prediction]"
     r"  $+\; 0.1 \cdot \mathrm{SIGReg}(Z)$"
     "  [anti-collapse]",
     "#FFFFF0", fs=8.8, lw=0.8, tc="#222")

# ══════════════════════════════════════════════════════════════════════════
# ARROWS — main flow
# ══════════════════════════════════════════════════════════════════════════
mid_y = 2.10

arr(1.08, mid_y, 1.30, mid_y)           # input → patch embed
arr(1.96, mid_y, 2.35, mid_y)           # patch embed → ViT
arr(3.35, mid_y, 3.65, mid_y)           # ViT → proj head
arr(4.55, mid_y, 4.80, mid_y)           # proj head → z_t
arr(5.58, mid_y, 5.82, mid_y)           # z_t → history
arr(6.44, mid_y, 7.00, mid_y)           # history → predictor
arr(8.10, mid_y, 9.10, mid_y)           # predictor → ẑ_{t+1}

# ẑ_{t+1} → surprise (down)
arr(9.49, 1.55, 9.49, 0.78)

# z_{t+1} feedback path: from latent (encoder output for next bar) to surprise box
# show as a curved path: z_t box loops around bottom to surprise
ax.annotate(
    "", xy=(10.45, 0.78), xytext=(5.19, 1.55),
    arrowprops=dict(
        arrowstyle="->", color="#888888", lw=1.0,
        connectionstyle="arc3,rad=0.28",
    ),
    zorder=1,
)

# ══════════════════════════════════════════════════════════════════════════
# Grouping braces — encoder vs predictor
# ══════════════════════════════════════════════════════════════════════════
# Thin lines over the top
ax.plot([0.08, 4.55], [3.55, 3.55], color="#AAAAAA", lw=0.8, ls="dotted")
ax.plot([6.80, 9.88], [3.55, 3.55], color="#AAAAAA", lw=0.8, ls="dotted")

ax.text(2.25, 3.42, "MarketEncoder  (888K params)",
        ha="center", fontsize=7.2, color="#555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
ax.text(8.34, 3.42, "CausalPredictor  (1.4M params)",
        ha="center", fontsize=7.2, color="#555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════
fig.savefig("./experiments/figure1.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
print("Saved: experiments/figure1.png")
