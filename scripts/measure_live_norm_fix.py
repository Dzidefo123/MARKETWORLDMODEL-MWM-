"""
scripts/measure_live_norm_fix.py — Quantify the live normalization fix.
=======================================================================

For each of the last K closed H1 bars, replicate exactly what live_trader does
per bar — fetch the trailing N bars, run the gold feature engineer (warm-up trim
250), apply the 500-bar rolling z-score, take the final window and encode it —
at the OLD shallow fetch depth (298) vs the NEW deep depth (798), using the same
production encoder and the same MT5 XAUUSD data.

If the diagnosis is right, the OLD depth normalizes each bar over only ~48 rows
(vs the 500 the encoder trained on), inflating |z| toward/past the ±5 clip and
pushing the encoding off-manifold — the live S_t morning spike. The NEW depth
should keep |z| in the trained range and match the offline-normalized encoding.

Usage:  python -m scripts.measure_live_norm_fix
Requires the MT5 terminal running.
"""

import sys
sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd
import torch

from data.features import GoldFeatureEngineer
from execution.live_trader import _load_system, _rolling_zscore, _LOOKBACK, _Z_SCORE_WIN, _WARM_UP_BARS
from scripts.retrain_gold_h1_mt5 import fetch_gold_h1_mt5, fetch_h1_macro

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OLD_FETCH = _LOOKBACK + _WARM_UP_BARS                 # 298 — the old _on_bar depth
NEW_FETCH = _Z_SCORE_WIN + _WARM_UP_BARS + _LOOKBACK  # 798 — the fixed depth
K_BARS    = 40                                        # recent as-of bars to evaluate


def _encode_asof(encoder, ohlcv, macro, t, n_fetch):
    """Replicate live per-bar features+encoding for as-of bar t at fetch depth n_fetch."""
    o = ohlcv.iloc[t - n_fetch + 1 : t + 1]
    m = macro.iloc[t - n_fetch + 1 : t + 1]
    eng = GoldFeatureEngineer(lookback_warm=_WARM_UP_BARS)
    raw_feat, ts, _ = eng.compute(o, m)
    feat_df = pd.DataFrame(raw_feat, columns=GoldFeatureEngineer.FEATURE_NAMES, index=ts)
    feat = _rolling_zscore(feat_df, window=_Z_SCORE_WIN).values.astype(np.float32)
    window = feat[-_LOOKBACK:]                                # (48, 52) — the encoded window
    with torch.no_grad():
        z = encoder(torch.tensor(window, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    n_norm_rows = len(feat)                                   # rows available to the 500-window
    mean_abs = float(np.abs(window).mean())
    clip_pct = float((np.abs(window) >= 4.999).mean() * 100)  # share of values at the +-5 clip
    return z, mean_abs, clip_pct, n_norm_rows


def main():
    ckpt  = "./experiments/checkpoints/best_model.pt"
    heads = "./experiments/heads_gold"
    encoder, _, _ = _load_system(ckpt, heads, device="cpu")
    encoder.eval()

    n_pull = NEW_FETCH + K_BARS + 10
    price_df = fetch_gold_h1_mt5("XAUUSD", n_pull)
    macro_df = fetch_h1_macro(price_df.index)

    n = len(price_df)
    idxs = range(n - K_BARS, n)

    rows = []
    for t in idxs:
        z_old, mean_old, clip_old, nr_old = _encode_asof(encoder, price_df, macro_df, t, OLD_FETCH)
        z_new, mean_new, clip_new, nr_new = _encode_asof(encoder, price_df, macro_df, t, NEW_FETCH)
        perturb = float(((z_old - z_new) ** 2).mean())   # how far OLD encoding lands from the
                                                          # properly-normalized (NEW) encoding
        rows.append({"ts": price_df.index[t], "hour": price_df.index[t].hour,
                     "mean_old": mean_old, "mean_new": mean_new,
                     "clip_old": clip_old, "clip_new": clip_new,
                     "perturb": perturb, "nr_old": nr_old, "nr_new": nr_new})

    df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print(f"LIVE NORMALIZATION FIX — {K_BARS} recent H1 bars  (OLD depth {OLD_FETCH} "
          f"vs NEW depth {NEW_FETCH})")
    print("=" * 72)
    print(f"  Normalized rows feeding the 500-bar window (need >=500 for full window):")
    print(f"    OLD: {df['nr_old'].iloc[0]:.0f} rows   NEW: {df['nr_new'].iloc[0]:.0f} rows")

    print(f"\n  Normalized-feature distortion over the encoded 48x52 window")
    print(f"  ('% clipped' = share of feature values saturated at the +-5 training clip):")
    print(f"  {'depth':<6} {'mean|z|':>9} {'% clipped':>11}")
    print(f"  {'-'*6} {'-'*9} {'-'*11}")
    for label, mc, cc in [("OLD", "mean_old", "clip_old"), ("NEW", "mean_new", "clip_new")]:
        print(f"  {label:<6} {df[mc].mean():>9.3f} {df[cc].mean():>10.1f}%")

    print(f"\n  Encoding perturbation ||z_old - z_new||^2 (offline June S_t baseline ~0.0005):")
    pv = df["perturb"].values
    print(f"    mean={pv.mean():.4f}   p90={np.percentile(pv,90):.4f}   max={pv.max():.4f}")
    ratio = pv.mean() / 0.0005
    print(f"    => OLD normalization injects ~{ratio:.0f}x the offline S_t scale of "
          f"spurious surprise into the encoding.")

    # Worst offenders (where the old path most distorts)
    worst = df.nlargest(5, "perturb")[["ts", "hour", "clip_old", "clip_new", "perturb"]]
    print(f"\n  Largest distortions (old vs new), with % of features clipped:")
    for _, r in worst.iterrows():
        print(f"    {str(r['ts'])[:16]}  h{int(r['hour']):02d}  "
              f"clipped {r['clip_old']:4.1f}% -> {r['clip_new']:4.1f}%   perturb={r['perturb']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
