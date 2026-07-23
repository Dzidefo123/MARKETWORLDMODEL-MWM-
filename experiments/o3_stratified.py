"""
o3_stratified.py -- C3 executed on O3: regime-stratified surprise statistics
============================================================================

regen_o3_matched.py settles the AGGREGATE O3 numbers (extreme-return ratio,
Spearman vs implied and realized vol). This script runs the C3 control on top
of them, i.e. it recomputes each effect WITHIN volatility strata instead of
pooled, and adds the vol-persistence control that is the most plausible
spurious generator of the residual surprise/realized-vol coupling.

Same data path, splits and checkpoints as regen_o3_matched (so the aggregate
column here reproduces o3_matched_{inst}.json exactly).

Per instrument it reports, on the held-out test span:
  * rho(S, RV) pooled, and within terciles of RV  (C3: does the coupling
    survive inside a volatility regime, or is it only the regime drift?)
  * partial rho(S, RV | RV_{t-24}), controlling for volatility persistence
  * median surprise ratio on extreme-return bars, pooled and within RV
    terciles (C3 on the violation-of-expectation claim: extreme returns
    cluster in high-vol windows where baseline surprise already differs)

Writes o3_stratified_{inst}.json and, with --dump-arrays, the (S, RV, IV,
extreme) columns used by make_o3_figure.py.

USAGE
  python experiments/o3_stratified.py all --dump-arrays
  python experiments/o3_stratified.py gold
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm
import torch

sys.path.insert(0, ".")
import experiments.o1b_boundary_slide as O
from experiments.retrain_long import REGIME_SPLITS, _load_models
from data.pipeline import DataPipeline
from data.long_history import build_long
from evaluation.surprise import (compute_surprise_timeseries, detect_extreme_events,
                                 IDX_RV32, IDX_GVZ_LEVEL)

CKPT = "experiments/checkpoints_long/{}/best_model.pt"
N_STRATA = 3
LAG = 24          # bars; vol-persistence control (one trading day of H1 bars)
N_BOOT = 2000
SEED = 0


def _spearman_ci(x, y, n_boot=N_BOOT, rng=None):
    """Bootstrap CI for Spearman rho; returns (rho, lo, hi, p)."""
    rng = rng or np.random.default_rng(SEED)
    res = spearmanr(x, y)
    boots = np.empty(n_boot)
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = spearmanr(x[idx], y[idx]).correlation
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(res.correlation), float(lo), float(hi), float(res.pvalue)


def _partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-linear)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    R = np.corrcoef(np.vstack([rx, ry, rz]))
    rxy, rxz, ryz = R[0, 1], R[0, 2], R[1, 2]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    r = (rxy - rxz * ryz) / (denom + 1e-12)
    # Fisher z for a p-value at n-3 df
    n = len(x)
    zstat = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(n - 4)
    p = 2 * (1 - norm.cdf(abs(zstat)))
    return float(r), float(p)


def _ratio_ci(S, mask, n_boot=N_BOOT, rng=None):
    """median(S|event)/median(S|non-event) with a bootstrap CI."""
    rng = rng or np.random.default_rng(SEED)
    ev, bg = S[mask], S[~mask]
    if len(ev) < 10:
        return float("nan"), float("nan"), float("nan"), int(mask.sum())
    ratio = float(np.median(ev) / (np.median(bg) + 1e-12))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        e = ev[rng.integers(0, len(ev), len(ev))]
        g = bg[rng.integers(0, len(bg), len(bg))]
        boots[b] = np.median(e) / (np.median(g) + 1e-12)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return ratio, float(lo), float(hi), int(mask.sum())


def run(inst, dump_arrays=False):
    dev = torch.device("cpu")
    start, end = O.CACHED_RANGE[inst]
    data = build_long(inst, start, end)
    pipe = DataPipeline(instrument=inst, lookback=48, norm_window=500)
    res = pipe.build_from_frames(data["price"], data["macro"],
                                 split_dates=REGIME_SPLITS[inst], history_len=3, stride=1)
    enc, pred, cfg = _load_models(CKPT.format(inst), dev)
    ts = compute_surprise_timeseries(enc, pred, res["test"], batch_size=128, history_len=3)

    S = np.asarray(ts["surprise"], dtype=float)
    xl, xn = ts["x_last"], ts["x_next"]
    RV = xl[:, IDX_RV32].astype(float)        # realized vol, same yardstick all assets
    IV = xl[:, IDX_GVZ_LEVEL].astype(float)   # implied vol proxy (GVZ/EVZ)
    events = detect_extreme_events(xl, xn, vol_feature_idx=IDX_GVZ_LEVEL)
    extreme = np.asarray(events["extreme_return"], dtype=bool)

    rng = np.random.default_rng(SEED)

    # ---- C3 on the surprise/realized-vol coupling -------------------------
    rho, lo, hi, p = _spearman_ci(S, RV, rng=rng)
    edges = np.quantile(RV, np.linspace(0, 1, N_STRATA + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    strata = np.digitize(RV, edges[1:-1])

    strat_rows = []
    for k in range(N_STRATA):
        m = strata == k
        r_k, lo_k, hi_k, p_k = _spearman_ci(S[m], RV[m], rng=rng)
        ratio_k, rlo_k, rhi_k, n_ev = _ratio_ci(S[m], extreme[m], rng=rng)
        strat_rows.append({
            "stratum": ["low", "mid", "high"][k], "n": int(m.sum()),
            "rho_S_RV": r_k, "rho_ci": [lo_k, hi_k], "rho_p": p_k,
            "extreme_ratio": ratio_k, "ratio_ci": [rlo_k, rhi_k], "n_extreme": n_ev,
        })

    # Fisher-z average of the within-stratum rhos (equal-sized strata)
    zs = [np.arctanh(np.clip(r["rho_S_RV"], -0.999, 0.999)) for r in strat_rows]
    rho_within = float(np.tanh(np.mean(zs)))

    # ---- vol-persistence control -----------------------------------------
    RV_lag = pd.Series(RV).shift(LAG).bfill().values
    rho_partial, p_partial = _partial_spearman(S, RV, RV_lag)

    # ---- C3 on the violation-of-expectation claim ------------------------
    ratio, rlo, rhi, n_ext = _ratio_ci(S, extreme, rng=rng)
    rho_iv, ivlo, ivhi, p_iv = _spearman_ci(S, IV, rng=rng)

    out = {
        "instrument": inst, "n_test": int(len(S)), "range": f"{start}..{end}",
        "trained_through": REGIME_SPLITS[inst]["train_end"],
        "rho_S_RV_pooled": rho, "rho_S_RV_ci": [lo, hi], "rho_S_RV_p": p,
        "rho_S_RV_within_strata": rho_within,
        "rho_S_RV_partial_lag24": rho_partial, "rho_S_RV_partial_p": p_partial,
        "rho_S_IV_pooled": rho_iv, "rho_S_IV_ci": [ivlo, ivhi], "rho_S_IV_p": p_iv,
        "extreme_ratio_pooled": ratio, "extreme_ratio_ci": [rlo, rhi],
        "n_extreme": n_ext, "strata": strat_rows,
    }
    Path(f"experiments/o3_stratified_{inst}.json").write_text(json.dumps(out, indent=2))

    print(f"\n[{inst}] n={len(S)}  {start}..{end}")
    print(f"  rho(S,RV) pooled   = {rho:+.3f}  [{lo:+.3f},{hi:+.3f}]  p={p:.2g}")
    print(f"  rho(S,RV) within   = {rho_within:+.3f}  (Fisher-z mean of terciles)")
    for r in strat_rows:
        print(f"      {r['stratum']:<4} n={r['n']:>6}  rho={r['rho_S_RV']:+.3f} "
              f"[{r['rho_ci'][0]:+.3f},{r['rho_ci'][1]:+.3f}]   "
              f"extreme ratio={r['extreme_ratio']:.2f} "
              f"[{r['ratio_ci'][0]:.2f},{r['ratio_ci'][1]:.2f}]  n_ev={r['n_extreme']}")
    print(f"  rho(S,RV | RV-{LAG}) = {rho_partial:+.3f}  p={p_partial:.2g}")
    print(f"  rho(S,IV) pooled   = {rho_iv:+.3f}  [{ivlo:+.3f},{ivhi:+.3f}]  p={p_iv:.2g}")
    print(f"  extreme ratio      = {ratio:.3f}  [{rlo:.3f},{rhi:.3f}]  n_ev={n_ext}")

    if dump_arrays:
        np.savez_compressed(f"experiments/o3_arrays_{inst}.npz",
                            S=S, RV=RV, IV=IV, extreme=extreme, strata=strata)
        print(f"  arrays -> experiments/o3_arrays_{inst}.npz")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    ap.add_argument("--dump-arrays", action="store_true")
    args = ap.parse_args()
    targets = (["gold", "eurusd", "usdjpy"] if args.instrument == "all"
               else [args.instrument])
    for t in targets:
        run(t, args.dump_arrays)
