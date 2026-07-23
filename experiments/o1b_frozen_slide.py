"""
o1b_frozen_slide.py -- O1b: SIGReg regime diagnostic via a frozen early encoder
================================================================================

THE CLAIM (paper abstract)
--------------------------
When a training window spans a structural regime change, validation SIGReg
diverges from training SIGReg. Mechanistically: SIGReg training Gaussianizes the
embedding distribution of the TRAIN regime; embeddings of a window drawn from a
DIFFERENT regime stay non-Gaussian, so their SIGReg stays high. The train/val
SIGReg gap is therefore a regime-change diagnostic.

WHY THIS DESIGN (vs retrain-per-boundary)
-----------------------------------------
The gap only appears once train SIGReg has actually COLLAPSED (~0.03-0.04). On
CPU, collapsing a fresh encoder at every boundary is ~50 min/boundary -> a 12h+
slide. Instead we pay the collapse cost ONCE: train a single encoder to full
collapse on an EARLY window (before the big shocks), freeze it, then slide the
validation window across the rest of history. The train/val boundary still
slides; val SIGReg(t) is the diagnostic time-series. This gives a dense curve in
~1h and directly tests the claim: does val SIGReg rise when the sliding window
enters a new regime (COVID 2020, rate shock 2022, ...) and fall when markets
calm again?

AVOIDING CIRCULARITY / THE CALENDAR CONFOUND
--------------------------------------------
val SIGReg is validated against an ENCODER-FREE distributional-shift metric
between the FIXED train window and each val window (two-sample KS + vol ratio on
raw H1 log returns). This metric is NON-monotonic (regimes come and go), so if
val SIGReg tracks its ups AND downs -- not merely rising with calendar time --
that is genuine O1b evidence. We report the partial: val_sig ~ shift controlling
for calendar index.

USAGE
  python experiments/o1b_frozen_slide.py eurusd
  python experiments/o1b_frozen_slide.py eurusd --train-end 2019-06-01 --epochs 40 \
        --wval 2000 --step 400
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from models.sigreg import SIGReg
from models.mwm_loss import MWMLoss, MarketWindowDataset
import experiments.o1b_boundary_slide as O   # reuse build_full_arrays, make_models, get_lr, KNOWN_BREAKS

LOOKBACK = O.LOOKBACK
HISTORY_LEN = O.HISTORY_LEN
Z_DIM = O.Z_DIM


def train_collapse(norm, act, lo, hi, epochs, device, stride=1, batch_size=64,
                   lr=3e-4, warmup=4, log_every=5):
    """Train a fresh encoder+predictor to SIGReg collapse on [lo, hi). Return the
    frozen encoder, the SIGReg module, and the train DataLoader (for baseline)."""
    ds = MarketWindowDataset(norm[lo:hi], act[lo:hi], LOOKBACK, stride=stride,
                             history_len=HISTORY_LEN)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    enc, pred = O.make_models(device, seed=42)
    sigreg = SIGReg(d_model=Z_DIM).to(device)
    loss_fn = MWMLoss(sigreg=sigreg, lambda_weight=0.1)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    print(f"  train_collapse: {len(ds):,} samples, {len(loader)} steps/epoch, "
          f"{epochs} epochs", flush=True)
    t0 = time.time()
    for epoch in range(epochs):
        cur = O.get_lr(epoch, epochs, warmup, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur
        enc.train(); pred.train()
        sig_acc = []
        for batch in loader:
            x_hist = batch["x_hist"].to(device)
            x_t1 = batch["x_t1"].to(device)
            a_t = batch["a_t"].to(device)
            opt.zero_grad()
            z_hist = enc.encode_sequence(x_hist)
            z_t = z_hist[:, -1, :]
            z_t1 = enc(x_t1)
            z_hat = pred(z_hist, a_t)
            losses = loss_fn(z_t, z_hat, z_t1)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            sig_acc.append(losses["sigreg"].item())
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    ep{epoch+1:02d} train_sigreg(batch)={np.mean(sig_acc):.4f} "
                  f"lr={cur:.1e} {(time.time()-t0)/60:.1f}m", flush=True)
    enc.eval()
    return enc, sigreg, loader


@torch.no_grad()
def window_sigreg(enc, sigreg, norm, act, lo, hi, device, m=400, n_draws=6, seed=123):
    """Frozen-encoder SIGReg on window [lo, hi): encode z_t over the window (stride
    4), then SIGReg on a fixed-size subsample m, averaged over n_draws projection
    seeds. Same m + seeds for train baseline and every val window => comparable."""
    ds = MarketWindowDataset(norm[lo:hi], act[lo:hi], LOOKBACK, stride=4,
                             history_len=HISTORY_LEN)
    if len(ds) < 16:
        return float("nan"), 0
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    zs = []
    for batch in loader:
        z_hist = enc.encode_sequence(batch["x_hist"].to(device))
        zs.append(z_hist[:, -1, :].cpu())
    Z = torch.cat(zs, 0)
    N = Z.shape[0]
    mm = min(m, N)
    vals = []
    for d in range(n_draws):
        g = torch.Generator().manual_seed(seed + d)
        idx = torch.randperm(N, generator=g)[:mm]
        torch.manual_seed(seed + d)
        vals.append(sigreg(Z[idx].to(device)).item())
    return float(np.mean(vals)), N


def run(instrument, train_end, epochs, W_val, step, train_stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#" * 72)
    print(f"# O1b FROZEN-SLIDE  [{instrument.upper()}]  device={device}")
    print(f"# train_end={train_end} epochs={epochs} W_val={W_val} step={step}")
    print("#" * 72, flush=True)

    norm, act, tstamps, prices = O.build_full_arrays(instrument)
    N = len(norm)
    ts_np = tstamps.values
    b_train_end = int(np.searchsorted(ts_np, np.datetime64(pd.Timestamp(train_end, tz="UTC"))))
    print(f"[{instrument}] {N:,} bars; train [0,{b_train_end}) "
          f"({tstamps[0].date()} -> {tstamps[b_train_end-1].date()})", flush=True)

    # 1) collapse a single encoder on the early window
    enc, sigreg, _ = train_collapse(norm, act, 0, b_train_end, epochs, device,
                                    stride=train_stride)

    # 2) fixed train-window baseline SIGReg (same m/seeds as val windows)
    train_sig, _ = window_sigreg(enc, sigreg, norm, act, 0, b_train_end, device)
    r_train = np.diff(np.log(np.maximum(prices[:b_train_end], 1e-12)))
    r_train = r_train[np.isfinite(r_train)]
    print(f"[{instrument}] frozen train-window SIGReg baseline = {train_sig:.4f}\n", flush=True)

    # 3) slide the val window across the post-train history
    from scipy.stats import ks_2samp
    rows = []
    starts = list(range(b_train_end, N - W_val, step))
    t0 = time.time()
    for k, v in enumerate(starts):
        vs, ve = tstamps[v], tstamps[min(v + W_val, N - 1)]
        val_sig, n = window_sigreg(enc, sigreg, norm, act, v, v + W_val, device)
        rv = np.diff(np.log(np.maximum(prices[v:v + W_val], 1e-12)))
        rv = rv[np.isfinite(rv)]
        vol_ratio = float(np.std(rv) / (np.std(r_train) + 1e-12)) if len(rv) > 30 else float("nan")
        ks = float(ks_2samp(r_train, rv).statistic) if len(rv) > 30 else float("nan")
        breaks = O.known_break_overlap(vs, ve)
        rows.append({"val_idx": int(v), "val_start": str(vs.date()), "val_end": str(ve.date()),
                     "val_mid": str(tstamps[min(v + W_val // 2, N - 1)].date()),
                     "val_sig": val_sig, "gap": val_sig - train_sig,
                     "ratio": val_sig / (train_sig + 1e-8),
                     "vol_ratio": vol_ratio, "ks": ks, "known_breaks": breaks})
        if k % 10 == 0 or breaks:
            el = time.time() - t0
            print(f"  [{k+1}/{len(starts)}] {vs.date()}..{ve.date()} "
                  f"val_sig={val_sig:.4f} gap={val_sig-train_sig:+.4f} ratio={val_sig/(train_sig+1e-8):.2f} "
                  f"| vol_ratio={vol_ratio:.2f} ks={ks:.3f} "
                  f"{'<'+','.join(breaks)+'>' if breaks else ''} | {el/60:.1f}m", flush=True)

    # 4) correlations (val_sig / gap vs encoder-free shift, plus calendar control)
    out = {"instrument": instrument,
           "config": {"train_end": train_end, "epochs": epochs, "W_val": W_val,
                      "step": step, "train_stride": train_stride,
                      "train_sig_baseline": train_sig,
                      "train_start": str(tstamps[0].date()),
                      "train_end_date": str(tstamps[b_train_end-1].date())},
           "rows": rows}
    if len(rows) >= 5:
        from scipy.stats import spearmanr, pearsonr
        gap = np.array([r["gap"] for r in rows])
        vs_arr = np.array([r["val_sig"] for r in rows])
        volr = np.array([r["vol_ratio"] for r in rows])
        ks = np.array([r["ks"] for r in rows])
        cal = np.arange(len(rows), dtype=float)
        corr = {}
        for xn, x in [("vol_ratio", volr), ("ks", ks), ("calendar_index", cal)]:
            for yn, y in [("val_sig", vs_arr), ("gap", gap)]:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 5:
                    sr = spearmanr(x[mask], y[mask]); pr = pearsonr(x[mask], y[mask])
                    corr[f"{yn}~{xn}"] = {"spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
                                          "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
                                          "n": int(mask.sum())}
        # partial: does shift explain val_sig BEYOND calendar? residualize both on calendar index.
        def _resid(y):
            A = np.vstack([np.ones_like(cal), cal]).T
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - A @ beta
        m2 = np.isfinite(volr) & np.isfinite(vs_arr)
        if m2.sum() >= 6:
            rv_ = spearmanr(_resid(volr[m2]), _resid(vs_arr[m2]))
            rk_ = spearmanr(_resid(ks[m2]), _resid(vs_arr[m2]))
            corr["val_sig~vol_ratio|calendar"] = {"spearman": float(rv_.statistic), "spearman_p": float(rv_.pvalue)}
            corr["val_sig~ks|calendar"] = {"spearman": float(rk_.statistic), "spearman_p": float(rk_.pvalue)}
        out["correlations"] = corr
        print(f"\n[{instrument}] CORRELATIONS (frozen val SIGReg vs encoder-free shift):")
        for kk, vv in corr.items():
            extra = f" pearson={vv['pearson']:+.3f}" if "pearson" in vv else ""
            print(f"    {kk:<32} spearman={vv['spearman']:+.3f} (p={vv['spearman_p']:.2g}){extra}")

    out_path = f"./experiments/o1b_frozen_slide_{instrument}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[{instrument}] wrote {out_path}  ({len(rows)} val windows)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="eurusd")
    ap.add_argument("--train-end", default=None,
                    help="date; default: instrument-specific pre-shock cutoff")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--wval", type=int, default=2000)
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    default_end = {"eurusd": "2019-06-01", "usdjpy": "2021-06-01", "gold": "2019-06-01"}
    te = args.train_end or default_end.get(args.instrument, "2019-06-01")
    run(args.instrument, te, args.epochs, args.wval, args.step, args.stride)
