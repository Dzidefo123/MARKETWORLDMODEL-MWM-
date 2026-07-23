"""
o1b_boundary_slide.py -- O1b: SIGReg train/val divergence as a regime-change diagnostic
=======================================================================================

THE CLAIM (paper abstract, last sentence)
-----------------------------------------
When a training window spans a structural regime change, the *validation* SIGReg
diverges from the *training* SIGReg. I.e. the encoder learns to Gaussianize the
train-window embeddings (train SIGReg -> ~0), but if the validation window is
drawn from a different regime, its embeddings stay non-Gaussian (val SIGReg stays
high). The train/val SIGReg gap is therefore a *diagnostic* for regime change.

WHY THE EARLIER CHECK WAS NOT ENOUGH
------------------------------------
The long-history checkpoints already show a large fixed-boundary gap (eurusd
train_sig 0.029 vs val_sig 0.123). But a single fixed boundary cannot separate
"val is a different regime" from "val is simply out-of-sample / later in time".
This is the *within-instrument boundary-slide*: we slide the train/val boundary
across the whole timeline, retrain a fresh encoder at each position, and test
whether the gap tracks an ENCODER-FREE measure of how much the val window's
return distribution differs from the train window's. If O1b is real, the SIGReg
gap should rise exactly where the raw-return distribution shifts, and stay low in
stationary stretches -- not merely grow monotonically with calendar time.

DESIGN (per instrument)
-----------------------
  * Compute the 52 features + 5-D actions + rolling z-score ONCE over the full
    history (reusing DataPipeline.build_from_frames, the exact paper path).
  * Rolling window: at boundary bar b, train on [b - W_train, b), validate on
    [b, b + W_val). Slide b by `step` bars.
  * At each b: fresh encoder+predictor (same init seed every time, so the only
    thing that varies is the data), train E epochs of the pure JEPA objective
    (MSE + 0.1*SIGReg, no aux loss), then measure final train/val SIGReg in eval
    mode on a FIXED number of embeddings (same N and same projection seeds for
    train and val, so the two are directly comparable).
  * Encoder-free regime-shift metrics between the train and val windows, computed
    from raw H1 log returns (NOT the encoder): volatility ratio and a two-sample
    KS statistic on the return distributions. These are the independent x-axis
    against which the SIGReg gap is validated.

PRIMARY RESULT
--------------
Spearman correlation between the SIGReg gap D(b) = val_sig - train_sig and the
encoder-free shift metrics across all boundaries. O1b predicts a strong positive
correlation. Also saved: the full per-boundary series for the paper figure, and
overlap flags against a curated list of known structural breaks.

USAGE
  python experiments/o1b_boundary_slide.py eurusd            # full slide, cached range
  python experiments/o1b_boundary_slide.py eurusd --smoke    # 3 boundaries, quick check
  python experiments/o1b_boundary_slide.py eurusd --wtrain 8000 --wval 2500 --step 3000 --epochs 14
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data.long_history import build_long
from data.pipeline import DataPipeline
from models.encoder import MarketEncoder
from models.predictor import CausalPredictor
from models.sigreg import SIGReg
from models.mwm_loss import MWMLoss, MarketWindowDataset

# Build ranges MATCHED to each long checkpoint's actual training range
# (per experiments/RETRAIN_LONG_RESULTS.md: gold trained 2015-start, EUR/USD &
# USD/JPY 2020-start). Matching the build to training makes the "train" segment
# in the artifact check genuinely in-sample. EUR/USD & USD/JPY 2020-2026 have
# offline combined caches; gold 2015-2026 fetches daily macro from Yahoo once
# (Dukascopy H1 chunks are already cached), then writes its own combined cache.
CACHED_RANGE = {
    "eurusd": ("2020-01-01", "2026-07-01"),
    "usdjpy": ("2020-01-01", "2026-07-01"),
    "gold":   ("2015-01-01", "2026-07-01"),
}

# Known structural breaks, for overlap flags on the figure (NOT used to compute
# the primary correlation, which rests on the encoder-free shift metrics).
KNOWN_BREAKS = {
    "2015-01-15": "SNB CHF unpeg",
    "2016-06-24": "Brexit vote",
    "2018-02-05": "Volmageddon",
    "2020-03-12": "COVID crash",
    "2022-03-16": "Fed liftoff / rate shock",
    "2022-09-26": "GBP/gilt crisis",
    "2023-03-10": "SVB / banking stress",
    "2024-08-05": "JPY carry unwind",
}

LOOKBACK = 48
HISTORY_LEN = 3
Z_DIM = 128


def build_full_arrays(instrument):
    """Run the paper feature path once; return (norm_features, actions, timestamps,
    prices) all warm-up trimmed and mutually aligned."""
    start, end = CACHED_RANGE.get(instrument, (None, None))
    data = build_long(instrument, start, end)
    pipe = DataPipeline(instrument=instrument, lookback=LOOKBACK, norm_window=500)
    # A throwaway split; we only want the normalized feature matrix + actions.
    ts_all = data["price"].index
    dummy = {"train_end": str(ts_all[len(ts_all) // 2].date()),
             "val_end":   str(ts_all[int(len(ts_all) * 0.75)].date())}
    res = pipe.build_from_frames(data["price"], data["macro"],
                                 split_dates=dummy, history_len=HISTORY_LEN, stride=1)
    meta = res["meta"]
    norm = meta["norm_features"].astype(np.float32)
    act = meta["macro_vecs"].astype(np.float32)
    tstamps = pd.DatetimeIndex(meta["timestamps"])
    prices = np.asarray(meta["prices"], dtype=np.float64)
    n = min(len(norm), len(act), len(tstamps), len(prices))
    return norm[:n], act[:n], tstamps[:n], prices[:n]


def make_models(device, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = MarketEncoder(lookback=LOOKBACK, n_features=52, patch_size=4, d_model=128,
                        n_heads=4, n_layers=4, dropout=0.1, proj_hidden=256, z_dim=Z_DIM).to(device)
    pred = CausalPredictor(z_dim=Z_DIM, d_model=128, n_heads=4, n_layers=6,
                           action_dim=5, history_len=HISTORY_LEN, dropout=0.1).to(device)
    return enc, pred


def get_lr(epoch, n_epochs, warmup, base):
    import math
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / max(1, (n_epochs - warmup))
    return base * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def eval_sigreg(enc, pred, loader, sigreg, loss_fn, device, n_eval=1536, n_draws=4, seed=123):
    """Collect z_t embeddings over the loader (eval mode) and compute SIGReg on a
    FIXED-size subsample, averaged over n_draws projection seeds. Using the same
    n_eval and the same seed sequence for train and val makes the two comparable
    (SIGReg's value depends on N and on the random projections)."""
    enc.eval()
    zs = []
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        z_hist = enc.encode_sequence(x_hist)
        zs.append(z_hist[:, -1, :].cpu())
    if not zs:
        return float("nan")
    Z = torch.cat(zs, 0)
    N = Z.shape[0]
    m = min(n_eval, N)
    vals = []
    for d in range(n_draws):
        g = torch.Generator().manual_seed(seed + d)
        idx = torch.randperm(N, generator=g)[:m]
        torch.manual_seed(seed + d)   # fix SIGReg's internal projection sampling
        vals.append(sigreg(Z[idx].to(device)).item())
    return float(np.mean(vals))


def train_one_boundary(norm, act, b, W_train, W_val, epochs, train_stride,
                       device, batch_size=64, lr=3e-4, warmup=3, verbose=False):
    """Train a fresh encoder on [b-W_train, b), validate on [b, b+W_val).
    Return dict with final train/val SIGReg (eval mode) and pred/total losses."""
    tr_lo, tr_hi = b - W_train, b
    vl_lo, vl_hi = b, b + W_val
    tr_ds = MarketWindowDataset(norm[tr_lo:tr_hi], act[tr_lo:tr_hi], LOOKBACK,
                                stride=train_stride, history_len=HISTORY_LEN)
    vl_ds = MarketWindowDataset(norm[vl_lo:vl_hi], act[vl_lo:vl_hi], LOOKBACK,
                                stride=4, history_len=HISTORY_LEN)
    if len(tr_ds) < batch_size or len(vl_ds) < 8:
        return None
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size * 2, shuffle=False)

    enc, pred = make_models(device, seed=42)
    sigreg = SIGReg(d_model=Z_DIM).to(device)
    loss_fn = MWMLoss(sigreg=sigreg, lambda_weight=0.1)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))

    for epoch in range(epochs):
        cur = get_lr(epoch, epochs, warmup, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur
        enc.train(); pred.train()
        for batch in tr_loader:
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
        if verbose:
            tr_s = eval_sigreg(enc, pred, tr_loader, sigreg, loss_fn, device)
            vl_s = eval_sigreg(enc, pred, vl_loader, sigreg, loss_fn, device)
            print(f"      ep{epoch+1:02d} train_sig={tr_s:.4f} val_sig={vl_s:.4f} "
                  f"gap={vl_s-tr_s:.4f}", flush=True)

    tr_sig = eval_sigreg(enc, pred, tr_loader, sigreg, loss_fn, device)
    vl_sig = eval_sigreg(enc, pred, vl_loader, sigreg, loss_fn, device)
    return {"train_sig": tr_sig, "val_sig": vl_sig,
            "gap": vl_sig - tr_sig, "ratio": vl_sig / (tr_sig + 1e-8),
            "n_train": len(tr_ds), "n_val": len(vl_ds)}


def regime_shift_metrics(prices, b, W_train, W_val):
    """Encoder-free distributional shift between the train and val windows,
    from raw H1 log returns. Returns vol_ratio and a two-sample KS statistic."""
    from scipy.stats import ks_2samp
    r = np.diff(np.log(np.maximum(prices, 1e-12)))
    # returns index i corresponds to bar i+1; align conservatively by bar range
    tr = r[max(0, b - W_train): b]
    vl = r[b: b + W_val]
    tr = tr[np.isfinite(tr)]; vl = vl[np.isfinite(vl)]
    if len(tr) < 50 or len(vl) < 50:
        return {"vol_ratio": float("nan"), "ks": float("nan")}
    vol_ratio = float(np.std(vl) / (np.std(tr) + 1e-12))
    ks = float(ks_2samp(tr, vl).statistic)
    return {"vol_ratio": vol_ratio, "ks": ks}


def known_break_overlap(val_start, val_end):
    hits = []
    for d, name in KNOWN_BREAKS.items():
        ts = pd.Timestamp(d, tz="UTC")
        if val_start <= ts <= val_end:
            hits.append(name)
    return hits


def run(instrument, W_train, W_val, step, epochs, train_stride, smoke=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#" * 72)
    print(f"# O1b BOUNDARY-SLIDE  [{instrument.upper()}]  device={device}")
    print(f"# W_train={W_train} W_val={W_val} step={step} epochs={epochs} "
          f"train_stride={train_stride}")
    print("#" * 72, flush=True)

    norm, act, tstamps, prices = build_full_arrays(instrument)
    N = len(norm)
    print(f"[{instrument}] full arrays: {N:,} bars "
          f"({tstamps[0].date()} -> {tstamps[-1].date()})", flush=True)

    boundaries = list(range(W_train, N - W_val, step))
    if smoke:
        # 3 boundaries spread across the timeline for a quick end-to-end check
        boundaries = [boundaries[len(boundaries) // 6],
                      boundaries[len(boundaries) // 2],
                      boundaries[-2]]
    print(f"[{instrument}] {len(boundaries)} boundaries\n", flush=True)

    rows = []
    t0 = time.time()
    for k, b in enumerate(boundaries):
        val_start, val_end = tstamps[b], tstamps[min(b + W_val, N - 1)]
        res = train_one_boundary(norm, act, b, W_train, W_val, epochs,
                                 train_stride, device, verbose=(smoke and k == 0))
        if res is None:
            print(f"  [{k+1}/{len(boundaries)}] b={b} skipped (too few samples)")
            continue
        shift = regime_shift_metrics(prices, b, W_train, W_val)
        breaks = known_break_overlap(val_start, val_end)
        row = {"boundary_idx": int(b),
               "boundary_date": str(tstamps[b].date()),
               "val_start": str(val_start.date()), "val_end": str(val_end.date()),
               **res, **shift, "known_breaks": breaks}
        rows.append(row)
        el = time.time() - t0
        eta = el / (k + 1) * (len(boundaries) - k - 1)
        print(f"  [{k+1}/{len(boundaries)}] {row['boundary_date']} | "
              f"train_sig={res['train_sig']:.4f} val_sig={res['val_sig']:.4f} "
              f"gap={res['gap']:.4f} ratio={res['ratio']:.2f} | "
              f"vol_ratio={shift['vol_ratio']:.2f} ks={shift['ks']:.3f} "
              f"{'<' + ','.join(breaks) + '>' if breaks else ''} "
              f"| {el/60:.1f}m elapsed, ETA {eta/60:.1f}m", flush=True)

    # ---- correlations: SIGReg gap vs encoder-free shift ----
    out = {"instrument": instrument,
           "config": {"W_train": W_train, "W_val": W_val, "step": step,
                      "epochs": epochs, "train_stride": train_stride,
                      "lookback": LOOKBACK, "z_dim": Z_DIM},
           "rows": rows}
    if len(rows) >= 4:
        from scipy.stats import spearmanr, pearsonr
        gap = np.array([r["gap"] for r in rows])
        ratio = np.array([r["ratio"] for r in rows])
        volr = np.array([r["vol_ratio"] for r in rows])
        ks = np.array([r["ks"] for r in rows])
        idx = np.arange(len(rows))
        corr = {}
        for xname, x in [("vol_ratio", volr), ("ks", ks), ("calendar_index", idx)]:
            for yname, y in [("gap", gap), ("ratio", ratio)]:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 4:
                    sr = spearmanr(x[mask], y[mask])
                    pr = pearsonr(x[mask], y[mask])
                    corr[f"{yname}~{xname}"] = {
                        "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
                        "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue)}
        out["correlations"] = corr
        print(f"\n[{instrument}] CORRELATIONS (SIGReg divergence vs encoder-free shift):")
        for k_, v in corr.items():
            print(f"    {k_:<22} spearman={v['spearman']:+.3f} (p={v['spearman_p']:.2g})  "
                  f"pearson={v['pearson']:+.3f} (p={v['pearson_p']:.2g})")
        # The decisive contrast for O1b: gap must track the DISTRIBUTIONAL shift
        # (vol_ratio/ks) more than it tracks mere calendar position.
        print(f"\n[{instrument}] O1b reads as SUPPORTED if gap~vol_ratio / gap~ks are "
              f"positive & significant, ideally stronger than gap~calendar_index.")

    out_path = f"./experiments/o1b_boundary_slide_{instrument}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[{instrument}] wrote {out_path}  ({len(rows)} boundaries)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="eurusd")
    ap.add_argument("--wtrain", type=int, default=8000)
    ap.add_argument("--wval", type=int, default=2500)
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    targets = ["eurusd", "usdjpy", "gold"] if args.instrument == "all" else [args.instrument]
    for t in targets:
        run(t, args.wtrain, args.wval, args.step, args.epochs, args.stride, smoke=args.smoke)
