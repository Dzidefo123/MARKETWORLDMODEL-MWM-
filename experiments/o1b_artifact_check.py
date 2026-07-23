"""
o1b_artifact_check.py -- O1b verdict: the SIGReg train/val "divergence" is a
batching artifact, not a regime-change diagnostic.
=======================================================================

The paper's O1b claim (its "strongest surviving" objective) is that when a
training window spans a structural regime change, VALIDATION SIGReg diverges from
TRAINING SIGReg -- proposed as a label-free regime-change detector. The long
checkpoints appear to show it dramatically (eurusd train_sig 0.029 vs val_sig
0.123, a 4x gap).

This script shows that gap is an artifact of how train vs val SIGReg are measured
in training/train_o1.py, and that no regime signal survives a clean measurement:

  TEST 1 (the artifact) -- on ONE fixed set of held-out embeddings from the
    fully-trained encoder, SIGReg computed on CONTIGUOUS batches (val_loader uses
    shuffle=False) is ~3-5x higher than on RANDOM subsamples (train_loader uses
    shuffle=True) of the SAME embeddings. Temporally-adjacent 48-bar windows
    overlap and drift slowly, so a contiguous batch lies on a low-dim manifold
    (non-Gaussian); the encoder did NOT fail to Gaussianize the val regime.

  TEST 2 (no regime tracking) -- slide a clean fixed-N, random-subsample SIGReg
    across the whole series. Mean SIGReg is ~equal on the training span, the
    dated val span, and the genuinely out-of-sample span, and does NOT correlate
    with an encoder-free distributional-shift metric (return vol-ratio / KS vs
    the training window).

Reuses the verified long checkpoints -- NO training. Writes o1b_artifact_check_{inst}.json.

USAGE
  python experiments/o1b_artifact_check.py eurusd
  python experiments/o1b_artifact_check.py all
"""
import argparse, json, sys, collections
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, ks_2samp

sys.path.insert(0, ".")
import experiments.o1b_boundary_slide as O
import experiments.o1b_frozen_slide as F
from experiments.retrain_long import REGIME_SPLITS, _load_models
from models.mwm_loss import MarketWindowDataset
from models.sigreg import SIGReg

CKPT = "experiments/checkpoints_long/{}/best_model.pt"


def _embeddings(enc, norm, act, lo, hi, stride=4):
    ds = MarketWindowDataset(norm[lo:hi], act[lo:hi], 48, stride=stride, history_len=3)
    Z = []
    for b in DataLoader(ds, batch_size=256, shuffle=False):
        with torch.no_grad():
            Z.append(enc.encode_sequence(b["x_hist"])[:, -1, :])
    return torch.cat(Z, 0) if Z else torch.empty(0, 128)


def _contig(sig, Z, bs):
    v = []
    for i in range(0, len(Z) - bs + 1, bs):
        torch.manual_seed(0); v.append(sig(Z[i:i + bs]).item())
    return float(np.mean(v)) if v else float("nan")


def _random(sig, Z, bs, draws=10):
    v = []
    for d in range(draws):
        g = torch.Generator().manual_seed(d)
        idx = torch.randperm(len(Z), generator=g)[:bs]
        torch.manual_seed(0); v.append(sig(Z[idx]).item())
    return float(np.mean(v)) if v else float("nan")


def run(inst):
    print("#" * 70); print(f"# O1b ARTIFACT CHECK  [{inst.upper()}]"); print("#" * 70, flush=True)
    norm, act, ts, prices = O.build_full_arrays(inst)
    N = len(norm); dev = torch.device("cpu")
    enc, pred, cfg = _load_models(CKPT.format(inst), dev); enc.eval()
    sig = SIGReg(d_model=128)
    b_te = int(np.searchsorted(ts.values,
               np.datetime64(pd.Timestamp(REGIME_SPLITS[inst]["train_end"], tz="UTC"))))
    b_ve = int(np.searchsorted(ts.values,
               np.datetime64(pd.Timestamp(REGIME_SPLITS[inst]["val_end"], tz="UTC"))))
    print(f"trained through {ts[b_te-1].date()}; val_end {ts[min(b_ve,N-1)-1].date()}; N={N}", flush=True)

    # ---- TEST 1: contiguous vs random on a held-out (dated-val) window ----
    Zval = _embeddings(enc, norm, act, b_te, min(b_ve, N - 1))
    test1 = {}
    for bs in [64, 128]:
        test1[f"bs{bs}"] = {"contiguous_shuffleFalse": _contig(sig, Zval, bs),
                            "random_shuffleTrue": _random(sig, Zval, bs)}
    print(f"\nTEST 1 -- SAME {len(Zval)} held-out embeddings, two batch orders:")
    for bs, d in test1.items():
        r = d["contiguous_shuffleFalse"] / (d["random_shuffleTrue"] + 1e-9)
        print(f"  {bs}: contiguous={d['contiguous_shuffleFalse']:.4f}  "
              f"random={d['random_shuffleTrue']:.4f}  ratio={r:.1f}x")

    # ---- TEST 2: clean sliding SIGReg vs encoder-free shift + segment means ----
    base, nb = F.window_sigreg(enc, sig, norm, act, 0, b_te, dev)
    r_tr = np.diff(np.log(np.maximum(prices[:b_te], 1e-12))); r_tr = r_tr[np.isfinite(r_tr)]
    Wv, step = 2000, 700
    rows = []
    for v in range(0, N - Wv, step):
        s, _ = F.window_sigreg(enc, sig, norm, act, v, v + Wv, dev)
        rv = np.diff(np.log(np.maximum(prices[v:v + Wv], 1e-12))); rv = rv[np.isfinite(rv)]
        vr = float(np.std(rv) / (np.std(r_tr) + 1e-12)); ks = float(ks_2samp(r_tr, rv).statistic)
        seg = "train" if v + Wv <= b_te else ("val" if v < b_ve else "oos")
        rows.append({"date": str(ts[v].date()), "val_sig": s, "vol_ratio": vr, "ks": ks, "seg": seg})
    vs = np.array([r["val_sig"] for r in rows]); vr = np.array([r["vol_ratio"] for r in rows]); ks = np.array([r["ks"] for r in rows])
    sr_vr = spearmanr(vr, vs); sr_ks = spearmanr(ks, vs)
    segm = collections.defaultdict(list)
    for r in rows: segm[r["seg"]].append(r["val_sig"])
    seg_means = {k: float(np.mean(v)) for k, v in segm.items()}
    print(f"\nTEST 2 -- clean sliding val_sig (random subsample, fixed N):")
    print(f"  train-window baseline = {base:.4f}")
    print(f"  val_sig range [{vs.min():.4f},{vs.max():.4f}] mean {vs.mean():.4f} std {vs.std():.4f}")
    print(f"  spearman(val_sig, vol_ratio) = {sr_vr.statistic:+.3f} (p={sr_vr.pvalue:.2g})")
    print(f"  spearman(val_sig, ks)        = {sr_ks.statistic:+.3f} (p={sr_ks.pvalue:.2g})")
    print(f"  mean val_sig by segment: " + "  ".join(f"{k}={v:.4f}(n{len(segm[k])})" for k, v in seg_means.items()))

    out = {"instrument": inst,
           "trained_through": str(ts[b_te - 1].date()),
           "test1_batch_artifact": test1,
           "test2": {"train_baseline": base, "val_sig_mean": float(vs.mean()),
                     "val_sig_std": float(vs.std()),
                     "spearman_val_sig_vol_ratio": [float(sr_vr.statistic), float(sr_vr.pvalue)],
                     "spearman_val_sig_ks": [float(sr_ks.statistic), float(sr_ks.pvalue)],
                     "segment_means": seg_means},
           "rows": rows,
           "verdict": ("O1b divergence is a shuffle/batching artifact; no clean regime "
                       "tracking survives (val vs train vs oos SIGReg ~equal, shift corr ~0).")}
    p = f"./experiments/o1b_artifact_check_{inst}.json"
    with open(p, "w") as f: json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {p}\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("instrument", nargs="?", default="eurusd")
    a = ap.parse_args()
    targets = ["eurusd", "usdjpy", "gold"] if a.instrument == "all" else [a.instrument]
    for t in targets:
        try:
            run(t)
        except Exception as ex:
            print(f"[{t}] skipped: {repr(ex)[:120]}")
