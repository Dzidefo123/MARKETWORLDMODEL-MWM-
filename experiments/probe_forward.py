"""
probe_forward.py -- Honest O2 test: does the encoder predict the FUTURE
=======================================================================

The O2 rescue. The paper's probe targets are input features (session flags,
rv_32, gvz level), so a random projection preserves them and the trained
encoder does not beat a random one (see project_o2_probe_finding). That does
not test whether JEPA learned anything -- it tests whether inputs survive a
projection (they do).

This probes FORWARD realized volatility: for each embedding z_t, predict the
RMS of 1-bar returns over the strictly-future bars [t+1 .. t+k]. That quantity
is NOT in the encoder's input window [.. t], so a random projection has no
trivial access to it. The random encoder still sees backward vol (rv_32 is an
input), so it can exploit volatility persistence -- which is exactly the
baseline we must beat. If the trained encoder beats random on forward vol, it
encoded forward-predictive structure beyond input persistence, and O2 lives in
a defensible form. If not, O2 is done.

Alignment (must be exact or the test is meaningless):
  MarketWindowDataset sample j -> anchor bar t = ds.indices[j] + lookback - 1,
  computed within that split's own feature array (ds.features). extract_
  embeddings uses shuffle=False, so Z[j] <-> ds.indices[j] one-to-one.
  Forward target uses bars strictly greater than t.

Outputs -> probe_forward_{inst}.json  (does not touch any existing results).

USAGE (Colab, after mount/unzip/deps/symlink cells)
  !python experiments/probe_forward.py all
  !python experiments/probe_forward.py gold --horizons 1,4,12,24
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from data.long_history import build_long
from data.pipeline import DataPipeline
import experiments.retrain_long as R
from models.encoder import MarketEncoder
from evaluation.probing import extract_embeddings, run_single_probe, IDX

DRIVE_RESULTS = "/content/drive/MyDrive/MWM_run/results"
RET1_IDX = IDX["ret_1"]   # normalized 1-bar log return (should be 0)


def _encoder(cfg, device):
    return MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256),
        z_dim=cfg.get("z_dim", 128),
    ).to(device)


def forward_vol(ds, horizon):
    """RMS of future normalized 1-bar returns over (t, t+horizon], per sample.
    Returns (y, valid_mask) aligned to extract_embeddings' sample order."""
    feat = ds.features.numpy()          # this split's normalized feature matrix
    lb = ds.lookback
    n = feat.shape[0]
    y = np.full(len(ds.indices), np.nan)
    for j, i in enumerate(ds.indices):
        t = i + lb - 1                  # anchor bar of z_t (== dataset's t_now)
        a, b = t + 1, t + 1 + horizon   # strictly future
        if b <= n:
            r = feat[a:b, RET1_IDX]
            y[j] = np.sqrt(np.mean(r * r))
    return y, ~np.isnan(y)


def run(instrument, horizons, results_dir=None):
    print("\n" + "#" * 70)
    print(f"# FORWARD-VOL PROBE  [{instrument.upper()}]  horizons={horizons}")
    print("#" * 70)

    device = torch.device("cpu")
    ckpt_path = Path(f"{R.CKPT_ROOT}/{instrument}/best_model.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    print(f"  checkpoint epoch {ckpt.get('epoch')} val={float(ckpt.get('val_loss', float('nan'))):.4f}"
          f"  |  ret_1 feature idx = {RET1_IDX}")

    data = build_long(instrument)
    splits = R.REGIME_SPLITS[instrument]
    pipe = DataPipeline(instrument=instrument, lookback=48, norm_window=500)
    result = pipe.build_from_frames(data["price"], data["macro"],
                                    split_dates=splits, history_len=3, stride=1)
    train_ds, val_ds = result["train"], result["val"]

    trained = _encoder(cfg, device); trained.load_state_dict(ckpt["encoder"]); trained.eval()
    random_ = _encoder(cfg, device); random_.eval()

    Z_tr_t, _, _ = extract_embeddings(trained, train_ds)
    Z_vl_t, _, _ = extract_embeddings(trained, val_ds)
    Z_tr_r, _, _ = extract_embeddings(random_, train_ds)
    Z_vl_r, _, _ = extract_embeddings(random_, val_ds)

    rows = []
    print(f"\n  {'horizon':>7} {'probe':>6} {'trained_r':>10} {'random_r':>10} {'T-R':>8}  verdict")
    for k in horizons:
        ytr, mtr = forward_vol(train_ds, k)
        yvl, mvl = forward_vol(val_ds, k)
        # trim each split to samples with a valid future target
        Ztr_t, Zvl_t = Z_tr_t[mtr], Z_vl_t[mvl]
        Ztr_r, Zvl_r = Z_tr_r[mtr], Z_vl_r[mvl]
        y = np.concatenate([ytr[mtr], yvl[mvl]])
        tgt = {"y": y, "task": "regression", "classes": 1,
               "label": f"forward_vol_{k}"}

        for probe in ["linear", "mlp"]:
            m_t = run_single_probe(Ztr_t, Zvl_t, tgt, probe)
            m_r = run_single_probe(Ztr_r, Zvl_r, tgt, probe)
            rt, rr = m_t.get("pearson_r", float("nan")), m_r.get("pearson_r", float("nan"))
            d = rt - rr
            verdict = "TRAINED WINS" if d > 0.02 else ("~tie" if abs(d) <= 0.02 else "random wins")
            print(f"  {k:>7} {probe:>6} {rt:>10.3f} {rr:>10.3f} {d:>+8.3f}  {verdict}")
            rows.append({"instrument": instrument, "horizon": k, "probe": probe,
                         "trained_r": rt, "random_r": rr, "delta_TR": d,
                         "n_samples": int(len(y))})

    out_json = f"./experiments/probe_forward_{instrument}.json"
    json.dump(rows, open(out_json, "w"), indent=2)
    dest = results_dir or (DRIVE_RESULTS if Path("/content/drive").is_dir() else None)
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(out_json, dest)
        print(f"\n  saved -> {dest}/probe_forward_{instrument}.json")
    else:
        print(f"\n  wrote {out_json}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    ap.add_argument("--horizons", default="1,4,12,24",
                    help="comma-separated forward horizons in bars (H1)")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    targets = ["gold", "eurusd", "usdjpy"] if args.instrument == "all" else [args.instrument]
    all_rows = []
    for t in targets:
        all_rows += run(t, horizons, args.results_dir)

    print("\n" + "=" * 70)
    print("VERDICT (max Δ(T-R) across horizons/probes, per instrument):")
    df = pd.DataFrame(all_rows)
    for inst in df["instrument"].unique():
        d = df[df["instrument"] == inst]["delta_TR"].max()
        tag = "O2 LIVES" if d > 0.02 else "no lift over random"
        print(f"  {inst:8s} best Δ(T-R) = {d:+.3f}   -> {tag}")
    print("\nIf trained beats random on forward vol, the encoder learned forward-")
    print("predictive structure beyond input persistence -- a defensible O2.")
