"""
probe_absolute.py -- absolute probe performance: inputs / random / trained
=========================================================================

regen_probing.py produced probing_results_long_{inst}.json (trained vs random,
5 targets x 2 probes). Table 2 of the paper reports only the DELTA. This script
re-emits the same evaluation in absolute terms and adds two reference columns
that settle whether the trained-ties-random result is a floor or ceiling effect:

  chance   majority-class rate on the val split (classification) / 0 (regression)
  inputs   the SAME probe trained on the raw 52 input features at the last bar,
           i.e. the information the encoder is given. This is the ceiling, and
           for every input-derived target it is near-perfect -- which is exactly
           the Johnson-Lindenstrauss argument of C1 made empirical: a random
           projection retains most of it, and prediction-driven compression in
           the trained encoder discards some.

RANGE NOTE
  probing_results_long_{inst}.json (2026-07-17) called build_long(inst) with no
  dates, i.e. HONEST_START -> today: gold from 2008, EUR/USD from 2004, USD/JPY
  from 2006. The checkpoints were trained on the shorter matched windows (gold
  2015-, FX 2020-, see RETRAIN_LONG_RESULTS.md), which is also the range every
  O3/O1b table uses. So the old probe table fit its probes over years the
  encoder never saw. This script uses o1b_boundary_slide.CACHED_RANGE, matching
  the encoder and the rest of the paper; the deltas move by <=0.06 and no
  conclusion changes, but the paper is now one consistent data path.

The random encoder is averaged over N_SEEDS initializations (a single random
draw is itself a sample) and the per-cell std is reported.

Writes probe_absolute_{inst}.json and prints a LaTeX-ready summary.

USAGE
  python experiments/probe_absolute.py all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import experiments.o1b_boundary_slide as O
import experiments.retrain_long as R
from data.long_history import build_long
from data.pipeline import DataPipeline
from models.encoder import MarketEncoder
from evaluation.probing import (extract_embeddings, build_probe_targets,
                                run_single_probe)

CKPT = "experiments/checkpoints_long/{}/best_model.pt"
N_SEEDS = 5


def _encoder_from_cfg(cfg, device):
    return MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256),
        z_dim=cfg.get("z_dim", 128),
    ).to(device)


def run(inst):
    device = torch.device("cpu")
    ckpt_path = Path(CKPT.format(inst))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})

    start, end = O.CACHED_RANGE[inst]
    data = build_long(inst, start, end)
    pipe = DataPipeline(instrument=inst, lookback=48, norm_window=500)
    res = pipe.build_from_frames(data["price"], data["macro"],
                                 split_dates=R.REGIME_SPLITS[inst],
                                 history_len=3, stride=1)
    train_ds, val_ds = res["train"], res["val"]

    trained = _encoder_from_cfg(cfg, device)
    trained.load_state_dict(ckpt["encoder"]); trained.eval()

    Z_tr_t, Xl_tr, Xn_tr = extract_embeddings(trained, train_ds)
    Z_vl_t, Xl_vl, Xn_vl = extract_embeddings(trained, val_ds)

    randoms = []
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        r = _encoder_from_cfg(cfg, device); r.eval()
        Z_tr_r, _, _ = extract_embeddings(r, train_ds)
        Z_vl_r, _, _ = extract_embeddings(r, val_ds)
        randoms.append((Z_tr_r, Z_vl_r))
        print(f"  random encoder seed {s} done")

    vol_idx = 43 if inst in ("gold", "eurusd") else 10
    n_train = len(Z_tr_t)
    Xl_all, Xn_all = np.vstack([Xl_tr, Xl_vl]), np.vstack([Xn_tr, Xn_vl])
    targets = build_probe_targets(Xl_all, Xn_all, vol_feature_idx=vol_idx)

    def _score(Ztr, Zvl, tinfo, probe_type):
        m = run_single_probe(Ztr, Zvl, tinfo, probe_type)
        return float(m["pearson_r"] if tinfo["task"] == "regression" else m["accuracy"])

    rows = []
    for tname, tinfo in targets.items():
        y_vl = tinfo["y"][n_train:]
        # empirical floor: always-predict-majority on the val split
        chance = (float(np.bincount(y_vl.astype(int)).max() / len(y_vl))
                  if tinfo["task"] == "classification" else 0.0)
        for probe_type in ["linear", "mlp"]:
            rand_scores = [_score(a, b, tinfo, probe_type) for a, b in randoms]
            for src, val, sd in [
                ("inputs",  _score(Xl_tr, Xl_vl, tinfo, probe_type), 0.0),
                ("random",  float(np.mean(rand_scores)), float(np.std(rand_scores))),
                ("trained", _score(Z_tr_t, Z_vl_t, tinfo, probe_type), 0.0),
            ]:
                rows.append({"target": tname, "probe": probe_type, "source": src,
                             "task": tinfo["task"], "score": val, "std": sd,
                             "chance": chance, "n_val": int(len(y_vl)),
                             "n_train": int(n_train),
                             "random_seeds": rand_scores if src == "random" else None})

    df = pd.DataFrame(rows)
    Path(f"experiments/probe_absolute_{inst}.json").write_text(
        df.to_json(orient="records", indent=2))

    piv = df.pivot_table(index=["target", "probe"], columns="source", values="score")
    piv = piv[["inputs", "random", "trained"]]
    piv["rand_sd"] = df[df.source == "random"].set_index(["target", "probe"])["std"]
    piv["chance"] = df.groupby(["target", "probe"])["chance"].first()
    piv["delta"] = piv["trained"] - piv["random"]
    print(f"\n[{inst}]  n_train={n_train}  n_val={len(Z_vl_t)}  "
          f"range={start}..{end}  train_end={R.REGIME_SPLITS[inst]['train_end']}")
    print(piv.round(3).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    args = ap.parse_args()
    for t in (["gold", "eurusd", "usdjpy"] if args.instrument == "all"
              else [args.instrument]):
        run(t)
