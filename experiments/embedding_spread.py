"""
embedding_spread.py -- O1 anti-collapse on the matched range
============================================================

Table 1 of the paper (mean per-dimension embedding variance, trained vs random)
came from regen_probing.py, which built data over HONEST_START->today rather
than the range each checkpoint was trained on. This recomputes it on
o1b_boundary_slide.CACHED_RANGE, matching probe_absolute.py, o3_stratified.py
and the O1b artifact checks, so every table in the paper describes one data path.

Only the validation split is needed, so this is fast. The random encoder is
averaged over N_SEEDS initializations, as the caption claims.

USAGE
  python experiments/embedding_spread.py all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
import experiments.o1b_boundary_slide as O
import experiments.retrain_long as R
from data.long_history import build_long
from data.pipeline import DataPipeline
from models.encoder import MarketEncoder
from evaluation.probing import extract_embeddings

CKPT = "experiments/checkpoints_long/{}/best_model.pt"
N_SEEDS = 10


def _enc(cfg, device):
    return MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256),
        z_dim=cfg.get("z_dim", 128),
    ).to(device)


def run(inst):
    device = torch.device("cpu")
    ckpt = torch.load(CKPT.format(inst), map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})

    start, end = O.CACHED_RANGE[inst]
    data = build_long(inst, start, end)
    pipe = DataPipeline(instrument=inst, lookback=48, norm_window=500)
    res = pipe.build_from_frames(data["price"], data["macro"],
                                 split_dates=R.REGIME_SPLITS[inst],
                                 history_len=3, stride=1)
    val_ds = res["val"]

    trained = _enc(cfg, device)
    trained.load_state_dict(ckpt["encoder"]); trained.eval()
    Z_t, _, _ = extract_embeddings(trained, val_ds)
    var_t = float(np.mean(np.var(Z_t, axis=0)))

    var_r, ratios = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        r = _enc(cfg, device); r.eval()
        Z_r, _, _ = extract_embeddings(r, val_ds)
        v = float(np.mean(np.var(Z_r, axis=0)))
        var_r.append(v); ratios.append(var_t / (v + 1e-12))

    out = {"instrument": inst, "range": f"{start}..{end}", "n_val": int(len(Z_t)),
           "var_trained": var_t,
           "var_random_mean": float(np.mean(var_r)),
           "var_random_std": float(np.std(var_r)),
           "ratio_mean": float(np.mean(ratios)), "ratio_std": float(np.std(ratios)),
           "n_seeds": N_SEEDS}
    Path(f"experiments/embedding_spread_{inst}.json").write_text(json.dumps(out, indent=2))
    print(f"[{inst}] n_val={len(Z_t)}  trained={var_t:.3f}  "
          f"random={np.mean(var_r):.4f}+-{np.std(var_r):.4f}  "
          f"ratio={np.mean(ratios):.1f}x +-{np.std(ratios):.1f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    args = ap.parse_args()
    for t in (["gold", "eurusd", "usdjpy"] if args.instrument == "all"
              else [args.instrument]):
        run(t)
