"""
regen_probing.py -- Recompute O1/O2 (spread + probes) on the long checkpoints
=============================================================================

The paper's O2 numbers (probing_results_*.json, dated May) were computed by
evaluation/probing.py, which loads ./experiments/checkpoints/{inst} -- the
SHORT-history checkpoints, and stale ones at that (final-epoch overwrite bug,
see [project_colab_retrain_loss]). This regenerates them on the verified
long-history best_model.pt so every number in the paper describes ONE model.

It cannot just call run_probing_experiments with a new checkpoint_path: that
function builds data via pipeline.build(use_real_data=True), which is capped at
~730 days and would mismatch the long split dates. So we mirror regen_surprise:
build_long -> build_from_frames(REGIME_SPLITS) -> reuse the probing internals.

Outputs -> probing_results_long_{inst}.json  (canonical May files untouched).

USAGE (Colab, after the mount/unzip/deps/symlink cells)
  !python experiments/regen_probing.py all
  !python experiments/regen_probing.py gold
"""

import argparse
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
from evaluation.probing import (
    extract_embeddings, build_probe_targets, run_single_probe,
    embedding_statistics, _print_summary_table, CHANCE_BASELINES,
)

DRIVE_RESULTS = "/content/drive/MyDrive/MWM_run/results"


def _encoder_from_cfg(cfg, device):
    return MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256),
        z_dim=cfg.get("z_dim", 128),
    ).to(device)


def regen(instrument, results_dir=None):
    print("\n" + "#" * 70)
    print(f"# REGEN O1/O2  [{instrument.upper()}]  (long checkpoint, no training)")
    print("#" * 70)

    ckpt_path = Path(f"{R.CKPT_ROOT}/{instrument}/best_model.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    print(f"  checkpoint: epoch {ckpt.get('epoch')} val={float(ckpt.get('val_loss', float('nan'))):.4f}")

    # 1) long data + identical splits/pipeline to retrain_long
    data = build_long(instrument)
    splits = R.REGIME_SPLITS[instrument]
    pipe = DataPipeline(instrument=instrument, lookback=48, norm_window=500)
    result = pipe.build_from_frames(
        data["price"], data["macro"],
        split_dates=splits, history_len=3, stride=1,
    )
    train_ds, val_ds = result["train"], result["val"]

    # 2) trained encoder (from long ckpt) + random control
    trained = _encoder_from_cfg(cfg, device)
    trained.load_state_dict(ckpt["encoder"]); trained.eval()
    random_ = _encoder_from_cfg(cfg, device); random_.eval()

    # 3) embeddings on train/val
    Z_tr_t, Xl_tr, Xn_tr = extract_embeddings(trained, train_ds)
    Z_vl_t, Xl_vl, Xn_vl = extract_embeddings(trained, val_ds)
    Z_tr_r, _, _ = extract_embeddings(random_, train_ds)
    Z_vl_r, _, _ = extract_embeddings(random_, val_ds)

    # 4) O1 spread + temporal-straightness stats (paper's "N.Nx more spread")
    stats = embedding_statistics(Z_vl_t, Z_vl_r)
    spread_ratio = stats["trained_var_mean"] / (stats["random_var_mean"] + 1e-12)
    print(f"\n[{instrument}] O1 embedding spread:")
    print(f"    trained var={stats['trained_var_mean']:.4f}  "
          f"random var={stats['random_var_mean']:.4f}  ratio={spread_ratio:.1f}x")
    print(f"    temporal cos  trained={stats['trained_temporal_cos']:.3f}  "
          f"random={stats['random_temporal_cos']:.3f}")

    # 5) O2 probes: 5 targets x 2 probes x 2 encoders
    vol_idx = 43 if instrument in ("gold", "eurusd") else 10
    n_train = len(Z_tr_t)
    Xl_all, Xn_all = np.vstack([Xl_tr, Xl_vl]), np.vstack([Xn_tr, Xn_vl])
    targets = build_probe_targets(Xl_all, Xn_all, vol_feature_idx=vol_idx)

    rows = []
    for tname, tinfo in targets.items():
        y_tr, y_vl = tinfo["y"][:n_train], tinfo["y"][n_train:]
        for probe_type in ["linear", "mlp"]:
            for enc_type, Z_tr, Z_vl in [("trained", Z_tr_t, Z_vl_t),
                                         ("random", Z_tr_r, Z_vl_r)]:
                m = run_single_probe(Z_tr, Z_vl,
                                     {**tinfo, "y": np.concatenate([y_tr, y_vl])},
                                     probe_type)
                rows.append({"target": tname, "label": tinfo["label"],
                             "probe": probe_type, "encoder": enc_type,
                             "task": tinfo["task"], **m})

    df = pd.DataFrame(rows)
    print(f"\n[{instrument}] O2 probes (trained vs random vs chance):")
    _print_summary_table(df)

    # 6) save -- long-suffixed, never overwrite canonical May files
    out_json = f"./experiments/probing_results_long_{instrument}.json"
    df.to_json(out_json, orient="records", indent=2)

    # stash spread stats alongside so the paper's O1 numbers are reproducible
    with open(f"./experiments/embedding_stats_long_{instrument}.json", "w") as f:
        import json
        json.dump({**stats, "spread_ratio": spread_ratio,
                   "epoch": ckpt.get("epoch"), "val_loss": float(ckpt.get("val_loss", float("nan")))},
                  f, indent=2)

    dest = results_dir or (DRIVE_RESULTS if Path("/content/drive").is_dir() else None)
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        for f in [out_json, f"./experiments/embedding_stats_long_{instrument}.json"]:
            shutil.copy(f, dest)
        print(f"    saved -> {dest}/probing_results_long_{instrument}.json (+ embedding_stats)")
    else:
        print(f"    wrote {out_json} (+ embedding_stats; no Drive mounted -- copy somewhere durable)")

    return df, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    targets = ["gold", "eurusd", "usdjpy"] if args.instrument == "all" else [args.instrument]
    for t in targets:
        regen(t, args.results_dir)
    print("\nDone. O1/O2 now describe the verified long best_model.pt weights "
          "(probing_results_long_*.json). Compare against the May canonical files.")
