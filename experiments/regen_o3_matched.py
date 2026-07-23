"""
regen_o3_matched.py -- O3 recomputed on the MATCHED training ranges, canonical method
=====================================================================================
Reproduces the paper's O3 evaluation (evaluation/surprise.detect_extreme_events +
run_statistical_tests -- rolling-100 2-sigma extreme returns, MEDIAN surprise
ratio, rank-biserial effect size, Spearman) but on the SAME matched ranges as
every other table in mwm_audit_paper.tex (gold 2015, EUR/USD & USD/JPY 2020),
using the verified long best_model.pt (no training). Settles the O3 numbers.

Reports per instrument:
  extreme_return ratio + effect_size |r|          (violation-of-expectation test)
  Spearman(surprise, implied vol = x_last[:,43])  (gold=GVZ, eurusd=EVZ)
  Spearman(surprise, realized vol = x_last[:,10]) + rank-biserial |r|  (all three)

Writes o3_matched_{inst}.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import torch

sys.path.insert(0, ".")
import experiments.o1b_boundary_slide as O
from experiments.retrain_long import REGIME_SPLITS, _load_models
from data.pipeline import DataPipeline
from data.long_history import build_long
from evaluation.surprise import (compute_surprise_timeseries, detect_extreme_events,
                                 run_statistical_tests, IDX_RV32, IDX_GVZ_LEVEL)

CKPT = "experiments/checkpoints_long/{}/best_model.pt"


def run(inst):
    dev = torch.device("cpu")
    start, end = O.CACHED_RANGE[inst]
    data = build_long(inst, start, end)
    pipe = DataPipeline(instrument=inst, lookback=48, norm_window=500)
    res = pipe.build_from_frames(data["price"], data["macro"],
                                 split_dates=REGIME_SPLITS[inst], history_len=3, stride=1)
    enc, pred, cfg = _load_models(CKPT.format(inst), dev)
    ts = compute_surprise_timeseries(enc, pred, res["test"], batch_size=128, history_len=3)
    S = np.asarray(ts["surprise"]); xl, xn = ts["x_last"], ts["x_next"]

    # canonical events + tests (implied-vol proxy = idx 43)
    events = detect_extreme_events(xl, xn, vol_feature_idx=IDX_GVZ_LEVEL)
    stats = run_statistical_tests(S, events, xl[:, IDX_GVZ_LEVEL])
    er = stats["extreme_return"]
    rho_impl = stats["spearman_surprise_gvz"]["rho"]

    # realized-vol coupling (idx 10), same yardstick for all three assets
    rho_real = float(spearmanr(S, xl[:, IDX_RV32]).correlation)
    hi = xl[:, IDX_RV32] > np.median(xl[:, IDX_RV32])
    U = mannwhitneyu(S[hi], S[~hi], alternative="two-sided").statistic
    r_real = abs(1.0 - 2.0 * U / (hi.sum() * (~hi).sum()))

    out = {"instrument": inst, "n_test": int(len(S)), "range": f"{start}..{end}",
           "trained_through": REGIME_SPLITS[inst]["train_end"],
           "extreme_return_ratio_median": er["ratio"],
           "extreme_return_effect_r": er["effect_size_r"],
           "extreme_return_p": er["p_value"], "n_extreme": int(events["extreme_return"].sum()),
           "spearman_surprise_impliedvol_idx43": float(rho_impl),
           "spearman_surprise_realizedvol_idx10": rho_real,
           "rankbiserial_r_realizedvol_idx10": float(r_real)}
    Path(f"experiments/o3_matched_{inst}.json").write_text(json.dumps(out, indent=2))
    print(f"[{inst}] n={len(S)}  extreme_ratio(median)={er['ratio']:.3f} "
          f"|r|_ext={abs(er['effect_size_r']):.3f} p={er['p_value']:.2g} | "
          f"rho(impl idx43)={float(rho_impl):+.3f} | rho(real idx10)={rho_real:+.3f} "
          f"|r|_real={r_real:.3f}")
    return out


if __name__ == "__main__":
    for t in (["gold", "eurusd", "usdjpy"] if (len(sys.argv) < 2 or sys.argv[1] == "all")
              else [sys.argv[1]]):
        run(t)
