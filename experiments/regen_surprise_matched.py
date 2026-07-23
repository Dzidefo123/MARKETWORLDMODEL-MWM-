"""
regen_surprise_matched.py -- surprise timeseries + event precision/recall on the
matched range and the CURRENT checkpoints
================================================================================

Why this exists
---------------
surprise_timeseries_long_{inst}.json and event_precision_recall_results.json in
this repo are dated 2026-07-01, i.e. they were produced from the superseded
short-run checkpoints, BEFORE the 60-epoch Colab checkpoints now in
experiments/checkpoints_long (2026-07-16/17). Every other number in the paper
was regenerated against those checkpoints; the recall figure quoted in
Section 5.3 was not. This closes that gap.

It also fixes the range bug that regen_surprise.py has: that script calls
build_long(inst) with no dates, which resolves to HONEST_START -> today and
therefore builds a different (longer) span than the one each encoder was trained
on. Here we use o1b_boundary_slide.CACHED_RANGE, matching probe_absolute.py,
embedding_spread.py and o3_stratified.py.

Writes
  surprise_timeseries_matched_{inst}.json   (schema event_precision_recall reads)
  event_precision_recall_matched.json       (recall/precision per asset)

USAGE
  python experiments/regen_surprise_matched.py all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
import experiments.o1b_boundary_slide as O
from experiments.retrain_long import REGIME_SPLITS, _load_models
from data.long_history import build_long
from data.pipeline import DataPipeline
from evaluation.surprise import (compute_surprise_timeseries, detect_extreme_events,
                                 smooth_surprise, IDX_GVZ_LEVEL)

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
    S = np.asarray(ts["surprise"], dtype=float)
    xl, xn = ts["x_last"], ts["x_next"]
    events = detect_extreme_events(xl, xn, vol_feature_idx=IDX_GVZ_LEVEL)

    # Timestamps for the test split, aligned to the surprise vector.
    meta = res.get("meta", {})
    all_ts = meta.get("timestamps")
    test_ts = ts.get("timestamps")
    if test_ts is None and all_ts is not None:
        test_ts = all_ts[-len(S):]
    if test_ts is None:
        raise SystemExit("no timestamps available from the pipeline for %s" % inst)

    out = {
        "timestamps":     [str(t) for t in list(test_ts)[:len(S)]],
        "surprise_raw":   S.tolist(),
        "surprise_smooth": smooth_surprise(S).tolist(),
        "extreme_return": events["extreme_return"].astype(int).tolist(),
        "high_gvz":       events["high_gvz"].astype(int).tolist(),
        "extreme_any":    events["extreme_any"].astype(int).tolist(),
        "vol_jump":       events["vol_jump"].astype(int).tolist(),
        # ret_z is metadata on the EVENT dict, not on the surprise dict. Taking it
        # from the wrong place silently zeroes event_precision_recall's reaction
        # filter, which then drops every calendar event and reports recall 0/0.
        "ret_z":          np.asarray(events["ret_z"], float).tolist(),
        "gvz_signal":     xl[:, IDX_GVZ_LEVEL].tolist(),
    }
    p = "experiments/surprise_timeseries_matched_%s.json" % inst
    Path(p).write_text(json.dumps(out))
    print("[%s] wrote %s  (n=%d, %s..%s)" % (inst, p, len(S), out["timestamps"][0][:10],
                                             out["timestamps"][-1][:10]))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    args = ap.parse_args()
    targets = (["gold", "eurusd", "usdjpy"] if args.instrument == "all"
               else [args.instrument])
    paths = {t: run(t) for t in targets}

    from experiments.event_precision_recall import load_calendar, analyse
    cal = load_calendar()
    summary = {}
    print("\n=== precision / recall vs macro calendar (current checkpoints) ===")
    for t, p in paths.items():
        r = analyse(t, p, cal, window=2, spike_pct=95.0, react_sigma=3.0,
                    top_n=15, merge_days=3)
        summary[t] = r
        rec = r["recall_hits"] / r["recall_den"] if r["recall_den"] else float("nan")
        prec = r["precision_hits"] / r["precision_den"] if r["precision_den"] else float("nan")
        print("  %-8s recall %2d/%-3d = %.2f   precision %2d/%-3d = %.2f"
              % (t, r["recall_hits"], r["recall_den"], rec,
                 r["precision_hits"], r["precision_den"], prec))
    Path("experiments/event_precision_recall_matched.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("\nwrote experiments/event_precision_recall_matched.json")
