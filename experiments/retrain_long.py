"""
retrain_long.py -- End-to-end retrain + evaluate MWM on long H1 history
=======================================================================

Wires the long-history data (data/long_history.py) through the existing
training loop (training/train_o1.py) with REGIME-SPANNING date splits, then
re-runs the O3 surprise evaluation on the held-out test set and the
precision/recall audit -- all on the same long series.

Pipeline per instrument:
  build_long()               -> long H1 price + daily-macro (tz-aware UTC)
  DataPipeline.build_from_frames(split_dates=REGIME_SPLITS)   -> train/val/test
  train_o1.train(result)     -> checkpoint (experiments/checkpoints_long/{inst})
  compute_surprise_timeseries on result["test"]  -> surprise_timeseries_long_{inst}.json
  event_precision_recall.analyse                 -> recall/precision on calendar overlap

REGIME-SPANNING SPLITS (date-based; test held out, multi-year, calendar-overlapping):
  gold    train ->2021-12-31 | val 2022 (rate shock) | test 2023-01-01 -> now
  eurusd  train ->2022-12-31 | val 2023             | test 2024-01-01 -> now
  usdjpy  train ->2022-12-31 | val 2023             | test 2024-01-01 -> now
Val windows are now thousands of bars (vs 48/136 in the paper) -> reliable probes.

Training config matches the paper's pure JEPA objective for a clean O1-O3
reproduction: from scratch, MSE + 0.1*SIGReg, no directional aux loss.

USAGE
  python experiments/retrain_long.py eurusd                 # one instrument, full run
  python experiments/retrain_long.py all                    # all three
  python experiments/retrain_long.py eurusd --epochs 20     # shorter proof run
  python experiments/retrain_long.py eurusd --start 2015-01-01 --end 2016-06-01 --epochs 3
                                                            # fast smoke test (cached chunks)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from data.long_history import build_long
from data.pipeline import DataPipeline
import training.train_o1 as T
from evaluation.surprise import (
    compute_surprise_timeseries, detect_extreme_events, run_statistical_tests,
    smooth_surprise, IDX_RET1, IDX_RV32, IDX_GVZ_LEVEL,
)
from models.encoder import MarketEncoder
from models.predictor import CausalPredictor

CKPT_ROOT = "./experiments/checkpoints_long"

REGIME_SPLITS = {
    "gold":   {"train_end": "2021-12-31", "val_end": "2022-12-31"},
    "eurusd": {"train_end": "2022-12-31", "val_end": "2023-12-31"},
    "usdjpy": {"train_end": "2022-12-31", "val_end": "2023-12-31"},
}


def _load_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    enc = MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256),
        z_dim=cfg.get("z_dim", 128),
    ).to(device)
    enc.load_state_dict(ckpt["encoder"])
    pred = CausalPredictor(
        z_dim=cfg.get("z_dim", 128), d_model=cfg.get("pred_d_model", 128),
        n_heads=cfg.get("pred_n_heads", 4), n_layers=cfg.get("pred_n_layers", 6),
        action_dim=cfg.get("action_dim", 5), history_len=cfg.get("history_len", 3),
        dropout=0.0,
    ).to(device)
    pred.load_state_dict(ckpt["predictor"])
    enc.eval(); pred.eval()
    return enc, pred, cfg


def evaluate_surprise(instrument, result, ckpt_path, out_json, vol_idx=None):
    """Compute O3 surprise on the long test set; save timeseries JSON in the
    same schema as evaluation/surprise.py so event_precision_recall can read it.

    vol_idx selects the feature whose level defines volatility events. Default
    (None) keeps the per-instrument choice: an external implied-vol index where
    one exists (GVZ for gold, EVZ for eurusd -- both sit at IDX_GVZ_LEVEL), and
    realized vol otherwise (usdjpy has no JPY vol index in the macro set).

    That default makes the instruments non-comparable: RV32 is computed from the
    same returns the model predicts, so surprise tracking it is partly mechanical,
    while GVZ/EVZ are independent. Pass vol_idx explicitly to hold the definition
    fixed across instruments and find out which the O3 claim actually rests on.
    """
    device = torch.device("cpu")
    enc, pred, cfg = _load_models(ckpt_path, device)

    if vol_idx is None:
        vol_idx = IDX_GVZ_LEVEL if instrument in ("gold", "eurusd") else IDX_RV32
    test_ds = result["test"]
    meta = result["meta"]
    n_train, n_val = meta["n_train"], meta["n_val"]
    all_ts = meta["timestamps"]
    test_start = n_train + n_val
    test_ts = all_ts[test_start: test_start + len(test_ds) + 1]

    ts_data = compute_surprise_timeseries(enc, pred, test_ds, batch_size=128,
                                          history_len=cfg.get("history_len", 3))
    surprise = ts_data["surprise"]
    x_last, x_next = ts_data["x_last"], ts_data["x_next"]
    events = detect_extreme_events(x_last, x_next, vol_feature_idx=vol_idx)
    stats = run_statistical_tests(surprise, events, x_last[:, vol_idx])

    ts_json = {
        "timestamps":      [str(t) for t in test_ts[:len(surprise)]],
        "surprise_raw":    surprise.tolist(),
        "surprise_smooth": smooth_surprise(surprise, window=12).tolist(),
        "extreme_return":  events["extreme_return"].tolist(),
        "high_gvz":        events["high_gvz"].tolist(),
        "vol_jump":        events["vol_jump"].tolist(),
        "extreme_any":     events["extreme_any"].tolist(),
        "ret_z":           events["ret_z"].tolist(),
        "gvz_signal":      x_last[:, vol_idx].tolist(),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(ts_json, f, default=str)
    return stats, ts_json


def run(instrument, start, end, epochs):
    print("\n" + "#" * 70)
    print(f"# RETRAIN-LONG  [{instrument.upper()}]  epochs={epochs}")
    print("#" * 70)

    # 1) long data
    data = build_long(instrument, start, end)

    # 2) regime-spanning splits
    splits = REGIME_SPLITS[instrument]
    # fail loudly if the pulled data does not extend past val_end (else the
    # test split is empty and build_from_frames crashes deep in a print)
    data_end_ts = data["price"].index.max()
    val_end_ts = pd.Timestamp(splits["val_end"])
    if val_end_ts.tzinfo is None and data_end_ts.tzinfo is not None:
        val_end_ts = val_end_ts.tz_localize("UTC")
    if data_end_ts <= val_end_ts:
        raise RuntimeError(
            f"[{instrument}] data ends {data_end_ts.date()} but val_end is "
            f"{splits['val_end']} -> empty test split. Check the pulled range / "
            f"delete a stale combined cache in data/cache/long/.")
    pipe = DataPipeline(instrument=instrument, lookback=48, norm_window=500)
    result = pipe.build_from_frames(
        data["price"], data["macro"],
        split_dates=splits, history_len=3, stride=1,
    )

    # 3) train (paper's pure JEPA objective, from scratch)
    T.CFG.update({
        "instrument":    instrument,
        "split_dates":   splits,
        "n_epochs":      epochs,
        "lr":            3e-4,
        "warmup_epochs":  5,
        "lambda_dir":    0.0,        # pure MSE + 0.1*SIGReg (paper O1)
        "finetune_from": None,       # from scratch
        "checkpoint_dir": CKPT_ROOT,
        "log_file":      f"./experiments/retrain_long_{instrument}.log",
    })
    T.train(result)

    # 4) O3 surprise on held-out long test set
    ckpt_path = f"{CKPT_ROOT}/{instrument}/best_model.pt"
    out_json = f"./experiments/surprise_timeseries_long_{instrument}.json"
    stats, ts_json = evaluate_surprise(instrument, result, ckpt_path, out_json)

    print(f"\n[{instrument}] O3 surprise (long test set):")
    for k in ["extreme_return", "high_gvz", "vol_jump", "extreme_any"]:
        r = stats.get(k, {})
        if "p_value" in r:
            print(f"    {k:<16} ratio={r['ratio']:.2f}x  r={r['effect_size_r']:.3f}  p={r['p_value']:.2g}")
    sp = stats.get("spearman_surprise_gvz", {})
    print(f"    Spearman(S,vol) rho={sp.get('rho',float('nan')):.3f} p={sp.get('p_value',float('nan')):.2g}")

    # 5) precision/recall over calendar overlap (needs experiments/macro_calendar.json)
    try:
        from experiments.event_precision_recall import load_calendar, analyse
        cal = load_calendar()
        r = analyse(instrument, out_json, cal, window=2, spike_pct=95.0,
                    react_sigma=3.0, top_n=15, merge_days=3)
        if r["recall_den"] > 0 or r["precision_den"] > 0:
            print(f"    RECALL={r['recall_hits']}/{r['recall_den']}  "
                  f"PRECISION={r['precision_hits']}/{r['precision_den']}  "
                  f"(calendar overlap {r['window_start']}..{r['window_end']})")
        else:
            print("    precision/recall: no calendar overlap in this test window "
                  "(extend macro_calendar.json back to test start)")
    except Exception as ex:
        print(f"    precision/recall skipped: {repr(ex)[:100]}")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="eurusd")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    targets = ["gold", "eurusd", "usdjpy"] if args.instrument == "all" else [args.instrument]
    for t in targets:
        run(t, args.start, args.end, args.epochs)
