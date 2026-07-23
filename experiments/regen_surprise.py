"""
regen_surprise.py -- Rebuild the O3 surprise JSONs from existing checkpoints
===========================================================================

Same pipeline as retrain_long.run(), with the training step removed. Use it when
the checkpoints are good but surprise_timeseries_long_{inst}.json is stale or was
lost with a Colab container.

The July 1 JSONs in this repo were computed from final-epoch weights (the old
train_o1.py overwrote best_model.pt on the last epoch), so every O3 / precision-
recall number derived from them describes the wrong model. This regenerates them
against the verified best_model.pt.

Split/pipeline params are imported from retrain_long rather than restated, so the
test window here cannot drift from the one used at training time.

USAGE (in Colab, after the mount/unzip/deps/symlink cells so the Drive cache is live)
  !python experiments/regen_surprise.py all
  !python experiments/regen_surprise.py gold
  !python experiments/regen_surprise.py all --results-dir /content/drive/MyDrive/MWM_run/results

  # hold the volatility definition fixed across instruments (see --vol-idx below)
  !python experiments/regen_surprise.py all --vol-idx 10   # rv_32   (realized)
  !python experiments/regen_surprise.py all --vol-idx 43   # gvz/evz (implied)

WHY --vol-idx
  By default each instrument keys its volatility events to a different feature:
  gold->GVZ, eurusd->EVZ (both idx 43, external implied vol), usdjpy->rv_32
  (idx 10, realized vol from its own returns). The 2026-07-17 O3 run came out
  null on gold (ratio 0.81) and marginal on eurusd (|r|=0.124) but strong on
  usdjpy (|r|=0.433) -- i.e. strong only where the vol proxy is derived from the
  same returns the model predicts, which is close to tautological. Forcing one
  index across all three separates "surprise tracks uncertainty" from "surprise
  tracks its own target's variance". Overridden runs are written to a suffixed
  filename so they never overwrite the canonical JSONs.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
from data.features import GoldFeatureEngineer
from data.long_history import build_long
from data.pipeline import DataPipeline
import experiments.retrain_long as R

DRIVE_RESULTS = "/content/drive/MyDrive/MWM_run/results"


def _vol_label(idx):
    names = getattr(GoldFeatureEngineer, "FEATURE_NAMES", [])
    return names[idx] if idx is not None and idx < len(names) else "?"


def regen(instrument, results_dir=None, vol_idx=None):
    print("\n" + "#" * 70)
    print(f"# REGEN SURPRISE  [{instrument.upper()}]  (no training)")
    if vol_idx is not None:
        print(f"#   vol_idx OVERRIDE = {vol_idx} ({_vol_label(vol_idx)}) -- same for every instrument")
    print("#" * 70)

    ckpt = Path(f"{R.CKPT_ROOT}/{instrument}/best_model.pt")
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    # 1) long data (cached -> fast; the Dukascopy pull already lives on Drive)
    data = build_long(instrument)

    # 2) identical splits + pipeline params to retrain_long.run
    splits = R.REGIME_SPLITS[instrument]
    pipe = DataPipeline(instrument=instrument, lookback=48, norm_window=500)
    result = pipe.build_from_frames(
        data["price"], data["macro"],
        split_dates=splits, history_len=3, stride=1,
    )

    # 3) O3 surprise from the verified best checkpoint.
    # An overridden vol_idx describes different events, so it gets its own file --
    # never overwrite the canonical per-instrument JSONs with a variant.
    suffix = "" if vol_idx is None else f"_vol{vol_idx}"
    out_json = f"./experiments/surprise_timeseries_long_{instrument}{suffix}.json"
    stats, _ = R.evaluate_surprise(instrument, result, str(ckpt), out_json, vol_idx=vol_idx)

    used = vol_idx if vol_idx is not None else (43 if instrument in ("gold", "eurusd") else 10)
    print(f"\n[{instrument}] O3 surprise (long test set)  "
          f"[vol events keyed to idx {used} = {_vol_label(used)}]:")
    for k in ["extreme_return", "high_gvz", "vol_jump", "extreme_any"]:
        r = stats.get(k, {})
        if "p_value" in r:
            print(f"    {k:<16} ratio={r['ratio']:.2f}x  r={r['effect_size_r']:.3f}  p={r['p_value']:.2g}")
    sp = stats.get("spearman_surprise_gvz", {})
    print(f"    Spearman(S,vol) rho={sp.get('rho', float('nan')):.3f} "
          f"p={sp.get('p_value', float('nan')):.2g}")

    # 4) precision/recall over calendar overlap
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
            print("    precision/recall: no calendar overlap in this test window")
    except Exception as ex:
        print(f"    precision/recall skipped: {repr(ex)[:100]}")

    # 5) persist -- ./experiments is NOT the Drive-symlinked dir, so copy out
    dest = results_dir or (DRIVE_RESULTS if Path("/content/drive").is_dir() else None)
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(out_json, dest)
        print(f"    saved -> {Path(dest) / Path(out_json).name}")
    else:
        print(f"    wrote {out_json} (no Drive mounted; copy it somewhere durable)")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--vol-idx", type=int, default=None,
                    help="Feature index defining volatility events, applied to EVERY "
                         "instrument. 10=rv_32 (realized, internal), 43=gvz/evz "
                         "(implied, external). Omit for the per-instrument default. "
                         "Overridden runs write to *_vol{N}.json.")
    args = ap.parse_args()
    targets = ["gold", "eurusd", "usdjpy"] if args.instrument == "all" else [args.instrument]
    for t in targets:
        regen(t, args.results_dir, args.vol_idx)

    if args.vol_idx is None:
        print("\nDone. These JSONs now describe the verified best_model.pt weights.")
    else:
        print(f"\nDone. vol_idx={args.vol_idx} ({_vol_label(args.vol_idx)}) held fixed across "
              f"{', '.join(targets)} -- results in *_vol{args.vol_idx}.json.")
        print("Compare |effect_size_r| and Spearman rho across instruments: the vol")
        print("definition is now the same, so differences are the model, not the proxy.")
