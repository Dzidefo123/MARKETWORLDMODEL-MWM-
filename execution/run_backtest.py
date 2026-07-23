"""
execution/run_backtest.py — Run the full backtest on the gold test split.

Usage:
    python -m execution.run_backtest [--instrument gold] [--checkpoint PATH]
                                     [--max-holding BARS] [--dir-threshold FLOAT]
                                     [--spread FLOAT]

Output:
    - Full performance report (Sharpe, drawdown, win rate, ...)
    - Session breakdown (Asian / London / NY)
    - Vol-regime breakdown (low / medium / high)
    - S_t quintile breakdown (Q1 = most predictable → Q5 = most surprising)
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

from models.encoder  import MarketEncoder
from models.predictor import CausalPredictor
from data.pipeline import DataPipeline
from execution.environment import TradingEnvironment
from execution.train_heads import load_heads
from execution.heads import encode_split
from execution.backtest import run_backtest, print_report

# Gold H1 futures: ~11 383 bars over 2 years → ~5 690 bars/year.
# backtest.py docstring estimates 5 938; we use the empirical figure.
_GOLD_BARS_PER_YEAR = 5690


# ---------------------------------------------------------------------------
# Extra breakdown helpers
# ---------------------------------------------------------------------------

def vol_regime_breakdown(records: list, vol_classes: np.ndarray,
                         lookback: int = 48, bars_per_year: int = _GOLD_BARS_PER_YEAR) -> dict:
    """
    Split records by vol-head predicted regime (0=low, 1=med, 2=high).
    vol_classes[i] corresponds to decision bar_idx = i + lookback - 1.
    """
    bar_to_regime = {}
    for i, vc in enumerate(vol_classes):
        bar_to_regime[i + lookback - 1] = int(vc)

    labels = {0: "Low vol ", 1: "Med vol ", 2: "High vol"}
    result = {}
    for regime, name in labels.items():
        subset = [r for r in records if bar_to_regime.get(r["bar_idx"], -1) == regime]
        if not subset:
            result[name] = {"n_bars": 0, "n_active": 0, "sharpe": 0.0,
                            "win_rate": 0.0, "total_pnl": 0.0}
            continue
        pnl    = np.array([r["pnl"] for r in subset])
        eff    = np.array([r["effective_action"] for r in subset])
        active = np.abs(eff) > 1e-9
        n_act  = int(active.sum())
        sharpe = float(pnl.mean() / pnl.std() * np.sqrt(bars_per_year)) if pnl.std() > 1e-12 else 0.0
        wr     = float((pnl[active] > 0).mean()) if n_act > 0 else 0.0
        result[name] = {
            "n_bars":    len(subset),
            "n_active":  n_act,
            "sharpe":    round(sharpe, 4),
            "win_rate":  round(wr, 4),
            "total_pnl": round(float(pnl.sum()), 6),
        }
    return result


def st_quintile_breakdown(records: list, bars_per_year: int = _GOLD_BARS_PER_YEAR) -> dict:
    """Split records into five equal S_t buckets (Q1 = low surprise, Q5 = high surprise)."""
    return _st_bucket_breakdown(records, n_buckets=5, bars_per_year=bars_per_year)


def st_decile_breakdown(records: list, bars_per_year: int = _GOLD_BARS_PER_YEAR) -> dict:
    """Split records into ten equal S_t buckets for fine-grained boundary detection."""
    return _st_bucket_breakdown(records, n_buckets=10, bars_per_year=bars_per_year)


def _st_bucket_breakdown(records: list, n_buckets: int,
                          bars_per_year: int = _GOLD_BARS_PER_YEAR) -> dict:
    st_vals = np.array([r["S_t"] for r in records])
    pct_pts = np.linspace(0, 100, n_buckets + 1)[1:-1]   # interior edges
    edges   = np.percentile(st_vals, pct_pts)
    width   = 100 // n_buckets

    def bidx(s):
        return int(np.searchsorted(edges, s))  # 0..n_buckets-1

    result = {}
    for b in range(n_buckets):
        lo = b * width
        hi = lo + width
        label  = f"D{b+1:02d} (p{lo:02d}-p{hi:02d})" if n_buckets == 10 \
                 else f"Q{b+1} (p{lo:02d}-p{hi:02d})"
        subset = [r for r in records if bidx(r["S_t"]) == b]
        if not subset:
            result[label] = {"n_bars": 0, "n_active": 0, "sharpe": 0.0,
                             "win_rate": 0.0, "total_pnl": 0.0}
            continue
        pnl    = np.array([r["pnl"] for r in subset])
        eff    = np.array([r["effective_action"] for r in subset])
        active = np.abs(eff) > 1e-9
        n_act  = int(active.sum())
        sharpe = float(pnl.mean() / pnl.std() * np.sqrt(bars_per_year)) if pnl.std() > 1e-12 else 0.0
        wr     = float((pnl[active] > 0).mean()) if n_act > 0 else 0.0
        result[label] = {
            "n_bars":    len(subset),
            "n_active":  n_act,
            "sharpe":    round(sharpe, 4),
            "win_rate":  round(wr, 4),
            "total_pnl": round(float(pnl.sum()), 6),
        }
    return result


def _print_breakdown_table(title: str, bd: dict) -> None:
    print(f"\n  {title}:")
    col_w = max(len(k) for k in bd) + 1
    print(f"  {'':<{col_w}} {'Bars':>6} {'Active':>6} {'Sharpe':>8} {'WinRate':>8} {'PnL':>10}")
    print(f"  {'-'*col_w} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for name, s in bd.items():
        if s.get("n_bars", 0) == 0:
            print(f"  {name:<{col_w}}   —")
            continue
        neg = " **" if s["sharpe"] < 0 else ""
        print(f"  {name:<{col_w}} {s['n_bars']:>6,} {s['n_active']:>6,} "
              f"{s['sharpe']:>8.3f} {s['win_rate']:>8.2%} {s['total_pnl']:>10.5f}{neg}")


def print_extra_breakdowns(vol_bd: dict, st_bd: dict, st_decile_bd: dict) -> None:
    _print_breakdown_table("Vol-regime breakdown", vol_bd)
    _print_breakdown_table("S_t quintile breakdown  (Q1=calm, Q5=chaotic)", st_bd)
    _print_breakdown_table("S_t decile breakdown    (boundary detection)", st_decile_bd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument",    default="gold",
                        choices=["gold", "eurusd", "usdjpy"])
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Encoder checkpoint path. Defaults to root best_model.pt.")
    parser.add_argument("--heads-dir",     type=str, default=None,
                        help="Directory with saved execution layer. "
                             "Defaults to experiments/heads_{instrument}.")
    parser.add_argument("--max-holding",   type=int, default=None,
                        help="Force-flat after this many bars. None = no limit.")
    parser.add_argument("--dir-threshold", type=float, default=0.53,
                        help="dir_prob threshold for entry (default 0.53).")
    parser.add_argument("--spread",        type=float, default=0.0003,
                        help="One-way spread cost as fraction of position (default 0.0003).")
    parser.add_argument("--skip-sessions", type=str, default=None,
                        help="Comma-separated session names to skip, e.g. NY,Asian.")
    args = parser.parse_args()

    # --- Resolve paths ---
    ckpt_path  = args.checkpoint or "./experiments/checkpoints/best_model.pt"
    heads_dir  = args.heads_dir  or f"./experiments/heads_{args.instrument}"
    skip_sess  = [s.strip() for s in args.skip_sessions.split(",")] if args.skip_sessions else None

    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not Path(heads_dir).exists():
        print(f"Heads directory not found: {heads_dir}")
        print(f"Run: python -m execution.train_heads --instrument {args.instrument}")
        sys.exit(1)

    # --- Load encoder + predictor ---
    print(f"Loading encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", {})

    encoder = MarketEncoder(
        lookback    = cfg.get("lookback",       48),
        n_features  = cfg.get("n_features",     52),
        patch_size  = 4,
        d_model     = cfg.get("enc_d_model",   128),
        n_heads     = cfg.get("enc_n_heads",     4),
        n_layers    = cfg.get("enc_n_layers",    4),
        dropout     = 0.0,
        proj_hidden = cfg.get("enc_proj_hidden", 256),
        z_dim       = cfg.get("z_dim",          128),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    predictor = CausalPredictor(
        z_dim       = cfg.get("z_dim",         128),
        d_model     = cfg.get("pred_d_model",  128),
        n_heads     = cfg.get("pred_n_heads",    4),
        n_layers    = cfg.get("pred_n_layers",   6),
        action_dim  = cfg.get("action_dim",      5),
        history_len = cfg.get("history_len",     3),
        dropout     = 0.0,
    )
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    print(f"  Encoder + predictor loaded (epoch {ckpt.get('epoch', '?')})")

    # --- Load execution layer ---
    print(f"Loading execution layer from: {heads_dir}")
    layer = load_heads(heads_dir)
    layer.eval()

    with open(Path(heads_dir) / "metrics.json") as f:
        head_metrics = json.load(f)
    print(f"  Dir AUC={head_metrics['direction']['auc']:.3f}  "
          f"Vol AUC={head_metrics['vol']['auc']:.3f}")

    # --- Build data pipeline ---
    print("\nBuilding data pipeline...")
    split_dates = cfg.get("split_dates")
    pipeline    = DataPipeline(instrument=args.instrument, lookback=48, norm_window=500)
    result      = pipeline.build(
        use_real_data=True, stride=1, history_len=3, split_dates=split_dates
    )
    meta        = result["meta"]
    lookback    = 48
    history_len = 3

    # Test split slice
    n_train = meta["n_train"]
    n_val   = meta["n_val"]
    test_features  = meta["norm_features"][n_train + n_val:]
    test_macros    = meta["macro_vecs"][n_train + n_val:]
    test_prices    = meta["prices"][n_train + n_val:]

    print(f"  Test split: {len(test_features)} bars")

    # --- Encode test split for vol-regime breakdown ---
    print("  Encoding test split for vol-regime analysis...")
    test_feat_t = torch.as_tensor(test_features, dtype=torch.float32)
    Z_test      = encode_split(encoder, test_feat_t, lookback=lookback)  # (n_test - lb + 1, 128)
    with torch.no_grad():
        vol_logits  = layer.vol_head(torch.FloatTensor(Z_test))
        vol_classes = vol_logits.argmax(dim=-1).numpy()  # (n_z,) — 0/1/2

    # --- Build environment ---
    env = TradingEnvironment(
        encoder     = encoder,
        predictor   = predictor,
        features    = test_features,
        macro_vecs  = test_macros,
        prices      = test_prices,
        spread_cost = args.spread,
        lookback    = lookback,
        history_len = history_len,
    )

    # --- Run backtest ---
    print(f"\nRunning backtest  (dir_thresh={args.dir_threshold}  "
          f"spread={args.spread:.4f}  max_hold={args.max_holding}  "
          f"skip={skip_sess})...")
    bt = run_backtest(
        env,
        layer,
        bars_per_year    = _GOLD_BARS_PER_YEAR,
        max_holding_bars = args.max_holding,
        skip_sessions    = skip_sess,
        dir_threshold    = args.dir_threshold,
        verbose          = True,
    )

    # --- Standard report ---
    print_report(bt, instrument=args.instrument)

    # --- Extra breakdowns ---
    vol_bd      = vol_regime_breakdown(bt["records"], vol_classes, lookback=lookback)
    st_bd       = st_quintile_breakdown(bt["records"])
    st_decile   = st_decile_breakdown(bt["records"])
    print_extra_breakdowns(vol_bd, st_bd, st_decile)
    print()
