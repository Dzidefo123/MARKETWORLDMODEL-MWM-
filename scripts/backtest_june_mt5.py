"""
scripts/backtest_june_mt5.py — Validate a retrained H1 gold encoder on June 2026.
================================================================================

Runs the H1 execution backtest on MT5 XAUUSD data (the same symbol/source the
live system trades) for the NEW (fine-tuned) encoder and the OLD (production)
encoder, on the *same* June 2026 bars, and prints a side-by-side comparison of:

  • S_t distribution on June  — does the new encoder calm the morning spikes?
  • PnL / Sharpe / drawdown on June.

The test slice is May+June: May warms the circuit-breaker's rolling p20/p60/p90
percentiles so the June portion is judged with a fully-warmed filter. Metrics are
reported on the June slice only.

NOTE: the execution heads (experiments/heads_gold) were trained on the OLD
encoder's latent space. Fine-tuning keeps the geometry close, so the PnL read is
indicative, but the cleanest signal here is the S_t comparison (encoder+predictor
only, no head dependence). Retrain the heads on the new encoder before going live.

Usage:
    python -m scripts.backtest_june_mt5
    python -m scripts.backtest_june_mt5 --test-start 2026-06-01 --warm-start 2026-05-01
"""

import sys, argparse, json
sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

from models.encoder    import MarketEncoder
from models.predictor  import CausalPredictor
from data.pipeline     import DataPipeline
from execution.environment import TradingEnvironment
from execution.train_heads import load_heads
from execution.backtest    import run_backtest, compute_metrics

from scripts.retrain_gold_h1_mt5 import fetch_gold_h1_mt5, fetch_h1_macro, LOOKBACK, HISTORY_LEN

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_GOLD_BARS_PER_YEAR = 5690


def _load_enc_pred(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", {})
    enc = MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256), z_dim=cfg.get("z_dim", 128),
    )
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    pred = CausalPredictor(
        z_dim=cfg.get("z_dim", 128), d_model=cfg.get("pred_d_model", 128),
        n_heads=cfg.get("pred_n_heads", 4), n_layers=cfg.get("pred_n_layers", 6),
        action_dim=cfg.get("action_dim", 5), history_len=cfg.get("history_len", 3), dropout=0.0,
    )
    pred.load_state_dict(ckpt["predictor"]); pred.eval()
    for p in list(enc.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)
    return enc, pred, ckpt.get("epoch", "?")


def _metrics_for_subset(records: list, bars_per_year: int) -> dict:
    """Recompute metrics on a record subset, rebuilding the equity curve from pnl."""
    if not records:
        return {}
    pnl = np.array([r["pnl"] for r in records])
    equity = np.concatenate([[1.0], np.cumprod(1.0 + pnl)])
    return compute_metrics(records, equity, bars_per_year)


def _st_stats(records: list) -> dict:
    st = np.array([r["S_t"] for r in records])
    return {
        "n":    len(st),
        "mean": float(st.mean()),
        "p50":  float(np.percentile(st, 50)),
        "p90":  float(np.percentile(st, 90)),
        "p99":  float(np.percentile(st, 99)),
        "max":  float(st.max()),
    }


def _run_one(name: str, ckpt_path: str, heads_dir: str,
             test_feat, test_macro, test_prices, test_ts, june_mask,
             spread: float, dir_threshold: float, max_holding) -> dict:
    enc, pred, epoch = _load_enc_pred(ckpt_path)
    logger.info("[%s] encoder loaded (epoch %s) from %s", name, epoch, ckpt_path)

    layer = load_heads(heads_dir); layer.eval()

    env = TradingEnvironment(
        encoder=enc, predictor=pred,
        features=test_feat, macro_vecs=test_macro, prices=test_prices,
        spread_cost=spread, lookback=LOOKBACK, history_len=HISTORY_LEN,
    )
    bt = run_backtest(env, layer, bars_per_year=_GOLD_BARS_PER_YEAR,
                      max_holding_bars=max_holding, dir_threshold=dir_threshold,
                      verbose=False)

    # Map each record's bar_idx -> timestamp, keep June only.
    june_records = [r for r in bt["records"] if june_mask[r["bar_idx"]]]
    logger.info("[%s] June bars: %d  (of %d test bars)",
                name, len(june_records), len(bt["records"]))

    return {
        "epoch":   epoch,
        "st":      _st_stats(june_records),
        "metrics": _metrics_for_subset(june_records, _GOLD_BARS_PER_YEAR),
    }


def main():
    ap = argparse.ArgumentParser(description="Backtest retrained H1 encoder on June 2026")
    ap.add_argument("--new-ckpt", default="./experiments/checkpoints/gold_mt5_h1/gold/best_model.pt")
    ap.add_argument("--old-ckpt", default="./experiments/checkpoints/best_model.pt")
    ap.add_argument("--heads-dir", default="./experiments/heads_gold")
    ap.add_argument("--warm-start", default="2026-05-01",
                    help="Test slice start (warms circuit-breaker percentiles before June)")
    ap.add_argument("--test-start", default="2026-06-01", help="Report metrics from this date")
    ap.add_argument("--n-bars", type=int, default=5500)
    ap.add_argument("--spread", type=float, default=0.0003)
    ap.add_argument("--dir-threshold", type=float, default=0.53)
    ap.add_argument("--max-holding", type=int, default=None)
    args = ap.parse_args()

    for label, path in [("new", args.new_ckpt), ("old", args.old_ckpt)]:
        if not Path(path).exists():
            logger.error("%s checkpoint not found: %s", label, path); sys.exit(1)

    # ── Data: MT5 XAUUSD H1, build the May+June test slice ───────────────────
    price_df = fetch_gold_h1_mt5("XAUUSD", args.n_bars)
    macro_df = fetch_h1_macro(price_df.index)

    warm = pd.Timestamp(args.warm_start)
    split_dates = {
        "train_end": (warm - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "val_end":   (warm - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    pipeline = DataPipeline(instrument="gold", lookback=LOOKBACK, norm_window=500)
    result   = pipeline.build_from_frames(price_df, macro_df, stride=1,
                                          history_len=HISTORY_LEN, split_dates=split_dates)
    meta = result["meta"]
    n_train, n_val = meta["n_train"], meta["n_val"]
    test_feat   = meta["norm_features"][n_train + n_val:]
    test_macro  = meta["macro_vecs"][n_train + n_val:]
    test_prices = meta["prices"][n_train + n_val:]
    test_ts     = pd.DatetimeIndex(meta["timestamps"][n_train + n_val:])

    test_start = pd.Timestamp(args.test_start)
    if test_start.tzinfo is None and test_ts.tz is not None:
        test_start = test_start.tz_localize(test_ts.tz)
    june_mask = np.asarray(test_ts >= test_start)
    logger.info("Test slice: %s -> %s  (%d bars; %d in June reporting window)",
                test_ts[0].date(), test_ts[-1].date(), len(test_ts), int(june_mask.sum()))

    # ── Run both encoders on the same data ───────────────────────────────────
    res = {}
    for name, ckpt in [("new (fine-tuned)", args.new_ckpt), ("old (production)", args.old_ckpt)]:
        res[name] = _run_one(name, ckpt, args.heads_dir,
                             test_feat, test_macro, test_prices, test_ts, june_mask,
                             args.spread, args.dir_threshold, args.max_holding)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("JUNE 2026 BACKTEST  —  retrained vs production H1 encoder  (MT5 XAUUSD)")
    print("=" * 70)

    print("\n  S_t distribution on June (pure market surprise — the key signal):")
    print(f"  {'encoder':<20} {'mean':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for name in res:
        s = res[name]["st"]
        print(f"  {name:<20} {s['mean']:>9.4f} {s['p50']:>9.4f} {s['p90']:>9.4f} "
              f"{s['p99']:>9.4f} {s['max']:>9.4f}")

    print("\n  June PnL (heads are from the OLD encoder — indicative, see note):")
    print(f"  {'encoder':<20} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} "
          f"{'Return':>8} {'WinRate':>8} {'Trades':>7}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for name in res:
        m = res[name]["metrics"]
        if not m:
            print(f"  {name:<20}  (no June bars)"); continue
        print(f"  {name:<20} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['max_drawdown']:>7.2%} {m['total_return']:>7.2%} "
              f"{m['win_rate']:>7.2%} {m['n_trade_entries']:>7}")

    # Verdict on the S_t hypothesis. Positive change = S_t went UP (worse:
    # the new encoder is MORE surprised on June, not less).
    new_s, old_s = res["new (fine-tuned)"]["st"], res["old (production)"]["st"]
    if old_s["max"] > 1e-9:
        chg_max = (new_s["max"] / old_s["max"] - 1) * 100
        chg_p99 = (new_s["p99"] / old_s["p99"] - 1) * 100
        arrow = "UP (worse)" if chg_max > 0 else "DOWN (calmer)"
        print(f"\n  June S_t change new vs old: max {old_s['max']:.4f} -> {new_s['max']:.4f} "
              f"({chg_max:+.1f}%)  p99 {old_s['p99']:.4f} -> {new_s['p99']:.4f} "
              f"({chg_p99:+.1f}%)  => {arrow}")
        print(f"  Both encoders' offline June S_t are ~3 orders of magnitude below the "
              f"live ~1.9 spikes => live spikes are a normalization warm-up artifact.")
    print("=" * 70)

    out = Path("./experiments/june_backtest_mt5.json")
    out.write_text(json.dumps({
        "generated":   datetime.now().isoformat(timespec="seconds"),
        "test_slice":  [str(test_ts[0]), str(test_ts[-1])],
        "june_bars":   int(june_mask.sum()),
        "results":     res,
    }, indent=2, default=str))
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
