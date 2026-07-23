"""
scripts/retrain_gold_h1_mt5.py — Retrain the H1 gold encoder on recent MT5 data.
================================================================================

WHY
---
The production H1 encoder (experiments/checkpoints/best_model.pt) was trained on
GC=F H1 bars from Yahoo Finance, which is capped at the last ~730 days and which
is a *futures* proxy — not the XAUUSD spot we actually trade. It never saw the
2026 regime: the all-time high, the January crash, the February Iran-conflict
onset, or the post-NFP June shift. Every morning S_t spike is the encoder hitting
a market state with no analog in its training data.

This script fixes that by pulling H1 XAUUSD straight from MT5 (no 730-day cap,
and it's the *same* symbol we trade), then retraining the encoder on the most
recent window so it learns the current regime. It reuses the exact training loop
and checkpoint format from training.train_o1, so the output is a drop-in
replacement for best_model.pt.

USAGE
-----
    # Verify the data path only (fast — fetch + features + split, no training):
    python -m scripts.retrain_gold_h1_mt5 --dry-run

    # Full retrain, fine-tuning from the current production encoder:
    python -m scripts.retrain_gold_h1_mt5 --finetune

    # Full retrain from scratch:
    python -m scripts.retrain_gold_h1_mt5 --scratch

Requires the MT5 terminal to be running. Writes the new checkpoint to
--out (default experiments/checkpoints/gold_mt5_h1/gold/best_model.pt) so the
live production checkpoint is never clobbered — backtest the new model before
swapping it in.
"""

import sys, argparse
sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from data.fetcher  import fallback_tickers
from data.pipeline import DataPipeline

# Durable, timestamped run logs live here (one file per run, never overwritten).
LOG_DIR = Path("./experiments/logs")


def _setup_logging() -> Path:
    """Log to console AND a timestamped file under experiments/logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"retrain_gold_h1_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return log_path


logger = logging.getLogger(__name__)

# Architecture must match the production checkpoint so _load_system in
# execution.live_trader (which builds MarketEncoder() with defaults) can load it.
LOOKBACK     = 48
HISTORY_LEN  = 3
NORM_WINDOW  = 500

# yfinance H1 hard limit — macro can only go back ~728 days.
YF_H1_DAYS   = 728


# ---------------------------------------------------------------------------
# Step 1 — H1 gold from MT5 (the symbol we actually trade, no 730-day cap)
# ---------------------------------------------------------------------------

def fetch_gold_h1_mt5(symbol: str, n_bars: int) -> pd.DataFrame:
    """Return the last `n_bars` completed H1 bars for `symbol` from MT5."""
    from execution.mt5_connector import MT5Connector

    connector = MT5Connector(symbol=symbol)
    connector.connect()
    try:
        attempt = n_bars
        price = None
        while attempt >= min(100, n_bars):
            try:
                logger.info("Fetching %d H1 bars of %s from MT5...", attempt, symbol)
                price = connector.fetch_ohlcv_bars(n=attempt, timeframe="H1")
                break
            except RuntimeError as exc:
                if "Invalid params" in str(exc):
                    logger.warning("  MT5 rejected n=%d, halving...", attempt)
                    attempt //= 2
                else:
                    raise
        if price is None or len(price) == 0:
            raise RuntimeError("Could not fetch H1 data from MT5 — all attempts failed")
    finally:
        connector.disconnect()

    logger.info("  Got %d H1 bars: %s -> %s", len(price), price.index[0], price.index[-1])
    return price


# ---------------------------------------------------------------------------
# Step 2 — H1 macro from yfinance (DXY/GVZ/TLT/Silver), aligned to gold bars
# ---------------------------------------------------------------------------

def fetch_h1_macro(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch H1 macro closes and forward-fill them onto the gold H1 index.

    Returns a DataFrame with columns [dxy, gvz, tlt, silver] aligned 1:1 to
    price_index. dxy goes through fallback_tickers so a single delisting can't
    break the feed. Any series Yahoo can't supply is filled with a constant
    (its features then contribute ~0 after rolling z-scoring).
    """
    import yfinance as yf

    macro_tickers = {"dxy": "DX-Y.NYB", "gvz": "^GVZ",
                     "tlt": "TLT",      "silver": "SI=F"}

    idx_utc   = price_index.tz_convert("UTC") if price_index.tz is not None \
        else price_index.tz_localize("UTC")
    yf_floor  = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=YF_H1_DAYS)
    px_start  = idx_utc[0].tz_localize(None)
    start = max(px_start, yf_floor).strftime("%Y-%m-%d")
    end   = (idx_utc[-1].tz_localize(None) + pd.Timedelta(hours=2)).strftime("%Y-%m-%d")
    if px_start < yf_floor:
        logger.warning("  Gold history starts %s but yfinance macro only reaches %s; "
                       "earlier bars use back-filled macro.",
                       px_start.date(), yf_floor.date())

    logger.info("Fetching H1 macro from yfinance (%s -> %s)...", start, end)
    macro = pd.DataFrame(index=idx_utc)
    for name, ticker in macro_tickers.items():
        series = pd.Series(dtype=float)
        for candidate in fallback_tickers(ticker):
            try:
                raw = yf.download(candidate, start=start, end=end, interval="1h",
                                  progress=False, auto_adjust=True)
                if raw.empty:
                    continue
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.squeeze()
                close.index = (close.index.tz_localize("UTC")
                               if close.index.tzinfo is None
                               else close.index.tz_convert("UTC"))
                series = close
                tag = "" if candidate == ticker else f" (fallback {candidate})"
                logger.info("  %-7s %d H1 bars%s", name + ":", len(close), tag)
                break
            except Exception as exc:
                logger.warning("  Failed to fetch %s (%s): %s", name, candidate, exc)
        if series.empty:
            logger.warning("  %-7s empty — using constant fill (100.0)", name + ":")
            macro[name] = 100.0
        else:
            macro[name] = series.reindex(idx_utc, method="ffill").bfill()

    macro = macro.ffill().bfill().fillna(100.0)
    macro.index = price_index   # restore the gold bar index exactly
    assert len(macro) == len(price_index), "macro/price length mismatch"
    assert not macro.isna().any().any(), "NaN remaining in macro after fill"
    return macro


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_splits():
    """train = recent ~12 months, val = ~3 weeks, test = most recent ~2-3 weeks."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return {
        "train_end": (today - timedelta(days=35)).strftime("%Y-%m-%d"),
        "val_end":   (today - timedelta(days=14)).strftime("%Y-%m-%d"),
    }


def main():
    ap = argparse.ArgumentParser(description="Retrain H1 gold encoder on recent MT5 data")
    ap.add_argument("--symbol",     default="XAUUSD")
    ap.add_argument("--n-bars",     type=int, default=8760,
                    help="H1 bars to pull from MT5 (8760 ~= 1 trading year incl. warm-up)")
    ap.add_argument("--train-end",  default=None, help="YYYY-MM-DD (default: today-35d)")
    ap.add_argument("--val-end",    default=None, help="YYYY-MM-DD (default: today-14d)")
    ap.add_argument("--epochs",     type=int, default=120)
    ap.add_argument("--lr",         type=float, default=None,
                    help="Override LR (default: 3e-5 finetune, 3e-4 scratch)")
    ap.add_argument("--lambda-dir", type=float, default=0.0,
                    help="Auxiliary directional loss weight (0 = off, matches prod best_model.pt)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--finetune", action="store_true",
                      help="Fine-tune from the production encoder (lower LR, preserves geometry)")
    mode.add_argument("--scratch",  action="store_true",
                      help="Train from random init")
    ap.add_argument("--finetune-from", default="./experiments/checkpoints/best_model.pt")
    ap.add_argument("--out",        default="./experiments/checkpoints/gold_mt5_h1",
                    help="Checkpoint dir; model saved under <out>/gold/best_model.pt")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Build data + splits and report shapes, then stop (no training)")
    args = ap.parse_args()

    run_log = _setup_logging()
    logger.info("Run log: %s", run_log)

    split_dates = _default_splits()
    if args.train_end: split_dates["train_end"] = args.train_end
    if args.val_end:   split_dates["val_end"]   = args.val_end

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1/3 — Fetch H1 %s from MT5", args.symbol)
    logger.info("=" * 60)
    price_df = fetch_gold_h1_mt5(args.symbol, args.n_bars)

    logger.info("=" * 60)
    logger.info("Step 2/3 — Fetch + align H1 macro")
    logger.info("=" * 60)
    macro_df = fetch_h1_macro(price_df.index)

    logger.info("=" * 60)
    logger.info("Step 3/3 — Feature engineering + split  (train<=%s, val<=%s)",
                split_dates["train_end"], split_dates["val_end"])
    logger.info("=" * 60)
    pipeline = DataPipeline(instrument="gold", lookback=LOOKBACK, norm_window=NORM_WINDOW)
    result = pipeline.build_from_frames(
        price_df, macro_df,
        stride       = 1,
        history_len  = HISTORY_LEN,
        split_dates  = split_dates,
    )

    if args.dry_run:
        logger.info("DRY RUN complete — train=%d val=%d test=%d samples. No training performed.",
                    len(result["train"]), len(result["val"]), len(result["test"]))
        return

    if len(result["val"]) == 0:
        logger.warning("Val split has 0 samples — checkpointing will fall back to train loss. "
                       "Widen the gap between --train-end and --val-end.")

    # ── Train (reuse training.train_o1's loop + checkpoint format) ───────────
    finetune = not args.scratch   # default to fine-tuning unless --scratch
    import training.train_o1 as t

    t.CFG.update({
        "instrument":     "gold",
        "lookback":       LOOKBACK,
        "history_len":    HISTORY_LEN,
        "norm_window":    NORM_WINDOW,
        "n_epochs":       args.epochs,
        "lambda_dir":     args.lambda_dir,
        "finetune_from":  args.finetune_from if finetune else None,
        "lr":             args.lr if args.lr is not None else (3e-5 if finetune else 3e-4),
        "checkpoint_dir": args.out,
        # Tee the training loop's per-epoch output into the same run log.
        "log_file":       str(run_log),
    })

    logger.info("Training: mode=%s  epochs=%d  lr=%.1e  lambda_dir=%.2f",
                "fine-tune" if finetune else "scratch",
                t.CFG["n_epochs"], t.CFG["lr"], t.CFG["lambda_dir"])
    logger.info("Output checkpoint: %s/gold/best_model.pt", args.out)

    t.train(result=result)

    logger.info("=" * 60)
    logger.info("Retrain complete. New encoder: %s/gold/best_model.pt", args.out)
    logger.info("NEXT: backtest this checkpoint before swapping it into run_trader.ps1.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
