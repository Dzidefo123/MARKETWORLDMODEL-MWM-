"""
scripts/js_mwm_attribution.py — Does the MWM S_t gate add value to the Jane Street system?
============================================================================================

The Tier-5 architecture hypothesis: the Jane Street Asian-range breakout supplies
direction/timing (validated edge, but regime-dependent), and the MWM encoder's S_t
"surprise" supplies a regime-quality gate that should skip breakouts occurring in
market states the world model finds unfamiliar.

This script TESTS that hypothesis instead of assuming it:
  1. Run JS on real XAUUSD H1 (validated params) -> per-trade records (entry_ts, net_pnl).
  2. Compute MWM S_t *rank* (causal rolling percentile, the live four-zone signal) on the
     same bars, using the production encoder/predictor.
  3. Join S_t-rank to each trade by entry bar.
  4. Compare JS-ALL vs JS gated at S_t-rank thresholds (skip Q5 / skip Q4+Q5 / Q1-2 only),
     overall and per regime window.

If gating improves risk-adjusted return → the architecture is real. If it only thins
trades, or hurts → it isn't. Trade counts are low, so read effect sizes, not noise.

Usage:  python -m scripts.js_mwm_attribution
"""

import sys, argparse
sys.path.insert(0, ".")
_JS_SRC = r"C:/Users/kalom/Downloads/janestreet/janestreet/src"
_JS_ROOT = r"C:/Users/kalom/Downloads/janestreet/janestreet"
sys.path.insert(0, _JS_SRC)

import logging
import numpy as np
import pandas as pd
import torch

from janestreet_mvp.config import load_config
from janestreet_mvp.data import load_candles
from janestreet_mvp.backtest import run_backtest as js_backtest

from data.pipeline import DataPipeline
from models.encoder import MarketEncoder
from models.predictor import CausalPredictor
from execution.heads import encode_split, compute_surprise_features
from scripts.retrain_gold_h1_mt5 import fetch_h1_macro

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LOOKBACK, HISTORY_LEN = 48, 3
START = "2024-08-01"   # JS warmup + within yfinance's 728-day macro window

# Regime windows from the multifold (anchored OOS folds) for per-period attribution
FOLDS = [
    ("F1 2024-11..2025-03", "2024-11-25", "2025-03-17"),
    ("F2 2025-03..2025-07", "2025-03-17", "2025-07-08"),
    ("F3 2025-07..2025-10", "2025-07-08", "2025-10-24"),
    ("F4 2025-10..2026-02", "2025-10-24", "2026-02-16"),
    ("F5 2026-02..2026-06", "2026-02-16", "2026-06-30"),
]


def _metrics(pnl: np.ndarray) -> dict:
    """Per-trade metrics for a set of trade net PnLs."""
    n = len(pnl)
    if n == 0:
        return {"n": 0, "total": 0.0, "win": 0.0, "avg": 0.0, "sharpe": 0.0, "pf": 0.0}
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gl = -losses.sum()
    return {
        "n": n,
        "total": float(pnl.sum()),
        "win": float((pnl > 0).mean()),
        "avg": float(pnl.mean()),
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(n)) if pnl.std() > 1e-9 else 0.0,
        "pf": float(wins.sum() / gl) if gl > 1e-9 else float("inf"),
    }


def _row(label, m):
    pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (f"  {label:<22} n={m['n']:<4} total={m['total']:>9.1f} win={m['win']:>5.1%} "
            f"avg={m['avg']:>7.2f} sharpe(pt)={m['sharpe']:>6.2f} pf={pf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./experiments/checkpoints/best_model.pt",
                    help="MWM encoder/predictor checkpoint for the S_t gate")
    args = ap.parse_args()

    # ── 1. JS trades ─────────────────────────────────────────────────────────
    cfg = load_config(f"{_JS_ROOT}/config/config.yaml")
    df = load_candles(f"{_JS_ROOT}/data/real_xauusd_h1.csv", start_utc=START)
    logger.info("JS data: %d H1 bars %s -> %s", len(df), df['ts'].iloc[0], df['ts'].iloc[-1])
    res = js_backtest(df, cfg)
    trades = pd.DataFrame(res.trade_records)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    logger.info("JS trades: %d  (validated Sharpe on this slice=%.2f, total_ret=%.2f%%)",
                len(trades), res.sharpe, res.total_return_pct)

    # ── 2. MWM S_t-rank on the same bars ─────────────────────────────────────
    price_df = df.set_index("ts")[["open", "high", "low", "close", "volume"]].copy()
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize("UTC")
    macro_df = fetch_h1_macro(price_df.index)

    pipe = DataPipeline(instrument="gold", lookback=LOOKBACK, norm_window=500)
    result = pipe.build_from_frames(price_df, macro_df, stride=1, history_len=HISTORY_LEN,
                                    split_dates=None)
    meta = result["meta"]
    feats = meta["norm_features"]
    acts = meta["macro_vecs"]
    ts = pd.DatetimeIndex(meta["timestamps"])

    logger.info("S_t gate encoder: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = MarketEncoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    pred = CausalPredictor(); pred.load_state_dict(ckpt["predictor"]); pred.eval()
    for p in list(enc.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)

    logger.info("Encoding %d bars + computing S_t-rank...", len(feats))
    Z = encode_split(enc, torch.tensor(feats, dtype=torch.float32), lookback=LOOKBACK)
    S = compute_surprise_features(Z, acts, pred, lookback=LOOKBACK, history_len=HISTORY_LEN)
    s_rank = S[:, 1]                                  # s_rank[i] = rank of surprise REALIZED at bar i+1
    z_ts = ts[LOOKBACK - 1 : LOOKBACK - 1 + len(Z)]   # Z[i] <-> bar ts[i+lookback-1]
    # Causal alignment (no lookahead): bar k's realized-surprise rank is s_rank[k-1]
    # (the surprise of the k-1 -> k transition, known when bar k closes). Mapping s_rank[i]
    # to z_ts[i] would assign bar i+1's surprise to a trade at bar i — a one-bar leak.
    srank_by_ts = pd.Series(s_rank[:-1], index=z_ts[1:])

    # ── 3. Join S_t-rank to each trade ───────────────────────────────────────
    trades["s_rank"] = trades["entry_ts"].map(srank_by_ts.to_dict())
    matched = trades.dropna(subset=["s_rank"]).copy()
    logger.info("Matched S_t to %d/%d trades (rest precede the S_t warmup window)",
                len(matched), len(trades))

    pnl_all = matched["net_pnl"].values

    # ── 4. Gate comparisons ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("JS × MWM S_t ATTRIBUTION  — does the regime gate add value?")
    print("=" * 78)
    print(f"\n  S_t-rank zones (causal percentile of surprise at entry):")
    print(f"    Q1<0.20  Q2 0.20-0.60  Q4 0.60-0.90  Q5>0.90")
    print(f"  Trades by zone: "
          f"Q1={int((matched.s_rank<0.2).sum())} "
          f"Q2={int(((matched.s_rank>=0.2)&(matched.s_rank<0.6)).sum())} "
          f"Q4={int(((matched.s_rank>=0.6)&(matched.s_rank<0.9)).sum())} "
          f"Q5={int((matched.s_rank>=0.9).sum())}")

    print(f"\n  OVERALL (all matched trades):")
    print(_row("JS-ALL (no gate)", _metrics(pnl_all)))
    for label, keep in [("gate: skip Q5 (<=0.9)", matched.s_rank <= 0.9),
                        ("gate: skip Q4+Q5(<=0.6)", matched.s_rank <= 0.6),
                        ("gate: Q1 only (<=0.2)", matched.s_rank <= 0.2)]:
        print(_row(label, _metrics(matched[keep]["net_pnl"].values)))

    # Win rate by zone — is high S_t actually where JS loses?
    print(f"\n  Win rate & avg PnL by S_t zone (the gate's premise):")
    for zlabel, lo, hi in [("Q1 (<0.2)", 0.0, 0.2), ("Q2 (0.2-0.6)", 0.2, 0.6),
                           ("Q4 (0.6-0.9)", 0.6, 0.9), ("Q5 (>=0.9)", 0.9, 1.01)]:
        sub = matched[(matched.s_rank >= lo) & (matched.s_rank < hi)]["net_pnl"].values
        m = _metrics(sub)
        print(f"    {zlabel:<14} n={m['n']:<3} win={m['win']:>5.1%} avg={m['avg']:>7.2f} total={m['total']:>8.1f}")

    # ── 5. Per-fold (does the gate rescue the bad regime?) ───────────────────
    print(f"\n  PER-REGIME (JS-ALL vs skip-Q4+Q5):")
    for name, a, b in FOLDS:
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        fold = matched[(matched.entry_ts >= a) & (matched.entry_ts < b)]
        m_all = _metrics(fold["net_pnl"].values)
        m_gate = _metrics(fold[fold.s_rank <= 0.6]["net_pnl"].values)
        print(f"    {name:<22} ALL: n={m_all['n']:<3} tot={m_all['total']:>7.1f} win={m_all['win']:>5.1%} "
              f"| GATE: n={m_gate['n']:<3} tot={m_gate['total']:>7.1f} win={m_gate['win']:>5.1%}")

    print("=" * 78)


if __name__ == "__main__":
    main()
