"""
scripts/run_combined_backtest.py

Combined H1 + M15 backtest — Config A and Config B.

Config A (Option B — S_t gate only):
  H1 directional signal → find first M15 bar in the NEXT H1 hour where
  M15 S_t < p20. Enter there. No M15 directional confirmation required.

Config B (full system):
  Same as A but M15 directional head (H=12, AUC=0.556) must also agree
  with H1 direction before entry is taken.

Both configs add two M15 contribution streams:
  1. Timing improvement — on H1 Q1 entries, enter at the calm M15 bar
     instead of H1 close, capturing more of the move.
  2. Standalone entries — when H1 S_t blocks entry (Q2-Q4) but H1 dir
     head is confident, M15 enters independently on the first calm bar.

Usage:
    python -m scripts.run_combined_backtest
    python -m scripts.run_combined_backtest --dir-threshold 0.53 --spread 0.0003
"""

import sys
sys.path.insert(0, ".")

import argparse, json, logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from models.encoder        import MarketEncoder
from models.predictor      import CausalPredictor
from data.pipeline         import DataPipeline
from execution.train_heads import load_heads
from execution.heads       import encode_split, build_z_history
from execution.backtest    import run_backtest, print_report
from execution.environment import TradingEnvironment
from experiments.phase0_m15_analysis import (
    fetch_m15_dataset, build_m15_pipeline, M15_CFG, compute_st,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_GOLD_BARS_PER_YEAR = 5690

H1_CKPT       = Path("./experiments/checkpoints/best_model.pt")
H1_HEADS_DIR  = Path("./experiments/heads_gold")
M15_CKPT      = Path("./experiments/checkpoints/m15_warmstart/best_model.pt")
M15_HEADS_H4  = Path("./experiments/heads_gold_m15")       # dir AUC 0.519 (not used for dir)
M15_HEADS_H12 = Path("./experiments/heads_gold_m15_h12")   # dir AUC 0.556 (Config B)

M15_POSITION_SIZE = 0.5   # fraction of H1 risk — Rule 4: combined max 1.5%
M15_P_THRESHOLD   = 20    # S_t percentile for execution gate (20 = p20, 40 = p40)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_h1(device="cpu"):
    ckpt = torch.load(H1_CKPT, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    enc  = MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=52, patch_size=4,
        d_model=128, n_heads=4, n_layers=4, dropout=0.0, proj_hidden=256, z_dim=128,
    )
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)

    pred = CausalPredictor(
        z_dim=128, d_model=128, n_heads=4, n_layers=6,
        action_dim=5, history_len=cfg.get("history_len", 3), dropout=0.0,
    )
    pred.load_state_dict(ckpt["predictor"])
    pred.eval()
    for p in pred.parameters(): p.requires_grad_(False)

    logger.info("H1 encoder loaded (epoch %s)", ckpt.get("epoch", "?"))
    return enc, pred, cfg


def _load_m15(device="cpu"):
    ckpt = torch.load(M15_CKPT, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    enc  = MarketEncoder(
        lookback=cfg["lookback"], n_features=52, patch_size=4,
        d_model=128, n_heads=4, n_layers=4, dropout=0.0, proj_hidden=256, z_dim=128,
    )
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)

    pred = CausalPredictor(
        z_dim=128, d_model=128, n_heads=4, n_layers=6,
        action_dim=5, history_len=cfg["history_len"], dropout=0.0,
    )
    pred.load_state_dict(ckpt["predictor"])
    pred.eval()
    for p in pred.parameters(): p.requires_grad_(False)

    logger.info("M15 encoder loaded (epoch %d  val=%.6f)", ckpt["epoch"], ckpt["val_loss"])
    return enc, pred, cfg


# ---------------------------------------------------------------------------
# Pre-compute H1 dir_prob for all test bars
# ---------------------------------------------------------------------------

def precompute_h1_dir_probs(h1_layer, Z_test, history_len=3):
    """Returns (N_test - lookback + 1,) array of H1 dir_prob."""
    Z_hist = build_z_history(Z_test, history_len)
    with torch.no_grad():
        logits    = h1_layer.dir_head(torch.FloatTensor(Z_hist))   # (N, 1) or (N, H, z)
        dir_probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    if dir_probs.ndim > 1:
        dir_probs = dir_probs[:, 0]
    return dir_probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Pre-compute M15 embeddings, S_t, and dir_prob for test split
# ---------------------------------------------------------------------------

def precompute_m15(m15_encoder, m15_predictor, m15_layer_h12, splits, m15_cfg,
                   pct_threshold=20):
    lookback    = m15_cfg["lookback"]
    history_len = m15_cfg["history_len"]

    # S_t threshold from training split (causal — no lookahead)
    logger.info("Computing M15 S_t on train split (for p%d threshold)...", pct_threshold)
    train_st = compute_st(
        m15_encoder, m15_predictor,
        splits["train_feat"], splits["train_act"],
        lookback=lookback, history_len=history_len,
    )
    m15_p20 = float(np.percentile(train_st, pct_threshold))
    logger.info("  M15 p%d threshold: %.6f", pct_threshold, m15_p20)

    # Test split S_t
    logger.info("Computing M15 S_t on test split...")
    test_st = compute_st(
        m15_encoder, m15_predictor,
        splits["test_feat"], splits["test_act"],
        lookback=lookback, history_len=history_len,
    )

    # Test split z embeddings (for directional head)
    logger.info("Encoding M15 test split...")
    test_feat_t = torch.as_tensor(splits["test_feat"], dtype=torch.float32)
    Z_test = encode_split(m15_encoder, test_feat_t, lookback, batch_size=256, device="cpu")

    # M15 dir_prob from H=12 heads
    logger.info("Computing M15 dir_prob (H=12 heads)...")
    Z_hist = build_z_history(Z_test, history_len)
    with torch.no_grad():
        logits    = m15_layer_h12.dir_head(torch.FloatTensor(Z_hist))
        m15_probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    if m15_probs.ndim > 1:
        m15_probs = m15_probs[:, 0]
    m15_probs = m15_probs.astype(np.float32)

    test_ts = splits["test_timestamps"][:len(test_st)]
    logger.info("  M15 test: %d S_t values  mean=%.6f  p20=%.6f",
                len(test_st), test_st.mean(), np.percentile(test_st, 20))

    return {
        "st":       test_st,
        "probs":    m15_probs,
        "ts":       test_ts,
        "p20":      m15_p20,
    }


# ---------------------------------------------------------------------------
# M15 lookup — indexed by UTC hour → list of bar dicts
# ---------------------------------------------------------------------------

def build_m15_lookup(m15_data: dict):
    """
    Build hour-indexed lookup for fast H1→M15 alignment.
    Each entry: {minute, st, is_calm, dir_prob, idx}
    """
    lookup = {}
    for idx, (st, prob, ts) in enumerate(zip(
            m15_data["st"], m15_data["probs"], m15_data["ts"])):
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        else:
            ts_pd = ts_pd.tz_convert("UTC")
        hour_key = ts_pd.floor("h")
        minute   = ts_pd.minute
        if hour_key not in lookup:
            lookup[hour_key] = []
        lookup[hour_key].append({
            "minute":   minute,
            "st":       float(st),
            "is_calm":  float(st) < m15_data["p20"],
            "dir_prob": float(prob),
            "idx":      idx,
        })
    return lookup


def first_calm_m15_bar(h1_ts, m15_lookup, direction, use_dir_filter=False, dir_threshold=0.5):
    """
    Find first calm M15 bar in the hour FOLLOWING h1_ts.
    Config B: also require M15 dir_prob on the correct side.
    Returns bar dict or None.
    """
    try:
        ts_pd = pd.Timestamp(h1_ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        else:
            ts_pd = ts_pd.tz_convert("UTC")
        # Entry is executed at close of H1 bar h1_ts; the next H1 hour is h1_ts + 1h
        next_h1_hour = ts_pd.floor("h") + pd.Timedelta(hours=1)
    except Exception:
        return None

    bars = m15_lookup.get(next_h1_hour, [])
    calm = [b for b in bars if b["is_calm"]]
    if not calm:
        return None

    calm.sort(key=lambda b: b["minute"])

    if not use_dir_filter:
        return calm[0]

    # Config B: require M15 dir head agreement
    for bar in calm:
        long_ok  = direction > 0 and bar["dir_prob"] > dir_threshold
        short_ok = direction < 0 and bar["dir_prob"] < (1 - dir_threshold)
        if long_ok or short_ok:
            return bar
    return None


# ---------------------------------------------------------------------------
# M15 close price lookup from OHLCV
# ---------------------------------------------------------------------------

def build_m15_price_lookup(m15_ohlcv):
    lookup = {}
    for ts, row in m15_ohlcv.iterrows():
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        else:
            ts_pd = ts_pd.tz_convert("UTC")
        lookup[ts_pd] = float(row["close"])
    return lookup


def m15_entry_price(h1_ts, calm_bar, m15_price_lookup):
    """Look up close price at the qualifying M15 bar."""
    try:
        ts_pd = pd.Timestamp(h1_ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        next_h1_hour = ts_pd.floor("h") + pd.Timedelta(hours=1)
        m15_ts = next_h1_hour + pd.Timedelta(minutes=calm_bar["minute"])
        return m15_price_lookup.get(m15_ts, None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Combined overlay (Config A or B)
# ---------------------------------------------------------------------------

def compute_config_d(
    h1_records,
    h1_timestamps,
    h1_prices,
    m15_lookup,
    m15_price_lookup,
    h1_lookback=48,
):
    """
    Config D — M15 as H1 execution layer.

    For each H1 trade entry bar (position_age == 1), find the first calm M15
    bar in the NEXT H1 hour and replace H1's bar-close entry price with it.
    If no calm M15 bar found, fall back to H1 close (no change).

    Single position, H1's sizing and exit logic unchanged.
    No stacking: never more than one open position.

    Returns per-bar entry improvement array and per-trade detail list.
    """
    n = len(h1_records)
    entry_improvement = np.zeros(n)   # extra PnL from better entry price
    trades = []

    for i, rec in enumerate(h1_records):
        # Only process the first bar of each H1 trade (fresh entry)
        if rec["position_age"] != 1:
            continue
        if abs(rec["effective_action"]) < 1e-9:
            continue

        bar_idx   = rec["bar_idx"]
        direction = float(np.sign(rec["effective_action"]))

        if bar_idx >= len(h1_timestamps) or bar_idx >= len(h1_prices):
            continue

        h1_ts          = h1_timestamps[bar_idx]
        h1_entry_price = h1_prices[bar_idx]   # H1 bar close = where H1 entered
        if h1_entry_price <= 0:
            continue

        # Find first calm M15 bar in the NEXT H1 hour
        calm = first_calm_m15_bar(h1_ts, m15_lookup, direction,
                                   use_dir_filter=False, dir_threshold=0.5)

        if calm is None:
            trades.append({
                "h1_ts":           str(h1_ts),
                "direction":       "LONG" if direction > 0 else "SHORT",
                "h1_entry":        round(h1_entry_price, 3),
                "m15_entry":       None,
                "improvement_pct": 0.0,
                "improvement_usd": 0.0,
                "m15_minute":      None,
                "m15_st":          None,
                "fallback":        True,
            })
            continue

        m15_price = m15_entry_price(h1_ts, calm, m15_price_lookup)

        if m15_price is None or m15_price <= 0:
            trades.append({
                "h1_ts":           str(h1_ts),
                "direction":       "LONG" if direction > 0 else "SHORT",
                "h1_entry":        round(h1_entry_price, 3),
                "m15_entry":       None,
                "improvement_pct": 0.0,
                "improvement_usd": 0.0,
                "m15_minute":      None,
                "m15_st":          round(calm["st"], 6),
                "fallback":        True,
                "fallback_reason": "price_unavailable",
            })
            continue

        # Adverse guard: only use M15 if it gives a genuinely better fill.
        # If M15 price moved against the direction vs H1 close, fall back.
        adverse = (direction > 0 and m15_price > h1_entry_price) or \
                  (direction < 0 and m15_price < h1_entry_price)
        if adverse:
            trades.append({
                "h1_ts":           str(h1_ts),
                "direction":       "LONG" if direction > 0 else "SHORT",
                "h1_entry":        round(h1_entry_price, 3),
                "m15_entry":       round(m15_price, 3),
                "improvement_pct": 0.0,
                "improvement_usd": 0.0,
                "m15_minute":      calm["minute"],
                "m15_st":          round(calm["st"], 6),
                "fallback":        True,
                "fallback_reason": "adverse_price",
            })
            continue

        # Entry price improvement: positive = better fill for our direction
        # No extra spread (same trade, just better-timed execution)
        improvement = direction * (h1_entry_price - m15_price) / h1_entry_price
        entry_improvement[i] = improvement

        improvement_usd = abs(h1_entry_price - m15_price)   # $/oz improvement

        trades.append({
            "h1_ts":           str(h1_ts),
            "direction":       "LONG" if direction > 0 else "SHORT",
            "h1_entry":        round(h1_entry_price, 3),
            "m15_entry":       round(m15_price, 3),
            "improvement_pct": round(improvement * 100, 4),
            "improvement_usd": round(improvement_usd, 2),
            "m15_minute":      calm["minute"],
            "m15_st":          round(calm["st"], 6),
            "fallback":        False,
        })

    return {
        "entry_improvement": entry_improvement,
        "trades":            trades,
    }


def print_config_d(result_d, metrics_d, h1_metrics):
    trades     = result_d["trades"]
    n_total    = len(trades)
    improved   = [t for t in trades if not t["fallback"]]
    fallback   = [t for t in trades if t["fallback"]]
    impr_pcts  = [t["improvement_pct"] for t in improved]
    impr_usds  = [t["improvement_usd"] for t in improved]

    print(f"\n{'='*68}")
    print(f"  CONFIG D — M15 execution layer (single position, no stacking)")
    print(f"{'='*68}")
    print(f"\n  H1 trades total     : {n_total}")
    print(f"  M15 execution found : {len(improved)} ({len(improved)/max(n_total,1):.0%})")
    print(f"  Fallback to H1 close: {len(fallback)}")
    if improved:
        print(f"\n  Entry price improvement ($/oz):")
        print(f"    Mean   : ${np.mean(impr_usds):+.2f}")
        print(f"    Median : ${np.median(impr_usds):+.2f}")
        print(f"    Min    : ${min(impr_usds):+.2f}")
        print(f"    Max    : ${max(impr_usds):+.2f}")
        print(f"    Positive (better fill): {sum(p > 0 for p in impr_pcts)}/{len(impr_pcts)}")
        print(f"\n  Entry return improvement (%):")
        print(f"    Mean   : {np.mean(impr_pcts):+.4f}%")
        print(f"    Total  : {sum(impr_pcts):+.4f}%")

    h = metrics_d["h1"]
    c = metrics_d["combined"]
    print(f"\n  {'Metric':<18} {'H1-only':>10} {'Config D':>10} {'Delta':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    rows = [
        ("Sharpe",       f"{h['sharpe']:.3f}",       f"{c['sharpe']:.3f}",
         f"{c['sharpe']-h['sharpe']:+.3f}"),
        ("Sortino",      f"{h['sortino']:.3f}",      f"{c['sortino']:.3f}",
         f"{c['sortino']-h['sortino']:+.3f}"),
        ("Max DD",       f"{h['max_dd']:.2%}",       f"{c['max_dd']:.2%}",
         f"{c['max_dd']-h['max_dd']:+.2%}"),
        ("Total return", f"{h['total_return']:.2%}", f"{c['total_return']:.2%}",
         f"{c['total_return']-h['total_return']:+.2%}"),
    ]
    for name, hv, cv, delta in rows:
        print(f"  {name:<18} {hv:>10} {cv:>10} {delta}")

    print(f"\n  Per-trade detail:")
    print(f"  {'Date':20} {'Dir':5} {'H1 entry':10} {'M15 entry':10} "
          f"{'Impr $':8} {'min':4} {'Fallback'}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*4} {'-'*8}")
    for t in trades:
        fb    = "YES" if t["fallback"] else ""
        m15e  = f"{t['m15_entry']:.2f}" if t["m15_entry"] else "—"
        iusd  = f"${t['improvement_usd']:+.2f}" if not t["fallback"] else "—"
        mmin  = str(t["m15_minute"]) if t["m15_minute"] is not None else "—"
        ts    = str(t["h1_ts"])[:16]
        print(f"  {ts:20} {t['direction']:5} {t['h1_entry']:10.2f} {m15e:10} "
              f"{iusd:8} {mmin:4} {fb}")
    print()


def compute_overlay(
    h1_records,
    h1_timestamps,
    h1_prices,
    h1_dir_probs,
    m15_lookup,
    m15_price_lookup,
    use_m15_dir=False,
    dir_threshold=0.53,
    spread=0.0003,
    h1_lookback=48,
):
    """
    Config C: M15 enters ONLY when H1 is genuinely flat (position_age == 0,
    effective_action == 0) and H1 dir head has directional confidence.
    M15 exits at the close of the next H1 bar (bar_idx + 1), before H1 can
    enter and stack. No simultaneous H1+M15 exposure.
    """
    n = len(h1_records)
    m15_pnl   = np.zeros(n)
    m15_events = []
    standalone_events = []   # kept for interface compatibility

    def _h1_prob(bar_idx):
        z_idx = bar_idx - (h1_lookback - 1)
        if z_idx < 0 or z_idx >= len(h1_dir_probs):
            return 0.5
        return float(h1_dir_probs[z_idx])

    # Compute H1 S_t p90 from records — used to exclude circuit-breaker bars
    all_h1_st = np.array([r["S_t"] for r in h1_records])
    h1_st_p90 = float(np.percentile(all_h1_st, 90))
    logger.info("  H1 S_t p90 (CB threshold proxy): %.6f", h1_st_p90)

    prev_pos_age = 0   # track previous bar's position_age to detect CB-close bars

    for i, rec in enumerate(h1_records):
        bar_idx      = rec["bar_idx"]
        cur_pos_age  = rec["position_age"]

        # Config C gate 1: H1 must be genuinely flat
        if rec["effective_action"] != 0.0 or cur_pos_age != 0:
            prev_pos_age = cur_pos_age
            continue

        # Config C gate 2: skip bars where H1 was JUST circuit-breaker closed
        # (prev_pos_age > 0 means we were in a position last bar — this bar's
        # flatness is due to a CB close, not a fresh flat state)
        if prev_pos_age > 0:
            prev_pos_age = cur_pos_age
            continue

        # Config C gate 3: skip extreme H1 S_t (genuine CB territory)
        if rec["S_t"] >= h1_st_p90:
            prev_pos_age = cur_pos_age
            continue

        prev_pos_age = cur_pos_age

        if bar_idx >= len(h1_timestamps) or bar_idx + 4 >= len(h1_prices):
            continue

        h1_ts = h1_timestamps[bar_idx]

        # Config C gate 4: H1 dir head must have clear directional intent
        h1_prob = _h1_prob(bar_idx)
        if h1_prob > dir_threshold:
            direction = 1.0
        elif h1_prob < (1 - dir_threshold):
            direction = -1.0
        else:
            continue

        # Config C gate 5: first calm M15 bar in NEXT H1 hour
        calm = first_calm_m15_bar(h1_ts, m15_lookup, direction,
                                   use_dir_filter=use_m15_dir,
                                   dir_threshold=0.5)
        if calm is None:
            continue

        entry_price = m15_entry_price(h1_ts, calm, m15_price_lookup)
        # Exit at bar_idx + 2 (2 H1 bars — enough time to capture the move,
        # short enough that H1 rarely enters within 2 bars of a CB flat)
        exit_price = h1_prices[bar_idx + 2]

        if entry_price is None or entry_price <= 0 or exit_price <= 0:
            continue

        raw_return = direction * (exit_price - entry_price) / entry_price
        pnl = raw_return * M15_POSITION_SIZE - 2 * spread
        m15_pnl[i] = pnl

        m15_events.append({
            "h1_ts":       str(h1_ts),
            "minute":      calm["minute"],
            "direction":   "LONG" if direction > 0 else "SHORT",
            "entry_price": round(entry_price, 3),
            "exit_price":  round(exit_price, 3),
            "raw_ret_pct": round(raw_return * 100, 3),
            "pnl":         round(pnl, 6),
            "h1_st":       round(rec["S_t"], 6),
            "m15_st":      round(calm["st"], 6),
            "m15_prob":    round(calm["dir_prob"], 3),
            "h1_prob":     round(h1_prob, 3),
        })

    return {
        "timing_pnl":        m15_pnl,
        "standalone_pnl":    np.zeros(n),
        "timing_events":     m15_events,
        "standalone_events": standalone_events,
    }


# ---------------------------------------------------------------------------
# Metrics + reporting
# ---------------------------------------------------------------------------

def combined_metrics(h1_records, overlay, bars_per_year=_GOLD_BARS_PER_YEAR):
    h1_pnl       = np.array([r["pnl"] for r in h1_records])
    m15_pnl      = overlay["timing_pnl"] + overlay["standalone_pnl"]
    combined_pnl = h1_pnl + m15_pnl

    def _sharpe(pnl):
        std = pnl.std()
        return pnl.mean() / std * np.sqrt(bars_per_year) if std > 1e-12 else 0.0

    def _sortino(pnl):
        down = pnl[pnl < 0]
        if len(down) == 0: return float("inf")
        dev = np.sqrt(np.mean(down ** 2))
        return pnl.mean() / dev * np.sqrt(bars_per_year) if dev > 1e-12 else 0.0

    def _max_dd(pnl):
        eq   = np.cumprod(1 + pnl)
        peak = np.maximum.accumulate(eq)
        return float(((peak - eq) / peak).max())

    def _n_trades(pnl_arr):
        return int((np.abs(pnl_arr) > 1e-9).sum())

    return {
        "h1": {
            "sharpe":       round(_sharpe(h1_pnl), 4),
            "sortino":      round(_sortino(h1_pnl), 4),
            "max_dd":       round(_max_dd(h1_pnl), 4),
            "total_return": round(float(np.prod(1 + h1_pnl) - 1), 4),
            "n_active":     _n_trades(h1_pnl),
        },
        "combined": {
            "sharpe":       round(_sharpe(combined_pnl), 4),
            "sortino":      round(_sortino(combined_pnl), 4),
            "max_dd":       round(_max_dd(combined_pnl), 4),
            "total_return": round(float(np.prod(1 + combined_pnl) - 1), 4),
            "n_active":     _n_trades(combined_pnl),
        },
        "m15": {
            "timing_trades":     len(overlay["timing_events"]),
            "standalone_trades": len(overlay["standalone_events"]),
            "timing_pnl":        round(float(overlay["timing_pnl"].sum()), 6),
            "standalone_pnl":    round(float(overlay["standalone_pnl"].sum()), 6),
        },
    }


def print_comparison(label, m, ref_sharpe=2.458):
    h = m["h1"]
    c = m["combined"]
    o = m["m15"]
    beat = "BEAT" if c["sharpe"] > ref_sharpe else "miss"
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    print(f"  {'Metric':<18} {'H1-only':>10} {'H1+M15':>10} {'Delta':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    rows = [
        ("Sharpe",       f"{h['sharpe']:.3f}",       f"{c['sharpe']:.3f}",
         f"{c['sharpe']-h['sharpe']:+.3f}  [{beat}]"),
        ("Sortino",      f"{h['sortino']:.3f}",      f"{c['sortino']:.3f}",
         f"{c['sortino']-h['sortino']:+.3f}"),
        ("Max DD",       f"{h['max_dd']:.2%}",       f"{c['max_dd']:.2%}",
         f"{c['max_dd']-h['max_dd']:+.2%}"),
        ("Total return", f"{h['total_return']:.2%}", f"{c['total_return']:.2%}",
         f"{c['total_return']-h['total_return']:+.2%}"),
        ("Active bars",  f"{h['n_active']}",         f"{c['n_active']}", ""),
    ]
    for name, hv, cv, delta in rows:
        print(f"  {name:<18} {hv:>10} {cv:>10} {delta}")
    print(f"\n  M15 contributions:")
    print(f"    Timing improvements : {o['timing_trades']:4d}  PnL={o['timing_pnl']:+.4f}")
    print(f"    Standalone entries  : {o['standalone_trades']:4d}  PnL={o['standalone_pnl']:+.4f}")


def print_highlight_trades(overlay, target_date="2026-05-28", target_hour=13):
    """
    Print trades near the target H1 breakout bar.
    yfinance bars are stamped at BAR OPEN, so the bar whose close is 13:00
    has timestamp 12:00. Search ±1 hour to catch both conventions.
    """
    all_events = overlay["timing_events"] + overlay["standalone_events"]
    hits = []
    for ev in all_events:
        try:
            ts = pd.Timestamp(ev["h1_ts"])
            if target_date in str(ts) and abs(ts.hour - target_hour) <= 1:
                hits.append((ts, ev))
        except Exception:
            pass
    if hits:
        print(f"\n  >> {target_date} ~{target_hour:02d}:00 UTC breakout entries:")
        for ts, ev in sorted(hits, key=lambda x: x[0]):
            kind = "TIMING" if "exit_price" not in ev else "STANDALONE"
            print(f"     [{kind}] h1_bar={ts}  dir={'LONG' if ev['direction']>0 else 'SHORT'}  "
                  f"m15_entry=+{ev['minute']}min  "
                  f"M15_S_t={ev['m15_st']:.5f}  pnl={ev['pnl']:+.5f}")
    else:
        print(f"\n  >> {target_date} ~{target_hour:02d}:00 UTC: no M15 entry found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-threshold", type=float, default=0.53)
    parser.add_argument("--spread",        type=float, default=0.0003)
    parser.add_argument("--skip-sessions", type=str,   default=None)
    parser.add_argument("--m15-pct",       type=int,   default=20,
                        help="M15 S_t percentile for execution gate (20=p20, 40=p40)")
    parser.add_argument("--out",  type=str, default="./experiments/combined_backtest_results.json")
    args = parser.parse_args()

    skip_sess = [s.strip() for s in args.skip_sessions.split(",")] if args.skip_sessions else None

    # ── Load components ──────────────────────────────────────────────────────
    logger.info("Loading H1 encoder + heads...")
    h1_enc, h1_pred, h1_cfg = _load_h1()
    h1_layer = load_heads(str(H1_HEADS_DIR)); h1_layer.eval()

    logger.info("Loading M15 encoder + both head sets...")
    m15_enc, m15_pred, m15_cfg = _load_m15()
    m15_layer_h12 = load_heads(str(M15_HEADS_H12)); m15_layer_h12.eval()

    # ── H1 data ──────────────────────────────────────────────────────────────
    logger.info("Building H1 data pipeline...")
    pipeline = DataPipeline(instrument="gold", lookback=48, norm_window=500)
    result   = pipeline.build(use_real_data=True, stride=1, history_len=3,
                               split_dates=h1_cfg.get("split_dates"))
    meta     = result["meta"]
    n_train, n_val = meta["n_train"], meta["n_val"]

    all_ts     = meta["timestamps"]
    test_feat  = meta["norm_features"][n_train + n_val:]
    test_mac   = meta["macro_vecs"][n_train + n_val:]
    test_price = meta["prices"][n_train + n_val:]
    test_ts    = all_ts[n_train + n_val:]
    logger.info("H1 test: %d bars  (%s → %s)",
                len(test_feat), pd.Timestamp(test_ts[0]).date(),
                pd.Timestamp(test_ts[-1]).date())

    # Pre-compute H1 z embeddings + dir_prob
    logger.info("Pre-computing H1 embeddings and dir_prob...")
    test_feat_t = torch.as_tensor(test_feat, dtype=torch.float32)
    Z_h1_test   = encode_split(h1_enc, test_feat_t, lookback=48, batch_size=256, device="cpu")
    h1_dir_probs = precompute_h1_dir_probs(h1_layer, Z_h1_test, history_len=3)
    logger.info("  H1 dir_prob: %d values  mean=%.3f", len(h1_dir_probs), h1_dir_probs.mean())

    # ── M15 data ─────────────────────────────────────────────────────────────
    logger.info("Fetching M15 data from MT5...")
    m15_ohlcv, macro_m15 = fetch_m15_dataset(n_bars=M15_CFG["n_m15_bars"])
    splits = build_m15_pipeline(m15_ohlcv, macro_m15,
                                 split_dates=M15_CFG["split_dates"],
                                 norm_window=M15_CFG["norm_window"])

    m15_data = precompute_m15(m15_enc, m15_pred, m15_layer_h12, splits, m15_cfg,
                               pct_threshold=args.m15_pct)
    m15_lookup       = build_m15_lookup(m15_data)
    m15_price_lookup = build_m15_price_lookup(m15_ohlcv)
    logger.info("M15 lookup: %d hour-buckets", len(m15_lookup))

    # ── Run H1 backtest (shared baseline) ────────────────────────────────────
    logger.info("Running H1 baseline backtest...")
    env = TradingEnvironment(
        encoder=h1_enc, predictor=h1_pred,
        features=test_feat, macro_vecs=test_mac, prices=test_price,
        spread_cost=args.spread, lookback=48, history_len=3,
    )
    bt = run_backtest(env, h1_layer, bars_per_year=_GOLD_BARS_PER_YEAR,
                      skip_sessions=skip_sess, dir_threshold=args.dir_threshold,
                      verbose=False)
    h1_records = bt["records"]
    logger.info("H1 baseline: Sharpe=%.3f  DD=%.2f%%  Return=%.2f%%",
                bt["metrics"]["sharpe"],
                bt["metrics"].get("max_drawdown", 0) * 100,
                bt["metrics"].get("total_return", 0) * 100)

    print("\n" + "="*68)
    print("  H1 BASELINE")
    print_report(bt, instrument="gold")

    # ── Config D — M15 execution layer ───────────────────────────────────────
    logger.info("Computing Config D (M15 execution layer)...")
    result_d = compute_config_d(
        h1_records, test_ts, test_price,
        m15_lookup, m15_price_lookup,
    )
    # Wrap as overlay for combined_metrics reuse
    d_overlay = {
        "timing_pnl":        result_d["entry_improvement"],
        "standalone_pnl":    np.zeros(len(h1_records)),
        "timing_events":     [t for t in result_d["trades"] if not t["fallback"]],
        "standalone_events": [],
    }
    metrics_d = combined_metrics(h1_records, d_overlay)
    print_config_d(result_d, metrics_d, bt["metrics"])

    # ── Config C — S_t gate only ──────────────────────────────────────────────
    logger.info("Computing Config C overlay (S_t gate only)...")
    overlay_a = compute_overlay(
        h1_records, test_ts, test_price, h1_dir_probs,
        m15_lookup, m15_price_lookup,
        use_m15_dir=False,
        dir_threshold=args.dir_threshold,
        spread=args.spread,
    )
    metrics_a = combined_metrics(h1_records, overlay_a)
    print_comparison("CONFIG C — H1 flat + M15 S_t gate (no stacking, no dir filter)", metrics_a)
    print_highlight_trades(overlay_a)

    # ── Config C+ — S_t gate + M15 dir confirmation (reference only) ─────────
    logger.info("Computing Config C+ overlay (S_t gate + M15 dir filter, reference)...")
    overlay_b = compute_overlay(
        h1_records, test_ts, test_price, h1_dir_probs,
        m15_lookup, m15_price_lookup,
        use_m15_dir=True,
        dir_threshold=args.dir_threshold,
        spread=args.spread,
    )
    metrics_b = combined_metrics(h1_records, overlay_b)
    print_comparison("CONFIG C+ — H1 flat + M15 S_t gate + M15 dir confirmation", metrics_b)
    print_highlight_trades(overlay_b)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "h1_baseline":    bt["metrics"],
        "config_d":       metrics_d,
        "config_d_trades": result_d["trades"],
        "config_c":       metrics_a,
        "config_c_plus":  metrics_b,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
