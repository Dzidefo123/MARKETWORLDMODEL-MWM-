"""
scripts/replay_st_history.py — Reinterpret the May–June S_t spikes.
===================================================================

Marches the *actual* live S_t loop bar-by-bar over a historical window, holding
the encoder/predictor fixed, under two normalization regimes:

  OLD: per-bar fetch = _LOOKBACK + _WARM_UP_BARS (298) -> ~48 rows feed the
       500-bar rolling z-score (the bug).
  NEW: per-bar fetch = _Z_SCORE_WIN + _WARM_UP_BARS + _LOOKBACK (798) -> a full
       500-bar window (the fix).

For each bar it carries the live state forward (z_deque history + rolling
forecast z_hat) and computes S_t = ||z_hat - z_t||^2 exactly as live_trader does,
so the reconstructed S_t_old reproduces what the live system actually saw, and
S_t_new is what it would see after the fix. This gives a per-spike verdict
(genuine vs normalization artifact) and the S_t_new distribution for checking the
four-zone thresholds.

Usage:  python -m scripts.replay_st_history
        python -m scripts.replay_st_history --record-start 2026-05-26 --warmup-start 2026-05-20
Requires the MT5 terminal running. Uses the production encoder.
"""

import sys, argparse, json
sys.path.insert(0, ".")

import logging
from collections import deque
import numpy as np
import pandas as pd
import torch

from data.features import GoldFeatureEngineer
from execution.live_trader import (
    _load_system, _rolling_zscore, LiveTrader,
    _LOOKBACK, _Z_SCORE_WIN, _WARM_UP_BARS, _HISTORY_LEN,
)
from scripts.retrain_gold_h1_mt5 import fetch_gold_h1_mt5, fetch_h1_macro

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OLD_FETCH = _LOOKBACK + _WARM_UP_BARS                  # 298
NEW_FETCH = _Z_SCORE_WIN + _WARM_UP_BARS + _LOOKBACK   # 798

# Four-zone circuit-breaker floors (live_trader / backtest calibration)
_ZONE_FLOORS = {"p20": 0.0005, "p60": 0.003, "p90": 0.010}

# Named spikes from the paper-trading session (UTC bar timestamps, best-effort)
_NAMED_SPIKES = {
    "2026-05-27 21:00": "Iran strikes",
    "2026-06-02 22:00": "Jun 2 spike",
    "2026-06-05 01:00": "NFP",
    "2026-06-08 09:00": "Jun 8 morning",
    "2026-06-17 09:00": "Jun 17 morning",
}


def _engineer_window(encoder, ohlcv, macro, t_pos, n_fetch):
    """Replicate live per-bar feature engineering + encode for as-of row t_pos."""
    o = ohlcv.iloc[t_pos - n_fetch + 1 : t_pos + 1]
    m = macro.iloc[t_pos - n_fetch + 1 : t_pos + 1]
    eng = GoldFeatureEngineer(lookback_warm=_WARM_UP_BARS)
    raw_feat, ts, _ = eng.compute(o, m)
    if len(raw_feat) < _LOOKBACK:
        return None, None
    feat_df = pd.DataFrame(raw_feat, columns=GoldFeatureEngineer.FEATURE_NAMES, index=ts)
    feat = _rolling_zscore(feat_df, window=_Z_SCORE_WIN).values.astype(np.float32)
    window = feat[-_LOOKBACK:]
    with torch.no_grad():
        z = encoder(torch.tensor(window, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    a_t = LiveTrader._build_action_vecs(m.loc[ts])[-1]   # action at the current bar
    return z, torch.tensor(a_t, dtype=torch.float32)


def _march(encoder, predictor, ohlcv, macro, positions, n_fetch, label):
    """Run the live S_t loop over `positions` (row indices), return S_t per position."""
    z_deque = deque(maxlen=_HISTORY_LEN)
    z_hat   = None
    out = {}
    for k, t_pos in enumerate(positions):
        z_t, a_t = _engineer_window(encoder, ohlcv, macro, t_pos, n_fetch)
        if z_t is None:
            continue
        if z_hat is None:                       # seed on first bar
            for _ in range(_HISTORY_LEN):
                z_deque.append(z_t.clone())
        else:
            S_t = float(((z_hat - z_t) ** 2).mean())
            out[t_pos] = S_t
            z_deque.append(z_t)
        with torch.no_grad():
            z_hist = torch.stack(list(z_deque), dim=0).unsqueeze(0)
            z_hat  = predictor(z_hist, a_t.unsqueeze(0)).squeeze(0)
        if (k + 1) % 100 == 0:
            logger.info("  [%s] %d/%d bars", label, k + 1, len(positions))
    return out


def main():
    ap = argparse.ArgumentParser(description="Replay S_t under old vs new normalization")
    ap.add_argument("--checkpoint", default="./experiments/checkpoints/best_model.pt")
    ap.add_argument("--heads-dir",  default="./experiments/heads_gold")
    ap.add_argument("--warmup-start", default="2026-05-20", help="march starts here (warms z_deque)")
    ap.add_argument("--record-start", default="2026-05-26", help="record S_t from here")
    args = ap.parse_args()

    encoder, predictor, _ = _load_system(args.checkpoint, args.heads_dir, device="cpu")
    encoder.eval(); predictor.eval()

    # Pull deep enough that the earliest marched bar still has NEW_FETCH trailing bars.
    n_pull = 2600
    ohlcv = fetch_gold_h1_mt5("XAUUSD", n_pull)
    macro = fetch_h1_macro(ohlcv.index)

    idx = ohlcv.index
    warm = pd.Timestamp(args.warmup_start, tz=idx.tz)
    first_pos = max(int(np.searchsorted(idx, warm)), NEW_FETCH)  # need NEW_FETCH history
    positions = list(range(first_pos, len(idx)))
    logger.info("Marching %d bars: %s -> %s  (record from %s)",
                len(positions), idx[first_pos].date(), idx[-1].date(), args.record_start)

    st_old = _march(encoder, predictor, ohlcv, macro, positions, OLD_FETCH, "OLD")
    st_new = _march(encoder, predictor, ohlcv, macro, positions, NEW_FETCH, "NEW")

    rec_start = pd.Timestamp(args.record_start, tz=idx.tz)
    common = [p for p in st_old if p in st_new and idx[p] >= rec_start]
    ts_arr  = [idx[p] for p in common]
    old_arr = np.array([st_old[p] for p in common])
    new_arr = np.array([st_new[p] for p in common])

    # ── Distribution of S_t_new (for four-zone recalibration) ────────────────
    new_pcts = {q: float(np.percentile(new_arr, q)) for q in (20, 60, 90, 99)}
    old_pcts = {q: float(np.percentile(old_arr, q)) for q in (20, 60, 90, 99)}
    base_new = float(np.median(new_arr))

    print("\n" + "=" * 78)
    print(f"S_t REPLAY — old vs new normalization  ({ts_arr[0].date()} -> {ts_arr[-1].date()}, "
          f"{len(common)} bars)")
    print("=" * 78)

    print("\n  S_t distribution (per-bar surprise):")
    print(f"  {'regime':<6} {'mean':>9} {'p20':>9} {'p60':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  {'OLD':<6} {old_arr.mean():>9.4f} {old_pcts[20]:>9.4f} {old_pcts[60]:>9.4f} "
          f"{old_pcts[90]:>9.4f} {old_pcts[99]:>9.4f} {old_arr.max():>9.4f}")
    print(f"  {'NEW':<6} {new_arr.mean():>9.4f} {new_pcts[20]:>9.4f} {new_pcts[60]:>9.4f} "
          f"{new_pcts[90]:>9.4f} {new_pcts[99]:>9.4f} {new_arr.max():>9.4f}")

    # ── Per-spike verdict ────────────────────────────────────────────────────
    p90_new, p60_new = new_pcts[90], new_pcts[60]

    def _verdict(s_new):
        if s_new >= p90_new:           return "GENUINE  (still >p90 after fix)"
        if s_new <= p60_new:           return "ARTIFACT (collapses to baseline)"
        return "PARTIAL  (p60-p90 after fix)"

    ts_index = pd.DatetimeIndex(ts_arr)

    def _lookup(ts_str):
        t = pd.Timestamp(ts_str, tz=idx.tz)
        pos = int(np.argmin(np.abs(ts_index.asi8 - t.value)))
        return pos, ts_index[pos]

    print("\n  Named spikes (verdict: GENUINE if S_t_new still elevated, else ARTIFACT):")
    print(f"  {'bar (UTC)':<17} {'event':<14} {'S_t old':>8} {'S_t new':>8} {'x drop':>7}  verdict")
    print(f"  {'-'*17} {'-'*14} {'-'*8} {'-'*8} {'-'*7}  {'-'*32}")
    spike_rows = []
    for ts_str, name in _NAMED_SPIKES.items():
        pos, actual = _lookup(ts_str)
        o, nw = old_arr[pos], new_arr[pos]
        drop = o / nw if nw > 1e-9 else float("inf")
        v = _verdict(nw)
        spike_rows.append({"requested": ts_str, "matched": str(actual), "event": name,
                           "s_old": float(o), "s_new": float(nw), "verdict": v})
        print(f"  {str(actual)[:16]:<17} {name:<14} {o:>8.4f} {nw:>8.4f} {drop:>7.1f}  {v}")

    # Also surface the largest OLD spikes that the named list might miss
    top = np.argsort(old_arr)[::-1][:8]
    print("\n  Largest OLD-normalization spikes overall:")
    print(f"  {'bar (UTC)':<17} {'S_t old':>8} {'S_t new':>8} {'x drop':>7}  verdict")
    print(f"  {'-'*17} {'-'*8} {'-'*8} {'-'*7}  {'-'*32}")
    for pos in top:
        o, nw = old_arr[pos], new_arr[pos]
        drop = o / nw if nw > 1e-9 else float("inf")
        print(f"  {str(ts_arr[pos])[:16]:<17} {o:>8.4f} {nw:>8.4f} {drop:>7.1f}  {_verdict(nw)}")

    # ── Four-zone threshold check ────────────────────────────────────────────
    print("\n  Four-zone floors vs S_t_new percentiles (recalibration check):")
    for q, key in [(20, "p20"), (60, "p60"), (90, "p90")]:
        floor = _ZONE_FLOORS[key]
        emp   = new_pcts[q]
        flag  = "OK (floor binds)" if floor >= emp else "FLOOR TOO LOW -> raise"
        print(f"    {key}: floor={floor:.4f}  empirical S_t_new {key}={emp:.4f}   -> {flag}")

    out = {
        "window":       [str(ts_arr[0]), str(ts_arr[-1])],
        "n_bars":       len(common),
        "st_old":       {"mean": float(old_arr.mean()), **{f"p{q}": old_pcts[q] for q in old_pcts}, "max": float(old_arr.max())},
        "st_new":       {"mean": float(new_arr.mean()), **{f"p{q}": new_pcts[q] for q in new_pcts}, "max": float(new_arr.max())},
        "named_spikes": spike_rows,
        "series":       [{"ts": str(t), "s_old": float(o), "s_new": float(n)}
                         for t, o, n in zip(ts_arr, old_arr, new_arr)],
    }
    path = "./experiments/st_replay_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full per-bar series saved to {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
