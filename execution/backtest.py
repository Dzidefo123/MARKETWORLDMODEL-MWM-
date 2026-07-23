"""
execution/backtest.py — Backtesting loop integrating TradingEnvironment with
                        SupervisedExecutionLayer.

SIGNAL FLOW
-----------
  SupervisedExecutionLayer.entry_signal(z_t)
      ↓  filters 1-3 (direction, magnitude, vol regime)
  trade_action  ∈ {-1.0, -0.5, 0.0, 0.5, 1.0}
      ↓
  TradingEnvironment.step(trade_action)
      ↓  filter 4 (S_t circuit breaker)
  effective_action  — what was actually executed
      ↓
  PnL = effective_action × bar_return − |Δposition| × spread_cost

METRICS (Phase 3 targets from the build plan)
----------------------------------------------
  Sharpe ratio        > 1.0  annualised
  Max drawdown        < 15%  (gold) / < 10% (FX)
  Sortino ratio       > 1.5  annualised
  Profit factor       > 1.3  (win_rate × avg_win / avg_loss combined)
  Trade entries       > 200  for statistical reliability
  Session breakdown   London / NY / Asian

SESSION CLASSIFICATION
----------------------
Uses is_london (feature 38) and is_ny (feature 39) from the normalised feature
matrix.  Because these are binary features z-scored with a 500-bar rolling
window, the sign of the normalised value reliably indicates the original 0/1:
    positive → feature was 1 (session active)
    negative → feature was 0 (session inactive)
"""

import numpy as np
import torch
from collections import deque
from typing import Optional

from execution.heads import SESSION_ASIAN, SESSION_LONDON, SESSION_NY

# Feature index constants — must match GoldFeatureEngineer.FEATURE_NAMES
_IS_LONDON_IDX = 38
_IS_NY_IDX     = 39

_SESSION_ID = {"Asian": SESSION_ASIAN, "London": SESSION_LONDON, "NY": SESSION_NY}


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

def _session_at(features: torch.Tensor, bar_idx: int) -> str:
    """Return 'London', 'NY', or 'Asian' for a given bar."""
    is_london = features[bar_idx, _IS_LONDON_IDX].item() > 0
    is_ny     = features[bar_idx, _IS_NY_IDX].item() > 0
    if is_ny:
        return "NY"
    if is_london:
        return "London"
    return "Asian"


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    env,
    execution_layer,
    bars_per_year:    int   = 4320,
    max_holding_bars: Optional[int] = None,
    skip_sessions:    Optional[list] = None,
    dir_threshold:    float = 0.53,
    verbose:          bool  = True,
) -> dict:
    """
    Run a full backtest of the supervised execution layer on the environment.

    Args:
        env:               TradingEnvironment (already constructed for the test split).
        execution_layer:   Trained SupervisedExecutionLayer (eval mode, frozen).
        bars_per_year:     Used for Sharpe/Sortino annualisation.
                           Gold H1 ≈ 5938, FX H1 ≈ 4320.
        max_holding_bars:  Force-close a position after this many bars, regardless
                           of the direction signal. None = no limit.
                           Set to the label horizon (e.g. 4) to prevent the
                           directional head from riding a one-sided bias indefinitely.
        skip_sessions:     List of session names to force-flat during, e.g. ['NY'].
                           Useful when a session has consistently negative alpha.
        verbose:           Print progress every 500 bars.

    Returns dict with:
        records:         List of per-bar dicts.
        equity_curve:    np.ndarray, shape (n_steps + 1,), starts at 1.0.
        metrics:         Performance metrics dict.
        session_metrics: Per-session metrics dict.
    """
    execution_layer.eval()

    z    = env.reset()
    done = False

    # Build z-history deque; history_len defaults to 1 for non-history models.
    history_len  = getattr(execution_layer, "history_len", 1)
    z_deque      = deque([z.clone()] * history_len, maxlen=history_len)

    equity_curve     = [1.0]
    records          = []
    step_count       = 0
    position_age     = 0       # bars held in current non-flat position
    current_pos_sign = 0.0     # sign of the current position (+1, -1, or 0)

    while not done:
        decision_bar = env.bar_idx   # bar where the decision is made

        session = _session_at(env.features, decision_bar)

        session_id = _SESSION_ID[session]

        # (1, H, z_dim) history window for the directional head
        z_hist = torch.stack(list(z_deque), dim=0).unsqueeze(0)

        # Force flat when max holding period exceeded or session is filtered
        if max_holding_bars is not None and position_age >= max_holding_bars:
            signal = 0.0
        elif skip_sessions is not None and session in skip_sessions:
            signal = 0.0
        else:
            with torch.no_grad():
                signal = execution_layer.entry_signal(
                    z.unsqueeze(0), z_hist=z_hist,
                    session_id=session_id,
                    spread=env.spread_cost, dir_threshold=dir_threshold,
                ).item()

        z, reward, done, info = env.step(signal)
        z_deque.append(z.clone())
        step_count += 1

        # Track position age from effective_action (accounts for circuit-breaker flats)
        eff = info["effective_action"]
        if abs(eff) < 1e-9:
            position_age     = 0
            current_pos_sign = 0.0
        elif np.sign(eff) != current_pos_sign:   # new entry or direction reversal
            position_age     = 1
            current_pos_sign = float(np.sign(eff))
        else:
            position_age += 1

        records.append({
            "bar_idx":          decision_bar,
            "signal":           signal,
            "effective_action": eff,
            "pnl":              info["pnl"],
            "equity":           info["equity"],
            "S_t":              info["S_t"],
            "session":          session,
            "position_age":     position_age,
        })
        equity_curve.append(info["equity"])

        if verbose and step_count % 500 == 0:
            print(f"  bar {step_count:>5}  equity={info['equity']:.4f}  "
                  f"S_t={info['S_t']:.4f}")

    equity_curve = np.array(equity_curve)
    metrics      = compute_metrics(records, equity_curve, bars_per_year)
    sess_metrics = _session_breakdown(records, bars_per_year)

    return {
        "records":         records,
        "equity_curve":    equity_curve,
        "metrics":         metrics,
        "session_metrics": sess_metrics,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    records:       list,
    equity_curve:  np.ndarray,
    bars_per_year: int = 4320,
) -> dict:
    """
    Compute all Phase 3 performance metrics.

    Args:
        records:      Per-bar list from run_backtest().
        equity_curve: np.ndarray starting at 1.0.
        bars_per_year: H1 bars in a calendar year for annualisation.
    """
    pnl       = np.array([r["pnl"] for r in records])
    eff_act   = np.array([r["effective_action"] for r in records])
    signal    = np.array([r["signal"] for r in records])
    S_t_vals  = np.array([r["S_t"] for r in records])

    active_mask = np.abs(eff_act) > 1e-9      # bars with a non-zero position
    n_active    = int(active_mask.sum())
    n_bars      = len(pnl)

    # --- Sharpe (annualised on all bars) ---
    pnl_std = pnl.std()
    sharpe  = (pnl.mean() / pnl_std * np.sqrt(bars_per_year)
               if pnl_std > 1e-12 else 0.0)

    # --- Sortino (annualised, downside only) ---
    downside_pnl = pnl[pnl < 0]
    if len(downside_pnl) > 0:
        downside_dev = np.sqrt(np.mean(downside_pnl ** 2))
        sortino = (pnl.mean() / downside_dev * np.sqrt(bars_per_year)
                   if downside_dev > 1e-12 else 0.0)
    else:
        sortino = float("inf")

    # --- Maximum drawdown ---
    running_max = np.maximum.accumulate(equity_curve)
    drawdown    = (running_max - equity_curve) / (running_max + 1e-12)
    max_dd      = float(drawdown.max())

    # --- Profit factor (gross profit / gross loss) ---
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss   = float(np.abs(pnl[pnl < 0].sum()))
    profit_factor = (gross_profit / gross_loss
                     if gross_loss > 1e-12 else float("inf"))

    # --- Win rate (among active bars) ---
    if n_active > 0:
        active_pnl = pnl[active_mask]
        win_rate   = float((active_pnl > 0).mean())
        avg_win    = float(active_pnl[active_pnl > 0].mean()) if (active_pnl > 0).any() else 0.0
        avg_loss   = float(np.abs(active_pnl[active_pnl < 0].mean())) if (active_pnl < 0).any() else 0.0
    else:
        win_rate = avg_win = avg_loss = 0.0

    # --- Trade entry count (0 → non-zero position transitions) ---
    n_trade_entries = int(
        sum(1 for i in range(1, len(records))
            if abs(records[i - 1]["effective_action"]) < 1e-9
            and abs(records[i]["effective_action"]) > 1e-9)
    )

    # --- Average holding length (bars per non-flat sequence) ---
    avg_holding = float(n_active / n_trade_entries) if n_trade_entries > 0 else 0.0

    # --- Circuit breaker impact ---
    cb_overrides = int(
        sum(1 for r in records
            if abs(r["signal"]) > 1e-9
            and abs(r["effective_action"]) < abs(r["signal"]) - 1e-9)
    )

    # --- Total return ---
    total_return = float(equity_curve[-1] - 1.0)

    return {
        "sharpe":           round(float(sharpe),        4),
        "sortino":          round(float(sortino),       4),
        "max_drawdown":     round(max_dd,               4),
        "profit_factor":    round(profit_factor,        4),
        "win_rate":         round(win_rate,              4),
        "avg_win":          round(avg_win,               6),
        "avg_loss":         round(avg_loss,              6),
        "total_return":     round(total_return,          4),
        "n_bars":           n_bars,
        "n_active_bars":    n_active,
        "market_exposure":  round(n_active / n_bars,    4) if n_bars > 0 else 0.0,
        "n_trade_entries":  n_trade_entries,
        "avg_holding_bars": round(avg_holding,           2),
        "cb_overrides":     cb_overrides,
        "bars_per_year":    bars_per_year,
    }


# ---------------------------------------------------------------------------
# Session breakdown
# ---------------------------------------------------------------------------

def _session_breakdown(records: list, bars_per_year: int) -> dict:
    sessions = ["Asian", "London", "NY"]
    result   = {}

    for sess in sessions:
        subset = [r for r in records if r["session"] == sess]
        if not subset:
            result[sess] = {"n_bars": 0}
            continue

        pnl_s    = np.array([r["pnl"] for r in subset])
        eff_s    = np.array([r["effective_action"] for r in subset])
        active_s = np.abs(eff_s) > 1e-9

        sharpe_s = 0.0
        if pnl_s.std() > 1e-12:
            # Scale annualisation by fraction of year this session represents
            sess_fraction = len(subset) / bars_per_year
            sess_annual   = int(len(subset) / sess_fraction) if sess_fraction > 0 else bars_per_year
            sharpe_s = float(pnl_s.mean() / pnl_s.std() * np.sqrt(sess_annual))

        n_active = int(active_s.sum())
        wr       = float((pnl_s[active_s] > 0).mean()) if n_active > 0 else 0.0

        result[sess] = {
            "n_bars":    len(subset),
            "n_active":  n_active,
            "sharpe":    round(sharpe_s, 4),
            "win_rate":  round(wr, 4),
            "total_pnl": round(float(pnl_s.sum()), 6),
        }

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: dict, instrument: str = "") -> None:
    """Print a formatted backtest report."""
    m  = result["metrics"]
    sm = result["session_metrics"]
    ec = result["equity_curve"]

    header = f"BACKTEST RESULTS  [{instrument.upper()}]" if instrument else "BACKTEST RESULTS"
    print("\n" + "=" * 58)
    print(header)
    print("=" * 58)

    _tgt = lambda val, tgt, ge=True: ("OK" if (val >= tgt if ge else val <= tgt) else "!!")

    print(f"\n  Total return      : {m['total_return']:+.2%}")
    print(f"  Sharpe (annual)   : {m['sharpe']:>7.3f}   target >1.0   {_tgt(m['sharpe'], 1.0)}")
    print(f"  Sortino (annual)  : {m['sortino']:>7.3f}   target >1.5   {_tgt(m['sortino'], 1.5)}")
    print(f"  Max drawdown      : {m['max_drawdown']:>7.2%}  target <15%   {_tgt(m['max_drawdown'], 0.15, ge=False)}")
    print(f"  Profit factor     : {m['profit_factor']:>7.3f}   target >1.3   {_tgt(m['profit_factor'], 1.3)}")
    print(f"  Win rate (active) : {m['win_rate']:>7.2%}")
    print(f"  Avg win / loss    :  {m['avg_win']:.5f} / {m['avg_loss']:.5f}")

    print(f"\n  Bars total        : {m['n_bars']:,}")
    print(f"  Bars active       : {m['n_active_bars']:,}  ({m['market_exposure']:.1%} exposure)")
    print(f"  Trade entries     : {m['n_trade_entries']:,}   target >200   "
          f"{_tgt(m['n_trade_entries'], 200)}")
    print(f"  Avg holding       : {m['avg_holding_bars']:.1f} bars")
    print(f"  Circuit breaker   : {m['cb_overrides']:,} overrides")

    print(f"\n  Session breakdown:")
    print(f"  {'Session':<8} {'Bars':>6} {'Active':>6} {'Sharpe':>8} {'WinRate':>8} {'PnL':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for sess in ["Asian", "London", "NY"]:
        s = sm.get(sess, {})
        if not s or s.get("n_bars", 0) == 0:
            print(f"  {sess:<8}   —")
            continue
        print(f"  {sess:<8} {s['n_bars']:>6,} {s['n_active']:>6,} "
              f"{s['sharpe']:>8.3f} {s['win_rate']:>8.2%} {s['total_pnl']:>10.5f}")

    print(f"\n  Statistical note: {m['n_trade_entries']} entries < 200 threshold"
          if m['n_trade_entries'] < 200 else "")
    print("=" * 58)
