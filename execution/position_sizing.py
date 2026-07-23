"""
execution/position_sizing.py — ATR-based position sizing.

Core rule:  risk exactly `risk_frac` of account equity per one ATR of adverse
            price movement.

    position_size = min(1.0, risk_frac / atr_pct)

where  atr_pct = ATR_14 / close_price  (dimensionless fraction).

Example (gold H1, account $100k):
    ATR=$20 on $3000 gold  ->  atr_pct=0.0067  ->  size=0.01/0.0067=1.50  ->  capped 1.0
    ATR=$40 on $3000 gold  ->  atr_pct=0.0133  ->  size=0.01/0.0133=0.75
    ATR=$80 on $3000 gold  ->  atr_pct=0.0267  ->  size=0.01/0.0267=0.37

MT5 lot conversion (XAUUSD, 1 lot = 100 oz):
    lots = (equity_usd * position_size) / (price * contract_mult)
"""

import numpy as np

_RISK_FRAC      = 0.01   # 1% equity risk per ATR of adverse move
_ATR_PERIOD     = 14
_CONTRACT_MULT  = 100    # XAUUSD: 1 standard lot = 100 oz


def atr_from_ohlc(
    high:   np.ndarray,
    low:    np.ndarray,
    close:  np.ndarray,
    period: int = _ATR_PERIOD,
) -> np.ndarray:
    """
    Wilder's Average True Range from OHLC arrays.
    Returns array of same length; first `period` values are NaN.
    """
    n  = len(close)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = float(np.nanmean(tr[1 : period + 1]))
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def atr_from_closes(close: np.ndarray, period: int = _ATR_PERIOD) -> np.ndarray:
    """
    ATR proxy using close prices only: Wilder-smoothed |Δclose|.
    Less accurate than OHLC ATR; use when high/low are unavailable.
    """
    tr    = np.abs(np.diff(close, prepend=close[0])).astype(float)
    tr[0] = np.nan
    atr   = np.full(len(close), np.nan)
    if len(close) > period:
        atr[period] = float(np.nanmean(tr[1 : period + 1]))
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def size_by_atr_risk(
    current_atr_pct: float,
    risk_frac:       float = _RISK_FRAC,
    max_size:        float = 1.0,
) -> float:
    """
    Fractional equity position size for a given current ATR-pct.

    Returns a value in (0, max_size].  Never returns 0 — caller decides
    whether to enter; this function only determines the size conditional
    on entering.
    """
    if current_atr_pct <= 1e-10:
        return max_size
    return float(min(max_size, risk_frac / current_atr_pct))


def lots_from_size(
    position_size:  float,
    equity_usd:     float,
    price:          float,
    contract_mult:  float = float(_CONTRACT_MULT),
    min_lots:       float = 0.01,
) -> float:
    """
    Convert fractional equity position to MT5 lot size.

        lots = (equity * position_size) / (price * contract_mult)

    Rounds down to nearest 0.01 lots (standard MT5 minimum step).
    """
    raw   = (equity_usd * position_size) / (price * contract_mult)
    lots  = max(min_lots, round(raw / min_lots) * min_lots)
    return round(lots, 2)
