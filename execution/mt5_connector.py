"""
execution/mt5_connector.py — MT5 API wrapper for live gold trading.

Wraps MetaTrader5 (pip install MetaTrader5) with a thin interface that the
live_trader loop calls.  Also fetches macro data (DXY/GVZ/TLT/Silver) from
Yahoo Finance via yfinance so that the full feature pipeline can run on each
bar close.

Usage:
    connector = MT5Connector(symbol="XAUUSD", magic=20260101)
    connector.connect()
    bars = connector.fetch_ohlcv_bars(n=100)
    macro = connector.fetch_macro_data(n=100)
    eq = connector.get_equity()
    connector.place_order(lots=0.10, direction=1)   # 1 = long, -1 = short
    connector.close_position()
    connector.disconnect()
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Macro tickers — same symbols used by DataFetcher.
# dxy resolves through fallback_tickers (DX-Y.NYB -> DX=F -> UUP) so a single
# delisting can't break the macro feed.
_MACRO_TICKERS = {
    "dxy":    "DX-Y.NYB",
    "gvz":    "^GVZ",
    "tlt":    "TLT",
    "silver": "SI=F",
}

_GOLD_TICK_VALUE   = 0.01   # USD per 0.01 price move on 1 lot XAUUSD
_CONTRACT_MULT     = 100    # 1 lot = 100 oz


class MT5Connector:
    """
    Thin wrapper around the MetaTrader5 Python package.

    All orders go through market execution with deviation=20 points.
    Only one position at a time is supported (the live trader is a
    position-managed trend-follower, not a multi-trade system).

    Args:
        symbol:      MT5 symbol string (e.g. "XAUUSD").
        magic:       Expert Advisor magic number for order identification.
        deviation:   Max price deviation in points for market orders.
        spread_frac: Approximate spread as a fraction of price, used for
                     signal validation only (not for actual execution).
    """

    def __init__(
        self,
        symbol:      str   = "XAUUSD",
        magic:       int   = 20260101,
        deviation:   int   = 20,
        spread_frac: float = 0.0003,
    ):
        self.symbol      = symbol
        self.magic       = magic
        self.deviation   = deviation
        self.spread_frac = spread_frac
        self._mt5        = None   # lazily imported

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise the MT5 terminal connection."""
        import MetaTrader5 as mt5
        self._mt5 = mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        info = mt5.terminal_info()
        logger.info("Connected to MT5: %s build %s", info.name, info.build)

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if self._mt5 is not None:
            self._mt5.shutdown()
            logger.info("MT5 connection closed.")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def fetch_ohlcv_bars(self, n: int = 200, timeframe: str = "H1") -> pd.DataFrame:
        """
        Fetch the last `n` completed H1 bars for self.symbol.

        Returns a DataFrame with columns [open, high, low, close, volume]
        indexed by UTC datetime.  The most recent bar is the LAST row.
        """
        mt5 = self._mt5
        tf  = self._parse_timeframe(timeframe)

        # Request n+1 bars — MT5 may include the currently forming bar; we drop it.
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, n + 1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"copy_rates_from_pos failed for {self.symbol}: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={
            "open":    "open",
            "high":    "high",
            "low":     "low",
            "close":   "close",
            "tick_volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        # Drop the current (still forming) bar — last bar is always incomplete.
        return df.iloc[:-1].tail(n)

    def fetch_macro_data(self, n: int = 200) -> dict[str, pd.Series]:
        """
        Fetch the last `n` hourly closes for DXY, GVZ, TLT, and Silver via
        yfinance.  Returns a dict {name: pd.Series} aligned by UTC hour.

        Missing bars are forward-filled to match gold bar alignment.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance is required for macro data: pip install yfinance") from e

        end   = datetime.now(tz=timezone.utc)
        # Fetch extra history for warm-up; 250 h ≈ 10 days
        start = end - timedelta(hours=n + 50)

        from data.fetcher import fallback_tickers

        result: dict[str, pd.Series] = {}
        for name, ticker in _MACRO_TICKERS.items():
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
                    close.index = close.index.tz_convert("UTC")
                    series = close.tail(n)
                    if candidate != ticker:
                        logger.info("%s: used fallback ticker %s", name, candidate)
                    break
                except Exception as exc:
                    logger.warning("Failed to fetch %s (%s): %s", name, candidate, exc)
            if series.empty:
                logger.warning("yfinance returned no data for %s (tried %s)",
                               name, fallback_tickers(ticker))
            result[name] = series

        return result

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def get_equity(self) -> float:
        """Return current account equity in account currency."""
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {self._mt5.last_error()}")
        return float(info.equity)

    def get_current_position(self) -> float:
        """
        Return current net position in lots (positive = long, negative = short,
        0.0 = flat).  Aggregates all positions with self.magic.
        """
        positions = self._mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return 0.0
        net = 0.0
        for pos in positions:
            if pos.magic != self.magic:
                continue
            if pos.type == self._mt5.ORDER_TYPE_BUY:
                net += pos.volume
            else:
                net -= pos.volume
        return round(net, 2)

    def get_current_price(self) -> float:
        """Return the current bid/ask midpoint for self.symbol."""
        tick = self._mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({self.symbol}) failed: {self._mt5.last_error()}")
        return (tick.bid + tick.ask) / 2.0

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(self, lots: float, direction: int) -> dict:
        """
        Send a market order.

        Args:
            lots:      Position size (positive float).
            direction: +1 = buy, -1 = sell.

        Returns:
            MT5 order result as a dict.
        """
        if lots <= 0:
            raise ValueError(f"lots must be positive, got {lots}")
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")

        mt5      = self._mt5
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        tick     = mt5.symbol_info_tick(self.symbol)
        price    = tick.ask if direction == 1 else tick.bid

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    self.symbol,
            "volume":    round(lots, 2),
            "type":      order_type,
            "price":     price,
            "deviation": self.deviation,
            "magic":     self.magic,
            "comment":   "MWM",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            raise RuntimeError(f"order_send failed (retcode={code}): {mt5.last_error()}")

        logger.info("Order placed: %s %.2f lots @ %.2f",
                    "BUY" if direction == 1 else "SELL", lots, price)
        return result._asdict()

    def close_position(self, retries: int = 3) -> None:
        """
        Close all open positions for self.symbol with self.magic.
        Retries up to `retries` times on transient failures.
        """
        mt5 = self._mt5
        for attempt in range(retries):
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return   # already flat

            for pos in positions:
                if pos.magic != self.magic:
                    continue
                is_buy   = pos.type == mt5.ORDER_TYPE_BUY
                close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
                tick       = mt5.symbol_info_tick(self.symbol)
                price      = tick.bid if is_buy else tick.ask

                request = {
                    "action":    mt5.TRADE_ACTION_DEAL,
                    "symbol":    self.symbol,
                    "volume":    pos.volume,
                    "type":      close_type,
                    "position":  pos.ticket,
                    "price":     price,
                    "deviation": self.deviation,
                    "magic":     self.magic,
                    "comment":   "MWM-close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.warning(
                        "close_position attempt %d failed (retcode=%s)",
                        attempt + 1,
                        result.retcode if result else "None",
                    )
                    time.sleep(1.0)
                else:
                    logger.info("Closed position ticket=%d %.2f lots", pos.ticket, pos.volume)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def lots_from_equity_risk(
        self,
        atr_pct:       float,
        risk_frac:     float = 0.01,
        min_lots:      float = 0.01,
    ) -> float:
        """
        Compute MT5 lot size targeting risk_frac of equity per ATR adverse move.

            position_size = min(1.0, risk_frac / atr_pct)
            lots = (equity * position_size) / (price * contract_mult)

        Rounds down to nearest 0.01 lots.
        """
        from execution.position_sizing import size_by_atr_risk, lots_from_size

        equity = self.get_equity()
        price  = self.get_current_price()
        size   = size_by_atr_risk(atr_pct, risk_frac)
        return lots_from_size(size, equity, price, min_lots=min_lots)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timeframe(tf: str) -> int:
        """Map timeframe string to MT5 constant."""
        import MetaTrader5 as mt5
        mapping = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }
        if tf not in mapping:
            raise ValueError(f"Unknown timeframe '{tf}'. Choose from {list(mapping)}")
        return mapping[tf]
