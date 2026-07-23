"""
data/long_history.py -- Long H1 history beyond Yahoo's 730-day cap
==================================================================

WHY
---
The paper's binding limitation is data volume: Yahoo caps H1 downloads at
~730 days, leaving 579 training bars for USD/JPY and 48/136-sample probe
val sets. This module lifts that ceiling by ~10x by combining:

  * PRICE (H1 OHLCV)  from Dukascopy  -- free, tick-derived, back to ~2003
                                          for EUR/USD, USD/JPY, XAU/USD.
  * MACRO (daily)     from Yahoo      -- DXY/GVZ/VIX/TLT/SHY/SPY/... have
                                          10-25y of DAILY history (no 730-day
                                          cap on interval='1d'); we forward-fill
                                          the daily level onto the H1 price grid.

Forward-filling daily macro onto H1 is standard and lookahead-safe here: the
existing pipeline already z-scores macro with a shift(1) rolling window, and
these covariates (real-yield proxy, USD index, implied-vol indices) are daily
instruments whose intraday path is not part of the 52-feature design.

HONEST START DATES (gated by macro availability, verified 2026-07 via Yahoo):
  gold    2008-07  (GVZ starts 2008-06-03 -- the binding gold covariate)
  eurusd  2004-01  (GBP/USD 2003-12, TLT 2002-07, VIX 1990)
  usdjpy  2006-06  (AUD/USD 2006-05-16 -- the binding carry-proxy covariate)

So all three reach ~18-22 years spanning 2008 GFC, 2011 EU crisis, 2015 CHF,
2018 vol, 2020 COVID, 2022 rate shock -- the severe VoE regimes the paper
currently cannot test.

OUTPUT
------
data/cache/dukascopy/{inst}_{chunkstart}.parquet   per-chunk H1 price cache (resumable)
data/cache/long/{inst}_h1_long.parquet             combined price+macro, pipeline-ready

The combined frame has lowercase OHLCV columns plus one column per macro key,
matching what DataFetcher.fetch_all() returns so the rest of the pipeline
(features.py -> normalization -> splits) is unchanged. See build_long().

USAGE
-----
    python -m data.long_history eurusd                 # one instrument
    python -m data.long_history all                    # all three (long-running)
    python -m data.long_history eurusd 2015-01-01 2017-01-01   # bounded slice

Dependencies:  pip install dukascopy-python   (yfinance already required)
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data.fetcher import INSTRUMENT_CONFIGS  # reuse the macro ticker map

# ----------------------------------------------------------------------------
DUKAS_CACHE = Path("./data/cache/dukascopy")
LONG_CACHE  = Path("./data/cache/long")
DUKAS_CACHE.mkdir(parents=True, exist_ok=True)
LONG_CACHE.mkdir(parents=True, exist_ok=True)

HONEST_START = {
    "gold":   "2008-07-01",
    "eurusd": "2004-01-01",
    "usdjpy": "2006-06-01",
}

CHUNK_DAYS = 90  # Dukascopy H1 pull granularity (resumable per chunk)


def _dukas_instrument(instrument: str):
    import dukascopy_python.instruments as I
    return {
        "gold":   I.INSTRUMENT_FX_METALS_XAU_USD,
        "eurusd": I.INSTRUMENT_FX_MAJORS_EUR_USD,
        "usdjpy": I.INSTRUMENT_FX_MAJORS_USD_JPY,
    }[instrument]


# ----------------------------------------------------------------------------
# 1) H1 price from Dukascopy (chunked, cached, resumable)
# ----------------------------------------------------------------------------
def fetch_price_h1(instrument: str, start: str, end: str) -> pd.DataFrame:
    import dukascopy_python as dp
    inst = _dukas_instrument(instrument)
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")

    frames = []
    cur = s
    while cur < e:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), e)
        cache = DUKAS_CACHE / f"{instrument}_{cur.date()}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
        else:
            for attempt in range(3):
                try:
                    df = dp.fetch(inst, dp.INTERVAL_HOUR_1, dp.OFFER_SIDE_BID, cur, nxt)
                    break
                except Exception as ex:
                    if attempt == 2:
                        print(f"    [warn] {instrument} {cur.date()} failed: {repr(ex)[:80]}")
                        df = pd.DataFrame()
                    else:
                        time.sleep(2 * (attempt + 1))
            if df is not None and len(df):
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.to_parquet(cache)
                frames.append(df)
            print(f"    {instrument} {cur.date()} -> {nxt.date()}: "
                  f"{0 if df is None else len(df):>4} bars", flush=True)
        cur = nxt

    if not frames:
        return pd.DataFrame()
    price = pd.concat(frames).sort_index()
    price = price[~price.index.duplicated(keep="first")]
    price.index.name = "timestamp"
    return price


# ----------------------------------------------------------------------------
# 2) Long daily macro from Yahoo, forward-filled onto the H1 price grid
# ----------------------------------------------------------------------------
def fetch_macro_daily_ffill(instrument: str, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    import yfinance as yf
    macro_map = INSTRUMENT_CONFIGS[instrument]["macro"]
    # ensure UTC tz-aware price index
    idx = price_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    out = {}
    for name, ticker in macro_map.items():
        try:
            d = yf.download(ticker, period="max", interval="1d",
                            progress=False, auto_adjust=True)
        except Exception as ex:
            print(f"    [warn] macro {name}({ticker}) failed: {repr(ex)[:60]}")
            d = pd.DataFrame()
        if d is None or len(d) == 0:
            out[name] = pd.Series(0.0, index=idx, name=name)
            continue
        close = d["Close"]
        if hasattr(close, "columns"):        # flatten yfinance MultiIndex
            close = close.iloc[:, 0]
        close = close.copy()
        close.index = pd.to_datetime(close.index).tz_localize("UTC")
        close = close.sort_index()
        # daily value is valid from its (UTC 00:00) date forward -> merge_asof backward
        left  = pd.DataFrame({"ts": idx})
        right = pd.DataFrame({"d": close.index, "val": close.values})
        merged = pd.merge_asof(left, right, left_on="ts", right_on="d",
                               direction="backward")
        out[name] = pd.Series(merged["val"].values, index=idx, name=name)

    macro = pd.DataFrame(out)
    macro.index = idx
    macro.index.name = "timestamp"
    return macro.ffill().fillna(0.0)


# ----------------------------------------------------------------------------
# 3) Combined pipeline-ready frame
# ----------------------------------------------------------------------------
def build_long(instrument: str, start: str = None, end: str = None,
               use_cache: bool = True) -> dict:
    start = start or HONEST_START[instrument]
    end   = end   or datetime.today().strftime("%Y-%m-%d")
    # Combined-cache filename MUST encode the date range: otherwise a cached
    # slice (e.g. a 2015-16 smoke test) gets reused for a different requested
    # range, silently training on the wrong data. Per-chunk caches remain
    # range-independent (keyed by chunk start) and are reused across ranges.
    combined_path = LONG_CACHE / f"{instrument}_h1_{start}_{end}.parquet"

    if use_cache and combined_path.exists():
        c = pd.read_parquet(combined_path)
        price = c[["open", "high", "low", "close", "volume"]]
        macro = c.drop(columns=["open", "high", "low", "close", "volume"])
        print(f"[{instrument}] loaded combined cache: {len(price):,} H1 bars "
              f"({price.index[0].date()} -> {price.index[-1].date()})")
        return {"price": price, "macro": macro}

    print(f"[{instrument}] building long history {start} -> {end}")
    print(f"  (1/2) H1 price from Dukascopy ...")
    price = fetch_price_h1(instrument, start, end)
    if len(price) == 0:
        raise RuntimeError(f"No Dukascopy price returned for {instrument}")
    print(f"        {len(price):,} H1 bars  {price.index[0]} -> {price.index[-1]}")

    print(f"  (2/2) daily macro from Yahoo, ffill to H1 grid ...")
    macro = fetch_macro_daily_ffill(instrument, price.index)

    combined = price.join(macro)
    combined.to_parquet(combined_path)
    print(f"  saved -> {combined_path}  ({combined.shape[0]:,} rows x "
          f"{combined.shape[1]} cols)")
    return {"price": price, "macro": macro}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "eurusd"
    s = sys.argv[2] if len(sys.argv) > 2 else None
    e = sys.argv[3] if len(sys.argv) > 3 else None
    targets = ["gold", "eurusd", "usdjpy"] if which == "all" else [which]
    for t in targets:
        build_long(t, s, e)
