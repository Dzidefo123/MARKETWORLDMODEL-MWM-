"""
data/fetcher.py - Market Data Fetcher
======================================

Supported instruments:
  gold   : GC=F   (Gold futures, XAUUSD proxy)
  eurusd : EURUSD=X
  usdjpy : USDJPY=X

Each instrument has a primary ticker + a set of macro instruments
relevant to its dominant price drivers.

YFINANCE H1 HARD LIMIT
-----------------------
Yahoo Finance only provides H1 data for the LAST 730 DAYS from today.
Any date older than 730 days returns empty data.

So we compute the date range DYNAMICALLY:
    end   = today
    start = today - 729 days

For longer history use MT5 export or a paid data provider (Polygon.io,
FirstRate Data).

COLUMN NAMING - yfinance MultiIndex fix
----------------------------------------
Newer yfinance (>=0.2.28) returns MultiIndex columns:
    ('Close', 'GC=F'), ('Open', 'GC=F'), ...
We flatten these automatically in _flatten_columns().
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Instrument configs
# ---------------------------------------------------------------------------

INSTRUMENT_CONFIGS = {
    "gold": {
        "primary":     "GC=F",
        "start_price": 2300.0,
        "macro": {
            "dxy":    "DX-Y.NYB",
            "gvz":    "^GVZ",
            "tlt":    "TLT",
            "silver": "SI=F",
        },
    },
    "eurusd": {
        "primary":     "EURUSD=X",
        "start_price": 1.09,
        "macro": {
            "dxy":    "DX-Y.NYB",
            "evz":    "^VIX",    # ^EVZ is delisted; VIX correlates with FX vol
            "tlt":    "TLT",
            "shy":    "SHY",
            "gbpusd": "GBPUSD=X",
        },
    },
    "usdjpy": {
        "primary":     "USDJPY=X",
        "start_price": 150.0,
        "macro": {
            "dxy":    "DX-Y.NYB",
            "tlt":    "TLT",
            "spy":    "SPY",
            "audusd": "AUDUSD=X",
        },
    },
}

CHUNK_DAYS = 55   # safely under Yahoo's ~60-day chunk limit for 1h data


# ---------------------------------------------------------------------------
# Ticker fallbacks
# ---------------------------------------------------------------------------
# Yahoo Finance periodically delists / renames symbols, and macro fetches must
# survive any one of them going dark. We try a list of equivalents in order and
# keep the first that returns bars. Every DXY feature is built from returns /
# ratios / correlations (scale-invariant), so the different absolute levels
# (DX-Y.NYB ~100, DX=F ~100, UUP ~28) are harmless.
#
# Order is by data quality, NOT by the order in the original request:
#   DX-Y.NYB : ICE cash US Dollar Index — full ~24/5 H1 coverage (BEST). As of
#              2026-06 this is the only source returning round-the-clock bars,
#              so it stays first; the others are genuine fallbacks.
#   DX=F     : USD Index futures — currently 404s on Yahoo, kept in case it
#              returns / works on other days.
#   UUP      : Invesco DB USD Bullish ETF — very reliable but equity-hours only
#              (~7 bars/day), leaving overnight gaps; last-resort only.
DXY_TICKERS = ["DX-Y.NYB", "DX=F", "UUP"]

TICKER_FALLBACKS = {
    "DX-Y.NYB": DXY_TICKERS,
    "DX=F":     DXY_TICKERS,
    "UUP":      DXY_TICKERS,
}


def fallback_tickers(ticker: str) -> list:
    """Return an ordered list of candidate tickers to try for `ticker`."""
    return TICKER_FALLBACKS.get(ticker, [ticker])


def _get_default_dates() -> tuple:
    """Return (start, end) covering the last 729 days from today."""
    end   = datetime.today()
    start = end - timedelta(days=729)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle yfinance MultiIndex columns.
    Older yfinance: columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    Newer yfinance: columns = [('Open','GC=F'), ('High','GC=F'), ...]
    Always returns simple lowercase string columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df


class DataFetcher:
    """
    Downloads and aligns OHLCV + macro data for model training.

    Caches downloaded data to disk so re-runs are instant.

    Args:
        cache_dir: Where to save parquet cache files (default: ./data/cache/)
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fetch_all(
        self,
        start:      str | None = None,
        end:        str | None = None,
        interval:   str  = "1h",
        use_cache:  bool = True,
        instrument: str  = "gold",
    ) -> dict:
        """
        Fetch primary asset + macro instruments and return aligned DataFrames.

        Args:
            start, end:  Date strings "YYYY-MM-DD". If None, uses today-729 days.
            interval:    Bar size. "1h" is the primary timeframe for MWM.
            use_cache:   Load from disk if previously downloaded.
            instrument:  "gold", "eurusd", or "usdjpy".

        Returns:
            {
                "price": pd.DataFrame  - OHLCV for the primary instrument
                "macro": pd.DataFrame  - macro instruments aligned to price index
            }
        """
        if instrument not in INSTRUMENT_CONFIGS:
            raise ValueError(
                f"Unknown instrument '{instrument}'. "
                f"Choose from: {list(INSTRUMENT_CONFIGS.keys())}"
            )
        config = INSTRUMENT_CONFIGS[instrument]

        # Dynamic default: always within Yahoo's 730-day window
        if start is None or end is None:
            default_start, default_end = _get_default_dates()
            start = start or default_start
            end   = end   or default_end

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        cutoff   = datetime.today() - timedelta(days=730)

        if start_dt < cutoff:
            safe_start = (datetime.today() - timedelta(days=728)).strftime("%Y-%m-%d")
            print(f"\n  WARNING: Requested start {start} is older than Yahoo's 730-day")
            print(f"    H1 data limit. Adjusting start to {safe_start}.")
            print(f"    For older data, use MT5 export or a paid data provider.")
            start = safe_start

        print(f"\n{'='*55}")
        print(f"DataFetcher: {start} -> {end}  |  interval={interval}  |  {instrument.upper()}")
        print(f"{'='*55}")

        # Primary asset
        price_df = self._fetch_with_cache(
            instrument, config["primary"], start, end, interval, use_cache
        )

        if price_df is None or len(price_df) == 0:
            raise RuntimeError(
                f"\nFailed to fetch {instrument} data.\n"
                f"Most likely cause: the requested dates ({start} -> {end}) "
                f"are outside Yahoo Finance's 730-day H1 window.\n"
                f"Run with no dates to use the automatic window:\n"
                f"    fetcher.fetch_all(instrument='{instrument}')\n"
            )

        print(f"\n  {instrument.upper()} bars: {len(price_df):,}")
        print(f"  Date range: {price_df.index[0].date()} -> {price_df.index[-1].date()}")

        # Macro instruments
        macro_dfs = {}
        for name, ticker in config["macro"].items():
            df = self._fetch_with_cache(name, ticker, start, end, interval, use_cache)
            if df is not None and len(df) > 0:
                macro_dfs[name] = df["close"].rename(name)
                print(f"  {name:8s}: {len(df):,} bars")
            else:
                print(f"  {name:8s}: FAILED - filling with zeros")
                macro_dfs[name] = pd.Series(0.0, index=price_df.index, name=name)

        macro_df = self._align_macro(price_df, macro_dfs)

        print(f"\n  Macro columns:  {list(macro_df.columns)}")
        print(f"  Missing values: {macro_df.isna().sum().to_dict()}")

        return {"price": price_df, "macro": macro_df}

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _fetch_with_cache(
        self, name: str, ticker: str,
        start: str, end: str, interval: str,
        use_cache: bool,
    ) -> pd.DataFrame | None:
        """Return cached data if it exists, otherwise download."""
        safe_name  = ticker.replace("=", "_").replace("^", "").replace("-", "_")
        cache_path = self.cache_dir / f"{safe_name}_{interval}_{start}_{end}.parquet"

        if use_cache and cache_path.exists():
            print(f"  Loading {name} from cache ({cache_path.name})...")
            return pd.read_parquet(cache_path)

        print(f"  Downloading {name} ({ticker})...", end="", flush=True)
        df = self._chunked_download(ticker, start, end, interval)

        if df is not None and len(df) > 0:
            if use_cache:
                df.to_parquet(cache_path)
            print(f"  {len(df):,} bars downloaded")
        else:
            print("  0 bars (failed)")

        return df

    def _chunked_download(
        self, ticker: str, start: str, end: str, interval: str
    ) -> pd.DataFrame | None:
        """
        Download data, trying each fallback candidate for `ticker` in order
        and returning the first that yields any bars. See TICKER_FALLBACKS.
        """
        candidates = fallback_tickers(ticker)
        for i, candidate in enumerate(candidates):
            df = self._chunked_download_single(candidate, start, end, interval)
            if df is not None and len(df) > 0:
                if candidate != ticker:
                    print(f" [fallback: {candidate}]", end="")
                return df
            if len(candidates) > 1 and i < len(candidates) - 1:
                print(f" [empty {candidate} -> trying {candidates[i+1]}]", end="")
        return None

    def _chunked_download_single(
        self, ticker: str, start: str, end: str, interval: str
    ) -> pd.DataFrame | None:
        """
        Download a single ticker in CHUNK_DAYS-sized windows and concatenate.
        Yahoo limits each H1 request to ~60 days.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Run: pip install yfinance")

        chunks   = []
        current  = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        n_chunks = 0

        while current < end_dt:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_dt)
            try:
                df = yf.download(
                    ticker,
                    start       = current.strftime("%Y-%m-%d"),
                    end         = chunk_end.strftime("%Y-%m-%d"),
                    interval    = interval,
                    progress    = False,
                    auto_adjust = True,
                )
                if len(df) > 0:
                    df = _flatten_columns(df)
                    chunks.append(df)
                    n_chunks += 1
            except Exception:
                pass   # silent - missing chunks are rare and handled by ffill

            current = chunk_end
            time.sleep(0.25)

        if not chunks:
            return None

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")].sort_index()

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df   = df[keep].dropna(subset=["close"])

        print(f" ({n_chunks} chunks)", end="")
        return df

    def _align_macro(
        self,
        price_df: pd.DataFrame,
        macro_series: dict,
    ) -> pd.DataFrame:
        """
        Reindex macro instruments to match primary asset bar timestamps.
        Uses forward-fill (not interpolation) to avoid lookahead bias.
        Caps forward-fill at 24 bars.
        """
        macro_df = pd.DataFrame(index=price_df.index)
        for name, series in macro_series.items():
            if isinstance(series, pd.Series) and len(series) > 0:
                aligned = series.reindex(price_df.index, method="ffill", limit=24)
                macro_df[name] = aligned
            else:
                macro_df[name] = 0.0
        return macro_df


# ---------------------------------------------------------------------------
# Synthetic data (for testing without internet)
# ---------------------------------------------------------------------------

def make_realistic_ohlcv(
    n_bars:     int   = 10_000,
    seed:       int   = 42,
    instrument: str   = "gold",
) -> tuple:
    """
    Generate realistic synthetic OHLCV data for the given instrument + macro.

    Uses a regime-switching GBM model with 4 states:
        0: trending_up    (drift +0.0002/bar, vol 0.003)
        1: trending_down  (drift -0.0002/bar, vol 0.003)
        2: ranging        (drift  0.0000/bar, vol 0.001)
        3: crisis         (drift -0.0005/bar, vol 0.008)

    Macro correlations reflect each instrument's real-world drivers:
      gold:   DXY inverse, GVZ + during crisis, TLT positive
      eurusd: DXY strongly inverse, SPY mild positive, GBPUSD correlated
      usdjpy: DXY positive, TLT positive (rate differential), SPY positive
    """
    if instrument not in INSTRUMENT_CONFIGS:
        raise ValueError(f"Unknown instrument '{instrument}'")

    np.random.seed(seed)

    start_price = INSTRUMENT_CONFIGS[instrument]["start_price"]

    regimes = {
        0: {"drift": +0.0002, "vol": 0.003},
        1: {"drift": -0.0002, "vol": 0.003},
        2: {"drift":  0.0000, "vol": 0.001},
        3: {"drift": -0.0005, "vol": 0.008},
    }
    trans = np.array([
        [0.97, 0.01, 0.01, 0.01],
        [0.01, 0.97, 0.01, 0.01],
        [0.20, 0.20, 0.55, 0.05],
        [0.05, 0.40, 0.45, 0.10],
    ])

    regime_seq = np.zeros(n_bars, dtype=int)
    regime_seq[0] = 2
    for t in range(1, n_bars):
        regime_seq[t] = np.random.choice(4, p=trans[regime_seq[t-1]])

    log_returns = np.array([
        np.random.normal(regimes[regime_seq[t]]["drift"],
                         regimes[regime_seq[t]]["vol"])
        for t in range(n_bars)
    ])

    close  = start_price * np.exp(np.cumsum(log_returns))
    noise  = np.abs(np.random.normal(0, 0.002, n_bars))
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = np.roll(close, 1); open_[0] = start_price
    volume = np.random.lognormal(10, 1, n_bars).astype(int)

    ts_start   = datetime.today() - timedelta(days=728)
    timestamps = pd.date_range(ts_start, periods=n_bars * 2, freq="1h")
    timestamps = timestamps[timestamps.dayofweek < 5][:n_bars]

    price_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "regime": regime_seq},
        index=timestamps
    )

    # Macro correlations vary by instrument
    if instrument == "gold":
        dxy_ret    = -0.6 * log_returns + np.random.normal(0, 0.001, n_bars)
        tlt_ret    =  0.3 * log_returns + np.random.normal(0, 0.001, n_bars)
        silver_ret =  0.8 * log_returns + np.random.normal(0, 0.004, n_bars)
        gvz = np.clip(15 + np.random.normal(0, 2, n_bars) + 15*(regime_seq==3), 8, 80)
        macro_df = pd.DataFrame({
            "dxy":    100 * np.exp(np.cumsum(dxy_ret)),
            "gvz":    gvz,
            "tlt":    100 * np.exp(np.cumsum(tlt_ret)),
            "silver": 25  * np.exp(np.cumsum(silver_ret)),
        }, index=timestamps)

    elif instrument == "eurusd":
        # EURUSD is ~57% of DXY basket - strongly inverse
        # VIX (used as evz proxy since ^EVZ is delisted) spikes during EUR/USD stress
        # SHY (1-3yr Treasury) less volatile than TLT, tracks short-rate expectations
        dxy_ret    = -0.8 * log_returns + np.random.normal(0, 0.001, n_bars)
        tlt_ret    = -0.2 * log_returns + np.random.normal(0, 0.001, n_bars)
        shy_ret    =  0.3 * tlt_ret + np.random.normal(0, 0.0005, n_bars)
        gbpusd_ret =  0.75 * log_returns + np.random.normal(0, 0.003, n_bars)
        evz = np.clip(8 + np.random.normal(0, 1.5, n_bars) + 8 * (regime_seq == 3), 4, 40)
        macro_df = pd.DataFrame({
            "dxy":    100  * np.exp(np.cumsum(dxy_ret)),
            "evz":    evz,
            "tlt":    100  * np.exp(np.cumsum(tlt_ret)),
            "shy":    85   * np.exp(np.cumsum(shy_ret)),
            "gbpusd": 1.27 * np.exp(np.cumsum(gbpusd_ret)),
        }, index=timestamps)

    else:  # usdjpy
        # USDJPY driven by US-Japan rate differential (TLT proxy) + risk sentiment
        dxy_ret    =  0.5 * log_returns + np.random.normal(0, 0.001, n_bars)
        tlt_ret    = -0.4 * log_returns + np.random.normal(0, 0.001, n_bars)
        spy_ret    =  0.3 * log_returns + np.random.normal(0, 0.002, n_bars)
        audusd_ret = -0.5 * log_returns + np.random.normal(0, 0.003, n_bars)
        macro_df = pd.DataFrame({
            "dxy":    100  * np.exp(np.cumsum(dxy_ret)),
            "tlt":    100  * np.exp(np.cumsum(tlt_ret)),
            "spy":    450  * np.exp(np.cumsum(spy_ret)),
            "audusd": 0.65 * np.exp(np.cumsum(audusd_ret)),
        }, index=timestamps)

    return price_df, macro_df


# ---------------------------------------------------------------------------
# Run as script: python -m data.fetcher
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    INSTRUMENT   = "gold"    # change to "eurusd" or "usdjpy"
    USE_REAL_DATA = True

    print("=" * 55)
    print(f"DataFetcher Test  [{INSTRUMENT.upper()}]")
    print("=" * 55)

    if USE_REAL_DATA:
        fetcher  = DataFetcher()
        data     = fetcher.fetch_all(instrument=INSTRUMENT, use_cache=True)
        price_df = data["price"]
        macro_df = data["macro"]
    else:
        print("\nSynthetic mode (no internet needed)...")
        price_df, macro_df = make_realistic_ohlcv(n_bars=8_760, instrument=INSTRUMENT)

    print(f"\nPrice shape:  {price_df.shape}")
    print(f"Macro shape:  {macro_df.shape}")
    print(f"\nPrice sample:")
    print(price_df.head(3).to_string())
    print(f"\nMacro sample:")
    print(macro_df.head(3).to_string())

    assert len(price_df) == len(macro_df), "Length mismatch!"
    assert (price_df["close"] > 0).all(), "Non-positive close prices!"

    print(f"\nDataFetcher OK - {len(price_df):,} bars ready  [{INSTRUMENT.upper()}]")
