"""
data/features.py - 52-Feature Market Feature Engineer
=======================================================

Two engineers are provided, both producing exactly 52 features so the
same model architecture (n_features=52) works for all instruments:

  GoldFeatureEngineer   - for GC=F (XAUUSD proxy)
  ForexFeatureEngineer  - for EURUSD=X and USDJPY=X

Feature groups (same structure for all instruments):
  Group A - Returns          (6 features)   cols 0-5
  Group B - Volatility       (8 features)   cols 6-13
  Group C - Momentum / Trend (12 features)  cols 14-25
  Group D - Price Structure  (8 features)   cols 26-33
  Group E - Session / Time   (6 features)   cols 34-39
  Group F - Macro            (12 features)  cols 40-51   <-- differs per instrument
  -------------------------------------------------------
  TOTAL                      52 features

LOOKAHEAD BIAS
--------------
Every feature uses ONLY past and current data. No feature peeks at
future bars. Enforced by shift(1), rolling windows ending AT current bar,
and never filling NaN with future values.

NORMALIZATION
-------------
Raw features are NOT normalized here. Normalization happens in pipeline.py
using a rolling z-score so that no future statistics leak into training.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ===========================================================================
# Gold Feature Engineer
# ===========================================================================

class GoldFeatureEngineer:
    """
    Computes all 52 features for Gold (GC=F / XAUUSD proxy).

    Group F macro uses: DXY, GVZ, TLT, Silver.

    Args:
        lookback_warm: Bars to discard at the start due to indicator warm-up.
                       EMA(200) needs 200 bars before it is meaningful.
                       Default: 250 (safe buffer above 200).
    """

    FEATURE_NAMES = [
        # Group A - Returns (6)
        "ret_1", "ret_4", "ret_8", "ret_16", "ret_32", "gap",
        # Group B - Volatility (8)
        "atr_14", "atr_50", "rv_8", "rv_16", "rv_32",
        "bb_width", "hl_range", "parkinson_vol",
        # Group C - Momentum / Trend (12)
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "ema_ratio_8_21", "ema_ratio_21_55", "ema_ratio_55_200",
        "adx_14", "roc_5", "roc_10", "roc_20", "stoch_k",
        # Group D - Price Structure (8)
        "close_vs_ema200", "close_vs_ema55", "close_vs_ema21",
        "bar_body", "upper_wick", "lower_wick",
        "close_position", "pivot_distance",
        # Group E - Session / Time (6)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_london", "is_ny",
        # Group F - Macro / Gold-specific (12)
        "dxy_ret_1", "dxy_ret_4", "dxy_vs_ema20",
        "gvz_level", "gvz_vs_ma20",
        "tlt_ret_1", "tlt_ret_4",
        "silver_ret_1", "gold_silver_ratio",
        "corr_gold_dxy_20", "macro_tension", "real_yield_proxy",
    ]

    def __init__(self, lookback_warm: int = 250):
        self.lookback_warm = lookback_warm
        assert len(self.FEATURE_NAMES) == 52, \
            f"Expected 52 features, got {len(self.FEATURE_NAMES)}"

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def compute(
        self,
        price_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> tuple:
        """
        Compute all 52 features for gold.

        Args:
            price_df: DataFrame with open, high, low, close, volume columns
            macro_df: DataFrame with dxy, gvz, tlt, silver columns
                      (same index as price_df)

        Returns:
            features:   np.ndarray of shape (n_valid_bars, 52)
            timestamps: pd.DatetimeIndex of valid bar timestamps
            regimes:    np.ndarray of shape (n_valid_bars,) if 'regime' col exists,
                        else None. Used for probing labels in O2.
        """
        assert price_df.index.equals(macro_df.index), \
            "price_df and macro_df must have identical indices"

        # Groups A-E (generic price action + session)
        groups_ae, o, h, l, c, v, idx = self._compute_price_groups(price_df)

        # Group F - Gold macro (12 features)
        print("  Computing Group F: Macro...")
        dxy    = macro_df["dxy"].values.astype(float)
        gvz    = macro_df["gvz"].values.astype(float)
        tlt    = macro_df["tlt"].values.astype(float)
        silver = macro_df["silver"].values.astype(float)

        for arr in [dxy, gvz, tlt, silver]:
            mask = (arr == 0) | ~np.isfinite(arr)
            if mask.any():
                arr[mask] = np.nan
                nans = np.where(mask)[0]
                for i in nans:
                    arr[i] = arr[i-1] if i > 0 else 100.0

        ret_1_price  = self._log_returns(c, 1)   # for rolling corr
        dxy_ret_1    = self._log_returns(dxy, 1)
        dxy_ret_4    = self._log_returns(dxy, 4)
        dxy_ema20    = self._ema(dxy, 20)
        dxy_vs_ema20 = dxy / dxy_ema20 - 1

        gvz_ma20    = self._rolling_mean(gvz, 20)
        gvz_level   = gvz / 50.0
        gvz_vs_ma20 = gvz / (gvz_ma20 + 1e-8) - 1

        tlt_ret_1   = self._log_returns(tlt, 1)
        tlt_ret_4   = self._log_returns(tlt, 4)

        silver_ret_1      = self._log_returns(silver, 1)
        gold_silver       = c / (silver + 1e-8)
        gold_silver_ratio = self._log_returns(gold_silver, 1)

        corr_gold_dxy  = self._rolling_corr(ret_1_price, dxy_ret_1, 20)
        macro_tension  = np.abs(dxy_ret_1) + np.abs(tlt_ret_1)
        real_yield_proxy = -tlt_ret_1

        group_f = np.column_stack([
            dxy_ret_1, dxy_ret_4, dxy_vs_ema20,
            gvz_level, gvz_vs_ma20,
            tlt_ret_1, tlt_ret_4,
            silver_ret_1, gold_silver_ratio,
            corr_gold_dxy, macro_tension, real_yield_proxy,
        ])

        feature_matrix = np.hstack([groups_ae, group_f])

        assert feature_matrix.shape[1] == 52, \
            f"Expected 52 features, got {feature_matrix.shape[1]}"

        W = self.lookback_warm
        feature_matrix = feature_matrix[W:]
        timestamps     = price_df.index[W:]
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        regimes = None
        if "regime" in price_df.columns:
            regimes = price_df["regime"].values[W:]

        print(f"\n  Feature matrix shape: {feature_matrix.shape}")
        print(f"  NaN count after cleanup: {np.isnan(feature_matrix).sum()}")
        print(f"  Timestamp range: {timestamps[0]} -> {timestamps[-1]}")

        return feature_matrix, timestamps, regimes

    # -----------------------------------------------------------------------
    # Shared helper: Groups A-E (pure price action + session/time)
    # -----------------------------------------------------------------------

    def _compute_price_groups(self, price_df: pd.DataFrame) -> tuple:
        """
        Compute Groups A through E (40 features) from OHLCV + timestamps.

        Returns (group_ae_array, o, h, l, c, v, idx) so callers can use
        the raw arrays for Group F computation too.
        """
        o = price_df["open"].values.astype(float)
        h = price_df["high"].values.astype(float)
        l = price_df["low"].values.astype(float)
        c = price_df["close"].values.astype(float)
        v = price_df["volume"].values.astype(float)

        # Group A: Returns (6)
        print("  Computing Group A: Returns...")
        ret_1  = self._log_returns(c, 1)
        ret_4  = self._log_returns(c, 4)
        ret_8  = self._log_returns(c, 8)
        ret_16 = self._log_returns(c, 16)
        ret_32 = self._log_returns(c, 32)
        gap    = np.log(o / np.roll(c, 1))
        gap[0] = 0.0

        # Group B: Volatility (8)
        print("  Computing Group B: Volatility...")
        atr_14   = self._atr(h, l, c, 14) / c
        atr_50   = self._atr(h, l, c, 50) / c
        rv_8     = self._rolling_std(ret_1, 8)
        rv_16    = self._rolling_std(ret_1, 16)
        rv_32    = self._rolling_std(ret_1, 32)
        bb_width = self._bollinger_width(c, 20, 2.0) / c
        hl_range = (h - l) / c
        parkinson = self._parkinson_vol(h, l, 14)

        # Group C: Momentum / Trend (12)
        print("  Computing Group C: Momentum/Trend...")
        rsi_14  = self._rsi(c, 14) / 100.0
        macd_l, macd_s, macd_h = self._macd(c, 12, 26, 9)
        macd_line   = macd_l / c
        macd_signal = macd_s / c
        macd_hist   = macd_h / c
        ema8    = self._ema(c, 8)
        ema21   = self._ema(c, 21)
        ema55   = self._ema(c, 55)
        ema200  = self._ema(c, 200)
        ema_ratio_8_21   = ema8  / ema21  - 1
        ema_ratio_21_55  = ema21 / ema55  - 1
        ema_ratio_55_200 = ema55 / ema200 - 1
        adx_14  = self._adx(h, l, c, 14) / 100.0
        roc_5   = c / np.roll(c, 5)  - 1;  roc_5[:5]   = 0
        roc_10  = c / np.roll(c, 10) - 1;  roc_10[:10] = 0
        roc_20  = c / np.roll(c, 20) - 1;  roc_20[:20] = 0
        stoch_k = self._stochastic_k(h, l, c, 14)

        # Group D: Price Structure (8)
        print("  Computing Group D: Price Structure...")
        close_vs_ema200 = c / ema200 - 1
        close_vs_ema55  = c / ema55  - 1
        close_vs_ema21  = c / ema21  - 1
        bar_range  = h - l + 1e-8
        bar_body   = (c - o) / bar_range
        upper_wick = (h - np.maximum(o, c)) / bar_range
        lower_wick = (np.minimum(o, c) - l) / bar_range
        close_pos  = (c - l) / bar_range
        pivot_dist = self._pivot_distance(h, l, c)

        # Group E: Session / Time (6)
        print("  Computing Group E: Session/Time...")
        idx      = price_df.index
        hour     = idx.hour.values.astype(float)
        dow      = idx.dayofweek.values.astype(float)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin  = np.sin(2 * np.pi * dow / 5)
        dow_cos  = np.cos(2 * np.pi * dow / 5)
        is_london = ((hour >= 7)  & (hour < 16)).astype(float)
        is_ny     = ((hour >= 12) & (hour < 21)).astype(float)

        groups_ae = np.column_stack([
            # A
            ret_1, ret_4, ret_8, ret_16, ret_32, gap,
            # B
            atr_14, atr_50, rv_8, rv_16, rv_32, bb_width, hl_range, parkinson,
            # C
            rsi_14, macd_line, macd_signal, macd_hist,
            ema_ratio_8_21, ema_ratio_21_55, ema_ratio_55_200,
            adx_14, roc_5, roc_10, roc_20, stoch_k,
            # D
            close_vs_ema200, close_vs_ema55, close_vs_ema21,
            bar_body, upper_wick, lower_wick, close_pos, pivot_dist,
            # E
            hour_sin, hour_cos, dow_sin, dow_cos, is_london, is_ny,
        ])

        return groups_ae, o, h, l, c, v, idx

    # -----------------------------------------------------------------------
    # Technical indicator implementations (pure numpy)
    # -----------------------------------------------------------------------

    @staticmethod
    def _log_returns(prices: np.ndarray, n: int) -> np.ndarray:
        r = np.log(prices / np.roll(prices, n))
        r[:n] = 0.0
        return r

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> np.ndarray:
        k   = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * k + ema[i-1] * (1 - k)
        return ema

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr).rolling(window, min_periods=1).mean().values

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr).rolling(window, min_periods=2).std().fillna(0).values

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int) -> np.ndarray:
        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        return GoldFeatureEngineer._ema(tr, period)

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta    = np.diff(close, prepend=close[0])
        gain     = np.where(delta > 0, delta, 0.0)
        loss     = np.where(delta < 0, -delta, 0.0)
        avg_gain = GoldFeatureEngineer._ema(gain, period)
        avg_loss = GoldFeatureEngineer._ema(loss, period)
        rs       = avg_gain / (avg_loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd(close: np.ndarray, fast: int = 12, slow: int = 26,
              signal: int = 9) -> tuple:
        ema_f  = GoldFeatureEngineer._ema(close, fast)
        ema_s  = GoldFeatureEngineer._ema(close, slow)
        macd_l = ema_f - ema_s
        macd_s = GoldFeatureEngineer._ema(macd_l, signal)
        macd_h = macd_l - macd_s
        return macd_l, macd_s, macd_h

    @staticmethod
    def _bollinger_width(close: np.ndarray, period: int = 20,
                         n_std: float = 2.0) -> np.ndarray:
        ser = pd.Series(close)
        std = ser.rolling(period, min_periods=2).std().fillna(0).values
        return 2 * n_std * std

    @staticmethod
    def _parkinson_vol(high: np.ndarray, low: np.ndarray,
                       period: int = 14) -> np.ndarray:
        log_hl = np.log(high / (low + 1e-8)) ** 2
        factor = 1.0 / (4.0 * np.log(2.0))
        return pd.Series(np.sqrt(factor * log_hl)).rolling(period, min_periods=1).mean().values

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray:
        prev_high  = np.roll(high,  1); prev_high[0]  = high[0]
        prev_low   = np.roll(low,   1); prev_low[0]   = low[0]
        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        plus_dm  = np.where((high - prev_high) > (prev_low - low),
                            np.maximum(high - prev_high, 0), 0)
        minus_dm = np.where((prev_low - low) > (high - prev_high),
                            np.maximum(prev_low - low, 0), 0)
        tr       = np.maximum(high - low,
                   np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr_ema   = GoldFeatureEngineer._ema(tr,       period)
        pdm_ema  = GoldFeatureEngineer._ema(plus_dm,  period)
        ndm_ema  = GoldFeatureEngineer._ema(minus_dm, period)
        pdi = 100 * pdm_ema / (tr_ema + 1e-8)
        ndi = 100 * ndm_ema / (tr_ema + 1e-8)
        dx  = 100 * np.abs(pdi - ndi) / (pdi + ndi + 1e-8)
        return GoldFeatureEngineer._ema(dx, period)

    @staticmethod
    def _stochastic_k(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      period: int = 14) -> np.ndarray:
        ser_h        = pd.Series(high)
        ser_l        = pd.Series(low)
        highest_high = ser_h.rolling(period, min_periods=1).max().values
        lowest_low   = ser_l.rolling(period, min_periods=1).min().values
        k            = (close - lowest_low) / (highest_high - lowest_low + 1e-8) * 100
        return k / 100.0

    @staticmethod
    def _pivot_distance(high: np.ndarray, low: np.ndarray,
                        close: np.ndarray) -> np.ndarray:
        ser_h      = pd.Series(high)
        ser_l      = pd.Series(low)
        ser_c      = pd.Series(close)
        prev_high  = ser_h.rolling(24, min_periods=1).max().shift(1).fillna(high[0]).values
        prev_low   = ser_l.rolling(24, min_periods=1).min().shift(1).fillna(low[0]).values
        prev_close = ser_c.rolling(24, min_periods=1).mean().shift(1).fillna(close[0]).values
        pivot      = (prev_high + prev_low + prev_close) / 3.0
        return (close - pivot) / (close + 1e-8)

    @staticmethod
    def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(x).rolling(window, min_periods=5).corr(
            pd.Series(y)
        ).fillna(0).values


# ===========================================================================
# Forex Feature Engineer  (EUR/USD and USD/JPY)
# ===========================================================================

# Per-instrument Group B override: EUR/USD replaces parkinson_vol (col 13) with rv_64
# because Parkinson vol is redundant with rv_32 for FX but rv_64 adds a slower
# volatility regime signal that matches ECB/Fed decision cycle windows.
_FOREX_GROUP_B_NAMES = {
    "eurusd": ["atr_14", "atr_50", "rv_8", "rv_16", "rv_32", "bb_width", "hl_range", "rv_64"],
    "usdjpy": ["atr_14", "atr_50", "rv_8", "rv_16", "rv_32", "bb_width", "hl_range", "parkinson_vol"],
}

# Per-instrument Group F names (12 features each)
_FOREX_GROUP_F_NAMES = {
    "eurusd": [
        "dxy_ret_1", "dxy_ret_4", "dxy_vs_ema20",
        "evz_level", "evz_vs_ma20",
        "tlt_ret_1", "rate_diff_proxy",
        "cross_ret_1", "corr_asset_dxy_20",
        "macro_tension", "real_yield_proxy", "vol_risk_premium",
    ],
    "usdjpy": [
        "dxy_ret_1", "dxy_ret_4", "dxy_vs_ema20",
        "tlt_ret_1", "tlt_ret_4",
        "spy_ret_1", "spy_ret_4", "spy_vs_ema20",
        "cross_ret_1", "corr_asset_dxy_20",
        "macro_tension", "real_yield_proxy",
    ],
}

# Required macro_df columns per instrument
_FOREX_REQUIRED_COLS = {
    "eurusd": {"dxy", "evz", "tlt", "shy", "gbpusd"},
    "usdjpy": {"dxy", "tlt", "spy", "audusd"},
}

_FOREX_CROSS_COL = {
    "eurusd": "gbpusd",
    "usdjpy": "audusd",
}

# Groups A, C, D, E names (shared)
_GROUP_ACDE_NAMES = [
    # A
    "ret_1", "ret_4", "ret_8", "ret_16", "ret_32", "gap",
    # C
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "ema_ratio_8_21", "ema_ratio_21_55", "ema_ratio_55_200",
    "adx_14", "roc_5", "roc_10", "roc_20", "stoch_k",
    # D
    "close_vs_ema200", "close_vs_ema55", "close_vs_ema21",
    "bar_body", "upper_wick", "lower_wick",
    "close_position", "pivot_distance",
    # E
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_london", "is_ny",
]


class ForexFeatureEngineer(GoldFeatureEngineer):
    """
    Computes all 52 features for forex pairs (EUR/USD or USD/JPY).

    Groups A, C, D, E are identical to GoldFeatureEngineer.
    Group B: EUR/USD replaces parkinson_vol with rv_64; USD/JPY keeps parkinson_vol.
    Group F: fully per-instrument (see _FOREX_GROUP_F_NAMES).

    EUR/USD Group F (12):
      dxy_ret_1       DXY 1-bar log return
      dxy_ret_4       DXY 4-bar log return
      dxy_vs_ema20    DXY vs its 20-bar EMA
      evz_level       EVZ / 10.0  (FX vol regime; analogous to gvz_level)
      evz_vs_ma20     EVZ vs its 20-bar MA
      tlt_ret_1       TLT 1-bar return (US long-rate expectations)
      rate_diff_proxy tlt_ret_1 - shy_ret_1  (yield curve slope change; ECB-Fed proxy)
      cross_ret_1     GBP/USD 1-bar return
      corr_asset_dxy_20  rolling 20-bar corr of EUR/USD with DXY
      macro_tension   |dxy_ret_1| + |tlt_ret_1|
      real_yield_proxy   -tlt_ret_1
      vol_risk_premium   evz_level / rv_32 - 1  (implied vs realized vol spread)

    USD/JPY Group F (12):
      dxy_ret_1, dxy_ret_4, dxy_vs_ema20,
      tlt_ret_1, tlt_ret_4, spy_ret_1, spy_ret_4, spy_vs_ema20,
      cross_ret_1 (AUD/USD), corr_asset_dxy_20, macro_tension, real_yield_proxy

    Args:
        instrument:   "eurusd" or "usdjpy"
        lookback_warm: Bars to discard for indicator warm-up (default 250)
    """

    def __init__(self, instrument: str = "eurusd", lookback_warm: int = 250):
        if instrument not in _FOREX_REQUIRED_COLS:
            raise ValueError(
                f"ForexFeatureEngineer: unknown instrument '{instrument}'. "
                f"Choose from: {list(_FOREX_REQUIRED_COLS.keys())}"
            )
        self.instrument    = instrument
        self.lookback_warm = lookback_warm

        # Build instance-level FEATURE_NAMES (differs by instrument)
        group_a  = ["ret_1", "ret_4", "ret_8", "ret_16", "ret_32", "gap"]
        group_b  = _FOREX_GROUP_B_NAMES[instrument]
        group_c  = ["rsi_14", "macd_line", "macd_signal", "macd_hist",
                    "ema_ratio_8_21", "ema_ratio_21_55", "ema_ratio_55_200",
                    "adx_14", "roc_5", "roc_10", "roc_20", "stoch_k"]
        group_d  = ["close_vs_ema200", "close_vs_ema55", "close_vs_ema21",
                    "bar_body", "upper_wick", "lower_wick",
                    "close_position", "pivot_distance"]
        group_e  = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_london", "is_ny"]
        group_f  = _FOREX_GROUP_F_NAMES[instrument]

        self.FEATURE_NAMES = group_a + group_b + group_c + group_d + group_e + group_f
        assert len(self.FEATURE_NAMES) == 52, \
            f"Expected 52 features, got {len(self.FEATURE_NAMES)}"

    def compute(
        self,
        price_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> tuple:
        """
        Compute all 52 features for a forex pair.

        Args:
            price_df: DataFrame with open, high, low, close, volume columns
            macro_df: DataFrame with instrument-specific macro columns
                      (same index as price_df)

        Returns:
            features:   np.ndarray of shape (n_valid_bars, 52)
            timestamps: pd.DatetimeIndex of valid bar timestamps
            regimes:    np.ndarray or None
        """
        assert price_df.index.equals(macro_df.index), \
            "price_df and macro_df must have identical indices"

        required = _FOREX_REQUIRED_COLS[self.instrument]
        missing  = required - set(macro_df.columns)
        if missing:
            raise ValueError(
                f"macro_df missing columns for {self.instrument}: {missing}"
            )

        # Groups A-E (shared base)
        groups_ae, o, h, l, c, v, idx = self._compute_price_groups(price_df)

        # Group B override for EUR/USD: replace col 13 (parkinson_vol) with rv_64
        if self.instrument == "eurusd":
            ret_1_for_rv = self._log_returns(c, 1)
            rv_64 = self._rolling_std(ret_1_for_rv, 64)
            groups_ae[:, 13] = rv_64

        # Group F - per-instrument macro (12 features)
        print(f"  Computing Group F: Macro ({self.instrument})...")

        def _clean(arr: np.ndarray) -> np.ndarray:
            s = pd.Series(arr.astype(float))
            return s.ffill().bfill().fillna(0.0).values

        ret_1_price = self._log_returns(c, 1)
        dxy = _clean(macro_df["dxy"].values)
        tlt = _clean(macro_df["tlt"].values)

        dxy_ret_1    = self._log_returns(dxy, 1)
        dxy_ret_4    = self._log_returns(dxy, 4)
        dxy_ema20    = self._ema(dxy, 20)
        dxy_vs_ema20 = dxy / dxy_ema20 - 1
        tlt_ret_1    = self._log_returns(tlt, 1)

        corr_asset_dxy  = self._rolling_corr(ret_1_price, dxy_ret_1, 20)
        macro_tension   = np.abs(dxy_ret_1) + np.abs(tlt_ret_1)
        real_yield_proxy = -tlt_ret_1

        if self.instrument == "eurusd":
            evz    = _clean(macro_df["evz"].values)
            shy    = _clean(macro_df["shy"].values)
            cross  = _clean(macro_df["gbpusd"].values)

            evz_ma20    = self._rolling_mean(evz, 20)
            evz_level   = evz / 10.0
            evz_vs_ma20 = evz / (evz_ma20 + 1e-8) - 1

            shy_ret_1        = self._log_returns(shy, 1)
            rate_diff_proxy  = tlt_ret_1 - shy_ret_1

            cross_ret_1 = self._log_returns(cross, 1)

            rv_32 = self._rolling_std(ret_1_price, 32)
            vol_risk_premium = evz_level / (rv_32 + 1e-8) - 1

            group_f = np.column_stack([
                dxy_ret_1, dxy_ret_4, dxy_vs_ema20,
                evz_level, evz_vs_ma20,
                tlt_ret_1, rate_diff_proxy,
                cross_ret_1, corr_asset_dxy,
                macro_tension, real_yield_proxy, vol_risk_premium,
            ])

        else:  # usdjpy
            spy   = _clean(macro_df["spy"].values)
            cross = _clean(macro_df["audusd"].values)

            tlt_ret_4    = self._log_returns(tlt, 4)
            spy_ret_1    = self._log_returns(spy, 1)
            spy_ret_4    = self._log_returns(spy, 4)
            spy_ema20    = self._ema(spy, 20)
            spy_vs_ema20 = spy / spy_ema20 - 1
            cross_ret_1  = self._log_returns(cross, 1)

            group_f = np.column_stack([
                dxy_ret_1, dxy_ret_4, dxy_vs_ema20,
                tlt_ret_1, tlt_ret_4,
                spy_ret_1, spy_ret_4, spy_vs_ema20,
                cross_ret_1, corr_asset_dxy,
                macro_tension, real_yield_proxy,
            ])

        feature_matrix = np.hstack([groups_ae, group_f])

        assert feature_matrix.shape[1] == 52, \
            f"Expected 52 features, got {feature_matrix.shape[1]}"

        W = self.lookback_warm
        feature_matrix = feature_matrix[W:]
        timestamps     = price_df.index[W:]
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        regimes = None
        if "regime" in price_df.columns:
            regimes = price_df["regime"].values[W:]

        print(f"\n  Feature matrix shape: {feature_matrix.shape}")
        print(f"  NaN count after cleanup: {np.isnan(feature_matrix).sum()}")
        print(f"  Timestamp range: {timestamps[0]} -> {timestamps[-1]}")

        return feature_matrix, timestamps, regimes


# ===========================================================================
# M15 Gold Feature Engineer  (XAUUSD 15-minute bars)
# ===========================================================================

class M15GoldFeatureEngineer(GoldFeatureEngineer):
    """
    52-feature engineer for M15 XAUUSD bars.

    Two differences from GoldFeatureEngineer (H1):
      1. _pivot_distance uses a 96-bar rolling window (= 24 hours at M15,
         vs the H1 default of 24 bars = 24 hours at H1).
      2. Group E columns 36-37 are is_london_open / is_ny_open instead of
         dow_sin / dow_cos. These flag the first 30 minutes of London and
         NY session opens — the intraday events with the highest tactical
         significance at M15 resolution. Indices 38-39 (is_london, is_ny)
         are unchanged so heads.py index constants remain valid.
    """

    FEATURE_NAMES = [
        # Group A - Returns (6)
        "ret_1", "ret_4", "ret_8", "ret_16", "ret_32", "gap",
        # Group B - Volatility (8)
        "atr_14", "atr_50", "rv_8", "rv_16", "rv_32",
        "bb_width", "hl_range", "parkinson_vol",
        # Group C - Momentum / Trend (12)
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "ema_ratio_8_21", "ema_ratio_21_55", "ema_ratio_55_200",
        "adx_14", "roc_5", "roc_10", "roc_20", "stoch_k",
        # Group D - Price Structure (8)
        "close_vs_ema200", "close_vs_ema55", "close_vs_ema21",
        "bar_body", "upper_wick", "lower_wick",
        "close_position", "pivot_distance",
        # Group E - Session / Time (6)  ← cols 36-37 differ from H1
        "hour_sin", "hour_cos", "is_london_open", "is_ny_open",
        "is_london", "is_ny",
        # Group F - Macro / Gold-specific (12)
        "dxy_ret_1", "dxy_ret_4", "dxy_vs_ema20",
        "gvz_level", "gvz_vs_ma20",
        "tlt_ret_1", "tlt_ret_4",
        "silver_ret_1", "gold_silver_ratio",
        "corr_gold_dxy_20", "macro_tension", "real_yield_proxy",
    ]

    @staticmethod
    def _pivot_distance(high: np.ndarray, low: np.ndarray,
                        close: np.ndarray) -> np.ndarray:
        """96-bar rolling window = 24 hours at M15 resolution."""
        ser_h      = pd.Series(high)
        ser_l      = pd.Series(low)
        ser_c      = pd.Series(close)
        prev_high  = ser_h.rolling(96, min_periods=1).max().shift(1).fillna(high[0]).values
        prev_low   = ser_l.rolling(96, min_periods=1).min().shift(1).fillna(low[0]).values
        prev_close = ser_c.rolling(96, min_periods=1).mean().shift(1).fillna(close[0]).values
        pivot      = (prev_high + prev_low + prev_close) / 3.0
        return (close - pivot) / (close + 1e-8)

    def compute(
        self,
        price_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> tuple:
        """
        Compute all 52 M15 features.

        Calls parent for Groups A-D and F (which correctly uses the overridden
        _pivot_distance via Python's method resolution), then patches Group E
        cols 36-37 with M15-specific session open flags.
        """
        features, timestamps, regimes = super().compute(price_df, macro_df)

        # Replace cols 36-37 (dow_sin, dow_cos from H1) with session-open flags.
        # super().compute() has already trimmed lookback_warm leading bars.
        W      = self.lookback_warm
        idx    = price_df.index[W:]
        hour   = idx.hour.values.astype(float)
        minute = idx.minute.values.astype(float)

        # is_london_open: 07:00 and 07:15 UTC bars (first 30 min of London open)
        features[:, 36] = ((hour == 7)  & (minute <= 15)).astype(float)
        # is_ny_open:     12:00 and 12:15 UTC bars (first 30 min of NY open)
        features[:, 37] = ((hour == 12) & (minute <= 15)).astype(float)

        return features, timestamps, regimes


# ---------------------------------------------------------------------------
# Feature summary utility (for paper's Table 1)
# ---------------------------------------------------------------------------

def feature_summary(feature_matrix: np.ndarray, engineer=None) -> pd.DataFrame:
    """
    Compute summary statistics for all 52 features.
    """
    if engineer is None:
        names = GoldFeatureEngineer.FEATURE_NAMES
    else:
        names = engineer.FEATURE_NAMES
    df      = pd.DataFrame(feature_matrix, columns=names)
    summary = df.describe().T[["mean", "std", "min", "max"]]
    summary["nan_pct"] = df.isna().mean() * 100
    return summary.round(4)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.fetcher import make_realistic_ohlcv

    for instr in ["gold", "eurusd", "usdjpy"]:
        print("=" * 55)
        print(f"FeatureEngineer Validation  [{instr.upper()}]")
        print("=" * 55)

        n_bars = 5_000
        price_df, macro_df = make_realistic_ohlcv(n_bars=n_bars, instrument=instr)

        if instr == "gold":
            engineer = GoldFeatureEngineer(lookback_warm=250)
        else:
            engineer = ForexFeatureEngineer(instrument=instr, lookback_warm=250)

        features, timestamps, regimes = engineer.compute(price_df, macro_df)

        expected = n_bars - 250
        assert features.shape == (expected, 52), \
            f"Shape mismatch: {features.shape} vs ({expected}, 52)"
        assert np.isnan(features).sum() == 0, "NaN in features!"

        print(f"\nShape: {features.shape}  - OK")
        print(f"NaN: 0  - OK")
        print(f"Group F names: {engineer.FEATURE_NAMES[40:]}")
        print()
