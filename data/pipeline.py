"""
data/pipeline.py - Full Data Pipeline
======================================

Orchestrates the full journey from raw OHLCV to training-ready tensors
for any supported instrument (gold, eurusd, usdjpy):

    Raw OHLCV + Macro
         ->  fetcher.py
    Aligned DataFrames
         ->  features.py  (GoldFeatureEngineer or ForexFeatureEngineer)
    Raw feature matrix (n_bars, 52)
         ->  Rolling Z-score normalization  (this file)
    Normalized feature matrix (n_bars, 52)
         ->  Action vector construction     (this file)
    Action vectors (n_bars, 5)
         ->  Walk-forward train/val/test split (this file)
    MarketWindowDataset instances (from mwm_loss.py)

THE NORMALIZATION PROBLEM
--------------------------
Raw features have incompatible scales. Rolling z-score normalization:
    z_t = (x_t - mean(x_{t-W}...x_{t-1})) / std(x_{t-W}...x_{t-1})

W = 500 bars balances regime adaptation vs structural stability.
shift(1) ensures at bar t we use only PAST bars - no lookahead.

WALK-FORWARD SPLITS
-------------------
Time series require chronological (not random) splits:
    Train: 70% of available bars (oldest)
    Val:   15%
    Test:  15% (held out for final evaluation)

ACTION VECTORS
--------------
The "action" represents the macro event that causes the transition
from z_t to z_{t+1}. Always 5-dimensional so the same predictor
architecture works across instruments.

  Gold action:  [dxy_ret_1, gvz_change, tlt_ret_1, hour_sin, macro_tension]
  Forex action: [dxy_ret_1, spy_ret_1,  tlt_ret_1, hour_sin, macro_tension]
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, ".")

from data.fetcher   import DataFetcher, make_realistic_ohlcv, INSTRUMENT_CONFIGS
from data.features  import GoldFeatureEngineer, ForexFeatureEngineer
from models.mwm_loss import MarketWindowDataset


class DataPipeline:
    """
    Full data pipeline from raw data to PyTorch datasets.

    Args:
        instrument:     "gold", "eurusd", or "usdjpy"
        lookback:       Bars per observation window (default 48 = 2 days)
        norm_window:    Rolling normalization window in bars (default 500)
        lookback_warm:  Feature warm-up bars to discard (default 250)
        train_ratio:    Fraction for training (default 0.70)
        val_ratio:      Fraction for validation (default 0.15)
        cache_dir:      Directory for caching raw downloads
    """

    def __init__(
        self,
        instrument:    str   = "gold",
        lookback:      int   = 48,
        norm_window:   int   = 500,
        lookback_warm: int   = 250,
        train_ratio:   float = 0.70,
        val_ratio:     float = 0.15,
        cache_dir:     str   = "./data/cache",
    ):
        if instrument not in INSTRUMENT_CONFIGS:
            raise ValueError(
                f"Unknown instrument '{instrument}'. "
                f"Choose from: {list(INSTRUMENT_CONFIGS.keys())}"
            )
        self.instrument    = instrument
        self.lookback      = lookback
        self.norm_window   = norm_window
        self.lookback_warm = lookback_warm
        self.train_ratio   = train_ratio
        self.val_ratio     = val_ratio
        self.test_ratio    = 1.0 - train_ratio - val_ratio
        self.cache_dir     = cache_dir

        self.fetcher = DataFetcher(cache_dir=cache_dir)

        if instrument == "gold":
            self.engineer = GoldFeatureEngineer(lookback_warm=lookback_warm)
        else:
            self.engineer = ForexFeatureEngineer(
                instrument=instrument, lookback_warm=lookback_warm
            )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build(
        self,
        start:         str  = None,
        end:           str  = None,
        use_real_data: bool = True,
        use_cache:     bool = True,
        stride:        int  = 1,
        history_len:   int  = 1,
        split_dates:   dict = None,
    ) -> dict:
        """
        Run the full pipeline and return train/val/test datasets.

        Args:
            start, end:     Date range for data (None = Yahoo's dynamic 730-day window)
            use_real_data:  If False, uses synthetic data (for testing)
            use_cache:      Cache downloaded data to disk
            stride:         Window stride (1 = every bar)
            history_len:    Consecutive history windows per sample (for predictor)
            split_dates:    Optional date-based splits, e.g.
                            {"train_end": "2024-10-31", "val_end": "2024-11-14"}.
                            When set, overrides the ratio-based 70/15/15 split.

        Returns:
            {
              "train": MarketWindowDataset,
              "val":   MarketWindowDataset,
              "test":  MarketWindowDataset,
              "meta":  dict with timestamps, split indices, feature names, etc.
            }
        """
        print("\n" + "=" * 55)
        print(f"DataPipeline.build()  [{self.instrument.upper()}]")
        print("=" * 55)

        # Step 1: Fetch data
        print("\n[1/5] Fetching market data...")
        if use_real_data:
            data     = self.fetcher.fetch_all(
                start, end, use_cache=use_cache, instrument=self.instrument
            )
            price_df = data["price"]
            macro_df = data["macro"]
        else:
            print(f"  Using synthetic data (use_real_data=False)  [{self.instrument.upper()}]")
            if start is not None and end is not None:
                days   = (pd.Timestamp(end) - pd.Timestamp(start)).days
                n_bars = int(days * 24 * 5/7 * 0.85)
            else:
                n_bars = 10_000
            price_df, macro_df = make_realistic_ohlcv(
                n_bars=n_bars, seed=42, instrument=self.instrument
            )

        print(f"  Raw bars: {len(price_df):,}")

        return self.build_from_frames(
            price_df, macro_df,
            stride       = stride,
            history_len  = history_len,
            split_dates  = split_dates,
        )

    def build_from_frames(
        self,
        price_df:     pd.DataFrame,
        macro_df:     pd.DataFrame,
        stride:       int  = 1,
        history_len:  int  = 1,
        split_dates:  dict = None,
    ) -> dict:
        """
        Run steps 2–5 of the pipeline on already-fetched OHLCV + macro frames.

        Identical feature engineering, action construction, rolling
        normalization and walk-forward split as build(), but the caller
        supplies the raw data. Use this to train on data sources that
        bypass the Yahoo 730-day cap — e.g. H1 bars pulled straight from MT5.

        Args:
            price_df:  OHLCV DataFrame indexed by bar timestamp.
            macro_df:  Macro DataFrame aligned to price_df.index, containing the
                       columns this instrument's feature engineer expects
                       (gold: dxy, gvz, tlt, silver).
            stride, history_len, split_dates: as in build().
        """
        # Step 2: Feature engineering
        print("\n[2/5] Computing 52 features...")
        raw_features, timestamps, regimes = self.engineer.compute(price_df, macro_df)
        print(f"  After warm-up trim: {raw_features.shape}")

        # Step 3: Action vectors
        print("\n[3/5] Building action vectors...")
        if self.instrument == "gold":
            actions = self._build_actions_gold(price_df, macro_df, timestamps)
        else:
            actions = self._build_actions_forex(price_df, macro_df, timestamps)
        print(f"  Action shape: {actions.shape}  (5 macro signals per bar)")

        # Close prices aligned with the feature rows (warm-up bars already dropped).
        # Used by TradingEnvironment for PnL computation — not for model inputs.
        n_trim = len(price_df) - len(raw_features)
        prices_trimmed = price_df["close"].values[n_trim:].astype(np.float32)

        # Step 4: Rolling normalization
        print("\n[4/5] Applying rolling z-score normalization...")
        norm_features = self._rolling_zscore(raw_features)
        n_clipped     = (np.abs(norm_features) > 5).sum()
        norm_features = np.clip(norm_features, -5, 5)
        print(f"  Clipped {n_clipped:,} values beyond +-5 sigma")
        print(f"  Normalized stats: mean={norm_features.mean():.4f}  std={norm_features.std():.4f}")

        # Step 5: Walk-forward split
        print("\n[5/5] Creating walk-forward train/val/test splits...")
        n = len(norm_features)
        if split_dates is not None:
            ts          = pd.DatetimeIndex(timestamps)

            # Coerce split bounds to the timestamp index's tz. yfinance data is
            # tz-naive; MT5 data (via build_from_frames) is tz-aware UTC. Matching
            # the tz keeps the <= comparison valid for both sources.
            def _coerce(bound):
                b = pd.Timestamp(bound)
                if ts.tz is not None:
                    b = b.tz_localize("UTC") if b.tzinfo is None else b.tz_convert("UTC")
                elif b.tzinfo is not None:
                    b = b.tz_localize(None)
                return b

            train_end   = _coerce(split_dates["train_end"])
            val_end     = _coerce(split_dates["val_end"])
            n_train     = int((ts <= train_end).sum())
            n_val       = int(((ts > train_end) & (ts <= val_end)).sum())
            n_test      = n - n_train - n_val
            print(f"  Using date-based splits: train_end={split_dates['train_end']}, "
                  f"val_end={split_dates['val_end']}")
        else:
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)
            n_test  = n - n_train - n_val

        train_feat = norm_features[:n_train]
        train_act  = actions[:n_train]

        val_feat   = norm_features[n_train : n_train + n_val]
        val_act    = actions[n_train : n_train + n_val]

        test_feat  = norm_features[n_train + n_val:]
        test_act   = actions[n_train + n_val:]

        print(f"  Train: {n_train:,} bars  ({timestamps[0].date()} -> {timestamps[n_train-1].date()})")
        print(f"  Val:   {n_val:,} bars  ({timestamps[n_train].date()} -> {timestamps[n_train+n_val-1].date()})")
        print(f"  Test:  {n_test:,} bars  ({timestamps[n_train+n_val].date()} -> {timestamps[-1].date()})")

        # Warn when val window is too small to produce any samples.
        # Minimum bars needed = lookback * (history_len + 1); at stride=4 even more.
        min_val_bars = self.lookback * (history_len + 1)
        if n_val < min_val_bars:
            shortfall = min_val_bars - n_val
            print(f"\n  WARNING: Val window has {n_val} bars but needs >= {min_val_bars} "
                  f"to form samples (short by {shortfall}).")
            print(f"    Val dataset will have 0 samples — checkpointing will use train loss.")
            print(f"    Fix: extend val_end by at least {shortfall // 13 + 1} trading days, "
                  f"or use ratio-based splits.")

        train_ds = MarketWindowDataset(train_feat, train_act, self.lookback, stride,  history_len)
        val_ds   = MarketWindowDataset(val_feat,   val_act,   self.lookback, 4,       history_len)
        test_ds  = MarketWindowDataset(test_feat,  test_act,  self.lookback, stride,  history_len)

        print("\n" + "-" * 40)
        print("Pipeline complete.")
        print(f"  Train samples: {len(train_ds):,}")
        print(f"  Val samples:   {len(val_ds):,}")
        print(f"  Test samples:  {len(test_ds):,}")
        print(f"  Feature dim:   52")
        print(f"  Action dim:    5")
        print(f"  Lookback:      {self.lookback} bars")
        print("-" * 40)

        return {
            "train": train_ds,
            "val":   val_ds,
            "test":  test_ds,
            "meta": {
                "instrument":    self.instrument,
                "timestamps":    timestamps,
                "regimes":       regimes,
                "feature_names": self.engineer.FEATURE_NAMES,
                "n_train":       n_train,
                "n_val":         n_val,
                "n_test":        n_test,
                "raw_features":  raw_features,
                "norm_features": norm_features,
                "prices":        prices_trimmed,
                "macro_vecs":    actions,
            }
        }

    # -----------------------------------------------------------------------
    # Action vector builders
    # -----------------------------------------------------------------------

    def _build_actions_gold(
        self,
        price_df:   pd.DataFrame,
        macro_df:   pd.DataFrame,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        5-D action vector for gold.

        Dimensions:
            [0] dxy_ret_1:     DXY 1-bar log return
            [1] gvz_change:    GVZ fractional change (vol regime signal)
            [2] tlt_ret_1:     TLT 1-bar return (real yield proxy)
            [3] hour_sin:      sin(2*pi*hour/24)  (session effect)
            [4] macro_tension: |dxy_ret| + |tlt_ret| (stress indicator)
        """
        W = self.lookback_warm

        def _clean(arr):
            s = pd.Series(arr.astype(float))
            return s.ffill().bfill().fillna(0.0).values

        dxy = _clean(macro_df["dxy"].values)
        gvz = _clean(macro_df["gvz"].values)
        tlt = _clean(macro_df["tlt"].values)

        dxy_ret = np.log(np.maximum(dxy / np.roll(dxy, 1), 1e-10)); dxy_ret[0] = 0.0
        tlt_ret = np.log(np.maximum(tlt / np.roll(tlt, 1), 1e-10)); tlt_ret[0] = 0.0
        gvz_chg = np.diff(gvz, prepend=gvz[0]) / (gvz + 1e-8)

        hour     = pd.DatetimeIndex(macro_df.index).hour.values.astype(float)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        macro_tension = np.abs(dxy_ret) + np.abs(tlt_ret)

        actions = np.column_stack([
            dxy_ret[W:], gvz_chg[W:], tlt_ret[W:],
            hour_sin[W:], macro_tension[W:],
        ])
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions = (actions - actions.mean(axis=0)) / (actions.std(axis=0) + 1e-8)
        assert np.isnan(actions).sum() == 0
        return actions.astype(np.float32)

    def _build_actions_forex(
        self,
        price_df:   pd.DataFrame,
        macro_df:   pd.DataFrame,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        5-D action vector for forex.

        EUR/USD dimensions:
            [0] dxy_ret_1:   DXY 1-bar log return
            [1] evz_change:  EVZ fractional change (FX vol regime signal)
            [2] tlt_ret_1:   TLT 1-bar return (US rate expectations)
            [3] hour_sin:    sin(2*pi*hour/24)
            [4] macro_tension: |dxy_ret| + |tlt_ret|

        USD/JPY dimensions:
            [0] dxy_ret_1:   DXY 1-bar log return
            [1] spy_ret_1:   SPY 1-bar return (risk sentiment)
            [2] tlt_ret_1:   TLT 1-bar return
            [3] hour_sin:    sin(2*pi*hour/24)
            [4] macro_tension: |dxy_ret| + |tlt_ret|
        """
        W = self.lookback_warm

        def _clean(arr):
            s = pd.Series(arr.astype(float))
            return s.ffill().bfill().fillna(0.0).values

        dxy = _clean(macro_df["dxy"].values)
        tlt = _clean(macro_df["tlt"].values)

        dxy_ret = np.log(np.maximum(dxy / np.roll(dxy, 1), 1e-10)); dxy_ret[0] = 0.0
        tlt_ret = np.log(np.maximum(tlt / np.roll(tlt, 1), 1e-10)); tlt_ret[0] = 0.0

        hour     = pd.DatetimeIndex(macro_df.index).hour.values.astype(float)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        macro_tension = np.abs(dxy_ret) + np.abs(tlt_ret)

        if self.instrument == "eurusd":
            evz     = _clean(macro_df["evz"].values)
            evz_chg = np.diff(evz, prepend=evz[0]) / (evz + 1e-8)
            col1    = evz_chg
        else:
            spy     = _clean(macro_df["spy"].values)
            spy_ret = np.log(np.maximum(spy / np.roll(spy, 1), 1e-10)); spy_ret[0] = 0.0
            col1    = spy_ret

        actions = np.column_stack([
            dxy_ret[W:], col1[W:], tlt_ret[W:],
            hour_sin[W:], macro_tension[W:],
        ])
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions = (actions - actions.mean(axis=0)) / (actions.std(axis=0) + 1e-8)
        assert np.isnan(actions).sum() == 0
        return actions.astype(np.float32)

    def _rolling_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Rolling z-score normalization with shift(1) to prevent lookahead.

        At bar t, normalizes using bars t-W through t-1 only.
        """
        df        = pd.DataFrame(X)
        W         = self.norm_window
        roll_mean = df.rolling(W, min_periods=2).mean().shift(1)
        roll_std  = df.rolling(W, min_periods=2).std().shift(1)
        Z = ((df - roll_mean) / (roll_std + 1e-8)).fillna(0).values.astype(np.float32)
        return Z


if __name__ == "__main__":
    """Quick smoke test for all three instruments."""
    import torch
    from torch.utils.data import DataLoader

    for instr in ["gold", "eurusd", "usdjpy"]:
        print("\n" + "=" * 55)
        print(f"DataPipeline Smoke Test  [{instr.upper()}]")
        print("=" * 55)

        pipeline = DataPipeline(instrument=instr, lookback=48, norm_window=500)
        result   = pipeline.build(use_real_data=False, stride=1, history_len=3)

        sample = result["train"][0]
        assert sample["x_hist"].shape == (3, 48, 52), \
            f"x_hist shape wrong: {sample['x_hist'].shape}"
        assert sample["x_t1"].shape == (48, 52)
        assert sample["a_t"].shape  == (5,)

        loader = DataLoader(result["train"], batch_size=128, shuffle=False)
        batch  = next(iter(loader))
        x      = batch["x_t"]
        print(f"\n  x_t mean: {x.mean().item():.4f}  std: {x.std().item():.4f}")
        print(f"  x_t range: [{x.min().item():.2f}, {x.max().item():.2f}]")
        print(f"  action dim: {batch['a_t'].shape[1]}")
        print(f"  [{instr.upper()}] OK")
