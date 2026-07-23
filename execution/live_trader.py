"""
execution/live_trader.py — Production H1 bar-close loop for MWM gold trading.

Architecture
------------
On every H1 bar close:
  1.  Fetch the last `lookback + warm_up_bars` gold OHLCV bars from MT5.
  2.  Fetch macro data (DXY, GVZ, TLT, Silver) from Yahoo Finance.
  3.  Re-run the full feature pipeline on the rolling window.
  4.  Apply rolling z-score normalisation (500-bar window, no lookahead).
  5.  Encode the current 48-bar window -> z_t.
  6.  Compute S_t = MSE(z_hat_{t}, z_t) using the forecast from the prior bar.
  7.  Apply the four-zone S_t circuit breaker.
  8.  If entering: compute ATR-based lot size via position_sizing.
  9.  Send order to MT5 or close existing position.

Usage
-----
    python -m execution.live_trader \
        --checkpoint  ./experiments/checkpoints/best_model.pt \
        --heads-dir   ./experiments/heads_gold \
        --symbol      XAUUSD \
        --magic       20260101 \
        --risk-frac   0.01 \
        --spread      0.0003 \
        --dir-thresh  0.53 \
        --surprise-warmup 100 \
        --dry-run

Design notes
------------
- Rolling state (z_deque, surprise_history) is kept in Python memory.
  On restart, warm_up() rebuilds it from the last warm_up_bars candles.
- Macro data for GVZ/TLT has lower frequency (equity hours); missing hours
  are forward-filled exactly as the training pipeline does.
- The feature pipeline runs on a rolling window of warm_up_bars + _LOOKBACK
  bars so z-score normalisation is stable from the first live bar.
- All times are UTC.  MT5 server time must also be UTC.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOOKBACK     = 48    # H1 MarketEncoder window
_HISTORY_LEN  = 3     # CausalPredictor history depth
_Z_SCORE_WIN  = 500   # Rolling z-score normalisation window
_WARM_UP_BARS = 250   # Extra bars fetched to stabilise z-score + surprise
_Z_DIM        = 128

_M15_LOOKBACK    = 96    # M15 MarketEncoder window
_M15_WARM_BARS   = 250   # M15 warm-up bars (mirrors M15_CFG)
_M15_PCT_GATE    = 20    # S_t percentile gate for M15 execution (p20)

# Neutral macro levels used only when a macro source returns nothing AND we have
# no cached last-known value. Constant levels yield zero-signal macro features
# (returns≈0, ratios≈1) — a clean no-op — instead of the 0.0 fill that made every
# DXY-derived feature degenerate and blew up S_t. See _align_macro.
_MACRO_NEUTRAL = {"dxy": 100.0, "gvz": 15.0, "tlt": 90.0, "silver": 25.0}

# Four-zone circuit-breaker floors — lower bounds on the rolling p20/p60/p90 zone
# boundaries. Set just below the post-fix S_t distribution (replay 2026-05/06:
# p20=0.0001 p60=0.0003 p90=0.0014) so the rolling percentiles drive zone
# assignment in normal conditions and the floors only catch an early, unrepresentative
# buffer. The previous 0.0005/0.003/0.010 were calibrated against the pre-fix
# inflated S_t; against the now-calm distribution they sat ABOVE p90, collapsing
# nearly every bar into Q1 (breaker never fires). See project_live_st_spike_rootcause.
#
# Set JUST BELOW the replay percentiles (p20=0.00009 p60=0.00030 p90=0.00141) so the
# rolling percentiles drive zone assignment in normal conditions (floors bind <11% of
# bars) and only catch an early/calm unrepresentative buffer. Verified occupancy on the
# 2026-05/06 replay: Q1 22% / Q2-skip 42% / Q4-half 26% / Q5-CB 11% (~the 20/40/30/10
# p20/p60/p90 design). Re-check with scripts/check_zone_occupancy.py if S_t shifts.
_CB_FLOOR_P20 = 0.00008
_CB_FLOOR_P60 = 0.00028
_CB_FLOOR_P90 = 0.0013


# ---------------------------------------------------------------------------
# Rolling normaliser (mirrors data/pipeline.py)
# ---------------------------------------------------------------------------

def _rolling_zscore(df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """
    Causal rolling z-score normalisation.  shift(1) means bar t is normalised
    using statistics from bars 0..t-1 only — no lookahead.
    """
    roll  = df.rolling(window, min_periods=20)
    mu    = roll.mean().shift(1).bfill()
    sigma = roll.std().shift(1).bfill().fillna(1.0)
    sigma = sigma.clip(lower=1e-8)
    return (df - mu).div(sigma).clip(-5, 5)


# ---------------------------------------------------------------------------
# LiveTrader
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Full production pipeline that runs on every H1 gold bar close.

    Args:
        encoder:        Frozen MarketEncoder.
        predictor:      Frozen CausalPredictor.
        layer:          SupervisedExecutionLayer (loaded heads).
        connector:      MT5Connector.
        risk_frac:      Fraction of equity risked per ATR of adverse move.
        spread:         One-way spread fraction (for signal cost filter).
        dir_threshold:  Directional head threshold (long if prob > thresh).
        surprise_warmup: Bars before circuit breaker activates.
        warm_up_bars:   Extra history bars fetched for stable z-score / ATR.
        dry_run:        If True, log signals but never send MT5 orders.
        device:         Torch device string.
    """

    def __init__(
        self,
        encoder,
        predictor,
        layer,
        connector,
        risk_frac:       float = 0.01,
        spread:          float = 0.0003,
        dir_threshold:   float = 0.53,
        surprise_warmup: int   = 100,
        warm_up_bars:    int   = _WARM_UP_BARS,
        dry_run:         bool  = False,
        device:          str   = "cpu",
        # Config D — M15 execution layer (optional)
        m15_encoder              = None,
        m15_predictor            = None,
        m15_st_threshold: float  = 0.0,   # pre-computed p20 from M15 training set
    ):
        self.encoder         = encoder
        self.predictor       = predictor
        self.layer           = layer
        self.connector       = connector
        self.risk_frac       = risk_frac
        self.spread          = spread
        self.dir_threshold   = dir_threshold
        self.surprise_warmup = surprise_warmup
        self.warm_up_bars    = warm_up_bars
        self.dry_run         = dry_run
        self.device          = torch.device(device)

        self.encoder.to(self.device).eval()
        self.predictor.to(self.device).eval()
        self.layer.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        for p in self.layer.parameters():
            p.requires_grad_(False)

        # Mutable rolling state — populated by warm_up()
        self.z_deque:          deque = deque(maxlen=_HISTORY_LEN)
        self.z_hat_current:    Optional[torch.Tensor] = None
        self.surprise_history: list  = []
        self.current_position: float = 0.0
        self._bar_history:     list  = []   # last N bar dicts for dashboard
        self._status_path      = Path("experiments/live_status.json")
        self.post_shock_bars_remaining = 0
        self._last_macro:      dict  = {}   # last-known macro level per name (fallback fill)

        # Config D — M15 execution layer state
        self.m15_encoder       = m15_encoder
        self.m15_predictor     = m15_predictor
        self.m15_st_threshold  = m15_st_threshold
        self._m15_use          = m15_encoder is not None and m15_st_threshold > 0
        # M15 rolling state (populated by m15_warm_up)
        self._m15_z_deque:         deque = deque(maxlen=_HISTORY_LEN)
        self._m15_z_hat:           Optional[torch.Tensor] = None
        self._m15_surprise:        list  = []
        # Pending entry state: H1 signalled, waiting for M15 calm bar
        self._m15_pending:         bool  = False
        self._m15_pending_dir:     float = 0.0
        self._m15_pending_h1px:    float = 0.0   # H1 bar close price (fallback reference)
        self._m15_pending_signal:  float = 0.0   # effective H1 signal magnitude
        self._m15_pending_lots:    float = 0.0

        if self._m15_use:
            self.m15_encoder.to(self.device).eval()
            self.m15_predictor.to(self.device).eval()
            for p in self.m15_encoder.parameters():
                p.requires_grad_(False)
            for p in self.m15_predictor.parameters():
                p.requires_grad_(False)
            logger.info("Config D M15 execution layer enabled  S_t gate=%.6f", m15_st_threshold)

    # ------------------------------------------------------------------
    # Warm-up — rebuild rolling state from recent history
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Fetch history and rebuild z_deque + surprise_history from scratch.
        Must be called once before the first on_bar_close().

        Mirrors TradingEnvironment exactly:
          bar 0: init deque with z[0] repeated H times, predict z[1].
          bar i: compute S_t = MSE(z_hat, z[i]); push z[i]; predict z[i+1].
        """
        # Fetch deep enough that, after the engineer trims its warm-up, a full
        # _Z_SCORE_WIN of rows remains for the rolling z-score (matching offline
        # normalisation), plus extra bars to replay a long surprise history.
        n_fetch = _Z_SCORE_WIN + 2 * self.warm_up_bars + _LOOKBACK
        logger.info("Warming up: fetching %d bars for %s ...", n_fetch, self.connector.symbol)

        ohlcv, features, macro_vecs = self._fetch_and_engineer(n_bars=n_fetch)
        n_valid = len(features) - _LOOKBACK + 1
        zs      = self._encode_all(features, n_valid)

        # Reset state
        self.z_deque          = deque(maxlen=_HISTORY_LEN)
        self.surprise_history = []

        # Bar 0: seed deque, produce first forecast
        z0 = torch.tensor(zs[0], dtype=torch.float32, device=self.device)
        for _ in range(_HISTORY_LEN):
            self.z_deque.append(z0.clone())

        a0 = macro_vecs[_LOOKBACK - 1]
        self.z_hat_current = self._predict_from_deque(a0)

        # Bars 1 .. n_valid-1: advance incrementally
        for i in range(1, n_valid):
            z_i = torch.tensor(zs[i], dtype=torch.float32, device=self.device)

            S_t = float(((self.z_hat_current - z_i) ** 2).mean().item())
            self.surprise_history.append(S_t)

            self.z_deque.append(z_i)

            bar_idx = min(i + _LOOKBACK - 1, len(macro_vecs) - 1)
            self.z_hat_current = self._predict_from_deque(macro_vecs[bar_idx])

        logger.info("Warm-up complete. Surprise buffer: %d bars.", len(self.surprise_history))
        self._write_status({"status": "waiting", "message": "Warm-up complete — awaiting first bar close."})

    # ------------------------------------------------------------------
    # Main tick handler
    # ------------------------------------------------------------------

    def on_bar_close(self) -> dict:
        """
        Process one completed H1 bar.  Call this at each H1 bar close.

        Returns a dict with keys:
            timestamp, signal, effective_signal, lots, S_t, atr_pct, equity,
            dir_prob, vol_regime, action_taken.
        """
        # Full _Z_SCORE_WIN of rows must survive the engineer's warm-up trim so
        # the current bar is normalised over the same 500-bar window as offline.
        # Under-fetching here normalised over ~48 bars -> inflated z -> S_t spikes.
        n_fetch = _Z_SCORE_WIN + self.warm_up_bars + _LOOKBACK
        ohlcv, features, macro_vecs = self._fetch_and_engineer(n_bars=n_fetch)

        # Encode the current bar (last valid window)
        n_valid  = len(features) - _LOOKBACK + 1
        z_t      = self._encode_single(features, n_valid - 1)

        # Compute S_t against the forecast produced on the previous bar
        S_t = float(((self.z_hat_current - z_t) ** 2).mean().item())
        self.surprise_history.append(S_t)
        self.z_deque.append(z_t)

        # ATR-based position size (from unnormalised OHLCV)
        atr_pct = self._compute_atr_pct(ohlcv)

        z_cpu    = z_t.unsqueeze(0).cpu()
        z_hist_t = self._stack_history(list(self.z_deque)).unsqueeze(0).cpu()
        raw_out  = self.layer(z_cpu, z_hist_t)
        dir_prob  = float(raw_out["dir_prob"][0])
        vol_regime = int(raw_out["vol_regime"][0])

        signal_t = self.layer.entry_signal(
            z_cpu, z_hist_t,
            dir_threshold=self.dir_threshold,
            atr_pct=atr_pct,
            risk_frac=self.risk_frac,
        )
        raw_signal = float(signal_t[0])

        # Four-zone S_t circuit breaker (manages post_shock_bars_remaining)
        effective_signal = self._apply_circuit_breaker(raw_signal, S_t)

        # Post-shock cooldown gate
        if self.post_shock_bars_remaining > 0:
            bars_elapsed = 8 - self.post_shock_bars_remaining
            logger.info("Post-shock cooldown: %d/8 bars elapsed — skipping entry", bars_elapsed)
            effective_signal = 0.0

        # Advance predictor state for next bar
        last_bar = min(n_valid - 1 + _LOOKBACK - 1, len(macro_vecs) - 1)
        self.z_hat_current = self._predict_from_deque(macro_vecs[last_bar])

        # MT5 lot size
        equity = self.connector.get_equity()
        price  = self.connector.get_current_price()
        size   = abs(effective_signal)
        lots   = 0.0
        if size > 1e-6:
            from execution.position_sizing import lots_from_size
            lots = lots_from_size(size, equity, price)

        # Config D: defer new entries to M15 execution layer when enabled.
        # If M15 is active and H1 just signalled a new entry from flat,
        # pend the entry rather than executing at H1 bar close.
        new_entry = (abs(effective_signal) > 1e-6
                     and abs(self.current_position) < 1e-6
                     and abs(effective_signal) > abs(self.current_position))
        if self._m15_use and new_entry:
            self._m15_pending        = True
            self._m15_pending_dir    = float(np.sign(effective_signal))
            self._m15_pending_h1px   = float(price)
            self._m15_pending_signal = effective_signal
            self._m15_pending_lots   = lots
            action_taken = f"m15_pending(dir={'LONG' if effective_signal>0 else 'SHORT'} lots={lots:.2f})"
        else:
            # Fallback: also execute H1 if pending M15 entry found nothing this hour
            if self._m15_use and self._m15_pending:
                fb_signal = self._m15_pending_signal
                fb_lots   = self._m15_pending_lots
                self._m15_pending = False
                if abs(effective_signal) < 1e-6:
                    logger.info("H1 fallback skipped — signal dissolved during M15 scan")
                    action_taken = "m15_fallback_skipped(signal_dissolved)"
                else:
                    action_taken = self._execute(fb_signal, fb_lots) + " [h1_fallback]"
                    logger.info("M15 found no calm bar — executed H1 fallback: %s", action_taken)
            else:
                action_taken = self._execute(effective_signal, lots)

        info = {
            "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
            "signal":           raw_signal,
            "effective_signal": effective_signal,
            "lots":             lots,
            "S_t":              S_t,
            "atr_pct":          atr_pct,
            "equity":           equity,
            "dir_prob":         dir_prob,
            "vol_regime":       vol_regime,
            "action_taken":     action_taken,
        }
        logger.info(
            "Bar close: dir_prob=%.3f  S_t=%.5f  signal=%.3f  eff=%.3f  lots=%.2f  action=%s",
            dir_prob, S_t, raw_signal, effective_signal, lots, action_taken,
        )
        self._bar_history.append(info)
        self._bar_history = self._bar_history[-50:]
        self._write_status({"status": "running"})
        return info

    # ------------------------------------------------------------------
    # Config D — M15 execution layer
    # ------------------------------------------------------------------

    def m15_warm_up(self) -> None:
        """Warm up M15 surprise history from recent M15 OHLCV bars."""
        if not self._m15_use:
            return
        from data.features import M15GoldFeatureEngineer

        n_fetch = _Z_SCORE_WIN + 2 * _M15_WARM_BARS + _M15_LOOKBACK
        logger.info("M15 warm-up: fetching %d M15 bars...", n_fetch)
        m15_ohlcv = self.connector.fetch_ohlcv_bars(n=n_fetch, timeframe="M15")
        macro_data = self.connector.fetch_macro_data(n=n_fetch)
        macro_df   = self._align_macro(macro_data, m15_ohlcv.index)

        engineer = M15GoldFeatureEngineer(lookback_warm=_M15_WARM_BARS)
        raw_feat, timestamps, _ = engineer.compute(m15_ohlcv, macro_df)
        macro_trimmed = macro_df.loc[timestamps]
        macro_vecs    = self._build_action_vecs(macro_trimmed)

        feat_df  = pd.DataFrame(raw_feat, index=timestamps)
        features = _rolling_zscore(feat_df, window=_Z_SCORE_WIN).values.astype(np.float32)

        n_valid = len(features) - _M15_LOOKBACK + 1
        self._m15_z_deque   = deque(maxlen=_HISTORY_LEN)
        self._m15_surprise  = []

        z0 = self._m15_encode_single(features, 0)
        for _ in range(_HISTORY_LEN):
            self._m15_z_deque.append(z0.clone())
        self._m15_z_hat = self._m15_predict(macro_vecs[_M15_LOOKBACK - 1])

        for i in range(1, n_valid):
            z_i = self._m15_encode_single(features, i)
            S_t = float(((self._m15_z_hat - z_i) ** 2).mean().item())
            self._m15_surprise.append(S_t)
            self._m15_z_deque.append(z_i)
            bar_idx = min(i + _M15_LOOKBACK - 1, len(macro_vecs) - 1)
            self._m15_z_hat = self._m15_predict(macro_vecs[bar_idx])

        logger.info("M15 warm-up complete. Surprise buffer: %d bars.", len(self._m15_surprise))

    def on_m15_bar_close(self) -> Optional[str]:
        """
        Called at each M15 bar close (:15, :30, :45 past the hour) when a
        pending H1 entry is waiting for M15 execution.

        Returns an action string if execution fired, else None.
        """
        if not self._m15_use or not self._m15_pending:
            return None

        from data.features import M15GoldFeatureEngineer

        # Fetch enough M15 bars for a full _Z_SCORE_WIN after the warm-up trim,
        # so the fresh S_t uses the same normalisation as offline.
        n_fetch = _Z_SCORE_WIN + _M15_WARM_BARS + _M15_LOOKBACK
        try:
            m15_ohlcv  = self.connector.fetch_ohlcv_bars(n=n_fetch, timeframe="M15")
            macro_data = self.connector.fetch_macro_data(n=n_fetch)
            macro_df   = self._align_macro(macro_data, m15_ohlcv.index)

            engineer = M15GoldFeatureEngineer(lookback_warm=_M15_WARM_BARS)
            raw_feat, timestamps, _ = engineer.compute(m15_ohlcv, macro_df)
            feat_df  = pd.DataFrame(raw_feat, index=timestamps)
            features = _rolling_zscore(feat_df, window=_Z_SCORE_WIN).values.astype(np.float32)

            z_t = self._m15_encode_single(features, len(features) - _M15_LOOKBACK)
            S_t = float(((self._m15_z_hat - z_t) ** 2).mean().item())
            self._m15_surprise.append(S_t)
            self._m15_z_deque.append(z_t)
            macro_vecs = self._build_action_vecs(macro_df.loc[timestamps])
            self._m15_z_hat = self._m15_predict(macro_vecs[-1])
        except Exception as exc:
            logger.warning("M15 bar check failed: %s", exc)
            return None

        logger.info("M15 bar: S_t=%.6f  gate=%.6f  pending=%s",
                    S_t, self.m15_st_threshold,
                    "LONG" if self._m15_pending_dir > 0 else "SHORT")

        if S_t >= self.m15_st_threshold:
            return None   # M15 not calm — wait for next M15 bar

        # M15 is calm — check adverse guard
        m15_price = self.connector.get_current_price()
        direction = self._m15_pending_dir
        h1_price  = self._m15_pending_h1px

        adverse = (direction > 0 and m15_price > h1_price) or \
                  (direction < 0 and m15_price < h1_price)
        if adverse:
            logger.info("M15 bar calm but adverse (dir=%s m15=%.2f h1=%.2f) — waiting",
                        "LONG" if direction > 0 else "SHORT", m15_price, h1_price)
            return None   # wait for better bar

        # Execute at M15 price
        signal = self._m15_pending_signal
        lots   = self._m15_pending_lots
        self._m15_pending = False

        action = self._execute(signal, lots)
        impr   = abs(h1_price - m15_price)
        logger.info(
            "M15 EXECUTION: %s  m15_price=%.2f  h1_ref=%.2f  improvement=$%.2f/oz  S_t=%.6f",
            action, m15_price, h1_price, impr, S_t,
        )
        return action

    def _m15_encode_single(self, features: np.ndarray, valid_idx: int) -> torch.Tensor:
        feat_t = torch.tensor(
            features[valid_idx : valid_idx + _M15_LOOKBACK],
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            return self.m15_encoder(feat_t.unsqueeze(0)).squeeze(0)

    def _m15_predict(self, action: np.ndarray) -> torch.Tensor:
        z_hist = self._stack_history(list(self._m15_z_deque)).unsqueeze(0)
        a_t    = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.m15_predictor(z_hist, a_t).squeeze(0)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _fetch_and_engineer(self, n_bars: int):
        """
        Fetch OHLCV + macro, run the gold feature engineer, apply rolling
        z-score normalisation.

        Returns:
            ohlcv       pd.DataFrame  OHLCV bars aligned to feature rows
            features    np.ndarray   (n_valid, 52) normalised features
            macro_vecs  np.ndarray   (n_valid, 5)  action vectors
        """
        from data.features import GoldFeatureEngineer

        ohlcv_raw  = self.connector.fetch_ohlcv_bars(n=n_bars)
        macro_data = self.connector.fetch_macro_data(n=n_bars)

        macro_df = self._align_macro(macro_data, ohlcv_raw.index)

        engineer = GoldFeatureEngineer(lookback_warm=self.warm_up_bars)
        raw_feat, timestamps, _ = engineer.compute(ohlcv_raw, macro_df)

        ohlcv_trimmed = ohlcv_raw.loc[timestamps]
        macro_trimmed = macro_df.loc[timestamps]
        macro_vecs    = self._build_action_vecs(macro_trimmed)

        feat_df  = pd.DataFrame(raw_feat, columns=GoldFeatureEngineer.FEATURE_NAMES,
                                index=timestamps)
        features = _rolling_zscore(feat_df, window=_Z_SCORE_WIN).values.astype(np.float32)

        return ohlcv_trimmed, features, macro_vecs

    def _align_macro(self, macro_data: dict, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align macro Series dict to target_index via reindex + forward-fill.

        When a source returns nothing, fill with the last-known level (or a
        neutral constant) instead of 0.0. A constant series yields zero-signal
        macro features (returns≈0, ratios≈1); a 0.0 fill makes every derived
        feature degenerate (dxy/ema-1 = -1, etc.) and produces spurious S_t
        spikes. DXY additionally has a ticker fallback upstream (fetch_macro_data).
        """
        cols = {}
        for name, series in macro_data.items():
            if series is not None and not series.empty:
                aligned = series.reindex(target_index, method="ffill").bfill()
                last = aligned.dropna()
                if not last.empty:
                    self._last_macro[name] = float(last.iloc[-1])   # remember last good level
            else:
                fill = self._last_macro.get(name, _MACRO_NEUTRAL.get(name, 1.0))
                logger.warning("macro '%s' returned no data; filling constant %.4f "
                               "(last-known/neutral)", name, fill)
                aligned = pd.Series(fill, index=target_index)
            cols[name] = aligned.fillna(self._last_macro.get(name,
                                        _MACRO_NEUTRAL.get(name, 1.0)))
        return pd.DataFrame(cols, index=target_index)

    @staticmethod
    def _build_action_vecs(macro_df: pd.DataFrame) -> np.ndarray:
        """
        Build 5-dim action vectors from an already warm-up-trimmed macro DataFrame.
        Mirrors DataPipeline._build_actions_gold without the [W:] slice.
        """
        def _clean(arr):
            s = pd.Series(arr.astype(float))
            return s.ffill().bfill().fillna(0.0).values

        dxy = _clean(macro_df["dxy"].values)
        gvz = _clean(macro_df["gvz"].values)
        tlt = _clean(macro_df["tlt"].values)

        dxy_ret = np.log(np.maximum(dxy / np.roll(dxy, 1), 1e-10)); dxy_ret[0] = 0.0
        tlt_ret = np.log(np.maximum(tlt / np.roll(tlt, 1), 1e-10)); tlt_ret[0] = 0.0
        gvz_chg = np.diff(gvz, prepend=gvz[0]) / (gvz + 1e-8)

        hour_sin      = np.sin(2 * np.pi * pd.DatetimeIndex(macro_df.index).hour / 24)
        macro_tension = np.abs(dxy_ret) + np.abs(tlt_ret)

        actions = np.column_stack([dxy_ret, gvz_chg, tlt_ret, hour_sin, macro_tension])
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        mu = actions.mean(axis=0)
        sd = np.where(actions.std(axis=0) < 1e-8, 1.0, actions.std(axis=0))
        return ((actions - mu) / sd).astype(np.float32)

    def _compute_atr_pct(self, ohlcv: pd.DataFrame) -> float:
        """ATR_14 / close on unnormalised OHLCV. Returns the current bar's ATR-pct."""
        from execution.position_sizing import atr_from_ohlc
        high  = ohlcv["high"].values.astype(np.float64)
        low   = ohlcv["low"].values.astype(np.float64)
        close = ohlcv["close"].values.astype(np.float64)
        atr   = atr_from_ohlc(high, low, close, period=14)
        last_atr   = atr[-1]
        last_close = close[-1]
        if np.isnan(last_atr) or last_close < 1e-6:
            return 0.01   # conservative fallback
        return float(last_atr / last_close)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_all(self, features: np.ndarray, n_valid: int) -> np.ndarray:
        """Encode all valid windows -> (n_valid, z_dim)."""
        feat_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        zs = []
        with torch.no_grad():
            for i in range(n_valid):
                window = feat_t[i : i + _LOOKBACK].unsqueeze(0)
                zs.append(self.encoder(window).squeeze(0).cpu().numpy())
        return np.array(zs, dtype=np.float32)

    def _encode_single(self, features: np.ndarray, valid_idx: int) -> torch.Tensor:
        """Encode the window ending at features[valid_idx + _LOOKBACK - 1]."""
        feat_t = torch.tensor(
            features[valid_idx : valid_idx + _LOOKBACK],
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            return self.encoder(feat_t.unsqueeze(0)).squeeze(0)

    def _predict_from_deque(self, action: np.ndarray) -> torch.Tensor:
        """Predict z_{t+1} from the current z_deque + macro action vec."""
        z_hist = self._stack_history(list(self.z_deque)).unsqueeze(0)
        a_t    = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.predictor(z_hist, a_t).squeeze(0)

    def _stack_history(self, zs: list) -> torch.Tensor:
        """
        Stack the last _HISTORY_LEN embeddings into (H, z_dim).
        Pads the beginning with the oldest frame if fewer than H frames exist.
        """
        if not zs:
            return torch.zeros(_HISTORY_LEN, _Z_DIM, dtype=torch.float32, device=self.device)
        while len(zs) < _HISTORY_LEN:
            zs.insert(0, zs[0])
        zs = zs[-_HISTORY_LEN:]
        if isinstance(zs[0], torch.Tensor):
            return torch.stack(zs)
        return torch.tensor(np.array(zs), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Dashboard status writer
    # ------------------------------------------------------------------

    def _write_status(self, extra: dict | None = None) -> None:
        import json
        pos = self.current_position
        pos_label = "LONG" if pos > 1e-6 else ("SHORT" if pos < -1e-6 else "FLAT")
        payload = {
            "updated":          datetime.now(tz=timezone.utc).isoformat(),
            "symbol":           self.connector.symbol,
            "dry_run":          self.dry_run,
            "position":         pos_label,
            "surprise_buf_len": len(self.surprise_history),
            "history":          list(reversed(self._bar_history)),
            **(extra or {}),
        }
        try:
            self._status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._status_path)
        except Exception as exc:
            logger.warning("Could not write status file: %s", exc)

    # ------------------------------------------------------------------
    # Circuit breaker (mirrors environment.py _apply_circuit_breaker)
    # ------------------------------------------------------------------

    def _apply_circuit_breaker(self, trade_action: float, S_t: float) -> float:
        if len(self.surprise_history) < self.surprise_warmup:
            return trade_action

        p20 = max(float(np.percentile(self.surprise_history, 20)), _CB_FLOOR_P20)
        p60 = max(float(np.percentile(self.surprise_history, 60)), _CB_FLOOR_P60)
        p90 = max(float(np.percentile(self.surprise_history, 90)), _CB_FLOOR_P90)

        logger.info(
            "S_t check: S_t=%.6f  p20=%.6f  p60=%.6f  p90=%.6f  zone=%s  buffer_len=%d",
            S_t, p20, p60, p90,
            "Q1" if S_t < p20 else "Q2-skip" if S_t < p60 else "Q4-half" if S_t < p90 else "Q5-CB",
            len(self.surprise_history),
        )

        if S_t < p90:
            if self.post_shock_bars_remaining > 0:
                self.post_shock_bars_remaining -= 1
            if S_t < p20:
                return trade_action
            elif S_t < p60:
                if abs(trade_action) < 1e-9 or abs(self.current_position) < 1e-9:
                    return 0.0
                return self.current_position
            else:
                return trade_action * 0.5
        else:
            if self.post_shock_bars_remaining == 0:
                self.post_shock_bars_remaining = 8
            return 0.0

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute(self, effective_signal: float, lots: float) -> str:
        """
        Diff effective_signal vs current_position, send the MT5 order.
        Returns a short action string for logging.
        """
        prev = self.current_position
        new  = effective_signal

        if abs(new - prev) < 1e-6:
            return "hold"

        if self.dry_run:
            self.current_position = new
            label = "long" if new > 0 else ("short" if new < 0 else "flat")
            return f"dry_run({label} {lots:.2f})"

        if abs(prev) > 1e-6:
            self.connector.close_position()

        if abs(new) > 1e-6:
            direction = 1 if new > 0 else -1
            self.connector.place_order(lots=lots, direction=direction)
            action = f"{'long' if direction == 1 else 'short'} {lots:.2f} lots"
        else:
            action = "flat"

        self.current_position = new
        return action


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_system(checkpoint: str, heads_dir: str, device: str = "cpu"):
    """Load encoder, predictor, and execution layer from checkpoints."""
    import json
    from models.encoder   import MarketEncoder
    from models.predictor import CausalPredictor
    from execution.heads  import SupervisedExecutionLayer, ProbeBundle

    ckpt      = torch.load(checkpoint, map_location=device, weights_only=False)
    encoder   = MarketEncoder()
    predictor = CausalPredictor()
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    encoder.eval()
    predictor.eval()
    logger.info("Loaded encoder + predictor from %s (epoch %s)",
                checkpoint, ckpt.get("epoch", "?"))

    hdir       = Path(heads_dir)
    state_dict = torch.load(str(hdir / "execution_layer.pt"),
                            map_location=device, weights_only=False)

    with open(hdir / "metrics.json") as f:
        meta = json.load(f)

    history_len     = meta.get("history_len", 3)
    dir_in_dim      = meta.get("dir_in_dim", None)
    vol_thresholds  = meta.get("vol_thresholds", (0.0, 1.0))
    mag_threshold   = meta.get("mag_threshold", 0.0)

    layer = SupervisedExecutionLayer(
        history_len    = history_len,
        dir_in_dim     = dir_in_dim,
        vol_thresholds = tuple(vol_thresholds),
        mag_threshold  = mag_threshold,
    )
    if any(k.startswith("probe_bundle.") for k in state_dict):
        layer.probe_bundle = ProbeBundle()
    layer.load_state_dict(state_dict)
    layer.eval()

    dir_auc = meta.get("direction", {}).get("auc", "?")
    vol_auc = meta.get("vol", {}).get("auc", "?")
    logger.info("Loaded execution heads from %s  Dir AUC=%.3f  Vol AUC=%.3f",
                heads_dir, dir_auc, vol_auc)

    return encoder, predictor, layer


def _main() -> None:
    log_fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    log_date = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=log_date)

    log_path = Path("experiments") / f"live_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=log_date))
    logging.getLogger().addHandler(fh)
    logging.getLogger(__name__).info("Logging to %s", log_path)

    ap = argparse.ArgumentParser(description="MWM live gold trader")
    ap.add_argument("--checkpoint",      default="./experiments/checkpoints/best_model.pt")
    ap.add_argument("--heads-dir",       default="./experiments/heads_gold")
    ap.add_argument("--symbol",          default="XAUUSD")
    ap.add_argument("--magic",           type=int,   default=20260101)
    ap.add_argument("--risk-frac",       type=float, default=0.01)
    ap.add_argument("--spread",          type=float, default=0.0003)
    ap.add_argument("--dir-thresh",      type=float, default=0.53)
    ap.add_argument("--surprise-warmup", type=int,   default=100)
    ap.add_argument("--warm-up-bars",    type=int,   default=_WARM_UP_BARS)
    ap.add_argument("--device",          default="cpu")
    ap.add_argument("--dry-run",         action="store_true",
                    help="Log signals but do not send MT5 orders")
    ap.add_argument("--poll-seconds",    type=int, default=10,
                    help="Polling interval in seconds for the bar-close check loop")
    # Config D — M15 execution layer
    ap.add_argument("--m15-checkpoint", type=str, default=None,
                    help="Path to M15 warmstart encoder checkpoint for Config D execution. "
                         "If omitted, enters at H1 bar close (original behaviour).")
    ap.add_argument("--m15-st-threshold", type=float, default=0.000012,
                    help="M15 S_t calm threshold (p20 from M15 training set, default 0.000012)")
    args = ap.parse_args()

    from execution.mt5_connector import MT5Connector

    encoder, predictor, layer = _load_system(args.checkpoint, args.heads_dir, args.device)

    # Optionally load M15 encoder for Config D execution improvement
    m15_encoder = m15_predictor = None
    if args.m15_checkpoint:
        from models.encoder   import MarketEncoder
        from models.predictor import CausalPredictor
        m15_ckpt = torch.load(args.m15_checkpoint, map_location=args.device, weights_only=False)
        m15_cfg  = m15_ckpt["cfg"]
        m15_encoder = MarketEncoder(
            lookback=m15_cfg["lookback"], n_features=52, patch_size=4,
            d_model=128, n_heads=4, n_layers=4, dropout=0.0, proj_hidden=256, z_dim=128,
        )
        m15_encoder.load_state_dict(m15_ckpt["encoder"])
        m15_predictor = CausalPredictor(
            z_dim=128, d_model=128, n_heads=4, n_layers=6,
            action_dim=5, history_len=m15_cfg["history_len"], dropout=0.0,
        )
        m15_predictor.load_state_dict(m15_ckpt["predictor"])
        logger.info("Config D: loaded M15 encoder (epoch %d)", m15_ckpt["epoch"])

    connector = MT5Connector(symbol=args.symbol, magic=args.magic)
    connector.connect()

    trader = LiveTrader(
        encoder           = encoder,
        predictor         = predictor,
        layer             = layer,
        connector         = connector,
        risk_frac         = args.risk_frac,
        spread            = args.spread,
        dir_threshold     = args.dir_thresh,
        surprise_warmup   = args.surprise_warmup,
        warm_up_bars      = args.warm_up_bars,
        dry_run           = args.dry_run,
        device            = args.device,
        m15_encoder       = m15_encoder,
        m15_predictor     = m15_predictor,
        m15_st_threshold  = args.m15_st_threshold if m15_encoder else 0.0,
    )

    trader.warm_up()
    if trader._m15_use:
        trader.m15_warm_up()
    logger.info("Live trader started. symbol=%s  dry_run=%s  poll=%ds  m15=%s",
                args.symbol, args.dry_run, args.poll_seconds,
                "enabled" if trader._m15_use else "disabled")

    last_processed_hour:   Optional[int] = None
    last_processed_minute: Optional[int] = None

    def is_market_open() -> bool:
        now = datetime.now(timezone.utc)
        if now.weekday() == 5:                          # Saturday
            return False
        if now.weekday() == 6 and now.hour < 22:        # Sunday before 22:00 UTC
            return False
        if now.weekday() == 4 and now.hour >= 21:       # Friday close at 21:00 UTC
            return False
        return True

    try:
        while True:
            if not is_market_open():
                logger.info("Market closed — weekend. Sleeping 1 hour.")
                time.sleep(3600)
                continue

            now    = datetime.now(tz=timezone.utc)
            hour   = now.hour
            minute = now.minute

            # H1 bar close (minute 0)
            if hour != last_processed_hour and minute == 0 and now.second < args.poll_seconds:
                trader.on_bar_close()
                last_processed_hour   = hour
                last_processed_minute = 0

            # M15 bar close (minute 15, 30, 45) — only when pending M15 entry
            elif (minute in (15, 30, 45)
                  and minute != last_processed_minute
                  and now.second < args.poll_seconds
                  and trader._m15_pending):
                trader.on_m15_bar_close()
                last_processed_minute = minute

            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        logger.info("Interrupted — closing positions and disconnecting.")
        if not args.dry_run:
            connector.close_position()
        connector.disconnect()


if __name__ == "__main__":
    _main()
