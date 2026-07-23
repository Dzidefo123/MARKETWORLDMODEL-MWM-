"""
execution/environment.py — Gym-style backtesting environment for MWM-based trading.

TWO DISTINCT VECTORS — NEVER CONFLATE THEM
-------------------------------------------
macro_vec (a_t)  : 5-dim market context vector [dxy_ret, gvz/evz_chg, tlt_ret,
                   hour_sin, macro_tension]. This is MARKET DATA that the
                   CausalPredictor uses to model the transition z_t -> z_{t+1}.
                   It is sourced from pipeline macro_vecs at each bar and is
                   NEVER derived from the agent's decisions.

trade_action     : The agent's position sizing in [-1, 1]. Sign = direction,
                   magnitude = fraction of equity. This is the TRADING DECISION.
                   It is the parameter to step() and NEVER enters the predictor.

The predictor was trained with macro_vec as its conditioning signal. Feeding
trade_action into it would corrupt the surprise signal S_t — the circuit
breaker would fire at random and become meaningless.
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, Dict, Any


class TradingEnvironment:
    """
    One-bar-at-a-time backtesting environment wrapping frozen MWM components.

    The encoder and predictor are always used in eval mode with no_grad.
    They are never fine-tuned inside this environment — that would destroy
    the structural organisation SIGReg built into the latent space.

    Args:
        encoder:              Frozen MarketEncoder (lookback=48, z_dim=128).
        predictor:            Frozen CausalPredictor (history_len=3, action_dim=5).
        features:             (n_bars, 52) normalized feature matrix for the split.
        macro_vecs:           (n_bars, 5)  macro context vectors for the same split.
                              These are the ONLY vectors passed to the predictor.
        prices:               (n_bars,)    close prices for PnL computation.
        spread_cost:          One-way transaction cost as a fraction of position size.
        lookback:             Bars per encoder window. Must match encoder architecture.
        history_len:          Predictor history depth. Must match predictor architecture.
        surprise_warmup:      Minimum S_t samples before the circuit breaker activates.
        device:               Torch device string.
    """

    def __init__(
        self,
        encoder,
        predictor,
        features:             np.ndarray,
        macro_vecs:           np.ndarray,
        prices:               np.ndarray,
        spread_cost:          float = 0.0003,
        lookback:             int   = 48,
        history_len:          int   = 3,
        surprise_warmup:      int   = 100,
        device:               str   = "cpu",
    ):
        self.lookback        = lookback
        self.history_len     = history_len
        self.spread_cost     = spread_cost
        self.surprise_warmup = surprise_warmup
        self.device          = torch.device(device)

        self.encoder   = encoder.to(self.device).eval()
        self.predictor = predictor.to(self.device).eval()

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.predictor.parameters():
            p.requires_grad_(False)

        self.features   = torch.as_tensor(features,   dtype=torch.float32, device=self.device)
        self.macro_vecs = torch.as_tensor(macro_vecs, dtype=torch.float32, device=self.device)
        self.prices     = np.asarray(prices, dtype=np.float64)

        self.n_bars = len(features)
        assert self.n_bars == len(macro_vecs) == len(prices), \
            "features, macro_vecs, and prices must have the same length"
        assert self.n_bars > lookback, \
            f"Need at least {lookback+1} bars; got {self.n_bars}"

        # Mutable state — populated by reset()
        self.bar_idx:          int               = None
        self.z_deque:          deque             = None
        self.z_hat_current:    torch.Tensor      = None
        self.surprise_history: list              = None
        self.position:         float             = None
        self.equity:           float             = None
        self.done:             bool              = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """
        Initialise at the first bar that has a full lookback window.

        Returns:
            z_t: (128,) embedding for the first observable bar.
        """
        self.bar_idx          = self.lookback - 1
        self.position         = 0.0
        self.equity           = 1.0
        self.done             = False
        self.surprise_history = []

        # Encode the first valid bar and fill history with it.
        # Padding with the first embedding matches CausalPredictor.build_history().
        z_first = self._encode_bar(self.bar_idx)
        self.z_deque = deque(
            [z_first.clone() for _ in range(self.history_len)],
            maxlen=self.history_len,
        )

        # Pre-compute the predictor's forecast for bar bar_idx+1.
        # macro_vecs[bar_idx] is MARKET DATA — it never carries the agent's intent.
        self.z_hat_current = self._predict_next(self.bar_idx)

        return z_first

    def step(self, trade_action: float) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Advance one bar, execute the trade, and return the next observation.

        Args:
            trade_action: Position size in [-1, 1]. This is the AGENT'S DECISION.
                          It is used only for PnL — it never enters the predictor.

        Returns:
            z_next:  (128,) encoder embedding at the next bar.
            reward:  Realised PnL net of spread cost.
            done:    True when the split is exhausted.
            info:    {'S_t', 'pnl', 'equity', 'bar_idx', 'effective_action'}.
        """
        assert self.bar_idx is not None, "Call reset() before step()"

        # --- Surprise at the current bar ---
        # S_t = prediction error of z_hat_t (forecast made at bar t-1) vs actual z_t.
        # This is the circuit-breaker signal. It uses only market data, not trade_action.
        z_t = self.z_deque[-1]
        S_t = ((self.z_hat_current - z_t) ** 2).mean().item()
        self.surprise_history.append(S_t)

        # --- Risk circuit breaker (trade_action is overridden, never the predictor) ---
        effective_action = self._apply_circuit_breaker(trade_action, S_t)

        # --- Check boundary ---
        if self.bar_idx + 1 >= self.n_bars:
            self.done = True
            return z_t, 0.0, True, {
                "S_t": S_t, "pnl": 0.0, "equity": self.equity,
                "bar_idx": self.bar_idx, "effective_action": effective_action,
            }

        # --- PnL ---
        bar_return  = (self.prices[self.bar_idx + 1] - self.prices[self.bar_idx]) \
                      / self.prices[self.bar_idx]
        trade_cost  = abs(effective_action - self.position) * self.spread_cost
        pnl         = float(effective_action) * bar_return - trade_cost
        self.equity *= (1.0 + pnl)
        self.position = effective_action

        # --- Advance ---
        self.bar_idx += 1
        z_next = self._encode_bar(self.bar_idx)
        self.z_deque.append(z_next)

        # Pre-compute predictor forecast for the NEXT step's S computation.
        # macro_vecs[bar_idx] = macro context AT the current bar.
        # This is always MARKET DATA — the agent's trade_action has no part here.
        self.done = (self.bar_idx >= self.n_bars - 1)
        if not self.done:
            self.z_hat_current = self._predict_next(self.bar_idx)

        return z_next, pnl, self.done, {
            "S_t": S_t, "pnl": pnl, "equity": self.equity,
            "bar_idx": self.bar_idx, "effective_action": effective_action,
        }

    @property
    def observation_dim(self) -> int:
        return 128

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        encoder,
        predictor,
        pipeline_result: dict,
        split:           str   = "test",
        **kwargs,
    ) -> "TradingEnvironment":
        """
        Construct from the dict returned by DataPipeline.build().

        Args:
            split: One of 'train', 'val', 'test'.
        """
        meta    = pipeline_result["meta"]
        n_train = meta["n_train"]
        n_val   = meta["n_val"]

        slices = {
            "train": slice(0,               n_train),
            "val":   slice(n_train,         n_train + n_val),
            "test":  slice(n_train + n_val, None),
        }
        if split not in slices:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
        s = slices[split]

        return cls(
            encoder    = encoder,
            predictor  = predictor,
            features   = meta["norm_features"][s],
            macro_vecs = meta["macro_vecs"][s],
            prices     = meta["prices"][s],
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_bar(self, bar_idx: int) -> torch.Tensor:
        """Return z_t for a bar by encoding its 48-bar lookback window."""
        window = self.features[bar_idx - self.lookback + 1 : bar_idx + 1]  # (48, 52)
        with torch.no_grad():
            z = self.encoder(window.unsqueeze(0))  # (1, 128)
        return z.squeeze(0)  # (128,)

    def _predict_next(self, bar_idx: int) -> torch.Tensor:
        """
        Forecast z_{bar_idx+1} using the causal predictor.

        The macro_vec fed here is MARKET DATA from macro_vecs[bar_idx]:
        [dxy_ret, gvz/evz_chg, tlt_ret, hour_sin, macro_tension].
        This is NEVER the agent's trade_action.
        """
        z_hist    = torch.stack(list(self.z_deque), dim=0).unsqueeze(0)  # (1, history_len, 128)
        macro_vec = self.macro_vecs[bar_idx].unsqueeze(0)                # (1, 5)
        with torch.no_grad():
            z_hat = self.predictor(z_hist, macro_vec)                    # (1, 128)
        return z_hat.squeeze(0)  # (128,)

    def _apply_circuit_breaker(self, trade_action: float, S_t: float) -> float:
        """
        Four-zone S_t filter. Operates on trade_action only — never touches macro_vec.

        Zone boundaries are causal rolling percentiles of the surprise history:
          < p20  : Q1 — calm, predictor reliable. Enter freely.
          p20–p60: Q2-Q3 — mid-range, signal unreliable (direction inverts in p20-p30).
                   Block NEW entries; hold existing positions.
                   Explicit flat signals (trade_action=0) are always honoured so that
                   skip_sessions / max_holding force-closes are not overridden.
          p60–p90: Q4 — elevated. Half-size new entries; reduce existing to half.
          >= p90 : Q5 — extreme regime break. Force flat.
        """
        if len(self.surprise_history) < self.surprise_warmup:
            return trade_action

        p20 = np.percentile(self.surprise_history, 20)
        p60 = np.percentile(self.surprise_history, 60)
        p90 = np.percentile(self.surprise_history, 90)

        if S_t < p20:
            # Q1: calm — full signal
            return trade_action

        elif S_t < p60:
            # Q2-Q3: mid-range — no new entries, hold existing.
            # trade_action=0 always passes through (honours force-flat from caller).
            if abs(trade_action) < 1e-9 or abs(self.position) < 1e-9:
                return 0.0
            return self.position  # preserve current position unchanged

        elif S_t < p90:
            # Q4: elevated — half size
            return trade_action * 0.5

        else:
            # Q5: extreme — force flat
            return 0.0
