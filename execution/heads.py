"""
execution/heads.py — Supervised execution heads on frozen z_t embeddings.

Three linear heads, each trained independently on the training split:

  VolatilityHead    — 3-class regime classifier (low / medium / high)
                      Label source: rv_32 of the current bar, bucketed by
                      training-set percentiles. Observable NOW, no lookahead.

  DirectionalHead   — P(forward horizon-bar return > 0)
                      Label source: prices[t+horizon] vs prices[t].
                      Forward-looking — labels use future data ONLY during training.

  MagnitudeHead     — Predicted |forward return| over the next horizon bars.
                      Same label source as DirectionalHead.

All heads take frozen z_t (128-dim) as input. The encoder is never fine-tuned.

LABEL INDEXING
--------------
For a split of shape (n_bars, 52):
  n_valid_z       = n_bars - lookback + 1   (bars with a full 48-bar window)
  n_valid_labeled = n_valid_z - horizon     (bars that also have future prices)

  z_matrix[i]  ↔  bar_idx = i + lookback - 1  (last bar of the encoding window)
  vol[i]       ↔  features[i + lookback - 1, RV32_IDX]
  direction[i] ↔  prices[i + lookback - 1 + horizon] vs prices[i + lookback - 1]
  magnitude[i] ↔  |prices[i + lookback - 1 + horizon] - prices[i + lookback - 1]|

ENTRY SIGNAL
------------
The SupervisedExecutionLayer combines all three heads into a position size [-1, 1]:
  1. dir_prob > 0.53 → long;  dir_prob < 0.47 → short;  else flat
  2. predicted magnitude > 1.5 × spread  (cost filter)
  3. vol_regime == 2 (high) → halve position size

The S_t circuit breaker (filter 4 from the build plan) is applied separately
by TradingEnvironment._apply_circuit_breaker(). It is not implemented here.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# rv_32 is at index 10 in GoldFeatureEngineer.FEATURE_NAMES and ForexFeatureEngineer.FEATURE_NAMES
RV32_IDX = 10

# Session feature indices (must match GoldFeatureEngineer.FEATURE_NAMES)
_IS_LONDON_IDX = 38
_IS_NY_IDX     = 39

# Session ID constants
SESSION_ASIAN  = 0
SESSION_LONDON = 1
SESSION_NY     = 2


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def compute_labels(
    features:     np.ndarray,
    prices:       np.ndarray,
    lookback:     int = 48,
    horizon:      int = 4,
    rv32_pct:     Optional[Tuple[float, float]] = None,
    trend_window: int = 0,
) -> dict:
    """
    Compute supervision labels for all valid bars in a split.

    Args:
        features:   (n_bars, 52) normalized features for the split.
        prices:     (n_bars,) close prices aligned to feature rows.
        lookback:   Encoder window length. Must match MarketEncoder.
        horizon:    Forward return window for direction/magnitude labels.
        rv32_pct:   (p33, p67) vol bucketing thresholds. Pass training-set
                    thresholds when labelling val/test to avoid leakage.
                    If None, computed from this split.

    Returns dict:
        vol:             int64  (n_valid_z,)       — 0=low, 1=medium, 2=high
        direction:       int64  (n_valid_labeled,) — 1=up, 0=down
        magnitude:       float32 (n_valid_labeled,) — |fwd return|
        n_valid_z:       int
        n_valid_labeled: int
        vol_thresholds:  (p33, p67) used for bucketing
    """
    n_bars          = len(features)
    n_valid_z       = n_bars - lookback + 1
    n_valid_labeled = n_valid_z - horizon

    assert n_valid_labeled > 0, (
        f"Split too short: {n_bars} bars, lookback={lookback}, horizon={horizon} "
        f"→ {n_valid_labeled} labeled samples"
    )

    # Vol labels — rv_32 at the last bar of each encoding window
    rv32 = features[lookback - 1 :, RV32_IDX]          # (n_valid_z,)

    if rv32_pct is None:
        p33 = float(np.percentile(rv32, 33))
        p67 = float(np.percentile(rv32, 67))
    else:
        p33, p67 = float(rv32_pct[0]), float(rv32_pct[1])

    vol = np.where(rv32 < p33, 0, np.where(rv32 < p67, 1, 2)).astype(np.int64)

    # Direction and magnitude — forward-looking from current close price
    # p_now[i] = prices at bar_idx = i + lookback - 1
    # p_fwd[i] = prices at bar_idx + horizon
    p_now  = prices[lookback - 1 : n_bars - horizon]            # (n_valid_labeled,)
    p_fwd  = prices[lookback - 1 + horizon : n_bars]            # (n_valid_labeled,)
    fwd    = (p_fwd - p_now) / (p_now + 1e-8)

    if trend_window > 0:
        # Trend-adjusted: beat the rolling median of recent returns, not beat zero.
        # shift(1) ensures the median is computed on PAST bars only — no lookahead.
        rolling_median = (pd.Series(fwd)
                          .rolling(trend_window, min_periods=max(1, trend_window // 10))
                          .median()
                          .shift(1)
                          .fillna(0.0)
                          .values)
        direction = (fwd > rolling_median).astype(np.int64)
    else:
        direction = (fwd > 0).astype(np.int64)

    # Session ID for each bar: 0=Asian, 1=London, 2=NY
    # bar_idx for z[i] = i + lookback - 1; use features at that bar
    feat_z    = features[lookback - 1:]                # (n_valid_z, 52)
    is_ny_arr = feat_z[:, _IS_NY_IDX]     > 0
    is_lo_arr = feat_z[:, _IS_LONDON_IDX] > 0
    session_z = np.where(is_ny_arr, SESSION_NY,
                np.where(is_lo_arr, SESSION_LONDON,
                SESSION_ASIAN)).astype(np.int64)       # (n_valid_z,)

    return {
        "vol":             vol,
        "direction":       direction,
        "magnitude":       np.abs(fwd).astype(np.float32),
        "session_z":       session_z,                  # (n_valid_z,) — for vol head
        "session":         session_z[:n_valid_labeled],# (n_valid_labeled,) — for dir/mag
        "n_valid_z":       n_valid_z,
        "n_valid_labeled": n_valid_labeled,
        "vol_thresholds":  (p33, p67),
    }


# ---------------------------------------------------------------------------
# Z-history utility
# ---------------------------------------------------------------------------

def build_z_history(Z: np.ndarray, history_len: int = 3) -> np.ndarray:
    """
    Build a sliding-window history tensor with boundary padding.

    Args:
        Z:           (N, z_dim) array of encoder embeddings.
        history_len: Number of consecutive frames to include (oldest → newest).

    Returns:
        (N, history_len, z_dim) — Z_hist[i, :] = [z_{i-H+1}, ..., z_i]
        with z_{<0} clamped to z_0.
    """
    N, z_dim = Z.shape
    Z_hist = np.zeros((N, history_len, z_dim), dtype=Z.dtype)
    for lag in range(history_len):
        src_idx = np.maximum(0, np.arange(N) - (history_len - 1 - lag))
        Z_hist[:, lag, :] = Z[src_idx]
    return Z_hist


# ---------------------------------------------------------------------------
# Batch embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_split(
    encoder,
    features:   torch.Tensor,
    lookback:   int = 48,
    batch_size: int = 256,
    device:     str = "cpu",
) -> np.ndarray:
    """
    Encode all valid bars in a split.

    Args:
        features: (n_bars, 52) normalized feature tensor.
        lookback: Encoder window length.

    Returns:
        (n_valid_bars, z_dim) — n_valid_bars = n_bars - lookback + 1.
        Row i encodes the window ending at bar_idx = i + lookback - 1.
    """
    encoder.eval()
    dev    = torch.device(device)
    feats  = features.to(dev)
    n_bars = len(feats)

    all_z = []
    for start in range(0, n_bars - lookback + 1, batch_size):
        end     = min(start + batch_size, n_bars - lookback + 1)
        windows = torch.stack([feats[t : t + lookback] for t in range(start, end)])
        all_z.append(encoder(windows).cpu().numpy())

    return np.vstack(all_z)                                     # (n_valid_bars, z_dim)


# ---------------------------------------------------------------------------
# Head modules
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with ReLU and LayerNorm for better gradient flow."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class VolatilityHead(nn.Module):
    """3-class vol regime classifier. Output: logits (B, 3)."""

    def __init__(self, z_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = _mlp(z_dim, hidden, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def predict_class(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z).argmax(dim=-1)


class DirectionalHead(nn.Module):
    """
    Binary classifier: P(forward return > trend median).

    Input:  z_hist (B, H, z_dim)  — current embedding + H-1 prior frames.
            z_dim can be 128 (raw) or 135 (augmented with probes + S_t).
            Flattened to (B, H*z_dim) before the MLP.
    Output: logit  (B, 1).
    """

    def __init__(self, z_dim: int = 128, hidden: int = 64, history_len: int = 1,
                 in_dim: int = None):
        super().__init__()
        _in_dim = in_dim if in_dim is not None else z_dim * history_len
        self.net = _mlp(_in_dim, hidden, 1)
        self.history_len = history_len

    def _make_features(self, z_hist: torch.Tensor) -> torch.Tensor:
        """(B, H, z_dim) → (B, H*z_dim): flatten raw history frames."""
        return z_hist.reshape(z_hist.shape[0], -1)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        return self.net(self._make_features(z_hist))

    def predict_prob(self, z_hist: torch.Tensor) -> torch.Tensor:
        """Probability in [0, 1], shape (B,)."""
        return torch.sigmoid(self.forward(z_hist)).squeeze(-1)


class MagnitudeHead(nn.Module):
    """Regressor: predicted |forward return|. Output: (B, 1), always >= 0."""

    def __init__(self, z_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = _mlp(z_dim, hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(z))      # softplus keeps output > 0

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Predicted magnitude, shape (B,)."""
        return self.forward(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Structured feature augmentation
# ---------------------------------------------------------------------------

class ProbeBundle(nn.Module):
    """
    Three lightweight linear probes trained on frozen z_t embeddings.
    Provides structural context that the directional head no longer needs to
    rediscover from raw 128-dim z:

      session_probs (3): softmax over Asian / London / NY trading session
      vol_prob      (1): P(rv_32 > median) — volatility regime signal
      rv_pred       (1): continuous realized-vol estimate

    Total: 5 extra dims.  z_aug = [z_t | session_probs | vol_prob | rv_pred]
    """

    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.session_probe = nn.Linear(z_dim, 3)
        self.vol_probe     = nn.Linear(z_dim, 1)
        self.rv_probe      = nn.Linear(z_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, z_dim) → (B, 5) structured context features."""
        session = torch.softmax(self.session_probe(z), dim=-1)  # (B, 3)
        vol     = torch.sigmoid(self.vol_probe(z))              # (B, 1)
        rv      = self.rv_probe(z)                              # (B, 1)
        return torch.cat([session, vol, rv], dim=-1)


def train_probe_bundle(
    Z_train:    np.ndarray,   # (n_valid_z, z_dim) frozen embeddings
    feat_train: np.ndarray,   # (n_train_bars, 52) normalized features
    lookback:   int   = 48,
    n_epochs:   int   = 200,
    lr:         float = 1e-3,
) -> ProbeBundle:
    """
    Train all three probes jointly on the training split.
    Labels are derived from features at bar_idx = i + lookback - 1 (no lookahead).
    Returns a frozen ProbeBundle (requires_grad=False on all params).
    """
    n        = len(Z_train)
    feat_z   = feat_train[lookback - 1 : lookback - 1 + n]  # (n, 52)

    # Session: is_ny (39) > 0 → NY, is_london (38) > 0 → London, else Asian
    is_ny   = feat_z[:, _IS_NY_IDX]     > 0
    is_lo   = feat_z[:, _IS_LONDON_IDX] > 0
    sess_y  = np.where(is_ny, SESSION_NY,
              np.where(is_lo, SESSION_LONDON, SESSION_ASIAN)).astype(np.int64)

    # Vol: rv_32 > median → high (binary)
    rv32    = feat_z[:, RV32_IDX].astype(np.float32)
    vol_y   = (rv32 > float(np.median(rv32))).astype(np.float32)

    bundle  = ProbeBundle(z_dim=Z_train.shape[1])
    Zt      = torch.FloatTensor(Z_train)
    opt     = torch.optim.Adam(bundle.parameters(), lr=lr)
    sess_yt = torch.LongTensor(sess_y)
    vol_yt  = torch.FloatTensor(vol_y).unsqueeze(1)
    rv_yt   = torch.FloatTensor(rv32).unsqueeze(1)

    bundle.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = (
            nn.CrossEntropyLoss()(bundle.session_probe(Zt), sess_yt)
            + nn.BCEWithLogitsLoss()(bundle.vol_probe(Zt), vol_yt)
            + nn.MSELoss()(bundle.rv_probe(Zt), rv_yt)
        )
        loss.backward()
        opt.step()

    bundle.eval()
    for p in bundle.parameters():
        p.requires_grad_(False)
    return bundle


@torch.no_grad()
def compute_surprise_features(
    Z:           np.ndarray,   # (n_valid_z, z_dim) pre-encoded embeddings
    actions:     np.ndarray,   # (n_bars_in_split, action_dim)
    predictor,                 # CausalPredictor (frozen)
    lookback:    int = 48,
    history_len: int = 3,
    window:      int = 500,
    batch_size:  int = 512,
) -> np.ndarray:
    """
    Compute causal surprise features [s_norm, s_rank] per bar.

    S_t = mean( (z_hat_{t+1} - z_{t+1})^2 ) — world model's prediction error.
    s_norm: rolling z-score over past `window` bars (causal, no lookahead).
    s_rank: fraction of past `window` bars with lower surprise (causal percentile).

    Returns (n_valid_z, 2) float32.  Last bar padded with zeros.
    """
    n       = len(Z)
    Z_hist  = build_z_history(Z, history_len)                       # (n, H, z_dim)
    act_idx = np.minimum(np.arange(n) + lookback - 1, len(actions) - 1)
    A       = actions[act_idx]                                       # (n, action_dim)

    Zh_t = torch.FloatTensor(Z_hist)
    At   = torch.FloatTensor(A)
    Zt   = torch.FloatTensor(Z)

    predictor.eval()
    z_hat_chunks = [
        predictor(Zh_t[s : s + batch_size], At[s : s + batch_size]).cpu()
        for s in range(0, n, batch_size)
    ]
    z_hat = torch.cat(z_hat_chunks)                                  # (n, z_dim)

    err  = ((z_hat[:-1] - Zt[1:]) ** 2).mean(dim=1).numpy().astype(np.float32)
    surp = np.append(err, 0.0)                                       # pad last bar

    # Causal rolling z-score (shift(1) so bar t uses only bars 0..t-1)
    s      = pd.Series(surp)
    roll   = s.rolling(window, min_periods=20)
    mu     = roll.mean().shift(1).bfill().values.astype(np.float32)
    sd     = roll.std().shift(1).bfill().fillna(1.0).values.astype(np.float32)
    sd     = np.where(sd < 1e-8, 1.0, sd)
    s_norm = ((surp - mu) / sd).astype(np.float32)

    # Causal rolling rank (fraction of past `window` bars with lower surprise)
    s_rank = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        past      = surp[max(0, i - window) : i]
        s_rank[i] = float(np.searchsorted(np.sort(past), surp[i])) / len(past)

    return np.column_stack([s_norm, s_rank])                         # (n, 2)


# ---------------------------------------------------------------------------
# Combined execution layer
# ---------------------------------------------------------------------------

class SupervisedExecutionLayer(nn.Module):
    """
    Combines the three heads into a single entry signal.

    Usage:
        layer = SupervisedExecutionLayer(vol_thresholds=(p33, p67))
        # after training the sub-heads, call .entry_signal(z_t) at each bar.

    Vol thresholds are stored as non-trainable buffers so they are
    saved/loaded with the model state_dict.
    """

    def __init__(
        self,
        z_dim:          int   = 128,
        hidden:         int   = 64,
        vol_thresholds: Tuple[float, float] = (0.0, 1.0),
        mag_threshold:  float = 0.0,
        history_len:    int   = 1,
        dir_in_dim:     Optional[int] = None,  # if set, overrides z_dim*history_len for dir head
    ):
        super().__init__()
        self.history_len = history_len
        self.vol_head = VolatilityHead(z_dim, hidden)
        self.dir_head = DirectionalHead(z_dim, hidden=hidden, history_len=history_len,
                                        in_dim=dir_in_dim)
        self.mag_head = MagnitudeHead(z_dim, hidden)
        # probe_bundle is assigned externally after training; stored as sub-module
        # so it is included in state_dict automatically.

        self.register_buffer("vol_p33",        torch.tensor(vol_thresholds[0]))
        self.register_buffer("vol_p67",        torch.tensor(vol_thresholds[1]))
        self.register_buffer("mag_threshold",  torch.tensor(float(mag_threshold)))

    def forward(self, z: torch.Tensor, z_hist: Optional[torch.Tensor] = None) -> dict:
        """
        Returns all signals for a batch of embeddings.

        Args:
            z:      (B, z_dim) — current embedding (used by vol/mag heads).
            z_hist: (B, H, z_dim) — history including current frame. If None,
                    the current z is tiled to fill all history slots (flat prior).

        Output dict:
            vol_probs:  (B, 3)  softmax regime probabilities
            vol_regime: (B,)    predicted class (0=low, 1=medium, 2=high)
            dir_prob:   (B,)    P(forward return > trend median)
            magnitude:  (B,)    predicted |forward return|
        """
        if z_hist is None:
            z_hist = z.unsqueeze(1).expand(-1, self.history_len, -1)
        vol_logits = self.vol_head(z)
        return {
            "vol_probs":  torch.softmax(vol_logits, dim=-1),
            "vol_regime": vol_logits.argmax(dim=-1),
            "dir_prob":   self.dir_head.predict_prob(z_hist),
            "magnitude":  self.mag_head.predict(z),
        }

    @torch.no_grad()
    def entry_signal(
        self,
        z:             torch.Tensor,
        z_hist:        Optional[torch.Tensor] = None,
        session_id:    int   = SESSION_ASIAN,
        spread:        float = 0.0003,
        dir_threshold: float = 0.53,
        atr_pct:       Optional[float] = None,
        risk_frac:     float = 0.01,
    ) -> torch.Tensor:
        """
        Entry signal. Filter 4 (S_t circuit breaker) applied by TradingEnvironment.

        Position sizing (two modes):
          ATR mode (atr_pct provided):
            size = min(1.0, risk_frac / atr_pct)  — risk exactly risk_frac of equity
            per one ATR of adverse move. Ignores vol_regime.
          Legacy mode (atr_pct=None):
            ±1.0  when vol_regime is low or medium
            ±0.5  when vol_regime is high  (half-Kelly sizing)
           0.0  when dir_prob in [1-dir_threshold, dir_threshold]

        Args:
            z:             (B, z_dim) current encoder embedding.
            z_hist:        (B, H, z_dim) history window. If None, z is tiled.
            session_id:    Kept for API compatibility; not used internally.
            spread:        Kept for API compatibility; not used internally.
            dir_threshold: Long if dir_prob > thresh, short if dir_prob < 1-thresh.
            atr_pct:       ATR_14 / close_price (dimensionless). When provided,
                           ATR-based sizing replaces the vol-regime binary rule.
            risk_frac:     Fraction of equity risked per ATR move (default 0.01 = 1%).

        Returns:
            (B,) float tensor.  Values in {-size, 0, +size} where size ∈ (0, 1].
        """
        out      = self.forward(z, z_hist)
        B        = z.shape[0]
        dir_prob = out["dir_prob"]

        go_long  = dir_prob > dir_threshold
        go_short = dir_prob < (1.0 - dir_threshold)

        signal = torch.zeros(B, dtype=torch.float32, device=z.device)
        signal[go_long]  =  1.0
        signal[go_short] = -1.0

        if atr_pct is not None:
            size = float(min(1.0, risk_frac / max(float(atr_pct), 1e-8)))
            signal = signal * size
        else:
            high_vol = out["vol_regime"] == 2
            signal[high_vol] *= 0.5

        return signal
