"""
MWM Loss and Dataset for O1
============================

LOSS FUNCTION
-------------
The full O1 training objective (mirrors LeWM Eq. 3):

    L = L_pred + ? ? SIGReg(Z)

where:
    L_pred = MSE(z_hat_t+1, z_t+1)     ? prediction loss
    SIGReg(Z) = (1/M) ?_m T(h^(m))     ? anti-collapse regularizer

The prediction loss incentivizes the encoder to produce embeddings that
are PREDICTABLE from past context (i.e., market dynamics are learnable).
SIGReg prevents the trivial solution (constant embeddings).

DATASET
-------
A windowed time-series dataset that produces consecutive observation pairs
(x_t, x_t+1) for training the predictor. Each sample is:
    - x_t:   (lookback, n_features) ? current observation window
    - x_t+1: (lookback, n_features) ? next observation window (shifted by 1 bar)
    - a_t:   (action_dim,)          ? macro event vector at time t

For training, we also produce a sequence of length history_len+1 to provide
the predictor with historical context.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.sigreg import SIGReg
from models.encoder import MarketEncoder


# ????????????????????????????????????????????????????????????
# Loss Function
# ????????????????????????????????????????????????????????????

class MWMLoss(nn.Module):
    """
    Combined O1 training loss.
    
    L = MSE(z_hat_t+1, z_t+1) + ? * SIGReg(Z_all)
    
    where Z_all is the concatenation of all encoder outputs in the batch,
    applied at every time step (step-wise SIGReg, as in LeWM Algorithm 1).
    
    Args:
        sigreg:        Initialized SIGReg module
        lambda_weight: Weight for SIGReg (the ONE tunable hyperparameter)
    """
    
    def __init__(self, sigreg: SIGReg, lambda_weight: float = 0.1):
        super().__init__()
        self.sigreg = sigreg
        self.lambda_weight = lambda_weight
    
    def forward(
        self,
        z_t: torch.Tensor,       # Current embeddings: (batch, z_dim) or (batch, T, z_dim)
        z_t1_hat: torch.Tensor,  # Predicted next embeddings: same shape
        z_t1: torch.Tensor,      # Actual next embeddings: same shape
    ) -> dict:
        """
        Compute combined loss.
        
        Returns:
            dict with keys: 'total', 'pred', 'sigreg'
            Allows logging each component separately in MLflow.
        """
        # Prediction loss: how well does the predictor anticipate the next state?
        pred_loss = nn.functional.mse_loss(z_t1_hat, z_t1)
        
        # SIGReg: applied to ALL encoder outputs (both z_t and z_t1)
        # We concatenate along the batch dimension so SIGReg sees a larger sample
        # This is "step-wise SIGReg" from LeWM Algorithm 1
        if z_t.dim() == 3:
            # (batch, T, z_dim) -> flatten time into batch for SIGReg
            Z_all = torch.cat([z_t, z_t1], dim=1)   # (batch, 2T, z_dim)
        else:
            Z_all = torch.stack([z_t, z_t1], dim=1)  # (batch, 2, z_dim)
        
        sigreg_loss = self.sigreg(Z_all)
        
        # Combined loss
        total_loss = pred_loss + self.lambda_weight * sigreg_loss
        
        return {
            "total":  total_loss,
            "pred":   pred_loss,
            "sigreg": sigreg_loss,
        }


# ????????????????????????????????????????????????????????????
# Windowed Time-Series Dataset
# ????????????????????????????????????????????????????????????

class MarketWindowDataset(Dataset):
    """
    Sliding-window dataset for market data.
    
    Given a feature matrix of shape (n_bars_total, n_features), produces
    samples of consecutive windows for predictor training.
    
    Each sample contains:
        x_t:   (lookback, n_features)  ? window starting at bar i
        x_t1:  (lookback, n_features)  ? window starting at bar i+1
        a_t:   (action_dim,)           ? macro event at bar i+lookback
                                          (the event that happened AT bar i+lookback,
                                           which transitions x_t -> x_t1)
    
    WHY DO WE NEED CONSECUTIVE WINDOWS?
    The model learns: "Given the current market state z_t, and a macro event a_t,
    predict the next market state z_t+1."
    
    This requires pairs (x_t, x_t+1) where x_t+1 is x_t shifted forward by 1 bar.
    
    Args:
        features:   (n_bars, n_features) ? pre-computed, normalized feature matrix
        actions:    (n_bars, action_dim) ? macro event vectors (or zeros if unavailable)
        lookback:   Number of bars per window (e.g. 48)
        stride:     Step between windows (1 = every bar, larger = faster training)
    
    Example with lookback=48, n_bars=10000:
        Window 0: features[0:48]   -> features[1:49]
        Window 1: features[1:49]   -> features[2:50]
        ...
        Window 9951: features[9951:9999] -> features[9952:10000]
        
        Total windows: n_bars - lookback - 1 = 10000 - 48 - 1 = 9951
    """
    
    def __init__(
        self,
        features:    np.ndarray,   # (n_bars, n_features)
        actions:     np.ndarray,   # (n_bars, action_dim)
        lookback:    int = 48,
        stride:      int = 1,
        history_len: int = 1,      # consecutive windows to return as temporal history
    ):
        self.features    = torch.FloatTensor(features)
        self.actions     = torch.FloatTensor(actions)
        self.lookback    = lookback
        self.stride      = stride
        self.history_len = history_len

        n_bars = features.shape[0]
        self.indices = list(range(0, n_bars - lookback - 1, stride))

        print(f"  MarketWindowDataset: {n_bars} bars -> {len(self.indices)} samples "
              f"(lookback={lookback}, stride={stride}, history_len={history_len})")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict:
        i  = self.indices[idx]
        H  = self.history_len
        lb = self.lookback

        # Build history: H consecutive windows ending at x_t.
        # When i < H-1, pad by repeating window 0 ? matches predictor.build_history.
        x_hist_list = []
        for k in range(H - 1, -1, -1):   # k = H-1, H-2, ..., 0  (oldest -> newest)
            src = max(0, i - k)
            x_hist_list.append(self.features[src : src + lb])
        x_hist = torch.stack(x_hist_list)   # (H, lookback, n_features)
        x_t    = x_hist[-1]                  # (lookback, n_features) ? current window

        x_t1   = self.features[i + 1 : i + 1 + lb]   # (lookback, n_features)
        a_t    = self.actions[i + lb]                   # (action_dim,)

        return {"x_hist": x_hist, "x_t": x_t, "x_t1": x_t1, "a_t": a_t,
                "t_now": torch.tensor(i + lb - 1, dtype=torch.long)}


# ????????????????????????????????????????????????????????????
# Simple Predictor (Placeholder for O1 training)
# ????????????????????????????????????????????????????????????

class SimplePredictor(nn.Module):
    """
    Minimal predictor for O1 validation.
    
    Takes z_t and a_t, outputs z_hat_t+1.
    This is deliberately simple ? in O2 we replace this with a full
    causal transformer predictor with AdaLN conditioning.
    
    For O1, we just need something that creates a gradient signal
    through the encoder so we can verify the full training loop works.
    
    Architecture: [z_t || a_t] -> Linear -> GELU -> Linear -> z_hat_t+1
    """
    
    def __init__(self, z_dim: int = 128, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, z_dim),
        )
    
    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (batch, z_dim)
            a_t: (batch, action_dim)
        Returns:
            z_hat_t1: (batch, z_dim)
        """
        x = torch.cat([z_t, a_t], dim=-1)
        return self.net(x)


# ????????????????????????????????????????????????????????????
# End-to-End O1 Training Step
# ????????????????????????????????????????????????????????????

def make_synthetic_data(n_bars: int = 5000, n_features: int = 52, action_dim: int = 5):
    """
    Generate synthetic market-like data for testing the pipeline
    before real market data is integrated.
    
    We simulate:
    - Feature matrix: 3 regime clusters (trending, ranging, crisis)
    - Actions: random macro event vectors
    
    In production, this is replaced by the real feature pipeline.
    """
    np.random.seed(42)
    
    # Simulate 3 regime clusters with different means and variances
    n_trending = n_bars // 3
    n_ranging  = n_bars // 3
    n_crisis   = n_bars - n_trending - n_ranging
    
    trending = np.random.randn(n_trending, n_features) * 1.0 + 0.5   # higher mean
    ranging  = np.random.randn(n_ranging,  n_features) * 0.5         # lower variance
    crisis   = np.random.randn(n_crisis,   n_features) * 2.0 - 0.5   # higher variance, negative drift
    
    # Interleave the regimes to simulate realistic transitions
    # (not blocks ? mix them to mimic realistic regime switching)
    indices = np.argsort(np.random.rand(n_bars))
    all_data = np.vstack([trending, ranging, crisis])
    features = all_data[np.random.permutation(n_bars)]
    
    # Normalize to zero mean, unit std (critical for stable training)
    features = (features - features.mean(0)) / (features.std(0) + 1e-8)
    
    # Random macro event vectors
    actions = np.random.randn(n_bars, action_dim) * 0.1
    
    return features, actions


if __name__ == "__main__":
    """
    Full O1 pipeline integration test.
    Verifies: data -> encoder -> predictor -> loss -> backward -> weight update
    """
    print("=" * 60)
    print("O1 Pipeline Integration Test")
    print("=" * 60)
    
    # Config
    LOOKBACK     = 48
    N_FEATURES   = 52
    Z_DIM        = 128
    ACTION_DIM   = 5
    BATCH_SIZE   = 32
    LAMBDA       = 0.1
    
    # ?? Step 1: Synthetic data ??
    print("\n?? Step 1: Generating synthetic data ??")
    features, actions = make_synthetic_data(5000, N_FEATURES, ACTION_DIM)
    
    dataset = MarketWindowDataset(features, actions, lookback=LOOKBACK)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))
    
    x_t  = batch["x_t"]   # (32, 48, 52)
    x_t1 = batch["x_t1"]  # (32, 48, 52)
    a_t  = batch["a_t"]   # (32, 5)
    print(f"  x_t:  {tuple(x_t.shape)}")
    print(f"  x_t1: {tuple(x_t1.shape)}")
    print(f"  a_t:  {tuple(a_t.shape)}")
    
    # ?? Step 2: Build models ??
    print("\n?? Step 2: Building models ??")
    encoder   = MarketEncoder(lookback=LOOKBACK, n_features=N_FEATURES, z_dim=Z_DIM)
    predictor = SimplePredictor(z_dim=Z_DIM, action_dim=ACTION_DIM)
    sigreg    = SIGReg(d_model=Z_DIM)
    loss_fn   = MWMLoss(sigreg=sigreg, lambda_weight=LAMBDA)
    
    # ?? Step 3: Forward pass ??
    print("\n?? Step 3: Forward pass ??")
    encoder.train()
    predictor.train()
    
    z_t   = encoder(x_t)    # (32, 128)
    z_t1  = encoder(x_t1)   # (32, 128)
    z_hat = predictor(z_t, a_t)  # (32, 128)
    
    print(f"  z_t shape:   {tuple(z_t.shape)}")
    print(f"  z_hat shape: {tuple(z_hat.shape)}")
    
    # ?? Step 4: Loss ??
    print("\n?? Step 4: Computing loss ??")
    losses = loss_fn(z_t, z_hat, z_t1)
    print(f"  Total loss:  {losses['total'].item():.4f}")
    print(f"  Pred loss:   {losses['pred'].item():.4f}")
    print(f"  SIGReg loss: {losses['sigreg'].item():.4f}")
    
    # ?? Step 5: Backward pass ??
    print("\n?? Step 5: Backward pass ??")
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    optimizer.zero_grad()
    losses["total"].backward()
    
    # Gradient norm (diagnostic ? if too large, grad clipping is needed)
    total_norm = torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
    )
    print(f"  Gradient norm (pre-clip): {total_norm.item():.4f}")
    
    optimizer.step()
    print("  ? Weight update successful")
    
    # ?? Step 6: One training epoch ??
    print("\n?? Step 6: Mini training epoch (5 steps) ??")
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        
        optimizer.zero_grad()
        z_t   = encoder(batch["x_t"])
        z_t1  = encoder(batch["x_t1"])
        z_hat = predictor(z_t, batch["a_t"])
        losses = loss_fn(z_t, z_hat, z_t1)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), 1.0
        )
        optimizer.step()
        
        print(f"  Step {step+1}: total={losses['total'].item():.4f}  "
              f"pred={losses['pred'].item():.4f}  "
              f"sigreg={losses['sigreg'].item():.4f}")
    
    print("\n? Full O1 pipeline verified end-to-end")
