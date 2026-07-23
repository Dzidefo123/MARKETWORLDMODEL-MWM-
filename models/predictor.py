"""
models/predictor.py ? Causal Market Dynamics Predictor
========================================================

WHAT THIS DOES
--------------
Takes a history of market state embeddings z_{t-N+1}, ..., z_t and a
macro event vector a_t, and predicts the NEXT market state embedding z_{t+1}.

This is the "world model" proper. While the encoder (encoder.py) answers
"what is the current market state?", the predictor answers "given the current
state and a macro event, what happens next?"

ARCHITECTURE
------------

    Input:
        z_history: (batch, history_len, z_dim)  ? past N market states
        a_t:       (batch, action_dim)           ? macro event at time t

    Processing:
        1. Project z_history to predictor d_model
        2. Add learnable positional encodings
        3. Pass through N causal transformer layers
           ? Each layer uses AdaLN conditioning on a_t
           ? Causal masking: each position only sees earlier positions
        4. Extract last position output
        5. Project back to z_dim

    Output:
        z_hat_{t+1}: (batch, z_dim) ? predicted next market state

WHY CAUSAL MASKING?
-------------------
Without causal masking, the predictor at position t could "look at"
position t+1, t+2, etc. during training ? it would learn to just copy
future embeddings rather than predict them. This is lookahead bias at
the model architecture level.

Causal masking forces: position t can only attend to positions 0..t.
The last position (t) has seen all history but not the future.

WHY AdaLN? (Adaptive Layer Normalization)
------------------------------------------
Standard approach: concatenate [z_t, a_t] and feed to a linear layer.
Problem: this gives the action a fixed, additive effect on every layer.

AdaLN (from DiT ? Diffusion Transformers, Peebles & Xie 2023, same paper
LeWM cites as reference [37]):
    ? Project a_t -> (?, ?) for each transformer layer
    ? Apply: AdaLN(x) = LayerNorm(x) * (1 + ?) + ?
    ? ? controls the SCALE of each feature after normalization
    ? ? controls the SHIFT

This allows the macro event to modulate the dynamics model in a
multiplicative way ? much more expressive than simple concatenation.
The scale effect lets the action "amplify" or "suppress" specific
dimensions of the market state, which maps naturally to how macro events
work (e.g., an NFP beat amplifies DXY-related dimensions while suppressing
safe-haven dimensions).

INITIALIZATION
--------------
AdaLN parameters are initialized to zero (?=0, ?=0). At initialization,
AdaLN(x) = LayerNorm(x) * 1 + 0 = LayerNorm(x) ? equivalent to standard
LayerNorm. The action conditioning is learned gradually during training.
This is the same "zero-init" trick LeWM uses for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ??????????????????????????????????????????????????????????????
# AdaLN Block ? the core conditioning mechanism
# ??????????????????????????????????????????????????????????????

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Standard LayerNorm normalizes x to zero mean, unit variance,
    then applies learnable affine parameters (?_fixed, ?_fixed).

    AdaLN replaces those fixed parameters with DYNAMIC ones computed
    from the conditioning signal (macro event vector a_t):
        ?, ? = Linear(a_t)    ? both have shape (d_model,)
        output = LayerNorm(x) * (1 + ?) + ?

    The (1 + ?) ensures that at initialization (?=0) we get standard
    LayerNorm behaviour, making training stable from the start.

    Args:
        d_model:    Dimension of the input x
        cond_dim:   Dimension of the conditioning signal a_t
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # elementwise_affine=False: LayerNorm has no learnable params of its own.
        # All affine transformation comes from the conditioning signal.

        # Projects conditioning signal -> (scale ?, shift ?)
        # Output dim is 2 * d_model: first half is ?, second half is ?
        self.cond_proj = nn.Linear(cond_dim, 2 * d_model)

        # Zero-initialize: at the start of training, ?=0 and ?=0
        # so AdaLN reduces to plain LayerNorm. Conditioning grows gradually.
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, d_model) ? input to normalize
            cond: (batch, cond_dim)         ? conditioning signal (a_t)

        Returns:
            (batch, seq_len, d_model) ? conditioned normalized output
        """
        # Normalize x to zero mean, unit variance (no affine yet)
        x_norm = self.norm(x)   # (batch, seq_len, d_model)

        # Compute dynamic scale and shift from conditioning signal
        # cond: (batch, cond_dim) -> (batch, 2*d_model)
        gamma_beta = self.cond_proj(cond)              # (batch, 2*d_model)

        # Split into ? (scale) and ? (shift)
        # We unsqueeze to broadcast over seq_len dimension
        gamma, beta = gamma_beta.chunk(2, dim=-1)      # each: (batch, d_model)
        gamma = gamma.unsqueeze(1)                     # (batch, 1, d_model)
        beta  = beta.unsqueeze(1)                      # (batch, 1, d_model)

        # Apply adaptive affine transformation
        # (1 + gamma): multiplicative modulation centered at 1
        return x_norm * (1 + gamma) + beta


# ??????????????????????????????????????????????????????????????
# Causal Predictor Layer
# ??????????????????????????????????????????????????????????????

class CausalPredictorLayer(nn.Module):
    """
    One layer of the causal predictor transformer.

    Architecture (Pre-LN with AdaLN conditioning):
        x -> AdaLN(x, a_t) -> Causal MHA -> x + attn_out
          -> AdaLN(x, a_t) -> FFN        -> x + ffn_out

    Both normalization steps use the SAME action conditioning ? the
    macro event modulates both attention and the feed-forward network.

    Args:
        d_model:   Model dimension
        n_heads:   Number of attention heads
        cond_dim:  Conditioning signal dimension (action_dim after projection)
        dropout:   Dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()

        # Pre-attention AdaLN
        self.adaLN1 = AdaLN(d_model, cond_dim)

        # Causal multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,   # (batch, seq, d_model) convention
        )
        self.attn_drop = nn.Dropout(dropout)

        # Pre-FFN AdaLN
        self.adaLN2 = AdaLN(d_model, cond_dim)

        # Feed-forward network: d_model -> 4?d_model -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:           torch.Tensor,   # (batch, seq_len, d_model)
        cond:        torch.Tensor,   # (batch, cond_dim)
        causal_mask: torch.Tensor,   # (seq_len, seq_len) ? boolean upper-triangular
    ) -> torch.Tensor:

        # ?? Attention block ??????????????????????????????????
        residual = x
        x_norm   = self.adaLN1(x, cond)   # apply macro conditioning before attention

        # Causal self-attention
        # attn_mask: True = IGNORE this position (PyTorch convention)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask      = causal_mask,
            need_weights   = False,
        )
        x = residual + self.attn_drop(attn_out)

        # ?? FFN block ????????????????????????????????????????
        residual = x
        x = residual + self.ffn(self.adaLN2(x, cond))

        return x


# ??????????????????????????????????????????????????????????????
# Full Causal Predictor
# ??????????????????????????????????????????????????????????????

class CausalPredictor(nn.Module):
    """
    Full causal market dynamics predictor.

    Autoregressively models:
        z_{t+1} = predictor(z_{t-N+1}, ..., z_t, a_t)

    The action a_t is the macro event that "caused" the transition
    from market state t to market state t+1.

    Args:
        z_dim:       Embedding dimension from encoder (default 128)
        d_model:     Internal predictor dimension (default 128)
        n_heads:     Attention heads (default 4)
        n_layers:    Transformer depth (default 6 ? deeper than encoder)
        action_dim:  Macro event vector dimension (default 5)
        history_len: How many past z_t the predictor uses (default 3)
        dropout:     Dropout rate (default 0.1)

    Why n_layers=6 (deeper than encoder's 4)?
        Predicting dynamics is harder than encoding state.
        The encoder maps a fixed window to a summary.
        The predictor must model complex temporal relationships
        across history_len past states AND integrate the macro signal.
    """

    def __init__(
        self,
        z_dim:       int   = 128,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 6,
        action_dim:  int   = 5,
        history_len: int   = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.history_len = history_len
        self.d_model     = d_model

        # Project action to a richer conditioning space
        # Raw action_dim=5 is small; we expand it so AdaLN has more
        # expressive power to modulate the dynamics.
        cond_dim = max(64, d_model // 2)   # at least 64-dim conditioning
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection: z_dim -> d_model
        # (In our setup z_dim == d_model == 128, so this is identity-like,
        # but keeping it explicit allows changing z_dim independently.)
        self.input_proj = nn.Linear(z_dim, d_model)

        # Learnable positional encodings for the history window
        # Shape: (1, history_len, d_model) ? broadcasts over batch
        self.pos_embed = nn.Parameter(
            torch.randn(1, history_len, d_model) * 0.02
        )

        # Causal transformer layers with AdaLN conditioning
        self.layers = nn.ModuleList([
            CausalPredictorLayer(d_model, n_heads, cond_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final normalization before output projection
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: d_model -> z_dim
        self.output_proj = nn.Linear(d_model, z_dim)

        # Pre-compute causal mask as FLOAT additive mask (not bool).
        #
        # WHY FLOAT, NOT BOOL?
        # PyTorch 2.0+ uses scaled_dot_product_attention (SDPA) internally
        # inside nn.MultiheadAttention. SDPA and nn.MultiheadAttention use
        # OPPOSITE bool conventions:
        #   nn.MultiheadAttention: True  = blocked (cannot attend)
        #   SDPA (internal):       False = blocked (cannot attend)
        # This mismatch produces NaN at the very first forward pass.
        #
        # Float additive masks are unambiguous across all PyTorch versions:
        #   0.0  = position can attend normally
        #   -inf = position is blocked (softmax(-inf) = 0, never NaN)
        #
        # Shape: (history_len, history_len)
        mask = torch.triu(
            torch.full((history_len, history_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"  CausalPredictor: {total:,} parameters  "
              f"({n_layers} layers ? {n_heads} heads ? {d_model} dims  "
              f"history={history_len}  cond_dim={cond_dim})")

    def forward(
        self,
        z_history: torch.Tensor,   # (batch, history_len, z_dim)
        a_t:       torch.Tensor,   # (batch, action_dim)
    ) -> torch.Tensor:
        """
        Predict the next market state embedding.

        Args:
            z_history: Last history_len market state embeddings.
                       z_history[:, -1, :] is the CURRENT state z_t.
                       z_history[:, 0, :]  is the OLDEST state z_{t-N+1}.
            a_t:       Macro event vector at time t.

        Returns:
            z_hat_{t+1}: (batch, z_dim) ? predicted next market state

        The model only uses z_history[:, -1, :] for the final prediction
        (it's a causal model ? last position can attend to all history
        but outputs only at the last position).
        """
        batch = z_history.shape[0]

        # ?? 1. Encode action into conditioning signal ??????????
        cond = self.action_encoder(a_t)   # (batch, cond_dim)

        # ?? 2. Project z_history to d_model ???????????????????
        x = self.input_proj(z_history)    # (batch, history_len, d_model)

        # ?? 3. Add positional encoding ?????????????????????????
        x = x + self.pos_embed            # broadcasts over batch

        # ?? 4. Causal transformer layers ???????????????????????
        for layer in self.layers:
            x = layer(x, cond, self.causal_mask)

        # ?? 5. Extract last position (current time t) ??????????
        # The last position has attended to all history and is conditioned
        # on a_t. Its output represents the model's belief about z_{t+1}.
        x_last = self.final_norm(x[:, -1, :])   # (batch, d_model)

        # ?? 6. Project to z_dim ????????????????????????????????
        z_hat = self.output_proj(x_last)         # (batch, z_dim)

        return z_hat

    def build_history(
        self,
        z_seq: torch.Tensor,
        t:     int,
    ) -> torch.Tensor:
        """
        Extract a history window ending at time step t.

        During training, we encode a full sequence z_1, z_2, ..., z_T
        and then for each t we need z_{t-N+1}, ..., z_t as input.

        This method handles the boundary case: if t < history_len,
        we pad the start with the first embedding (repeat z_1).

        Args:
            z_seq: (batch, T, z_dim) ? full sequence of embeddings
            t:     Index of the CURRENT time step (0-indexed)

        Returns:
            (batch, history_len, z_dim)
        """
        N = self.history_len
        batch, T, z_dim = z_seq.shape

        if t >= N - 1:
            # Enough history available: take z_{t-N+1}, ..., z_t
            return z_seq[:, t - N + 1 : t + 1, :]
        else:
            # Not enough history: pad with first embedding
            available = z_seq[:, :t + 1, :]            # (batch, t+1, z_dim)
            pad_len   = N - (t + 1)
            pad       = z_seq[:, :1, :].expand(-1, pad_len, -1)  # repeat z_0
            return torch.cat([pad, available], dim=1)  # (batch, N, z_dim)


if __name__ == "__main__":
    """
    Architecture verification tests for CausalPredictor.

    Verifies:
    1. Output shape is correct
    2. Causal mask is upper-triangular
    3. AdaLN conditions on action (different actions -> different outputs)
    4. Gradient flows through all parameters
    5. build_history handles boundary cases
    """
    print("=" * 60)
    print("CausalPredictor Verification")
    print("=" * 60)

    torch.manual_seed(42)

    Z_DIM      = 128
    ACTION_DIM = 5
    HISTORY    = 3
    BATCH      = 16

    print("\nBuilding predictor:")
    predictor = CausalPredictor(
        z_dim       = Z_DIM,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 6,
        action_dim  = ACTION_DIM,
        history_len = HISTORY,
        dropout     = 0.1,
    )

    # ?? Test 1: Forward pass shape ??????????????????????????
    print("\n?? Test 1: Forward pass ??")
    z_hist = torch.randn(BATCH, HISTORY, Z_DIM)
    a_t    = torch.randn(BATCH, ACTION_DIM)

    predictor.train()
    z_hat  = predictor(z_hist, a_t)

    print(f"  Input z_history: {tuple(z_hist.shape)}")
    print(f"  Input a_t:       {tuple(a_t.shape)}")
    print(f"  Output z_hat:    {tuple(z_hat.shape)}  (expect: ({BATCH}, {Z_DIM}))")
    assert z_hat.shape == (BATCH, Z_DIM)
    print("  ? Shape correct")

    # ?? Test 2: Causal mask ?????????????????????????????????
    print("\n?? Test 2: Causal mask structure ??")
    import math
    mask = predictor.causal_mask
    print(f"  Mask dtype: {mask.dtype}  (expect float32, not bool)")
    print(f"  Mask shape: {tuple(mask.shape)}")
    print(f"  Mask values (0.0=allowed  -inf=blocked):")
    print(mask)
    # Float additive mask: -inf=blocked, 0.0=allowed
    assert mask.dtype == torch.float32,    "Mask must be float32 not bool"
    assert math.isinf(mask[0, 1].item()), "mask[0,1] must be -inf (blocked)"
    assert mask[0, 1].item() < 0,         "mask[0,1] must be -inf not +inf"
    assert mask[1, 0].item() == 0.0,      "mask[1,0] must be 0.0 (allowed)"
    assert mask[0, 0].item() == 0.0,      "mask[0,0] must be 0.0 (allowed)"
    print("  ? Float causal mask correct  (-inf=blocked  0.0=allowed)")


    # ?? Test 3: AdaLN zero-init + post-update conditioning ????
    print("\n?? Test 3: AdaLN conditioning ??")
    #
    # WHY 0.0000 AT INIT IS CORRECT
    # --------------------------------
    # AdaLN's cond_proj is zero-initialized (weight=0, bias=0).
    # This means at initialization:
    #   gamma = cond_proj(a_t)[:, :d_model] = 0   for ANY a_t
    #   beta  = cond_proj(a_t)[:, d_model:] = 0   for ANY a_t
    #   AdaLN(x, a_t) = LayerNorm(x) * (1+0) + 0 = LayerNorm(x)
    #
    # So before any gradient update, ALL actions produce the same output.
    # This is intentional ? it makes training stable from random init
    # (the predictor starts as a plain transformer, conditioning grows slowly).
    #
    # After even ONE gradient step the cond_proj weights become non-zero
    # and different actions produce different outputs.
    #
    predictor.eval()
    with torch.no_grad():
        z_out_1 = predictor(z_hist, a_t)
        z_out_2 = predictor(z_hist, -a_t)
        diff_before = (z_out_1 - z_out_2).abs().mean().item()

    print(f"  Diff before training: {diff_before:.4f}  (expect 0.0 ? zero-init)")
    assert diff_before == 0.0, f"Expected 0.0 at init, got {diff_before}"
    print("  ? Zero-init confirmed (AdaLN = plain LayerNorm at init)")

    # Now do one gradient step and verify conditioning activates
    predictor.train()
    optimizer_test = torch.optim.SGD(predictor.parameters(), lr=0.1)
    optimizer_test.zero_grad()
    dummy_loss = predictor(z_hist, a_t).mean()
    dummy_loss.backward()
    optimizer_test.step()

    predictor.eval()
    with torch.no_grad():
        z_out_1 = predictor(z_hist, a_t)
        z_out_2 = predictor(z_hist, -a_t)
        diff_after = (z_out_1 - z_out_2).abs().mean().item()

    print(f"  Diff after 1 gradient step: {diff_after:.6f}  (expect > 0)")
    assert diff_after > 0, "AdaLN still not conditioning after gradient update!"
    print("  ? AdaLN conditioning activates after training begins")

    # ?? Test 4: Gradient flow ???????????????????????????????
    print("\n?? Test 4: Gradient flow ??")
    predictor.train()
    z_hat = predictor(z_hist, a_t)
    loss  = z_hat.mean()
    loss.backward()

    no_grad = [n for n, p in predictor.named_parameters() if p.grad is None]
    if no_grad:
        print(f"  ? No gradient: {no_grad[:5]}")
    else:
        n_params = sum(1 for _ in predictor.parameters())
        print(f"  ? All {n_params} parameter tensors have gradients")

    # ?? Test 5: build_history boundary cases ????????????????
    print("\n?? Test 5: build_history ??")
    z_seq = torch.randn(4, 10, Z_DIM)   # batch=4, T=10

    # Normal case: t=5, history_len=3 -> positions [3,4,5]
    h = predictor.build_history(z_seq, t=5)
    assert h.shape == (4, HISTORY, Z_DIM)
    assert torch.allclose(h[:, -1, :], z_seq[:, 5, :])
    print(f"  t=5 (normal):   {tuple(h.shape)} ? last pos matches z_seq[:,5,:]  ?")

    # Boundary case: t=1, history_len=3 -> pad + z[0], z[1]
    h = predictor.build_history(z_seq, t=1)
    assert h.shape == (4, HISTORY, Z_DIM)
    assert torch.allclose(h[:, -1, :], z_seq[:, 1, :])
    print(f"  t=1 (boundary): {tuple(h.shape)} ? padded with z_0  ?")

    print("\n? CausalPredictor fully verified")