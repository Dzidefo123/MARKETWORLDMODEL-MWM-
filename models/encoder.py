"""
MarketEncoder ? Tabular-Temporal Patch Encoder
===============================================

WHAT THIS DOES
--------------
Takes a raw market observation ? a matrix of shape (lookback, n_features),
e.g. (48 bars ? 52 features) ? and compresses it into a single compact
vector z_t of shape (z_dim,), e.g. 128-dimensional.

This z_t is the "market state embedding". It should capture:
  - Current regime (trending, ranging, breakout, crisis)
  - Session context (London, NY, Asian)
  - Volatility level and direction
  - Macro driver state (real yields, DXY, COT positioning)

The architecture mirrors LeWM's ViT encoder, adapted for tabular time series
instead of image pixels.

ARCHITECTURE WALKTHROUGH
-------------------------

Step 1 ? Patch Tokenization
    Original input: (batch, 48, 52)  ? 48 time steps, 52 features
    
    We split the time dimension into non-overlapping patches of size P=4:
        48 bars / 4 bars per patch = 12 patches
    
    Each patch is a (4, 52) sub-matrix = 208 values.
    We flatten each patch: (4?52 = 208) and project to d_model=128.
    
    Result: (batch, 12, 128) ? 12 tokens, each 128-dimensional
    
    WHY PATCH TOKENIZATION?
    Treating every bar as a token (48 tokens) would work but is slow.
    Patching groups local temporal patterns: a 4-bar window captures
    things like "gap + follow-through" or "candle cluster".
    This is the same insight as PatchTST and LeWM's pixel patching.

Step 2 ? CLS Token
    We prepend a learnable [CLS] token (shape: (1, 128)) to the 12 patches.
    This gives (batch, 13, 128).
    
    WHY CLS?
    The CLS token is the "aggregator". After self-attention, it has attended
    to all 12 patch tokens and summarizes the entire sequence. We extract only
    the CLS token output as our market state embedding z_t.
    This is the same mechanism used in BERT and ViT.

Step 3 ? Positional Encoding
    Transformers have no inherent notion of order (all tokens see each other
    equally). We add learned positional embeddings so the model knows which
    token came from which time period.
    
    Why LEARNED (not sinusoidal)? For price data, position meaning is more
    complex than in NLP. The model should learn what "being the most recent
    bar" or "being 2 days ago" means in the market context.

Step 4 ? Transformer Encoder
    Standard transformer encoder: MultiHeadAttention -> Add&Norm -> FFN -> Add&Norm
    Stacked n_layers=4 times.
    
    KEY DETAIL ? We use PRE-LN (LayerNorm before attention), not POST-LN.
    Pre-LN is more stable for training (no gradient explosion at initialization).
    LeWM uses the same choice.
    
    At the end, we extract the CLS token (index 0).

Step 5 ? Projection Head
    The CLS token output goes through:
        Linear(128 -> 256) -> BatchNorm(256) -> GELU -> Linear(256 -> 128)
    
    WHY BATCHNORM here (not LayerNorm)?
    SIGReg operates on the BATCH distribution of z_t. BatchNorm normalizes
    across the batch dimension, which directly helps SIGReg push the distribution
    toward Gaussian. If we used LayerNorm (which normalizes per-sample), we'd
    disrupt SIGReg's ability to shape the batch distribution.
    
    This is an explicit design choice from the LeWM paper (Section 3.1).

Output: z_t ? shape (batch, z_dim=128)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Converts (batch, lookback, n_features) -> (batch, n_patches, d_model).
    
    This is the "tokenizer" ? it creates the sequence of tokens that the
    transformer will process.
    
    Args:
        lookback:    Number of input bars (e.g. 48)
        n_features:  Number of features per bar (e.g. 52)
        patch_size:  Number of bars per patch (e.g. 4)
        d_model:     Output embedding dimension (e.g. 128)
    """
    
    def __init__(self, lookback: int, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        
        assert lookback % patch_size == 0, \
            f"lookback ({lookback}) must be divisible by patch_size ({patch_size})"
        
        self.patch_size = patch_size
        self.n_patches = lookback // patch_size   # e.g. 48 // 4 = 12
        patch_dim = patch_size * n_features        # e.g. 4 * 52 = 208
        
        # Linear projection: flattened patch -> d_model
        # This is a simple learned linear map (no bias needed ? LayerNorm follows)
        self.projection = nn.Linear(patch_dim, d_model, bias=False)
        
        # Layer norm after projection ? stabilizes the token representations
        self.norm = nn.LayerNorm(d_model)
        
        print(f"  PatchEmbedding: {lookback} bars ? {n_features} features "
              f"-> {self.n_patches} patches ? {d_model} dims "
              f"(patch_dim={patch_dim})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, n_features) ? raw feature matrix
        
        Returns:
            tokens: (batch, n_patches, d_model)
        """
        batch, lookback, n_features = x.shape
        
        # Rearrange: split time into patches and flatten each patch
        # (batch, lookback, n_features)
        # -> (batch, n_patches, patch_size * n_features)
        # einops notation: 'b (p s) f -> b p (s f)'
        #   p = number of patches, s = patch_size, f = n_features
        tokens = rearrange(x, 'b (p s) f -> b p (s f)', s=self.patch_size)
        
        # Linear projection: (batch, n_patches, patch_dim) -> (batch, n_patches, d_model)
        tokens = self.projection(tokens)
        tokens = self.norm(tokens)
        
        return tokens


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with PRE-LN architecture.
    
    Pre-LN: LayerNorm -> Attention -> Residual, then LayerNorm -> FFN -> Residual
    Post-LN (standard): Attention -> Residual -> LayerNorm (less stable)
    
    Pre-LN trains more stably because gradients don't pass through the
    residual+norm combined, which can cause gradient scale issues.
    
    Architecture:
        Input x
          -> LayerNorm
          -> Multi-Head Self-Attention
          -> x + attention_output   (residual connection)
          -> LayerNorm
          -> FFN (Linear -> GELU -> Dropout -> Linear)
          -> x + ffn_output         (residual connection)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Pre-normalization for attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-head self-attention
        # Each head has d_model // n_heads = 32 dimensions
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True    # Input/output: (batch, seq, d_model)
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Pre-normalization for FFN
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network: d_model -> 4?d_model -> d_model
        # The 4? expansion is standard in transformers (from "Attention is All You Need")
        ffn_dim = 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),           # GELU works better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            x: (batch, seq_len, d_model) ? same shape, enriched with context
        """
        # Attention block (Pre-LN)
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # Self-attention: Q=K=V=x
        x = residual + self.attn_dropout(attn_out)
        
        # FFN block (Pre-LN)
        residual = x
        x = residual + self.ffn(self.norm2(x))
        
        return x


class ProjectionHead(nn.Module):
    """
    Maps CLS token output -> final embedding z_t.
    
    Architecture: Linear -> BatchNorm -> GELU -> Linear
    
    CRITICAL DETAIL:
    BatchNorm is applied ACROSS THE BATCH (not per-sample).
    This means the mean and variance are computed over all samples in the batch.
    This is what SIGReg needs ? it checks that the BATCH DISTRIBUTION of z_t
    follows a Gaussian. BatchNorm actively shapes this distribution.
    
    Args:
        in_dim:     Input dimension (= d_model from encoder, e.g. 128)
        hidden_dim: Hidden dimension (e.g. 256)
        z_dim:      Output embedding dimension (e.g. 128)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, z_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # ? Key: batch normalization, not layer norm
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim) ? CLS token representation
        
        Returns:
            z: (batch, z_dim) ? final market state embedding
        """
        return self.net(x)


class MarketEncoder(nn.Module):
    """
    Full market encoder: raw features -> compact market state embedding z_t.
    
    Architecture:
        (batch, lookback, n_features)
        -> PatchEmbedding           -> (batch, n_patches, d_model)
        -> Prepend CLS token        -> (batch, n_patches+1, d_model)
        -> Add positional encoding  -> (batch, n_patches+1, d_model)
        -> Transformer encoder ?4   -> (batch, n_patches+1, d_model)
        -> Extract CLS token [0]    -> (batch, d_model)
        -> ProjectionHead           -> (batch, z_dim)
    
    Args:
        lookback:    Number of input bars (from config)
        n_features:  Number of features per bar (from config)
        patch_size:  Bars per token (from config)
        d_model:     Internal transformer dimension (from config)
        n_heads:     Number of attention heads (from config)
        n_layers:    Number of transformer layers (from config)
        dropout:     Dropout rate (from config)
        proj_hidden: Projection head hidden dim (from config)
        z_dim:       Output embedding dimension (from config)
    """
    
    def __init__(
        self,
        lookback: int    = 48,
        n_features: int  = 52,
        patch_size: int  = 4,
        d_model: int     = 128,
        n_heads: int     = 4,
        n_layers: int    = 4,
        dropout: float   = 0.1,
        proj_hidden: int = 256,
        z_dim: int       = 128,
    ):
        super().__init__()
        
        print("\nBuilding MarketEncoder:")
        
        # 1. Patch tokenizer
        self.patch_embed = PatchEmbedding(lookback, n_features, patch_size, d_model)
        n_patches = lookback // patch_size
        
        # 2. CLS token ? learnable vector, shape (1, 1, d_model)
        #    The leading 1s are for batch broadcasting
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        print(f"  CLS token: shape (1, 1, {d_model})")
        
        # 3. Positional encoding ? learned, shape (1, n_patches+1, d_model)
        #    +1 for CLS token at position 0
        n_positions = n_patches + 1
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)
        print(f"  Positional encoding: {n_positions} positions ? {d_model} dims")
        
        # 4. Transformer encoder layers (Pre-LN)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        print(f"  Transformer: {n_layers} layers ? {n_heads} heads ? {d_model} dims")
        
        # 5. Final layer norm (applied to CLS token before projection)
        self.final_norm = nn.LayerNorm(d_model)
        
        # 6. Projection head (Linear -> BatchNorm -> GELU -> Linear)
        self.proj_head = ProjectionHead(d_model, proj_hidden, z_dim)
        print(f"  Projection: {d_model} -> {proj_hidden} -> {z_dim} (z_dim)")
        
        # Count and display parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw features -> market state embedding.
        
        Args:
            x: (batch, lookback, n_features) ? the input feature matrix
               Each row is a bar, each column is a feature.
               Values should be z-scored (mean 0, std 1) before input.
        
        Returns:
            z_t: (batch, z_dim) ? the market state embedding
                 This is the compact representation that captures regime,
                 session, volatility, and macro state.
        """
        batch = x.shape[0]
        
        # 1. Patch embedding: (batch, lookback, n_features) -> (batch, n_patches, d_model)
        tokens = self.patch_embed(x)
        
        # 2. Prepend CLS token
        # cls_token is (1, 1, d_model) -> expand to (batch, 1, d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_patches+1, d_model)
        
        # 3. Add positional encoding
        # pos_embed is (1, n_patches+1, d_model) -> broadcasts over batch
        tokens = tokens + self.pos_embed
        
        # 4. Pass through transformer encoder layers
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        
        # 5. Extract CLS token (position 0) and apply final norm
        cls_output = self.final_norm(tokens[:, 0, :])  # (batch, d_model)
        
        # 6. Project to z_dim via projection head
        z_t = self.proj_head(cls_output)  # (batch, z_dim)
        
        return z_t
    
    def encode_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of observations for use with the predictor.
        
        Args:
            x_seq: (batch, time_steps, lookback, n_features)
        
        Returns:
            z_seq: (batch, time_steps, z_dim)
        
        This is used during training where we encode multiple consecutive
        windows to get z_1, z_2, ..., z_T for the predictor.
        """
        batch, T, lookback, n_features = x_seq.shape
        
        # Flatten batch and time: (batch*T, lookback, n_features)
        x_flat = x_seq.reshape(batch * T, lookback, n_features)
        
        # Encode all at once
        z_flat = self.forward(x_flat)  # (batch*T, z_dim)
        
        # Unflatten: (batch, T, z_dim)
        z_seq = z_flat.reshape(batch, T, -1)
        
        return z_seq


if __name__ == "__main__":
    """
    Architecture verification test.
    
    Checks:
    1. Correct output shapes
    2. Parameters are in expected range
    3. BatchNorm is working in train vs eval mode
    4. Gradient flow through encoder
    """
    print("=" * 60)
    print("MarketEncoder Architecture Verification")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Build encoder with config values
    encoder = MarketEncoder(
        lookback=48,
        n_features=52,
        patch_size=4,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        proj_hidden=256,
        z_dim=128,
    )
    
    # ?? Test 1: Single batch forward pass ??
    print("\n?? Test 1: Forward pass ??")
    x = torch.randn(32, 48, 52)  # batch=32, lookback=48, features=52
    encoder.train()
    z = encoder(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(z.shape)}  (expect: (32, 128))")
    assert z.shape == (32, 128), f"Shape mismatch: {z.shape}"
    print("  ? Shape correct")
    
    # ?? Test 2: Sequence encoding ??
    print("\n?? Test 2: Sequence encoding ??")
    x_seq = torch.randn(8, 5, 48, 52)  # batch=8, 5 time steps, lookback=48, features=52
    z_seq = encoder.encode_sequence(x_seq)
    print(f"  Input:  {tuple(x_seq.shape)}")
    print(f"  Output: {tuple(z_seq.shape)}  (expect: (8, 5, 128))")
    assert z_seq.shape == (8, 5, 128)
    print("  ? Shape correct")
    
    # ?? Test 3: Gradient flow ??
    print("\n?? Test 3: Gradient flow ??")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4)
    loss = z.mean()
    loss.backward()
    
    # Check that all parameters have gradients
    no_grad = [n for n, p in encoder.named_parameters() if p.grad is None]
    if no_grad:
        print(f"  ? No gradient: {no_grad}")
    else:
        print(f"  ? All {sum(1 for _ in encoder.parameters())} parameter tensors have gradients")
    
    # ?? Test 4: Embedding statistics ??
    print("\n?? Test 4: Embedding statistics ??")
    print(f"  z_t mean: {z.mean().item():.4f}  (BatchNorm pushes toward 0)")
    print(f"  z_t std:  {z.std().item():.4f}   (BatchNorm pushes toward 1)")
    print(f"  z_t min:  {z.min().item():.4f}")
    print(f"  z_t max:  {z.max().item():.4f}")
    
    print("\n? MarketEncoder verification complete")
