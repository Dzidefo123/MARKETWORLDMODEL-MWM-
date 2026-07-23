"""
SIGReg ? Sketched Isotropic Gaussian Regularizer
=================================================

WHY THIS EXISTS
---------------
When we train a world model with only a prediction loss (MSE between predicted
and actual next embedding), the encoder finds a "trivial" shortcut: it maps
EVERY market state to the same vector. The prediction loss becomes 0 because
the predictor always guesses the same constant. This is called "representation
collapse", and it makes the learned embeddings completely useless.

SIGReg solves this by demanding that the distribution of all embeddings Z
looks like an isotropic Gaussian (a bell curve in every direction). If all
embeddings were the same constant, they would form a single point ? nothing
like a Gaussian. So the regularizer directly fights collapse.

THE MATH (step by step)
------------------------
Given: Z ? R^(N?d) ? a batch of N embeddings, each d-dimensional

Step 1 ? Random projections (Cram?r-Wold theorem)
    To check if Z follows an isotropic Gaussian in d dimensions, we use the
    Cram?r-Wold theorem: a d-dimensional distribution matches a target if and
    only if ALL its 1D projections match the target's 1D projections.
    
    So instead of working in d dimensions, we project Z onto M random unit
    vectors u^(m) ? S^(d-1):
        h^(m) = Z ? u^(m)    ->   shape (N,)
    
    Each h^(m) should follow a standard N(0,1) distribution.

Step 2 ? Epps-Pulley test statistic
    For each projection h^(m), we measure how far its distribution is from
    N(0,1) using the Epps-Pulley test. This test works via characteristic
    functions (Fourier transforms of probability distributions):
    
    The empirical characteristic function (ECF) of h is:
        ?_N(t; h) = (1/N) ?_n exp(i?t?h_n)
                   = (1/N) ?_n [cos(t?h_n) + i?sin(t?h_n)]
    
    The target N(0,1) characteristic function is:
        ?_0(t) = exp(-t?/2)    (purely real)
    
    The test statistic is the weighted squared distance:
        T(h) = ? w(t) ? |?_N(t;h) - ?_0(t)|? dt
    
    where w(t) = exp(-t?/(2??)) is a Gaussian weight that down-weights large t.
    
    The integral is approximated by the trapezoid rule over t ? [0.2, 4.0].
    T(h) = 0 iff h ~ N(0,1). We MINIMIZE T, so the encoder is pushed toward
    Gaussian embeddings.

Step 3 ? Aggregate over projections
    SIGReg(Z) = (1/M) ?_m T(h^(m))

Total loss:
    L = L_pred + ? ? SIGReg(Z)
    
    Only ? needs tuning. M and the quadrature nodes are insensitive (per LeWM).

TRADING INTERPRETATION
-----------------------
SIGReg ensures that different market states (trending gold vs ranging gold,
London session vs Asian session, high-volatility vs low-volatility) map to
DIFFERENT regions of the latent space. Without it, everything collapses to
one point and regime detection is impossible.
"""

import torch
import torch.nn as nn
import math


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.
    
    Args:
        d_model:            Embedding dimension (must match encoder z_dim)
        n_projections:      Number of random unit vectors M (default 1024)
        n_knots:            Number of trapezoid quadrature nodes (default 17)
        t_min, t_max:       Integration range [0.2, 4.0] per LeWM paper
        weight_lambda:      Width of the Gaussian weight function w(t)
    
    Usage:
        sigreg = SIGReg(d_model=128)
        Z = encoder(x)          # Z has shape (batch, time, 128) or (batch, 128)
        loss = sigreg(Z)        # scalar loss
    """
    
    def __init__(
        self,
        d_model: int,
        n_projections: int = 1024,
        n_knots: int = 17,
        t_min: float = 0.2,
        t_max: float = 4.0,
        weight_lambda: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_projections = n_projections
        self.n_knots = n_knots
        self.weight_lambda = weight_lambda
        
        # Pre-compute fixed quadrature nodes t ? [t_min, t_max]
        # These are the points at which we evaluate the integral.
        # shape: (n_knots,)
        t_nodes = torch.linspace(t_min, t_max, n_knots)
        self.register_buffer("t_nodes", t_nodes)
        
        # Pre-compute the Gaussian weighting function w(t) = exp(-t? / (2??))
        # Applied to each quadrature node. Shape: (n_knots,)
        weights = torch.exp(-t_nodes ** 2 / (2.0 * weight_lambda ** 2))
        self.register_buffer("weights", weights)
        
        # Pre-compute the target characteristic function ?_0(t) = exp(-t?/2)
        # This is the CF of a standard N(0,1) distribution.
        # It is purely real (imaginary part = 0).
        # Shape: (n_knots,)
        target_cf = torch.exp(-t_nodes ** 2 / 2.0)
        self.register_buffer("target_cf", target_cf)
        
        # NOTE: We do NOT register projections as a buffer.
        # We sample fresh random projections at each forward pass.
        # This "sketching" gives better coverage of the hypersphere
        # and prevents the encoder from gaming a fixed set of directions.
    
    def _epps_pulley_statistic(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Epps-Pulley test statistic T(h) for a 1D sample h.
        
        Args:
            h: (N,) ? projected embeddings (one projection direction)
        
        Returns:
            scalar ? T(h) ? 0, equals 0 iff h ~ N(0,1)
        
        Math:
            ?_N(t; h) = (1/N) ?_n [cos(t?h_n) + i?sin(t?h_n)]
            T(h) = ? w(t) ? [(Re ?_N - ?_0)? + (Im ?_N)?] dt
                 ? ?_k w(t_k) ? [(Re ?_N(t_k) - ?_0(t_k))? + Im ?_N(t_k)?] ? ?t
        """
        N = h.shape[0]
        
        # Outer product: t_nodes (n_knots,) ? h (N,) -> (n_knots, N)
        # th[k, n] = t_k * h_n
        th = self.t_nodes.unsqueeze(1) * h.unsqueeze(0)  # (n_knots, N)
        
        # Empirical characteristic function
        # Real part: ?_N_real(t_k) = (1/N) ?_n cos(t_k * h_n)
        # Imag part: ?_N_imag(t_k) = (1/N) ?_n sin(t_k * h_n)
        ecf_real = torch.cos(th).mean(dim=1)  # (n_knots,)
        ecf_imag = torch.sin(th).mean(dim=1)  # (n_knots,)
        
        # Squared distance from target CF at each quadrature node
        # |?_N(t_k) - ?_0(t_k)|? = (ecf_real - target_cf)? + ecf_imag?
        diff_sq = (ecf_real - self.target_cf) ** 2 + ecf_imag ** 2  # (n_knots,)
        
        # Weighted integrand: w(t_k) * |?_N(t_k) - ?_0(t_k)|?
        integrand = self.weights * diff_sq  # (n_knots,)
        
        # Approximate integral via trapezoid rule
        stat = torch.trapezoid(integrand, self.t_nodes)
        
        return stat  # scalar
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss for a batch of embeddings Z.
        
        Args:
            Z: Tensor of shape (N, d) or (B, T, d)
               B = batch size, T = time steps, d = embedding dim
               If 3D, we flatten B and T into N.
        
        Returns:
            scalar loss ? minimize this to push Z toward isotropic Gaussian
        
        What happens here:
            1. Flatten Z to (N, d)
            2. Sample M random unit vectors on S^(d-1)
            3. Project Z onto each: h^(m) = Z ? u^(m)  ->  (N,)
            4. Compute Epps-Pulley statistic for each h^(m)
            5. Return mean over M projections
        """
        # Step 1 ? Flatten to (N, d)
        if Z.dim() == 3:
            B, T, d = Z.shape
            Z_flat = Z.reshape(B * T, d)  # (N, d) where N = B*T
        elif Z.dim() == 2:
            Z_flat = Z                    # (N, d) already
        else:
            raise ValueError(f"Z must be 2D or 3D, got shape {Z.shape}")
        
        N, d = Z_flat.shape
        
        # Safety check
        assert d == self.d_model, \
            f"Z has embedding dim {d} but SIGReg expects {self.d_model}"
        
        # Step 2 ? Sample M random unit vectors on S^(d-1)
        # We sample from N(0,I) and normalize ? this gives uniform
        # distribution on the hypersphere.
        # Shape: (d, M)
        directions = torch.randn(d, self.n_projections, device=Z.device)
        directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-8)
        
        # Step 3 ? Project embeddings
        # Z_flat: (N, d) @ directions: (d, M) -> projections: (N, M)
        projections = Z_flat @ directions  # (N, M)
        
        # Step 4 ? Epps-Pulley statistic for each projection direction
        # We vectorize this by transposing: each column of projections is h^(m)
        # Process all M projections in batch for efficiency
        total_stat = torch.tensor(0.0, device=Z.device)
        
        # Vectorized computation across all M projections at once
        # projections.T: (M, N)
        # t_nodes:       (n_knots,)
        # th:            (n_knots, M, N) via broadcasting
        t = self.t_nodes.view(-1, 1, 1)          # (n_knots, 1, 1)
        h = projections.T.unsqueeze(0)            # (1, M, N)
        th_all = t * h                            # (n_knots, M, N) ? broadcasts
        
        ecf_real_all = torch.cos(th_all).mean(dim=2)   # (n_knots, M)
        ecf_imag_all = torch.sin(th_all).mean(dim=2)   # (n_knots, M)
        
        target = self.target_cf.view(-1, 1)             # (n_knots, 1)
        diff_sq_all = (ecf_real_all - target) ** 2 + ecf_imag_all ** 2  # (n_knots, M)
        
        weights = self.weights.view(-1, 1)               # (n_knots, 1)
        integrand_all = weights * diff_sq_all            # (n_knots, M)
        
        # Trapezoid integration over t for each projection -> (M,)
        stats = torch.trapezoid(integrand_all, self.t_nodes.unsqueeze(1).expand_as(integrand_all), dim=0)
        
        # Step 5 ? Mean over all M projections
        sigreg_loss = stats.mean()
        
        return sigreg_loss


if __name__ == "__main__":
    """
    Quick sanity check:
    - If Z ~ N(0, I), SIGReg loss should be near 0
    - If Z = constant (collapsed), SIGReg loss should be large
    """
    print("=== SIGReg Sanity Check ===\n")
    
    d, N = 128, 512
    sigreg = SIGReg(d_model=d)
    
    # Case 1: Gaussian embeddings (ideal ? loss should be small)
    Z_gaussian = torch.randn(N, d)
    loss_gaussian = sigreg(Z_gaussian)
    print(f"Z ~ N(0, I) -> SIGReg loss: {loss_gaussian.item():.4f}  (expect near 0)")
    
    # Case 2: Collapsed embeddings (all zeros ? loss should be large)
    Z_collapsed = torch.zeros(N, d)
    loss_collapsed = sigreg(Z_collapsed)
    print(f"Z = 0 (collapsed) -> SIGReg loss: {loss_collapsed.item():.4f}  (expect large)")
    
    # Case 3: 3D input (batch, time, d) ? shape used during training
    Z_3d = torch.randn(32, 48, d)
    loss_3d = sigreg(Z_3d)
    print(f"Z shape (32, 48, {d}) -> SIGReg loss: {loss_3d.item():.4f}  (expect near 0)")
    
    print("\n? All cases computed successfully")
