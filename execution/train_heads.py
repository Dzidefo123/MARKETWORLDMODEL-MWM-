"""
execution/train_heads.py — Training and evaluation for supervised execution heads.

Loads a trained encoder checkpoint, encodes the train/val splits once,
then trains all three heads on the frozen embeddings.

Usage:
    python -m execution.train_heads --instrument gold
    python -m execution.train_heads --instrument eurusd --horizon 4

Saves:
    experiments/heads_{instrument}/execution_layer.pt
    experiments/heads_{instrument}/metrics.json
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from data.pipeline      import DataPipeline
from models.encoder     import MarketEncoder
from models.predictor   import CausalPredictor
from execution.heads    import (
    SupervisedExecutionLayer,
    ProbeBundle,
    SESSION_NY,
    compute_labels,
    encode_split,
    build_z_history,
    train_probe_bundle,
    compute_surprise_features,
)


# ---------------------------------------------------------------------------
# Per-head training loops
# ---------------------------------------------------------------------------

def _train_classifier(
    head:       nn.Module,
    Z_train:    np.ndarray,
    y_train:    np.ndarray,
    Z_val:      np.ndarray,
    y_val:      np.ndarray,
    n_classes:  int,
    n_epochs:   int,
    lr:         float,
    weight_decay: float,
    device:     torch.device,
) -> dict:
    """Train a classification head; returns val metrics.

    Binary heads (n_classes=2) output (B, 1) and use BCEWithLogitsLoss.
    Multi-class heads (n_classes>2) output (B, n_classes) and use CrossEntropyLoss.
    """
    Zt = torch.FloatTensor(Z_train).to(device)
    Zv = torch.FloatTensor(Z_val).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    binary = (n_classes == 2)

    if binary:
        criterion = nn.BCEWithLogitsLoss()
        yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)   # (N, 1)
    else:
        criterion = nn.CrossEntropyLoss()
        yt = torch.LongTensor(y_train).to(device)                  # (N,)

    head.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = criterion(head(Zt), yt)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        logits_v = head(Zv)

    if binary:
        probs_pos = torch.sigmoid(logits_v).squeeze(1).cpu().numpy()  # (N,)
        preds     = (probs_pos > 0.5).astype(int)
        acc       = accuracy_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, probs_pos)
        except Exception:
            auc = float("nan")
    else:
        preds = logits_v.argmax(dim=-1).cpu().numpy()
        probs = torch.softmax(logits_v, dim=-1).cpu().numpy()
        acc   = accuracy_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")

    return {"accuracy": acc, "auc": auc}


def _train_dir_head(
    head:         nn.Module,
    Z_train:      np.ndarray,
    y_train:      np.ndarray,
    s_train:      np.ndarray,       # session IDs (int64)
    Z_val:        np.ndarray,
    y_val:        np.ndarray,
    s_val:        np.ndarray,
    n_epochs:     int,
    lr:           float,
    weight_decay: float,
    device:       torch.device,
) -> dict:
    """Train the session-conditioned DirectionalHead; returns val metrics."""
    Zt  = torch.FloatTensor(Z_train).to(device)
    Zv  = torch.FloatTensor(Z_val).to(device)
    St  = torch.LongTensor(s_train).to(device)
    Sv  = torch.LongTensor(s_val).to(device)
    yt  = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    opt       = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    head.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = criterion(head(Zt, St), yt)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        probs_pos = torch.sigmoid(head(Zv, Sv)).squeeze(1).cpu().numpy()

    preds = (probs_pos > 0.5).astype(int)
    acc   = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, probs_pos)
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "auc": auc}


def _train_regressor(
    head:       nn.Module,
    Z_train:    np.ndarray,
    y_train:    np.ndarray,
    Z_val:      np.ndarray,
    y_val:      np.ndarray,
    n_epochs:   int,
    lr:         float,
    weight_decay: float,
    device:     torch.device,
) -> dict:
    """Train a regression head; returns val metrics."""
    Zt = torch.FloatTensor(Z_train).to(device)
    Zv = torch.FloatTensor(Z_val).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    opt       = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    head.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = criterion(head(Zt), yt)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        preds = head(Zv).squeeze(1).cpu().numpy()           # (N,)

    mae = float(mean_absolute_error(y_val, preds))
    try:
        r, _ = pearsonr(y_val, preds)
        r = float(r)
    except Exception:
        r = float("nan")

    return {"mae": mae, "pearson_r": r}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_heads(
    encoder,
    pipeline_result: dict,
    predictor       = None,   # CausalPredictor — enables S_t surprise features
    horizon:      int   = 4,
    lookback:     int   = 48,
    n_epochs:     int   = 150,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size:   int   = 256,
    device:       str   = "cpu",
    trend_window: int   = 200,
    history_len:  int   = 3,
) -> tuple:
    """
    Train all three execution heads on frozen encoder embeddings.

    Args:
        encoder:         Frozen MarketEncoder (weights are not modified).
        pipeline_result: Dict from DataPipeline.build() — must contain meta with
                         norm_features, prices, n_train, n_val.
        horizon:         Forward return window for direction/magnitude labels.
        lookback:        Encoder window length.

    Returns:
        (execution_layer, metrics_dict)
    """
    meta    = pipeline_result["meta"]
    n_train = meta["n_train"]
    n_val   = meta["n_val"]
    dev     = torch.device(device)

    features_all = torch.as_tensor(meta["norm_features"], dtype=torch.float32)
    prices_all   = meta["prices"]

    train_feat   = features_all[:n_train]
    train_prices = prices_all[:n_train]
    val_feat     = features_all[n_train : n_train + n_val]
    val_prices   = prices_all[n_train : n_train + n_val]

    # --- Encode both splits (encoder frozen throughout) ---
    print("\n[1/4] Encoding train split...")
    Z_train = encode_split(encoder, train_feat, lookback, batch_size, device)
    print(f"  Z_train: {Z_train.shape}")

    print("\n[2/4] Encoding val split...")
    Z_val = encode_split(encoder, val_feat, lookback, batch_size, device)
    print(f"  Z_val:   {Z_val.shape}")

    # --- Structured feature augmentation ---
    # Augment z_t with 3 frozen linear probe outputs (5 dims) + S_t surprise (2 dims)
    # = 135-dim input for the directional head. Vol/mag heads keep raw 128-dim z.
    probe_bundle = None
    if predictor is not None:
        print("\n[2b/4] Building structured features (probes + surprise)...")

        # Train 3 linear probes on frozen Z_train
        probe_bundle = train_probe_bundle(
            Z_train, train_feat.numpy(), lookback=lookback
        )
        with torch.no_grad():
            ctx_tr = probe_bundle(torch.FloatTensor(Z_train)).numpy()  # (N_tr, 5)
            ctx_vl = probe_bundle(torch.FloatTensor(Z_val)).numpy()    # (N_vl, 5)
        print(f"  Probe features: session(3) + vol(1) + rv(1) = 5 dims")

        # Compute S_t surprise features from predictor
        train_actions = meta["macro_vecs"][:n_train]
        val_actions   = meta["macro_vecs"][n_train : n_train + n_val]
        S_tr = compute_surprise_features(
            Z_train, train_actions, predictor, lookback, history_len=history_len
        )
        S_vl = compute_surprise_features(
            Z_val,   val_actions,   predictor, lookback, history_len=history_len
        )
        print(f"  Surprise features: s_norm(1) + s_rank(1) = 2 dims")

        # Augmented arrays for the directional head only: (N, 135)
        # Vol and mag heads keep raw 128-dim Z — do NOT overwrite Z_train/Z_val.
        Z_dir_train = np.hstack([Z_train, ctx_tr, S_tr])
        Z_dir_val   = np.hstack([Z_val,   ctx_vl, S_vl])
        print(f"  Augmented dir arrays: train {Z_dir_train.shape}  val {Z_dir_val.shape}")
    else:
        Z_dir_train = Z_train
        Z_dir_val   = Z_val

    # Total flattened input dim = step_dim × history_len (e.g. 135 × 3 = 405)
    dir_in_dim = Z_dir_train.shape[1] * history_len if Z_dir_train.shape[1] != 128 else None

    # Always use full z-history — structured features are per-bar context that
    # augments each step, not a replacement for temporal dynamics.
    _dir_history = history_len
    Z_hist_train = build_z_history(Z_dir_train, _dir_history)   # (N_tr, H, z_dim_aug)
    Z_hist_val   = build_z_history(Z_dir_val,   _dir_history)   # (N_vl, H, z_dim_aug)

    # --- Compute labels ---
    print("\n[3/4] Computing labels...")
    if trend_window > 0:
        print(f"  Trend-adjusted labels: rolling median window={trend_window}")
    train_labels = compute_labels(
        train_feat.numpy(), train_prices, lookback, horizon,
        rv32_pct=None, trend_window=trend_window,
    )
    vol_thresholds = train_labels["vol_thresholds"]
    print(f"  Vol thresholds (train p33/p67): {vol_thresholds[0]:.3f} / {vol_thresholds[1]:.3f}")
    dir_balance = train_labels["direction"].mean()
    print(f"  Direction label balance (train): {dir_balance:.3f}  (0.5 = perfect)")
    print(f"  Train: {train_labels['n_valid_labeled']:,} labeled  "
          f"{train_labels['n_valid_z']:,} vol-labeled")

    # Val labels use training-set thresholds to avoid leakage
    val_labels = compute_labels(
        val_feat.numpy(), val_prices, lookback, horizon,
        rv32_pct=vol_thresholds, trend_window=trend_window,
    )
    print(f"  Val:   {val_labels['n_valid_labeled']:,} labeled  "
          f"{val_labels['n_valid_z']:,} vol-labeled")

    # --- Slice z arrays to match label counts ---
    # Vol head uses all valid_z bars; direction/magnitude heads use valid_labeled bars.
    N_tr_z  = train_labels["n_valid_z"]
    N_tr_lb = train_labels["n_valid_labeled"]
    N_vl_z  = val_labels["n_valid_z"]
    N_vl_lb = val_labels["n_valid_labeled"]

    Z_tr_vol      = Z_train[:N_tr_z]
    Z_tr_dir      = Z_train[:N_tr_lb]
    Z_vl_vol      = Z_val[:N_vl_z]
    Z_vl_dir      = Z_val[:N_vl_lb]

    # History-sliced arrays for direction head
    Z_hist_tr_dir = Z_hist_train[:N_tr_lb]    # (N_tr_lb, H, z_dim)
    Z_hist_vl_dir = Z_hist_val[:N_vl_lb]      # (N_vl_lb, H, z_dim)

    # --- Build and train execution layer ---
    print("\n[4/4] Training heads...")
    layer = SupervisedExecutionLayer(
        z_dim=128, hidden=64, vol_thresholds=vol_thresholds,
        history_len=_dir_history,
        dir_in_dim=dir_in_dim,
    ).to(dev)
    if probe_bundle is not None:
        layer.probe_bundle = probe_bundle

    train_kwargs = dict(
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, device=dev
    )

    # Vol head
    print("  VolatilityHead  (3-class)  ...", end="", flush=True)
    vol_metrics = _train_classifier(
        layer.vol_head,
        Z_tr_vol, train_labels["vol"],
        Z_vl_vol, val_labels["vol"],
        n_classes=3, **train_kwargs,
    )
    print(f"  val acc={vol_metrics['accuracy']:.3f}  auc={vol_metrics['auc']:.3f}")

    # Direction head — trained on ALL sessions with z-history (latent momentum).
    # NY included as contrastive "no-alpha" examples; suppressed at inference via
    # skip_sessions. Input shape: (N, H, z_dim) — head computes diffs internally.
    print("  DirectionalHead (binary)   ...", end="", flush=True)
    dir_metrics = _train_classifier(
        layer.dir_head,
        Z_hist_tr_dir, train_labels["direction"],
        Z_hist_vl_dir, val_labels["direction"],
        n_classes=2, **train_kwargs,
    )
    print(f"  val acc={dir_metrics['accuracy']:.3f}  auc={dir_metrics['auc']:.3f}")

    # Magnitude head
    print("  MagnitudeHead   (regress)  ...", end="", flush=True)
    mag_metrics = _train_regressor(
        layer.mag_head,
        Z_tr_dir, train_labels["magnitude"],
        Z_vl_dir, val_labels["magnitude"],
        **train_kwargs,
    )
    print(f"  val mae={mag_metrics['mae']:.5f}  r={mag_metrics['pearson_r']:.3f}")

    # Calibrate magnitude threshold to p50 of training-set predictions.
    # Replaces the ineffective 1.5×spread default (which is ~8% of typical
    # 4-bar gold returns and never blocks anything).
    layer.mag_head.eval()
    with torch.no_grad():
        train_mag_preds = layer.mag_head.predict(
            torch.FloatTensor(Z_tr_dir).to(dev)
        ).cpu().numpy()
    mag_threshold = float(np.percentile(train_mag_preds, 50))
    layer.mag_threshold.data.fill_(mag_threshold)
    print(f"  Magnitude threshold (p50 train preds): {mag_threshold:.6f}  "
          f"[train range {train_mag_preds.min():.4f}–{train_mag_preds.max():.4f}]")

    metrics = {
        "horizon":         horizon,
        "trend_window":    trend_window,
        "history_len":     _dir_history,
        "dir_in_dim":      dir_in_dim,
        "structured_features": dir_in_dim is not None,
        "vol":             vol_metrics,
        "direction":       dir_metrics,
        "magnitude":       mag_metrics,
        "vol_thresholds":  list(vol_thresholds),
        "mag_threshold":   mag_threshold,
        "n_train_labeled": int(N_tr_lb),
        "n_val_labeled":   int(N_vl_lb),
    }

    _print_summary(metrics)
    return layer, metrics


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_heads(layer: SupervisedExecutionLayer, metrics: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(layer.state_dict(), out / "execution_layer.pt")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved execution layer: {out / 'execution_layer.pt'}")
    print(f"Saved metrics:         {out / 'metrics.json'}")


def load_heads(
    out_dir: str,
    z_dim:   int = 128,
    device:  str = "cpu",
) -> SupervisedExecutionLayer:
    """
    Load a saved SupervisedExecutionLayer (with optional probe_bundle).
    Vol thresholds are restored from the state_dict buffers automatically.
    """
    p = Path(out_dir)
    with open(p / "metrics.json") as f:
        metrics = json.load(f)

    vol_thresholds = tuple(metrics["vol_thresholds"])
    history_len    = int(metrics.get("history_len", 1))
    dir_in_dim     = metrics.get("dir_in_dim", None)

    layer = SupervisedExecutionLayer(
        z_dim=z_dim, hidden=64,
        vol_thresholds=vol_thresholds,
        history_len=history_len,
        dir_in_dim=dir_in_dim,
    )

    # Restore probe_bundle sub-module before load_state_dict if weights are present
    ckpt = torch.load(p / "execution_layer.pt", map_location=device, weights_only=True)
    if any(k.startswith("probe_bundle.") for k in ckpt):
        layer.probe_bundle = ProbeBundle(z_dim)
    layer.load_state_dict(ckpt)
    layer.eval()
    return layer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    print("\n" + "=" * 55)
    print("EXECUTION HEAD TRAINING RESULTS")
    print("=" * 55)

    chance_note = "(chance=0.333)" if True else ""
    print(f"\n  VolatilityHead  (3-class)  {chance_note}")
    print(f"    val accuracy : {metrics['vol']['accuracy']:.3f}  "
          f"(chance=0.333)")
    print(f"    val AUC      : {metrics['vol']['auc']:.3f}")

    print(f"\n  DirectionalHead (binary)")
    print(f"    val accuracy : {metrics['direction']['accuracy']:.3f}  "
          f"(chance=0.500)")
    print(f"    val AUC      : {metrics['direction']['auc']:.3f}")

    print(f"\n  MagnitudeHead   (regression)")
    print(f"    val MAE      : {metrics['magnitude']['mae']:.5f}")
    print(f"    val Pearson r: {metrics['magnitude']['pearson_r']:.3f}")

    print(f"\n  Vol thresholds (p33/p67): {metrics['vol_thresholds'][0]:.3f} / "
          f"{metrics['vol_thresholds'][1]:.3f}")
    print(f"  Train labeled: {metrics['n_train_labeled']:,}  "
          f"Val labeled: {metrics['n_val_labeled']:,}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument",    default="gold",
                        choices=["gold", "eurusd", "usdjpy"])
    parser.add_argument("--horizon",       type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--trend-window",  type=int,   default=200,
                        help="Rolling-median window for trend-adjusted direction labels. "
                             "0 = raw (fwd > 0).")
    parser.add_argument("--out-dir",       type=str,   default=None,
                        help="Output directory for saved heads. Defaults to "
                             "./experiments/heads_{instrument}.")
    parser.add_argument("--seed",          type=int,   default=None,
                        help="Random seed for reproducible training.")
    parser.add_argument("--structured-features", action="store_true",
                        help="Augment directional head input with ProbeBundle + S_t surprise "
                             "(135-dim per step). Default: raw z-history only.")
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Explicit path to encoder checkpoint. "
                             "Default: experiments/checkpoints/best_model.pt (root baseline).")
    args = parser.parse_args()

    # Default to the root baseline checkpoint.
    # Instrument-specific subfolders may contain fine-tuned variants — pass
    # --checkpoint explicitly to use those.
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = "./experiments/checkpoints/best_model.pt"
        if not Path(ckpt_path).exists():
            ckpt_path = f"./experiments/checkpoints/{args.instrument}/best_model.pt"
    out_dir = args.out_dir or f"./experiments/heads_{args.instrument}"

    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found for instrument '{args.instrument}'.")
        print("Run 'python -m training.train_o1' first.")
        sys.exit(1)

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")

    print("=" * 55)
    print(f"Training execution heads  [{args.instrument.upper()}]")
    print("=" * 55)

    # Load encoder from checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", {})

    encoder = MarketEncoder(
        lookback    = cfg.get("lookback",      48),
        n_features  = cfg.get("n_features",    52),
        patch_size  = 4,
        d_model     = cfg.get("enc_d_model",  128),
        n_heads     = cfg.get("enc_n_heads",    4),
        n_layers    = cfg.get("enc_n_layers",   4),
        dropout     = 0.0,
        proj_hidden = cfg.get("enc_proj_hidden", 256),
        z_dim       = cfg.get("z_dim",         128),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Load predictor (for S_t surprise features)
    cfg       = ckpt.get("cfg", {})
    predictor = CausalPredictor(
        z_dim       = cfg.get("z_dim",         128),
        d_model     = cfg.get("pred_d_model",  128),
        n_heads     = cfg.get("pred_n_heads",    4),
        n_layers    = cfg.get("pred_n_layers",   6),
        action_dim  = cfg.get("action_dim",      5),
        history_len = cfg.get("history_len",     3),
        dropout     = 0.0,
    )
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    print(f"Loaded encoder + predictor from epoch {ckpt.get('epoch', '?')}")

    # Load data
    split_dates = cfg.get("split_dates")
    pipeline = DataPipeline(instrument=args.instrument, lookback=48, norm_window=500)
    result   = pipeline.build(
        use_real_data=True,
        stride=1,
        history_len=3,
        split_dates=split_dates,
    )

    # Train
    # Pass predictor only when --structured-features flag is set.
    # Default: clean baseline (raw z-history, no probe augmentation).
    _predictor = predictor if args.structured_features else None

    layer, metrics = train_heads(
        encoder, result,
        predictor=_predictor,
        horizon=args.horizon,
        n_epochs=args.epochs,
        lr=args.lr,
        trend_window=args.trend_window,
        history_len=3,
    )

    # Save
    save_heads(layer, metrics, out_dir)
