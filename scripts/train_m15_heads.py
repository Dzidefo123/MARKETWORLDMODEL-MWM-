"""
scripts/train_m15_heads.py

Train M15 execution heads (direction, vol, magnitude) on frozen M15 warmstart encoder.

Fetches M15 data from MT5, encodes with the frozen ep-172 warmstart encoder,
then trains heads on those embeddings. Runs locally (MT5 required).

Usage:
    python -m scripts.train_m15_heads
    python -m scripts.train_m15_heads --horizon 4 --epochs 200 --out-dir experiments/heads_gold_m15

Outputs:
    experiments/heads_gold_m15/execution_layer.pt
    experiments/heads_gold_m15/metrics.json
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path

from models.encoder   import MarketEncoder
from models.predictor import CausalPredictor
from execution.heads  import (
    SupervisedExecutionLayer,
    compute_labels,
    encode_split,
    build_z_history,
)
from execution.train_heads import (
    _train_classifier,
    _train_regressor,
    save_heads,
    _print_summary,
)
from experiments.phase0_m15_analysis import (
    fetch_m15_dataset,
    build_m15_pipeline,
    M15_CFG,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

WARMSTART_CKPT = Path("./experiments/checkpoints/m15_warmstart/best_model.pt")
DEFAULT_OUT    = "./experiments/heads_gold_m15"


def load_m15_encoder(ckpt_path: Path, device: str = "cpu"):
    """Load the frozen M15 warmstart encoder + predictor."""
    logger.info("Loading M15 warmstart encoder: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]

    encoder = MarketEncoder(
        lookback    = cfg["lookback"],
        n_features  = 52,
        patch_size  = 4,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 4,
        dropout     = 0.0,
        proj_hidden = 256,
        z_dim       = 128,
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    predictor = CausalPredictor(
        z_dim       = 128,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 6,
        action_dim  = 5,
        history_len = cfg["history_len"],
        dropout     = 0.0,
    )
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    logger.info("  Encoder loaded from epoch %d  val_loss=%.6f",
                ckpt["epoch"], ckpt["val_loss"])
    return encoder, predictor, cfg


def build_m15_prices(m15_ohlcv, n_train: int, n_val: int, lookback_warm: int = 250):
    """Extract close prices aligned with the feature matrix."""
    closes = m15_ohlcv["close"].values
    train_prices = closes[lookback_warm           : lookback_warm + n_train]
    val_prices   = closes[lookback_warm + n_train : lookback_warm + n_train + n_val]
    test_prices  = closes[lookback_warm + n_train + n_val :]
    return train_prices, val_prices, test_prices


def main():
    parser = argparse.ArgumentParser(description="Train M15 execution heads")
    parser.add_argument("--horizon",   type=int,   default=4,
                        help="Forward bars for direction/magnitude labels (default 4 = 1 hour)")
    parser.add_argument("--epochs",    type=int,   default=200,
                        help="Head training epochs (default 200)")
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--trend-window", type=int, default=200,
                        help="Rolling-median window for trend-adjusted labels")
    parser.add_argument("--out-dir",   type=str,   default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int,  default=256)
    args = parser.parse_args()

    device = "cpu"
    logger.info("=" * 60)
    logger.info("M15 EXECUTION HEAD TRAINING")
    logger.info("  Encoder:  %s", WARMSTART_CKPT)
    logger.info("  Horizon:  %d M15 bars (%.0f min)", args.horizon, args.horizon * 15)
    logger.info("  Epochs:   %d", args.epochs)
    logger.info("=" * 60)

    # 1 — Load encoder
    encoder, predictor, enc_cfg = load_m15_encoder(WARMSTART_CKPT, device)
    lookback    = enc_cfg["lookback"]     # 96
    history_len = enc_cfg["history_len"]  # 3

    # 2 — Fetch M15 data
    logger.info("Fetching M15 data from MT5...")
    m15_ohlcv, macro_m15 = fetch_m15_dataset(n_bars=M15_CFG["n_m15_bars"])

    logger.info("Building M15 feature splits...")
    splits = build_m15_pipeline(
        m15_ohlcv, macro_m15,
        split_dates = M15_CFG["split_dates"],
        norm_window = M15_CFG["norm_window"],
    )

    train_prices, val_prices, _ = build_m15_prices(
        m15_ohlcv,
        n_train     = splits["n_train"],
        n_val       = splits["n_val"],
        lookback_warm = 250,
    )

    train_feat = torch.as_tensor(splits["train_feat"], dtype=torch.float32)
    val_feat   = torch.as_tensor(splits["val_feat"],   dtype=torch.float32)

    # 3 — Encode splits (frozen encoder)
    logger.info("Encoding train split (%d bars)...", splits["n_train"])
    Z_train = encode_split(encoder, train_feat, lookback, args.batch_size, device)
    logger.info("  Z_train: %s", Z_train.shape)

    logger.info("Encoding val split (%d bars)...", splits["n_val"])
    Z_val = encode_split(encoder, val_feat, lookback, args.batch_size, device)
    logger.info("  Z_val:   %s", Z_val.shape)

    # 4 — Compute labels
    logger.info("Computing M15 labels (horizon=%d bars = %d min)...",
                args.horizon, args.horizon * 15)
    train_labels = compute_labels(
        splits["train_feat"], train_prices, lookback, args.horizon,
        rv32_pct=None, trend_window=args.trend_window,
    )
    vol_thresholds = train_labels["vol_thresholds"]
    dir_balance    = train_labels["direction"].mean()
    logger.info("  Dir label balance: %.3f  (0.5=perfect)", dir_balance)
    logger.info("  Vol thresholds p33/p67: %.4f / %.4f",
                vol_thresholds[0], vol_thresholds[1])
    logger.info("  Train labeled: %d  vol-labeled: %d",
                train_labels["n_valid_labeled"], train_labels["n_valid_z"])

    val_labels = compute_labels(
        splits["val_feat"], val_prices, lookback, args.horizon,
        rv32_pct=vol_thresholds, trend_window=args.trend_window,
    )
    logger.info("  Val   labeled: %d  vol-labeled: %d",
                val_labels["n_valid_labeled"], val_labels["n_valid_z"])

    # 5 — Slice Z arrays to label counts
    N_tr_lb = train_labels["n_valid_labeled"]
    N_tr_z  = train_labels["n_valid_z"]
    N_vl_lb = val_labels["n_valid_labeled"]
    N_vl_z  = val_labels["n_valid_z"]

    Z_tr_vol = Z_train[:N_tr_z]
    Z_vl_vol = Z_val[:N_vl_z]
    Z_tr_dir = Z_train[:N_tr_lb]
    Z_vl_dir = Z_val[:N_vl_lb]

    # Build z-history for directional head
    Z_hist_tr = build_z_history(Z_tr_dir, history_len)
    Z_hist_vl = build_z_history(Z_vl_dir, history_len)

    # 6 — Build and train heads
    logger.info("=" * 60)
    logger.info("Training heads  (epochs=%d  lr=%.0e)", args.epochs, args.lr)
    logger.info("=" * 60)

    dev   = torch.device(device)
    layer = SupervisedExecutionLayer(
        z_dim=128, hidden=64, vol_thresholds=vol_thresholds,
        history_len=history_len,
        dir_in_dim=None,   # raw z-history, no structured feature augmentation
    ).to(dev)

    train_kwargs = dict(
        n_epochs=args.epochs, lr=args.lr,
        weight_decay=1e-4, device=dev,
    )

    logger.info("  VolatilityHead  (3-class)...")
    vol_metrics = _train_classifier(
        layer.vol_head,
        Z_tr_vol, train_labels["vol"],
        Z_vl_vol, val_labels["vol"],
        n_classes=3, **train_kwargs,
    )
    logger.info("    val acc=%.3f  AUC=%.3f", vol_metrics["accuracy"], vol_metrics["auc"])

    logger.info("  DirectionalHead (binary)...")
    dir_metrics = _train_classifier(
        layer.dir_head,
        Z_hist_tr, train_labels["direction"],
        Z_hist_vl, val_labels["direction"],
        n_classes=2, **train_kwargs,
    )
    logger.info("    val acc=%.3f  AUC=%.3f", dir_metrics["accuracy"], dir_metrics["auc"])

    logger.info("  MagnitudeHead   (regress)...")
    mag_metrics = _train_regressor(
        layer.mag_head,
        Z_tr_dir, train_labels["magnitude"],
        Z_vl_dir, val_labels["magnitude"],
        **train_kwargs,
    )
    logger.info("    val MAE=%.5f  r=%.3f", mag_metrics["mae"], mag_metrics["pearson_r"])

    # Calibrate magnitude threshold to p50 of training predictions
    layer.mag_head.eval()
    with torch.no_grad():
        train_mag_preds = layer.mag_head.predict(
            torch.FloatTensor(Z_tr_dir).to(dev)
        ).cpu().numpy()
    mag_threshold = float(np.percentile(train_mag_preds, 50))
    layer.mag_threshold.data.fill_(mag_threshold)
    logger.info("  Magnitude threshold (p50): %.6f", mag_threshold)

    metrics = {
        "encoder_ckpt":     str(WARMSTART_CKPT),
        "encoder_epoch":    172,
        "encoder_val_loss": 0.005363,
        "horizon":          args.horizon,
        "horizon_minutes":  args.horizon * 15,
        "trend_window":     args.trend_window,
        "history_len":      history_len,
        "dir_in_dim":       None,
        "structured_features": False,
        "vol":              vol_metrics,
        "direction":        dir_metrics,
        "magnitude":        mag_metrics,
        "vol_thresholds":   list(vol_thresholds),
        "mag_threshold":    mag_threshold,
        "n_train_labeled":  int(N_tr_lb),
        "n_val_labeled":    int(N_vl_lb),
    }

    _print_summary(metrics)

    # 7 — Save
    save_heads(layer, metrics, args.out_dir)
    logger.info("Done. Heads saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
