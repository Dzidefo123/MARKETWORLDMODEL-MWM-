"""
scripts/runpod_warmstart.py

GPU warm restart for the M15 encoder.
Loads the epoch-54 checkpoint, runs a fresh cosine LR schedule for 200 epochs.
Reads pre-exported .npy splits — no MT5 required.

Usage (on RunPod pod):
    python runpod_warmstart.py

Outputs:
    checkpoints/m15_warmstart/best_model.pt
    checkpoints/m15_warmstart/loss_history.json
    logs/warmstart.log
"""

import sys, os, json, math, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, ".")

from models.encoder   import MarketEncoder
from models.predictor import CausalPredictor
from models.sigreg    import SIGReg
from models.mwm_loss  import MWMLoss, MarketWindowDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMSTART_CFG = {
    "n_epochs":      200,
    "lr":            1e-4,       # fresh cosine schedule starting here
    "warmup_epochs": 0,          # already trained — no warmup needed
    "weight_decay":  1e-4,
    "grad_clip":     1.0,
    "batch_size":    128,
    "lookback":      96,
    "history_len":   3,
}

DATA_DIR   = Path("./data")
CKPT_DIR   = Path("./checkpoints/m15_warmstart")
SEED_CKPT  = Path("./checkpoints/m15/best_model.pt")   # epoch-54 checkpoint
LOG_PATH   = Path("./logs/warmstart.log")

# ---------------------------------------------------------------------------
# Logging — tee to file
# ---------------------------------------------------------------------------

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _get_lr(epoch: int, n_epochs: int, base_lr: float) -> float:
    progress = epoch / n_epochs
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Load data
    logger.info("Loading pre-exported M15 splits from %s ...", DATA_DIR)
    train_feat = np.load(DATA_DIR / "train_feat.npy")
    train_act  = np.load(DATA_DIR / "train_act.npy")
    val_feat   = np.load(DATA_DIR / "val_feat.npy")
    val_act    = np.load(DATA_DIR / "val_act.npy")
    logger.info("  train=%s  val=%s", train_feat.shape, val_feat.shape)

    lkb = WARMSTART_CFG["lookback"]
    hln = WARMSTART_CFG["history_len"]

    train_ds = MarketWindowDataset(train_feat, train_act, lkb, stride=1, history_len=hln)
    val_ds   = MarketWindowDataset(val_feat,   val_act,   lkb, stride=4, history_len=hln)
    logger.info("  Train samples: %d    Val samples: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=WARMSTART_CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=WARMSTART_CFG["batch_size"] * 2,
                              shuffle=False, num_workers=2)

    # Build models
    encoder = MarketEncoder(
        lookback=lkb, n_features=52, patch_size=4,
        d_model=128, n_heads=4, n_layers=4,
        dropout=0.1, proj_hidden=256, z_dim=128,
    ).to(device)

    predictor = CausalPredictor(
        z_dim=128, d_model=128, n_heads=4, n_layers=6,
        action_dim=5, history_len=hln, dropout=0.1,
    ).to(device)

    sigreg  = SIGReg(d_model=128).to(device)
    loss_fn = MWMLoss(sigreg=sigreg, lambda_weight=0.1)

    # Load checkpoint — resume from warmstart best if it exists, else seed from original
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    warmstart_ckpt = CKPT_DIR / "best_model.pt"
    history_file   = CKPT_DIR / "loss_history.json"

    best_val    = float("inf")
    best_epoch  = 0
    start_epoch = 0
    seed_epoch  = None
    history     = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    if warmstart_ckpt.exists():
        logger.info("Resuming from warmstart checkpoint: %s", warmstart_ckpt)
        saved = torch.load(warmstart_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(saved["encoder"])
        predictor.load_state_dict(saved["predictor"])
        start_epoch = saved["epoch"]
        best_val    = saved["val_loss"]
        best_epoch  = start_epoch
        logger.info("  Resumed from warmstart epoch %d  val_loss=%.6f", start_epoch, best_val)
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
    else:
        if not SEED_CKPT.exists():
            raise FileNotFoundError(
                f"Seed checkpoint not found: {SEED_CKPT}\n"
                "Make sure you uploaded checkpoints/m15/best_model.pt to the pod."
            )
        logger.info("Loading seed checkpoint: %s", SEED_CKPT)
        saved = torch.load(SEED_CKPT, map_location=device, weights_only=False)
        encoder.load_state_dict(saved["encoder"])
        predictor.load_state_dict(saved["predictor"])
        seed_epoch = saved.get("epoch", "?")
        seed_val   = saved.get("val_loss", float("inf"))
        logger.info("  Seeded from epoch %s  val_loss=%.6f", seed_epoch, seed_val)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=WARMSTART_CFG["lr"], weight_decay=WARMSTART_CFG["weight_decay"],
    )

    n_epochs = WARMSTART_CFG["n_epochs"]
    logger.info("[WARM RESTART] epochs %d → %d  LR %.1e → ~0",
                start_epoch + 1, n_epochs, WARMSTART_CFG["lr"])
    logger.info("-" * 60)

    for epoch in range(start_epoch, n_epochs):
        lr = _get_lr(epoch, n_epochs, WARMSTART_CFG["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # --- train ---
        encoder.train(); predictor.train()
        train_losses = []
        for batch in train_loader:
            x_hist = batch["x_hist"].to(device)
            x_t1   = batch["x_t1"].to(device)
            a_t    = batch["a_t"].to(device)

            optimizer.zero_grad()
            z_hist = encoder.encode_sequence(x_hist)
            z_t    = z_hist[:, -1, :]
            z_t1   = encoder(x_t1)
            z_hat  = predictor(z_hist, a_t)
            losses = loss_fn(z_t, z_hat, z_t1)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()),
                WARMSTART_CFG["grad_clip"],
            )
            optimizer.step()
            train_losses.append(losses["total"].item())

        # --- validate ---
        encoder.eval(); predictor.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_hist = batch["x_hist"].to(device)
                x_t1   = batch["x_t1"].to(device)
                a_t    = batch["a_t"].to(device)
                z_hist = encoder.encode_sequence(x_hist)
                z_t    = z_hist[:, -1, :]
                z_t1   = encoder(x_t1)
                z_hat  = predictor(z_hist, a_t)
                losses = loss_fn(z_t, z_hat, z_t1)
                val_losses.append(losses["total"].item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses)) if val_losses else train_loss

        is_best = val_loss < best_val
        marker  = " *" if is_best else ""
        line = (f"  Epoch {epoch+1:3d}/{n_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}{marker}")
        logger.info(line.strip())

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["lr"].append(round(lr, 8))
        hist_path = CKPT_DIR / "loss_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if is_best:
            best_val   = val_loss
            best_epoch = epoch + 1
            torch.save({
                "epoch":      epoch + 1,
                "encoder":    encoder.state_dict(),
                "predictor":  predictor.state_dict(),
                "val_loss":   val_loss,
                "seed_epoch": seed_epoch,
                "cfg":        WARMSTART_CFG,
            }, CKPT_DIR / "best_model.pt")

    logger.info("=" * 60)
    logger.info("Warm restart complete. Best val=%.6f at epoch %d", best_val, best_epoch)
    logger.info("Checkpoint: %s/best_model.pt", CKPT_DIR)
    logger.info("Log: %s", LOG_PATH)

if __name__ == "__main__":
    main()
