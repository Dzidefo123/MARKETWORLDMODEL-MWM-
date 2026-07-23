"""
training/train_o1.py ? Full O1 Training Script
================================================

Ties together:
    DataPipeline  ->  MarketEncoder  ->  CausalPredictor  ->  MWMLoss (MSE + SIGReg)

Run from project root:
    python -m training.train_o1

Output:
    - Training loss curves printed to console
    - Checkpoints saved to ./experiments/checkpoints/
    - Best model saved as ./experiments/checkpoints/best_model.pt (lowest val — use this)
    - Final epoch saved as ./experiments/checkpoints/last_model.pt (inspection only)
"""

import sys, os
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from data.pipeline    import DataPipeline
from models.encoder   import MarketEncoder
from models.predictor import CausalPredictor
from models.sigreg    import SIGReg
from models.mwm_loss  import MWMLoss


# ??????????????????????????????????????????????????????????????
# Configuration
# ??????????????????????????????????????????????????????????????

CFG = {
    # Data
    "instrument":     "gold",  # "gold", "eurusd", or "usdjpy"
    "use_real_data":  True,
    "lookback":       48,
    "norm_window":    500,
    "stride":         1,
    "data_start":     None,
    "data_end":       None,
    "split_dates":    None,    # None = default 70/15/15 ratio

    # Model
    "z_dim":          128,
    "enc_d_model":    128,
    "enc_n_heads":    4,
    "enc_n_layers":   4,
    "enc_dropout":    0.1,
    "enc_proj_hidden":256,
    "pred_d_model":   128,
    "pred_n_heads":   4,
    "pred_n_layers":  6,
    "pred_dropout":   0.1,
    "history_len":    3,
    "action_dim":     5,
    "n_features":     52,

    # SIGReg
    "lambda_weight":  0.1,

    # Auxiliary directional loss — forces z_t to encode short-term trend direction.
    # Uses x_t1[:, -1, 0] (z-scored ret_1 at bar t+1) as the H=1 direction label.
    # Sign is preserved by z-scoring, so this is a valid (if noisy) direction signal.
    # Set to 0.0 to disable. Typical range: 0.05 – 0.2.
    "lambda_dir":     0.1,

    # Fine-tune from an existing checkpoint instead of training from scratch.
    # Points to the previous gold encoder (trained without directional signal).
    # Fine-tuning at a low LR preserves the predictor while adding dir signal.
    # Set to None to train from scratch.
    "finetune_from":  "./experiments/checkpoints/best_model.pt",

    # Training
    "batch_size":     64,
    "n_epochs":       120,      # Fine-tune run: shorter than full 200-epoch retrain
    "lr":             3e-5,     # 10x lower than scratch LR to preserve representations
    "weight_decay":   1e-4,
    "grad_clip":      1.0,
    "warmup_epochs":  3,

    # Paths
    "checkpoint_dir": "./experiments/checkpoints",
    "log_every":      20,
}


# ??????????????????????????????????????????????????????????????
# Learning Rate Scheduler: Linear Warmup + Cosine Decay
# ??????????????????????????????????????????????????????????????

def get_lr(epoch: int, n_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    """
    Linear warmup for the first warmup_epochs, then cosine decay.

    WHY WARMUP?
    At initialization, model weights are random. Large learning rates
    on random weights cause chaotic gradient updates. We ramp up slowly
    so the model first finds a reasonable region of parameter space.

    WHY COSINE DECAY?
    Near the end of training, large steps overshoot good minima.
    Cosine decay smoothly reduces the step size, allowing fine-tuning
    of the final solution.
    """
    import math
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ??????????????????????????????????????????????????????????????
# Training Loop
# ??????????????????????????????????????????????????????????????

class _Tee:
    """Write every print() to both stdout and a log file simultaneously."""
    def __init__(self, log_path: str, mode: str = "w"):
        import builtins
        self._file    = open(log_path, mode, buffering=1, encoding="utf-8")
        self._stdout  = sys.stdout
        sys.stdout    = self

    def write(self, data):
        self._stdout.write(data)
        self._stdout.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def train(result: dict = None):
    """
    Train the encoder + predictor.

    Args:
        result: Optional pre-built pipeline dict (train/val/test datasets + meta),
                as returned by DataPipeline.build / build_from_frames. When None,
                the data is fetched and built from CFG (default behaviour). Pass a
                pre-built result to train on a custom data source — e.g. H1 bars
                pulled from MT5 — while reusing this exact loop and checkpoint format.
    """
    instrument = CFG["instrument"]
    ckpt_dir   = f"{CFG['checkpoint_dir']}/{instrument}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Auto-tee all output to a log file. Callers (e.g. the MT5 retrain script)
    # can pass CFG["log_file"] to record each run to its own timestamped file
    # instead of overwriting the shared per-instrument log.
    log_path = CFG.get("log_file") or f"./experiments/train_{instrument}.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    # Append when the caller owns the (unique, per-run) log file so we don't
    # truncate lines it already wrote; truncate the shared default log as before.
    _tee = _Tee(log_path, mode="a" if CFG.get("log_file") else "w")

    print("=" * 60)
    print(f"MarketWorldModel - O1 Training  [{instrument.upper()}]")
    print(f"Log: {log_path}")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    print("\n[DATA]")
    if result is None:
        pipeline = DataPipeline(
            instrument  = instrument,
            lookback    = CFG["lookback"],
            norm_window = CFG["norm_window"],
        )
        result = pipeline.build(
            start          = CFG.get("data_start"),
            end            = CFG.get("data_end"),
            use_real_data  = CFG["use_real_data"],
            stride         = CFG["stride"],
            history_len    = CFG["history_len"],
            split_dates    = CFG.get("split_dates"),
        )
    else:
        print("  Using caller-supplied pre-built dataset (skipping fetch/build)")

    train_loader = DataLoader(
        result["train"],
        batch_size = CFG["batch_size"],
        shuffle    = True,
        num_workers= 0,
        pin_memory = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        result["val"],
        batch_size = CFG["batch_size"] * 2,
        shuffle    = False,
    )

    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val   batches per epoch: {len(val_loader)}")

    # ?? Models ??????????????????????????????????????????????
    print("\n[MODELS]")
    encoder = MarketEncoder(
        lookback    = CFG["lookback"],
        n_features  = CFG["n_features"],
        patch_size  = 4,
        d_model     = CFG["enc_d_model"],
        n_heads     = CFG["enc_n_heads"],
        n_layers    = CFG["enc_n_layers"],
        dropout     = CFG["enc_dropout"],
        proj_hidden = CFG["enc_proj_hidden"],
        z_dim       = CFG["z_dim"],
    ).to(device)

    predictor = CausalPredictor(
        z_dim       = CFG["z_dim"],
        d_model     = CFG["pred_d_model"],
        n_heads     = CFG["pred_n_heads"],
        n_layers    = CFG["pred_n_layers"],
        action_dim  = CFG["action_dim"],
        history_len = CFG["history_len"],
        dropout     = CFG["pred_dropout"],
    ).to(device)

    sigreg  = SIGReg(d_model=CFG["z_dim"]).to(device)
    loss_fn = MWMLoss(sigreg=sigreg, lambda_weight=CFG["lambda_weight"])

    # Auxiliary direction head: shallow MLP on z_t → direction logit.
    # Discarded after training; only the encoder/predictor weights matter.
    lambda_dir   = float(CFG.get("lambda_dir", 0.0))
    aux_dir_head = nn.Sequential(
        nn.Linear(CFG["z_dim"], 32), nn.ReLU(), nn.Linear(32, 1),
    ).to(device) if lambda_dir > 0 else None

    # Precompute trend-adjusted H=4 direction labels for the training set.
    # Label = 1 if fwd_return[t] > rolling_median_200(fwd_return), else 0.
    # Matches the DirectionalHead evaluation signal exactly (same as compute_labels
    # with trend_window=200). Raw H=4 labels (fwd > 0) are regime-dependent and
    # invert when macro trend changes — the trend-adjusted residual is stable.
    # Stored on CPU; indexed at batch time via batch["t_now"].
    # Last 4 bars get 0.5 (balanced, zero net gradient at the boundary).
    dir_labels_cpu = None
    if lambda_dir > 0:
        import pandas as _pd
        _DIR_HORIZON  = 4
        _TREND_WINDOW = 200
        train_prices = result["meta"]["prices"][:result["meta"]["n_train"]]
        fwd_raw = (train_prices[_DIR_HORIZON:] - train_prices[:-_DIR_HORIZON]) / (train_prices[:-_DIR_HORIZON] + 1e-8)
        rolling_med = (_pd.Series(fwd_raw)
                       .rolling(_TREND_WINDOW, min_periods=max(1, _TREND_WINDOW // 10))
                       .median()
                       .shift(1)
                       .fillna(0.0)
                       .values)
        dir_vec = (fwd_raw > rolling_med).astype(np.float32)
        dir_labels_np = np.concatenate(
            [dir_vec, np.full(_DIR_HORIZON, 0.5, dtype=np.float32)]
        )
        dir_labels_cpu = torch.FloatTensor(dir_labels_np)
        bal = dir_vec.mean()
        print(f"Direction labels (H=4, trend-adj window={_TREND_WINDOW}): "
              f"{len(dir_vec):,} bars  balance={bal:.3f}")

    # Optional: fine-tune from an existing checkpoint
    finetune_from = CFG.get("finetune_from")
    if finetune_from and Path(finetune_from).exists():
        print(f"\nFine-tuning from checkpoint: {finetune_from}")
        ck = torch.load(finetune_from, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        predictor.load_state_dict(ck["predictor"])
        print(f"  Loaded encoder + predictor from epoch {ck.get('epoch', '?')}")
    elif finetune_from:
        print(f"\nWARNING: finetune_from not found at {finetune_from} — training from scratch")

    backbone_params = list(encoder.parameters()) + list(predictor.parameters())
    all_params      = backbone_params[:]
    if aux_dir_head is not None:
        all_params += list(aux_dir_head.parameters())

    total_params = sum(p.numel() for p in all_params)
    print(f"\nTotal trainable parameters: {total_params:,}")
    if lambda_dir > 0:
        print(f"Auxiliary direction loss enabled  (lambda_dir={lambda_dir})")

    # ?? Optimizer ???????????????????????????????????????????
    # The aux head is randomly initialized, so it needs a much higher LR than the
    # pretrained encoder. Use separate param groups so the cosine schedule can scale
    # each group independently by multiplying from its own base LR.
    _AUX_LR = 1e-3  # ~33x backbone LR — enough for a new 3-layer head to learn
    if aux_dir_head is not None:
        param_groups = [
            {"params": backbone_params,                   "base_lr": CFG["lr"]},
            {"params": list(aux_dir_head.parameters()),   "base_lr": _AUX_LR},
        ]
        print(f"  Param groups: backbone lr={CFG['lr']:.1e}  aux_head lr={_AUX_LR:.1e}")
    else:
        param_groups = [{"params": backbone_params, "base_lr": CFG["lr"]}]

    for pg in param_groups:
        pg["lr"] = pg["base_lr"]  # initialise 'lr' field required by AdamW

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay = CFG["weight_decay"],
        betas        = (0.9, 0.999),
    )

    # ?? Training ????????????????????????????????????????????
    print(f"\n[TRAINING] {CFG['n_epochs']} epochs")
    print("-" * 60)

    history = {"train_total": [], "train_pred": [], "train_sigreg": [], "train_dir": [],
               "val_total": [], "val_pred": [], "val_sigreg": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(CFG["n_epochs"]):

        # Update learning rate — scale each group from its own base LR so the
        # backbone (3e-5) and aux head (1e-3) decay in proportion, not to the
        # same absolute value.
        lr_multiplier = get_lr(epoch, CFG["n_epochs"], CFG["warmup_epochs"], 1.0)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["base_lr"] * lr_multiplier
        lr = CFG["lr"] * lr_multiplier  # for logging only

        # ?? Train ??????????????????????????????????????????
        encoder.train()
        predictor.train()

        epoch_losses = {"total": [], "pred": [], "sigreg": [], "dir": []}

        for step, batch in enumerate(train_loader):
            x_hist = batch["x_hist"].to(device)  # (batch, H, 48, 52)
            x_t1   = batch["x_t1"].to(device)    # (batch, 48, 52)
            a_t    = batch["a_t"].to(device)      # (batch, 5)

            optimizer.zero_grad()

            # Encode full history sequence ? proper [z_{t-H+1}, ..., z_{t-1}, z_t]
            z_hist = encoder.encode_sequence(x_hist)  # (batch, H, z_dim)
            z_t    = z_hist[:, -1, :]                  # (batch, z_dim) ? current state

            # Encode next observation
            z_t1   = encoder(x_t1)                    # (batch, z_dim)

            # Predict next state using true temporal history
            z_hat  = predictor(z_hist, a_t)           # (batch, z_dim)

            # NaN guard
            if torch.isnan(z_hat).any():
                print(f"NaN detected at step {step}!")
                print(f"    x_hist NaN: {torch.isnan(x_hist).any()}")
                print(f"    z_hist NaN: {torch.isnan(z_hist).any()}  (encoder output)")
                print(f"    a_t NaN:    {torch.isnan(a_t).any()}")
                print(f"    z_hat NaN:  True  (predictor output)")
                raise RuntimeError("NaN in predictor output ? see diagnostics above")

            # Compute MWM loss (predictor + SIGReg)
            losses = loss_fn(z_t, z_hat, z_t1)
            total_loss = losses["total"]

            # Auxiliary direction loss: z_t must predict H=4 direction.
            # Labels precomputed from raw prices (price[t+4] > price[t]).
            if aux_dir_head is not None and dir_labels_cpu is not None:
                t_idx     = batch["t_now"].clamp(0, len(dir_labels_cpu) - 1)
                dir_label = dir_labels_cpu[t_idx].unsqueeze(1).to(device)   # (B, 1)
                dir_logit = aux_dir_head(z_t)
                dir_loss  = nn.functional.binary_cross_entropy_with_logits(
                                dir_logit, dir_label)
                total_loss = total_loss + lambda_dir * dir_loss
            else:
                dir_loss = torch.tensor(0.0)

            total_loss.backward()

            # Gradient clipping
            clip_params = list(encoder.parameters()) + list(predictor.parameters())
            if aux_dir_head is not None:
                clip_params += list(aux_dir_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, CFG["grad_clip"])
            optimizer.step()

            for k in ["total", "pred", "sigreg"]:
                epoch_losses[k].append(losses[k].item())
            epoch_losses["dir"].append(dir_loss.item())

            if step % CFG["log_every"] == 0:
                dir_str = f"  dir={dir_loss.item():.4f}" if lambda_dir > 0 else ""
                print(f"  E{epoch+1:03d} S{step:04d} | "
                      f"total={total_loss.item():.4f}  "
                      f"pred={losses['pred'].item():.4f}  "
                      f"sigreg={losses['sigreg'].item():.4f}"
                      f"{dir_str}  lr={lr:.2e}")

        # ?? Validate ???????????????????????????????????????
        val_is_empty = (len(val_loader) == 0)
        encoder.eval()
        predictor.eval()
        val_losses = {"total": [], "pred": [], "sigreg": []}

        if not val_is_empty:
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

                    for k in ["total", "pred", "sigreg"]:
                        val_losses[k].append(losses[k].item())

        # ?? Epoch summary ??????????????????????????????????
        train_total = np.mean(epoch_losses["total"])
        val_total   = np.mean(val_losses["total"]) if val_losses["total"] else float("nan")

        history["train_total"].append(train_total)
        history["train_pred"].append(np.mean(epoch_losses["pred"]))
        history["train_sigreg"].append(np.mean(epoch_losses["sigreg"]))
        history["train_dir"].append(np.mean(epoch_losses["dir"]))
        history["val_total"].append(val_total if not np.isnan(val_total) else train_total)
        history["val_pred"].append(np.mean(val_losses["pred"]) if val_losses["pred"] else float("nan"))
        history["val_sigreg"].append(np.mean(val_losses["sigreg"]) if val_losses["sigreg"] else float("nan"))
        history["lr"].append(lr)

        dir_train_str = (f"  dir={np.mean(epoch_losses['dir']):.4f}"
                         if lambda_dir > 0 else "")
        print(f"\n  ?? Epoch {epoch+1:3d}/{CFG['n_epochs']} ??")
        print(f"     Train: total={train_total:.4f}  "
              f"pred={np.mean(epoch_losses['pred']):.4f}  "
              f"sigreg={np.mean(epoch_losses['sigreg']):.4f}"
              f"{dir_train_str}")
        if val_is_empty:
            print(f"     Val:   (empty — using train loss for checkpointing)")
        else:
            print(f"     Val:   total={val_total:.4f}  "
                  f"pred={np.mean(val_losses['pred']):.4f}  "
                  f"sigreg={np.mean(val_losses['sigreg']):.4f}")
        print(f"     LR: {lr:.2e}\n")

        # ?? Checkpoint ????????????????????????????????????
        # When val is available: save on improved val loss (standard early stopping).
        # When val is empty: save on improved train loss (date boundary is the stopping criterion).
        ckpt_score   = train_total if val_is_empty else val_total
        is_last_epoch = (epoch + 1 == CFG["n_epochs"])

        is_best = ckpt_score < best_val_loss
        if is_best or is_last_epoch:
            if is_best:
                best_val_loss = ckpt_score
                best_epoch    = epoch + 1
            ckpt_payload = {
                "epoch":          epoch + 1,
                "encoder":        encoder.state_dict(),
                "predictor":      predictor.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "val_loss":       val_total,
                "train_loss":     train_total,
                "cfg":            CFG,
            }
            if aux_dir_head is not None:
                ckpt_payload["aux_dir_head"] = aux_dir_head.state_dict()
            # best_model.pt must only ever hold the best-scoring epoch. The final
            # epoch goes to last_model.pt — overwriting best with it discards the
            # better weights whenever the run overfits past its optimum.
            if is_best:
                torch.save(ckpt_payload, f"{ckpt_dir}/best_model.pt")
                print(f"  Checkpoint saved [{'train' if val_is_empty else 'val'}={ckpt_score:.4f}]")
            if is_last_epoch:
                torch.save(ckpt_payload, f"{ckpt_dir}/last_model.pt")
                print(f"  Final epoch saved [last_model.pt={ckpt_score:.4f}]")

    # ?? Final summary ????????????????????????????????????
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint:    {ckpt_dir}/best_model.pt")

    # Save loss history as JSON (for paper figures)
    history_path = f"{ckpt_dir}/loss_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Loss history:  {history_path}")
    print("=" * 60)

    _tee.close()
    return encoder, predictor, history


if __name__ == "__main__":
    train()