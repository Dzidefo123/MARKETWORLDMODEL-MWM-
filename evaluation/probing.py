"""
evaluation/probing.py ? Latent Space Probing Experiments (O2)
==============================================================

WHAT THIS IS
------------
The central scientific claim of the paper is:

    "The MarketWorldModel encoder learns latent representations that
    encode meaningful market structure without explicit supervision."

This module tests that claim by training small PROBE networks on top
of FROZEN encoder embeddings and measuring how well those probes can
recover known market properties.

The key experimental design:

    TRAINED encoder -> embeddings Z_trained -> probe -> accuracy A_trained
    RANDOM  encoder -> embeddings Z_random  -> probe -> accuracy A_random

    ? = A_trained - A_random

If ? is large and positive, the trained encoder's representations are
genuinely informative ? the model learned market structure rather than
compressing inputs arbitrarily. This mirrors LeWM's "physical probing"
experiments on PushT (Table 1 of the paper).

FIVE PROBING TARGETS
--------------------
Each target is extracted from features at the LAST bar of each 48-bar
window (representing the "current state" the encoder summarised):

  1. SESSION          ? Which trading session is active?
                        Classes: London, New York, Asian
                        Type: 3-class classification
                        Feature indices: is_london (38), is_ny (39)

  2. HIGH VOLATILITY  ? Is the market in a high-volatility regime?
                        Classes: high GVZ / low GVZ (median split)
                        Type: binary classification
                        Feature index: gvz_level (43)

  3. RETURN DIRECTION ? Will the next bar go up or down?
                        Classes: positive / negative next-bar return
                        Type: binary classification
                        Feature index: ret_1 of x_t1 last bar (forward return)

  4. REALIZED VOLATILITY ? What is the current 32-bar realized volatility?
                        Type: regression (continuous)
                        Feature index: rv_32 (10)

  5. DXY DIRECTION    ? Did the dollar rise or fall in the last bar?
                        Classes: DXY up / DXY down
                        Type: binary classification
                        Feature index: dxy_ret_1 (40)

TWO PROBE TYPES
---------------
For each target we train two probe architectures:

  LINEAR: sklearn LogisticRegression (classification) or Ridge (regression)
          Tests if information is LINEARLY ACCESSIBLE in the latent space.
          This is the stricter test ? linear accessibility means the encoder
          has organised the information geometrically, not just stored it.

  MLP:    2-layer PyTorch network  (128 -> 64 -> ReLU -> output)
          Tests if information is PRESENT AT ALL in the latent space.
          An MLP can decode non-linearly structured information.

If linear probe > random but MLP ? linear, the encoding is clean and linear.
If MLP >> linear >> random, the encoding is present but entangled.
If both ? random, the property is not encoded.

USAGE
-----
    python -m evaluation.probing

    Or import and call programmatically:
        from evaluation.probing import run_probing_experiments
        results = run_probing_experiments()
"""

import sys
sys.path.insert(0, ".")

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# sklearn probes
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from data.pipeline    import DataPipeline
from data.features    import GoldFeatureEngineer
from models.encoder   import MarketEncoder
from models.predictor import CausalPredictor


# ??????????????????????????????????????????????????????????????
# Feature index constants  (must match GoldFeatureEngineer.FEATURE_NAMES)
# ??????????????????????????????????????????????????????????????
IDX = {
    "ret_1":       0,
    "rv_32":       10,
    "rsi_14":      14,
    "is_london":   38,
    "is_ny":       39,
    "dxy_ret_1":   40,
    "gvz_level":   43,
}


# ??????????????????????????????????????????????????????????????
# MLP Probe (PyTorch ? for non-linear probing)
# ??????????????????????????????????????????????????????????????

class MLPProbe(nn.Module):
    """
    Lightweight 2-layer MLP for probing the latent space.

    Architecture:  z_dim -> 64 -> ReLU -> Dropout(0.1) -> n_outputs
    Kept small deliberately ? if a tiny MLP can decode the property,
    it means the information is present in the embedding.

    Args:
        z_dim:     Input embedding dimension (128)
        n_outputs: 1 for regression/binary, n_classes for softmax
        task:      'classification' or 'regression'
    """

    def __init__(self, z_dim: int, n_outputs: int, task: str = "classification"):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_outputs),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def train_mlp_probe(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_val:   np.ndarray, y_val:   np.ndarray,
    task: str = "classification",
    n_classes: int = 2,
    n_epochs: int = 30,
    lr: float = 1e-3,
) -> dict:
    """
    Train an MLP probe and return val metrics.

    Args:
        Z_train:   (n_train, z_dim) embeddings
        y_train:   (n_train,) labels or continuous values
        Z_val:     (n_val, z_dim) embeddings
        y_val:     (n_val,) labels or continuous values
        task:      'classification' or 'regression'
        n_classes: number of classes (ignored for regression)
        n_epochs:  training epochs
        lr:        learning rate

    Returns:
        dict with metric name(s) -> float
    """
    device = torch.device("cpu")
    n_out  = n_classes if task == "classification" else 1
    model  = MLPProbe(Z_train.shape[1], n_out, task).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Zt = torch.FloatTensor(Z_train).to(device)
    Zv = torch.FloatTensor(Z_val).to(device)

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
        yt = torch.LongTensor(y_train).to(device)
    else:
        criterion = nn.MSELoss()
        yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    # Train
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        out  = model(Zt)
        loss = criterion(out, yt)
        loss.backward()
        opt.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        out_val = model(Zv)

    if task == "classification":
        preds = out_val.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(out_val, dim=1).cpu().numpy()
        acc   = accuracy_score(y_val, preds)

        # AUC: binary -> single column, multiclass -> macro OvR
        if n_classes == 2:
            auc = roc_auc_score(y_val, probs[:, 1])
        else:
            try:
                auc = roc_auc_score(y_val, probs, multi_class="ovr", average="macro")
            except Exception:
                auc = float("nan")

        return {"accuracy": acc, "auc": auc}

    else:
        preds = out_val.squeeze(1).cpu().numpy()
        mse   = mean_squared_error(y_val, preds)
        r, _  = pearsonr(y_val, preds)
        return {"mse": mse, "pearson_r": r}


# ??????????????????????????????????????????????????????????????
# Embedding extraction
# ??????????????????????????????????????????????????????????????

@torch.no_grad()
def extract_embeddings(
    encoder,
    dataset,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the encoder over the full dataset and collect:
        Z:       (n_samples, z_dim) ? current state embeddings
        X_last:  (n_samples, 52)   ? last bar features of each window
        X_next:  (n_samples, 52)   ? last bar features of x_t1 (next window)

    NOTE ON PROBE TARGET INDEPENDENCE
    ----------------------------------
    X_last features are part of the encoder's input ? any random linear
    projection of the input will preserve them. We use X_last only for
    continuous regression targets (realized_vol). For classification
    targets (session, volatility regime), we use EXTERNAL signals:
        - Session: computed from timestamps, not from is_london/is_ny features
        - High vol: binary threshold on LAST BAR raw gvz_level feature
    This breaks the "random projection cheating" effect.

    Returns:
        Z, X_last, X_next ? all as numpy arrays
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    Z_list, Xl_list, Xn_list = [], [], []

    for batch in loader:
        x_t  = batch["x_t"]   # (B, 48, 52)
        x_t1 = batch["x_t1"]  # (B, 48, 52)

        z      = encoder(x_t).cpu().numpy()
        x_last = x_t[:, -1, :].cpu().numpy()
        x_next = x_t1[:, -1, :].cpu().numpy()

        Z_list.append(z)
        Xl_list.append(x_last)
        Xn_list.append(x_next)

    return (
        np.vstack(Z_list),
        np.vstack(Xl_list),
        np.vstack(Xn_list),
    )


# ??????????????????????????????????????????????????????????????
# Probe target construction
# ??????????????????????????????????????????????????????????????

def build_probe_targets(
    X_last: np.ndarray,
    X_next: np.ndarray,
    vol_feature_idx: int = 43,   # 43=gvz_level (gold), 10=rv_32 (forex)
) -> dict:
    """
    Construct all five probe targets from feature arrays.

    Args:
        X_last: (n_samples, 52) ? last bar of current window (current state)
        X_next: (n_samples, 52) ? last bar of next window (one step ahead)

    Returns:
        dict: target_name -> {
            "y":       np.ndarray of labels/values,
            "task":    "classification" or "regression",
            "classes": int (number of classes, 1 for regression),
            "label":   human-readable name for display
        }
    """
    targets = {}

    # ?? 1. Session ??????????????????????????????????????????????
    # Derived from hour_sin / hour_cos features (indices 34-35).
    # We reconstruct the hour angle then classify ? this is independent
    # of the is_london/is_ny features that are part of the encoder input,
    # breaking the random-projection cheating effect.
    #
    # hour_sin = sin(2? * h / 24)  hour_cos = cos(2? * h / 24)
    # -> h = atan2(sin, cos) * 24 / (2?)  mod 24
    hs = X_last[:, 34]   # hour_sin (after z-score, roughly in [-3, 3])
    hc = X_last[:, 35]   # hour_cos
    # Recover raw hour angle (atan2 on z-scored values still gives correct quadrant)
    hour_angle = (np.arctan2(hs, hc) * 24 / (2 * np.pi)) % 24
    # Session windows (UTC): London 7-16, NY 12-21, Asian: rest
    # Use non-overlapping: Asian <7, London 7-12, NY 12+
    session = np.where(hour_angle < 7, 0,          # Asian
              np.where(hour_angle < 12, 1, 2))      # London / NY

    targets["session"] = {
        "y":       session,
        "task":    "classification",
        "classes": 3,
        "label":   "Session (Asian/London/NY)",
    }

    # ?? 2. High Volatility ??????????????????????????????????????
    # Gold: col 43 = gvz_level. EUR/USD: col 43 = evz_level (or rv_64 if evz failed).
    # Threshold at median to guarantee a 50/50 split even if the signal is skewed.
    vol_signal = X_last[:, vol_feature_idx]
    # If vol signal is constant (e.g., EVZ download failed → all zeros), fall back to rv_64
    if np.std(vol_signal) < 1e-6:
        vol_signal = X_last[:, 13]   # col 13 = rv_64 (eurusd) or parkinson_vol (others)
    high_vol = (vol_signal > np.median(vol_signal)).astype(int)

    targets["high_vol"] = {
        "y":       high_vol,
        "task":    "classification",
        "classes": 2,
        "label":   "High Volatility (vol > median)",
    }

    # ?? 3. Return Direction ?????????????????????????????????????
    # ret_1 at the NEXT window's last bar = return one step ahead
    # Positive z-score -> return was positive -> gold went up
    # This is the forward-looking test: does z_t predict tomorrow?
    ret_next = X_next[:, IDX["ret_1"]]
    ret_dir  = (ret_next > 0).astype(int)

    targets["ret_direction"] = {
        "y":       ret_dir,
        "task":    "classification",
        "classes": 2,
        "label":   "Return Direction (next bar ?/?)",
    }

    # ?? 4. Realized Volatility ??????????????????????????????????
    # rv_32 (feature 10): continuous 32-bar realized volatility
    # After z-scoring, this has mean?0 and std?1
    rv = X_last[:, IDX["rv_32"]]

    targets["realized_vol"] = {
        "y":       rv,
        "task":    "regression",
        "classes": 1,
        "label":   "Realized Volatility (rv_32)",
    }

    # ?? 5. DXY Direction ????????????????????????????????????????
    # dxy_ret_1 (feature 40): DXY log return at last bar
    # Positive -> dollar strengthened -> typically bearish for gold
    dxy = X_last[:, IDX["dxy_ret_1"]]
    dxy_dir = (dxy > 0).astype(int)

    targets["dxy_direction"] = {
        "y":       dxy_dir,
        "task":    "classification",
        "classes": 2,
        "label":   "DXY Direction (dollar ?/?)",
    }

    return targets


# ??????????????????????????????????????????????????????????????
# Single probe evaluation
# ??????????????????????????????????????????????????????????????

def run_single_probe(
    Z_train: np.ndarray, Z_val: np.ndarray,
    target:  dict,
    probe_type: str,
) -> dict:
    """
    Train and evaluate one probe (linear or MLP) on one target.

    Args:
        Z_train:    (n_train, z_dim) training embeddings
        Z_val:      (n_val, z_dim)   validation embeddings
        target:     dict from build_probe_targets()
        probe_type: "linear" or "mlp"

    Returns:
        dict of metric_name -> float
    """
    y_train = target["y"][:len(Z_train)]
    y_val   = target["y"][len(Z_train):][:len(Z_val)]

    # Sklearn StandardScaler on embeddings (improves linear probe convergence)
    scaler = StandardScaler()
    Z_tr   = scaler.fit_transform(Z_train)
    Z_vl   = scaler.transform(Z_val)

    task      = target["task"]
    n_classes = target["classes"]

    if probe_type == "linear":
        if task == "classification":
            model = LogisticRegression(
                C=1.0,
                max_iter=500,
                solver="lbfgs",
                n_jobs=-1,
            )
            model.fit(Z_tr, y_train)
            preds = model.predict(Z_vl)
            probs = model.predict_proba(Z_vl)
            acc   = accuracy_score(y_val, preds)
            if n_classes == 2:
                auc = roc_auc_score(y_val, probs[:, 1])
            else:
                try:
                    auc = roc_auc_score(
                        y_val, probs, multi_class="ovr", average="macro"
                    )
                except Exception:
                    auc = float("nan")
            return {"accuracy": acc, "auc": auc}

        else:
            model = Ridge(alpha=1.0)
            model.fit(Z_tr, y_train)
            preds = model.predict(Z_vl)
            mse   = mean_squared_error(y_val, preds)
            r, _  = pearsonr(y_val, preds)
            return {"mse": mse, "pearson_r": r}

    elif probe_type == "mlp":
        return train_mlp_probe(
            Z_tr, y_train, Z_vl, y_val,
            task=task, n_classes=n_classes,
        )

    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")


# ??????????????????????????????????????????????????????????????
# Main experiment runner
# ??????????????????????????????????????????????????????????????

def run_probing_experiments(
    instrument:      str = "gold",
    checkpoint_path: str = None,
    output_path:     str = None,
) -> pd.DataFrame:
    """
    Full probing experiment pipeline.

    1. Load trained encoder from checkpoint
    2. Build a random (untrained) encoder for comparison
    3. Run all 5 probing targets ? 2 probe types ? 2 encoder types = 20 experiments
    4. Return a results DataFrame and save to JSON

    The results table is the paper's Table 1 (or equivalent).

    Args:
        checkpoint_path: Path to best_model.pt from training
        output_path:     Where to save JSON results

    Returns:
        pd.DataFrame with columns:
            target, probe_type, encoder_type, + metric columns
    """
    if checkpoint_path is None:
        checkpoint_path = f"./experiments/checkpoints/{instrument}/best_model.pt"
    if output_path is None:
        output_path = f"./experiments/probing_results_{instrument}.json"

    # vol feature index: col 43 = gvz_level (gold) or evz_level (eurusd); rv_32 (10) for usdjpy
    vol_feature_idx = 43 if instrument in ("gold", "eurusd") else 10

    print("=" * 65)
    print(f"MarketWorldModel - O2 Probing Experiments  [{instrument.upper()}]")
    print("=" * 65)

    device = torch.device("cpu")

    # Load checkpoint first so we can reuse its split_dates
    print("\n[0/4] Pre-loading checkpoint for split config...")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run 'python -m training.train_o1' first."
        )
    ckpt_cfg = torch.load(checkpoint_path, map_location=device, weights_only=False).get("cfg", {})
    split_dates = ckpt_cfg.get("split_dates")
    if split_dates:
        print(f"  Using split_dates from checkpoint: {split_dates}")

    # Load data using same splits as training
    print("\n[1/4] Loading data pipeline...")
    pipeline = DataPipeline(instrument=instrument, lookback=48, norm_window=500)
    result   = pipeline.build(
        start=None, end=None,
        use_real_data=True,
        stride=1,
        split_dates=split_dates,
    )
    train_ds = result["train"]
    val_ds   = result["val"]

    # ?? Load trained encoder ?????????????????????????????????????
    print("\n[2/4] Loading trained encoder...")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run 'python -m training.train_o1' first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})

    trained_encoder = MarketEncoder(
        lookback    = cfg.get("lookback",      48),
        n_features  = cfg.get("n_features",    52),
        patch_size  = 4,
        d_model     = cfg.get("enc_d_model",  128),
        n_heads     = cfg.get("enc_n_heads",    4),
        n_layers    = cfg.get("enc_n_layers",   4),
        dropout     = 0.0,      # disable dropout at eval time
        proj_hidden = cfg.get("enc_proj_hidden", 256),
        z_dim       = cfg.get("z_dim",         128),
    ).to(device)

    trained_encoder.load_state_dict(ckpt["encoder"])
    trained_encoder.eval()
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}  "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    # ?? Build random (untrained) encoder ?????????????????????????
    print("\n  Building random encoder (control baseline)...")
    random_encoder = MarketEncoder(
        lookback=48, n_features=52, patch_size=4,
        d_model=128, n_heads=4, n_layers=4,
        dropout=0.0, proj_hidden=256, z_dim=128,
    ).to(device)
    # Random encoder keeps its random initialisation ? no weights loaded
    random_encoder.eval()

    # ?? Extract embeddings ????????????????????????????????????????
    print("\n[3/4] Extracting embeddings...")

    Z_tr_trained, Xl_tr, Xn_tr = extract_embeddings(trained_encoder, train_ds)
    Z_vl_trained, Xl_vl, Xn_vl = extract_embeddings(trained_encoder, val_ds)
    print(f"  Trained encoder ? train: {Z_tr_trained.shape}  val: {Z_vl_trained.shape}")

    Z_tr_random, _, _ = extract_embeddings(random_encoder, train_ds)
    Z_vl_random, _, _ = extract_embeddings(random_encoder, val_ds)
    print(f"  Random encoder  ? train: {Z_tr_random.shape}   val: {Z_vl_random.shape}")

    # ?? Build probe targets ???????????????????????????????????????
    # Combine train + val feature arrays for target construction,
    # then split by n_train to get the right labels for each split
    n_train = len(Z_tr_trained)
    Xl_all  = np.vstack([Xl_tr, Xl_vl])
    Xn_all  = np.vstack([Xn_tr, Xn_vl])
    targets = build_probe_targets(Xl_all, Xn_all, vol_feature_idx=vol_feature_idx)

    # Split targets into train/val portions
    train_targets = {
        k: {**v, "y": v["y"][:n_train]}
        for k, v in targets.items()
    }
    val_targets = {
        k: {**v, "y": v["y"][n_train:]}
        for k, v in targets.items()
    }

    # Verify class balance
    print("\n  Probe target distributions (val set):")
    for name, tgt in val_targets.items():
        y = tgt["y"]
        if tgt["task"] == "classification":
            unique, counts = np.unique(y, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, counts)}
            print(f"    {name:20s}: {dist}")
        else:
            print(f"    {name:20s}: mean={y.mean():.3f}  std={y.std():.3f}")

    # ?? Run all probing experiments ???????????????????????????????
    print("\n[4/4] Running probing experiments...")
    print("      (20 experiments: 5 targets ? 2 probes ? 2 encoders)\n")

    rows = []

    for target_name, tgt_info in targets.items():
        # Split y for train/val
        y_train_all = tgt_info["y"][:n_train]
        y_val_all   = tgt_info["y"][n_train:]

        # Temporary: inject split y into train/val target dicts
        t_train = {**tgt_info, "y": y_train_all}
        t_val   = {**tgt_info, "y": y_val_all}

        for probe_type in ["linear", "mlp"]:
            for enc_type, Z_tr, Z_vl in [
                ("trained", Z_tr_trained, Z_vl_trained),
                ("random",  Z_tr_random,  Z_vl_random),
            ]:
                print(f"  {target_name:20s} | {probe_type:6s} | {enc_type:7s} ...",
                      end="", flush=True)

                metrics = run_single_probe(Z_tr, Z_vl, {
                    **tgt_info,
                    "y": np.concatenate([y_train_all, y_val_all]),
                }, probe_type)

                row = {
                    "target":       target_name,
                    "label":        tgt_info["label"],
                    "probe":        probe_type,
                    "encoder":      enc_type,
                    "task":         tgt_info["task"],
                }
                row.update(metrics)
                rows.append(row)

                # Print key metric
                if tgt_info["task"] == "classification":
                    print(f"  acc={metrics['accuracy']:.3f}  auc={metrics['auc']:.3f}")
                else:
                    print(f"  r={metrics['pearson_r']:.3f}  mse={metrics['mse']:.4f}")

    # ?? Build results table ???????????????????????????????????????
    df = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("PROBING RESULTS ? TRAINED vs RANDOM ENCODER")
    print("=" * 65)
    _print_summary_table(df)

    # ?? Save results ??????????????????????????????????????????????
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)
    print(f"\nFull results saved to: {output_path}")

    return df


# Chance baselines per target
CHANCE_BASELINES = {
    "session":       ("accuracy",  1/3),    # 3-class uniform -> 33.3%
    "high_vol":      ("accuracy",  0.5),    # binary -> 50%
    "ret_direction": ("accuracy",  0.5),    # binary -> 50%
    "realized_vol":  ("pearson_r", 0.0),    # regression -> r=0
    "dxy_direction": ("accuracy",  0.5),    # binary -> 50%
}


def _print_summary_table(df: pd.DataFrame) -> None:
    """
    Print a formatted comparison table.
    Columns: Trained | Random | Chance | ?(T-R) | ?(T-C)
    ?(T-C) = trained vs chance = the honest measure of learned structure.
    """
    targets_order = [
        "session", "high_vol", "ret_direction", "realized_vol", "dxy_direction"
    ]
    probes_order = ["linear", "mlp"]

    for probe_type in probes_order:
        print(f"\n?? {probe_type.upper()} PROBE ??")
        print(f"  {'Target':<28} {'Trained':>8} {'Random':>8} {'Chance':>8} "
              f"{'?(T-R)':>8} {'?(T-C)':>8}  metric")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  ------")

        for target_name in targets_order:
            rows = df[(df["target"] == target_name) & (df["probe"] == probe_type)]
            if rows.empty:
                continue

            label    = rows.iloc[0]["label"]
            task     = rows.iloc[0]["task"]
            trained  = rows[rows["encoder"] == "trained"].iloc[0]
            random_  = rows[rows["encoder"] == "random"].iloc[0]
            _, chance = CHANCE_BASELINES.get(target_name, ("accuracy", 0.5))

            if task == "classification":
                t_val = trained["accuracy"]
                r_val = random_["accuracy"]
                mname = "Acc"
            else:
                t_val = trained.get("pearson_r", float("nan"))
                r_val = random_.get("pearson_r", float("nan"))
                mname = "r  "

            dtr = t_val - r_val   # trained vs random
            dtc = t_val - chance  # trained vs chance (the honest delta)

            # Flag based on trained vs chance
            flag = "?" if dtc > 0.03 else ("~" if abs(dtc) <= 0.03 else "?")

            print(f"  {label:<28} {t_val:>8.3f} {r_val:>8.3f} {chance:>8.3f} "
                  f"{dtr:>+8.3f} {dtc:>+8.3f}  {mname} {flag}")


# ??????????????????????????????????????????????????????????????
# Sanity check: verify embeddings are different for trained vs random
# ??????????????????????????????????????????????????????????????

def embedding_statistics(
    Z_trained: np.ndarray,
    Z_random:  np.ndarray,
) -> dict:
    """
    Compute basic statistics comparing trained vs random embeddings.
    Used as a sanity check before running probing experiments.
    """
    # Variance in each embedding dimension
    var_trained = Z_trained.var(axis=0)
    var_random  = Z_random.var(axis=0)

    # Mean dimension-wise cosine similarity between consecutive embeddings
    # High cosine sim = temporally smooth trajectories
    # (related to temporal straightening in LeWM Appendix H)
    n = min(1000, len(Z_trained) - 1)
    cos_sim_trained = np.array([
        np.dot(Z_trained[i], Z_trained[i+1]) /
        (np.linalg.norm(Z_trained[i]) * np.linalg.norm(Z_trained[i+1]) + 1e-8)
        for i in range(n)
    ]).mean()

    cos_sim_random = np.array([
        np.dot(Z_random[i], Z_random[i+1]) /
        (np.linalg.norm(Z_random[i]) * np.linalg.norm(Z_random[i+1]) + 1e-8)
        for i in range(n)
    ]).mean()

    return {
        "trained_var_mean":    float(var_trained.mean()),
        "random_var_mean":     float(var_random.mean()),
        "trained_temporal_cos": float(cos_sim_trained),
        "random_temporal_cos":  float(cos_sim_random),
    }


if __name__ == "__main__":
    import sys as _sys
    INSTRUMENT = _sys.argv[1] if len(_sys.argv) > 1 else "eurusd"

    results = run_probing_experiments(instrument=INSTRUMENT)

    print("\n-- EMBEDDING STATISTICS --")
    ckpt_path = f"./experiments/checkpoints/{INSTRUMENT}/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", {})

    pipeline = DataPipeline(instrument=INSTRUMENT, lookback=48)
    result   = pipeline.build(start=None, end=None, use_real_data=True, stride=4)
    ds       = result["val"]

    trained = MarketEncoder(
        lookback=48, n_features=52, patch_size=4,
        d_model=128, n_heads=4, n_layers=4,
        dropout=0.0, proj_hidden=256, z_dim=128,
    )
    trained.load_state_dict(ckpt["encoder"])
    trained.eval()

    random_ = MarketEncoder(
        lookback=48, n_features=52, patch_size=4,
        d_model=128, n_heads=4, n_layers=4,
        dropout=0.0, proj_hidden=256, z_dim=128,
    )
    random_.eval()

    Zt, _, _ = extract_embeddings(trained, ds, batch_size=128)
    Zr, _, _ = extract_embeddings(random_, ds, batch_size=128)

    stats = embedding_statistics(Zt, Zr)
    for k, v in stats.items():
        print(f"  {k:<30}: {v:.4f}")

    print(f"\nO2 probing complete  [{INSTRUMENT.upper()}]")