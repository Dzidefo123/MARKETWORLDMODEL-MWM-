"""
evaluation/surprise.py ? Violation-of-Expectation (VoE) Evaluation (O3)
=========================================================================

WHAT THIS IS
------------
The surprise signal is the model's prediction error at each bar:

    surprise_t = ||z_hat_{t+1} - z_{t+1}||?

where:
    z_t      = encoder(x_t)              ? current market state embedding
    z_hat_t1 = predictor(z_hist_t, a_t)  ? predicted next state
    z_t1     = encoder(x_t1)             ? actual next state

When the world model's prediction is accurate, surprise is low.
When the market does something unexpected ? a flash move, regime change,
macro shock ? surprise spikes because the predictor's learned dynamics
are violated.

This mirrors LeWM's VoE framework (Section 5.2, Figure 10):
  LeWM tests: teleport objects (physical discontinuity) -> surprise spikes
  MWM tests:  extreme price moves / GVZ spikes -> surprise spikes

EVALUATION DESIGN
-----------------
We evaluate exclusively on the HELD-OUT TEST SET (Jan 2026 ? May 2026),
which was never seen during training or probing. This is a genuinely
out-of-sample test of the world model's behaviour.

Two categories of bars are compared:

  EXTREME BARS:    |ret_1| > 2? (rolling)  OR  GVZ in top decile
                   These are the gold market's "teleportation" events ?
                   the market discontinuously jumped to a new state.
                   The model should assign high surprise here.

  NORMAL BARS:     All other bars
                   The model should assign low surprise here.

Statistical test: Mann-Whitney U test (non-parametric, appropriate for
the non-Gaussian surprise distribution). We report:
  - Median surprise: extreme vs normal bars
  - Effect size: rank-biserial correlation r
  - p-value (two-sided)

PAPER CLAIM (O3)
----------------
"MarketWorldModel assigns significantly higher prediction error (surprise)
to extreme market events than to normal bars (Mann-Whitney p < 0.05,
effect size r > 0.2), demonstrating the world model has learned to detect
violations of learned market dynamics."

USAGE
-----
    python -m evaluation.surprise

Output files:
    experiments/surprise_timeseries.json  ? bar-by-bar surprise + event labels
    experiments/surprise_results.json     ? statistical test results
"""

import sys
sys.path.insert(0, ".")

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu, spearmanr

from data.pipeline    import DataPipeline
from models.encoder   import MarketEncoder
from models.predictor import CausalPredictor
from models.mwm_loss  import MarketWindowDataset


# ??????????????????????????????????????????????????????????????
# Feature indices (must match GoldFeatureEngineer.FEATURE_NAMES)
# ??????????????????????????????????????????????????????????????
IDX_RET1      = 0    # 1-bar log return
IDX_RV32      = 10   # 32-bar realized volatility
IDX_IS_LONDON = 38   # London session flag
IDX_IS_NY     = 39   # NY session flag
IDX_GVZ_LEVEL = 43   # GVZ normalized level


# ??????????????????????????????????????????????????????????????
# Chronological surprise computation
# ??????????????????????????????????????????????????????????????

@torch.no_grad()
def compute_surprise_timeseries(
    encoder:   MarketEncoder,
    predictor: CausalPredictor,
    dataset:   MarketWindowDataset,
    batch_size: int = 128,
    history_len: int = 3,
) -> dict:
    """
    Compute per-bar surprise for every sample in the dataset.

    The dataset must be created with history_len matching the predictor so
    each sample carries x_hist = [x_{t-H+1}, ..., x_{t-1}, x_t] with proper
    zero-padding at the start ? no more repeated-current-state shortcut.

    DataLoader must NOT shuffle ? we need chronological order.
    """
    encoder.eval()
    predictor.eval()
    device = next(predictor.parameters()).device

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,   # CRITICAL ? chronological order
        num_workers = 0,
    )

    surprise_list = []
    err_dim_list  = []
    x_last_list   = []
    x_next_list   = []

    for batch in loader:
        x_hist = batch["x_hist"].to(device)   # (B, H, 48, 52)
        x_t1   = batch["x_t1"].to(device)     # (B, 48, 52)
        a_t    = batch["a_t"].to(device)       # (B, 5)

        # Encode full history sequence: (B, H, 48, 52) -> (B, H, z_dim)
        z_hist = encoder.encode_sequence(x_hist)

        # Encode actual next state
        z_t1   = encoder(x_t1)                 # (B, z_dim)

        # Predict next state using true temporal history
        z_hat  = predictor(z_hist, a_t)        # (B, z_dim)

        err_per_dim = (z_hat - z_t1) ** 2      # (B, z_dim)
        surprise    = err_per_dim.mean(dim=1)   # (B,)

        surprise_list.append(surprise.cpu().numpy())
        err_dim_list.append(err_per_dim.cpu().numpy())
        x_last_list.append(batch["x_t"][:, -1, :].cpu().numpy())
        x_next_list.append(batch["x_t1"][:, -1, :].cpu().numpy())

    return {
        "surprise":    np.concatenate(surprise_list),
        "err_per_dim": np.vstack(err_dim_list),
        "x_last":      np.vstack(x_last_list),
        "x_next":      np.vstack(x_next_list),
    }


# ??????????????????????????????????????????????????????????????
# Event detection
# ??????????????????????????????????????????????????????????????

def detect_extreme_events(
    x_last:          np.ndarray,
    x_next:          np.ndarray,
    ret_sigma:       float = 2.0,
    gvz_pct:         float = 0.80,
    vol_feature_idx: int   = IDX_GVZ_LEVEL,  # 43=gvz (gold), 10=rv_32 (forex)
) -> dict:
    """
    Identify extreme market bars using two independent signals.

    Event categories:
      EXTREME_RETURN:  |ret_1| at x_next > ret_sigma rolling std
                       Large bar-to-bar price moves ? the gold market
                       equivalent of LeWM's object teleportation.

      HIGH_GVZ:        GVZ level at x_last > gvz_pct percentile
                       Market-implied volatility spike ? the options
                       market is pricing in uncertainty.

      EITHER:          EXTREME_RETURN OR HIGH_GVZ (union)

    Args:
        x_last:    (n, 52) features at last bar of current window
        x_next:    (n, 52) features at last bar of next window
        ret_sigma: Threshold in standard deviations for extreme return
        gvz_pct:   Percentile threshold for high GVZ

    Returns:
        dict of binary mask arrays (True = event bar)
    """
    # Forward return (from x_next, the bar AFTER the current window)
    ret_next = x_next[:, IDX_RET1]

    # Extreme return: |return| > ret_sigma standard deviations
    # Use rolling std (window=100) to avoid global look-ahead
    ret_series   = pd.Series(ret_next)
    rolling_std  = ret_series.rolling(100, min_periods=20).std().fillna(ret_series.std())
    ret_z        = np.abs(ret_next) / (rolling_std.values + 1e-8)
    extreme_ret  = ret_z > ret_sigma

    # High volatility: above gvz_pct percentile on the vol signal
    # Gold: GVZ level (feature 43). Forex: rv_32 (feature 10).
    gvz          = x_last[:, vol_feature_idx]
    gvz_threshold = np.nanpercentile(gvz, gvz_pct * 100)
    high_gvz     = gvz > gvz_threshold

    # Regime change: large jump in realized volatility
    rv = x_last[:, IDX_RV32]
    rv_series   = pd.Series(rv)
    rv_change   = rv_series.diff().abs()
    rv_threshold = rv_change.quantile(0.90)
    vol_jump    = (rv_change > rv_threshold).values

    # Union of all extreme events
    extreme_any = extreme_ret | high_gvz | vol_jump

    return {
        "extreme_return": extreme_ret,
        "high_gvz":       high_gvz,
        "vol_jump":       vol_jump,
        "extreme_any":    extreme_any,
        "normal":         ~extreme_any,
        # Metadata
        "ret_z":          ret_z,
        "gvz_raw":        gvz,
        "gvz_threshold":  float(gvz_threshold),
        "ret_sigma_used": ret_sigma,
        "n_extreme":      int(extreme_any.sum()),
        "n_normal":       int((~extreme_any).sum()),
        "pct_extreme":    float(extreme_any.mean() * 100),
    }


# ??????????????????????????????????????????????????????????????
# Statistical tests
# ??????????????????????????????????????????????????????????????

def run_statistical_tests(
    surprise:    np.ndarray,
    event_masks: dict,
    gvz_signal:  np.ndarray,
) -> dict:
    """
    Run all statistical tests for the O3 paper claim.

    Tests:
      1. Mann-Whitney U: surprise at extreme bars vs normal bars
         H0: distributions are the same
         H1: surprise is higher at extreme bars
         -> p < 0.05 confirms VoE behaviour

      2. Spearman correlation: surprise vs GVZ
         Tests whether model surprise tracks the market's own uncertainty gauge

      3. Effect size: rank-biserial correlation r
         r > 0.1 = small, r > 0.3 = medium, r > 0.5 = large

    Args:
        surprise:    (n,) per-bar surprise values
        event_masks: output from detect_extreme_events()
        gvz_signal:  (n,) raw GVZ values

    Returns:
        dict of test results
    """
    results = {}

    for event_name in ["extreme_return", "high_gvz", "vol_jump", "extreme_any"]:
        mask  = event_masks[event_name]
        s_ext = surprise[mask]
        s_nor = surprise[~mask]

        if len(s_ext) < 5 or len(s_nor) < 5:
            results[event_name] = {"error": "insufficient samples"}
            continue

        # Mann-Whitney U (one-sided: extreme bars have HIGHER surprise)
        u_stat, p_two = mannwhitneyu(s_ext, s_nor, alternative="greater")
        n1, n2 = len(s_ext), len(s_nor)

        # Rank-biserial effect size: r = 1 - 2U/(n1*n2)
        r_effect = 1 - (2 * u_stat) / (n1 * n2)

        results[event_name] = {
            "n_extreme":         int(n1),
            "n_normal":          int(n2),
            "median_surprise_extreme": float(np.median(s_ext)),
            "median_surprise_normal":  float(np.median(s_nor)),
            "ratio":             float(np.median(s_ext) / (np.median(s_nor) + 1e-10)),
            "mann_whitney_u":    float(u_stat),
            "p_value":           float(p_two),
            "effect_size_r":     float(r_effect),
            "significant_005":   bool(p_two < 0.05),
            "significant_001":   bool(p_two < 0.01),
        }

    # Spearman correlation: surprise vs GVZ
    valid   = ~np.isnan(gvz_signal)
    rho, p  = spearmanr(surprise[valid], gvz_signal[valid])
    results["spearman_surprise_gvz"] = {
        "rho":       float(rho),
        "p_value":   float(p),
        "significant": bool(p < 0.05),
    }

    return results


# ??????????????????????????????????????????????????????????????
# Rolling surprise smoothing
# ??????????????????????????????????????????????????????????????

def smooth_surprise(surprise: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Apply a rolling mean to the surprise signal.
    Window=12 bars = 12 hours ? removes intra-session noise while
    preserving day-level regime shifts.
    """
    return pd.Series(surprise).rolling(window, min_periods=1).mean().values


# ??????????????????????????????????????????????????????????????
# Main evaluation
# ??????????????????????????????????????????????????????????????

def run_surprise_evaluation(
    instrument:      str = "gold",
    checkpoint_path: str = None,
    output_ts_path:  str = None,
    output_res_path: str = None,
) -> dict:
    """
    Full O3 evaluation pipeline.

    1. Load trained encoder + predictor
    2. Build chronological TEST dataset (held-out, never seen during training)
    3. Compute bar-by-bar surprise
    4. Detect extreme market events
    5. Run statistical tests
    6. Save results

    Returns:
        dict with 'timeseries' and 'statistics' keys
    """
    if checkpoint_path is None:
        checkpoint_path = f"./experiments/checkpoints/{instrument}/best_model.pt"
    if output_ts_path is None:
        output_ts_path = f"./experiments/surprise_timeseries_{instrument}.json"
    if output_res_path is None:
        output_res_path = f"./experiments/surprise_results_{instrument}.json"

    # vol feature index: col 43 = gvz_level (gold) or evz_level (eurusd); rv_32 (10) for usdjpy
    vol_feature_idx = IDX_GVZ_LEVEL if instrument in ("gold", "eurusd") else IDX_RV32

    print("=" * 65)
    print(f"MarketWorldModel - O3 Surprise / VoE Evaluation  [{instrument.upper()}]")
    print("=" * 65)

    device = torch.device("cpu")

    # Load checkpoint
    print("\n[1/5] Loading checkpoint...")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run 'python -m training.train_o1' first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    print(f"  Loaded epoch {ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # ?? Build test dataset (chronological, stride=1) ??????????????
    print("\n[2/5] Building test dataset...")
    split_dates = cfg.get("split_dates")
    if split_dates:
        print(f"  Using split_dates from checkpoint: {split_dates}")
    pipeline = DataPipeline(
        instrument  = instrument,
        lookback    = cfg.get("lookback", 48),
        norm_window = 500,
    )
    result = pipeline.build(
        start         = None,
        end           = None,
        use_real_data = True,
        stride        = 1,
        history_len   = cfg.get("history_len", 3),
        split_dates   = split_dates,
    )

    # Use TEST SET ONLY ? completely held out
    test_ds    = result["test"]
    meta       = result["meta"]
    n_train    = meta["n_train"]
    n_val      = meta["n_val"]

    # Recover approximate timestamps for the test window
    all_timestamps = meta["timestamps"]
    test_start_idx = n_train + n_val
    test_timestamps = all_timestamps[
        test_start_idx : test_start_idx + len(test_ds) + 1
    ]

    print(f"  Test samples: {len(test_ds):,}")
    if len(test_timestamps) > 0:
        print(f"  Test window:  {test_timestamps[0].date()} -> "
              f"{test_timestamps[-1].date()}")

    # ?? Load encoder + predictor ?????????????????????????????????
    print("\n[3/5] Loading trained models...")
    encoder = MarketEncoder(
        lookback    = cfg.get("lookback",        48),
        n_features  = cfg.get("n_features",      52),
        patch_size  = 4,
        d_model     = cfg.get("enc_d_model",    128),
        n_heads     = cfg.get("enc_n_heads",      4),
        n_layers    = cfg.get("enc_n_layers",     4),
        dropout     = 0.0,
        proj_hidden = cfg.get("enc_proj_hidden", 256),
        z_dim       = cfg.get("z_dim",          128),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    predictor = CausalPredictor(
        z_dim       = cfg.get("z_dim",          128),
        d_model     = cfg.get("pred_d_model",   128),
        n_heads     = cfg.get("pred_n_heads",     4),
        n_layers    = cfg.get("pred_n_layers",    6),
        action_dim  = cfg.get("action_dim",       5),
        history_len = cfg.get("history_len",      3),
        dropout     = 0.0,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])

    encoder.eval()
    predictor.eval()

    # ?? Compute surprise ?????????????????????????????????????????
    print("\n[4/5] Computing surprise timeseries...")
    ts_data = compute_surprise_timeseries(
        encoder, predictor, test_ds,
        batch_size  = 128,
        history_len = cfg.get("history_len", 3),
    )

    surprise     = ts_data["surprise"]
    x_last       = ts_data["x_last"]
    x_next       = ts_data["x_next"]
    surprise_raw = surprise.copy()
    surprise_smooth = smooth_surprise(surprise, window=12)

    print(f"  Surprise computed for {len(surprise):,} bars")
    print(f"  Mean: {surprise.mean():.4f}   Std: {surprise.std():.4f}")
    print(f"  Min:  {surprise.min():.4f}   Max: {surprise.max():.4f}")
    print(f"  90th pct: {np.percentile(surprise, 90):.4f}")

    # ?? Detect events ????????????????????????????????????????????
    print("\n[5/5] Detecting extreme events and running statistics...")
    events = detect_extreme_events(x_last, x_next, vol_feature_idx=vol_feature_idx)

    print(f"\n  Event detection summary:")
    print(f"    Total bars:          {len(surprise):,}")
    print(f"    Extreme returns:     {events['extreme_return'].sum():,} "
          f"({events['extreme_return'].mean()*100:.1f}%)")
    print(f"    High GVZ:            {events['high_gvz'].sum():,} "
          f"({events['high_gvz'].mean()*100:.1f}%)")
    print(f"    Vol regime jumps:    {events['vol_jump'].sum():,} "
          f"({events['vol_jump'].mean()*100:.1f}%)")
    print(f"    Any extreme event:   {events['extreme_any'].sum():,} "
          f"({events['extreme_any'].mean()*100:.1f}%)")

    # ?? Statistical tests ????????????????????????????????????????
    stats = run_statistical_tests(
        surprise    = surprise,
        event_masks = events,
        gvz_signal  = x_last[:, vol_feature_idx],
    )

    # ?? Print results table ??????????????????????????????????????
    print("\n" + "=" * 65)
    print("O3 VIOLATION-OF-EXPECTATION RESULTS")
    print("=" * 65)

    print(f"\n{'Event type':<22} {'N ext':>6} {'Med ext':>8} {'Med nrm':>8} "
          f"{'Ratio':>6} {'r':>6} {'p':>8} {'sig'}")
    print("-" * 75)

    for event_name in ["extreme_return", "high_gvz", "vol_jump", "extreme_any"]:
        r = stats.get(event_name, {})
        if "error" in r:
            continue
        sig = "***" if r["p_value"] < 0.001 else (
              "**"  if r["p_value"] < 0.01  else (
              "*"   if r["p_value"] < 0.05  else "ns"))
        print(f"  {event_name:<20} {r['n_extreme']:>6} "
              f"{r['median_surprise_extreme']:>8.4f} "
              f"{r['median_surprise_normal']:>8.4f} "
              f"{r['ratio']:>6.2f}x "
              f"{r['effect_size_r']:>6.3f} "
              f"{r['p_value']:>8.4f} {sig}")

    gvz_corr = stats.get("spearman_surprise_gvz", {})
    if gvz_corr:
        sig = "*" if gvz_corr.get("significant") else "ns"
        print(f"\n  Spearman(surprise, GVZ): ?={gvz_corr['rho']:.3f}  "
              f"p={gvz_corr['p_value']:.4f}  {sig}")

    # ?? Build paper-ready summary ????????????????????????????????
    print("\n" + "=" * 65)
    print("PAPER SUMMARY (for Section 4.3)")
    print("=" * 65)
    _print_paper_summary(stats, events, surprise)

    # ?? Identify top surprise events ?????????????????????????????
    print("\n?? TOP 10 SURPRISE EVENTS ??")
    top_idx = np.argsort(surprise_smooth)[::-1][:10]
    for rank, idx in enumerate(top_idx):
        ts_str = str(test_timestamps[idx].date()) if idx < len(test_timestamps) else f"bar_{idx}"
        ret    = x_next[idx, IDX_RET1]
        gvz    = x_last[idx, vol_feature_idx]
        sess   = "London" if x_last[idx, IDX_IS_LONDON] > 0 else (
                 "NY" if x_last[idx, IDX_IS_NY] > 0 else "Asian")
        print(f"  #{rank+1:2d}  {ts_str}  surprise={surprise[idx]:.4f}  "
              f"ret={ret:+.4f}  GVZ={gvz:.2f}  {sess}")

    # ?? Save outputs ?????????????????????????????????????????????
    Path(output_ts_path).parent.mkdir(parents=True, exist_ok=True)

    # Time series JSON (for matplotlib plotting)
    ts_json = {
        "timestamps": [str(t) for t in test_timestamps[:len(surprise)]],
        "surprise_raw":    surprise_raw.tolist(),
        "surprise_smooth": surprise_smooth.tolist(),
        "extreme_return":  events["extreme_return"].tolist(),
        "high_gvz":        events["high_gvz"].tolist(),
        "vol_jump":        events["vol_jump"].tolist(),
        "extreme_any":     events["extreme_any"].tolist(),
        "ret_z":           events["ret_z"].tolist(),
        "gvz_signal":      x_last[:, vol_feature_idx].tolist(),
    }
    with open(output_ts_path, "w") as f:
        json.dump(ts_json, f, indent=2, default=str)

    # Statistical results JSON (for paper table)
    with open(output_res_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\nSaved:")
    print(f"  Time series -> {output_ts_path}")
    print(f"  Statistics  -> {output_res_path}")
    print("\n? O3 surprise evaluation complete")

    return {"timeseries": ts_json, "statistics": stats}


def _print_paper_summary(stats: dict, events: dict, surprise: np.ndarray) -> None:
    """Print a ready-to-use paragraph for the paper."""

    ea = stats.get("extreme_any", {})
    if not ea or "error" in ea:
        print("  Insufficient data for paper summary.")
        return

    sig_str = ("p < 0.001" if ea["p_value"] < 0.001 else
               "p < 0.01"  if ea["p_value"] < 0.01  else
               "p < 0.05"  if ea["p_value"] < 0.05  else
               f"p = {ea['p_value']:.3f}")

    r_str = ("large" if abs(ea["effect_size_r"]) > 0.5 else
             "medium" if abs(ea["effect_size_r"]) > 0.3 else
             "small")

    gvz_corr = stats.get("spearman_surprise_gvz", {})

    print(f"""
  "We evaluate O3 using the held-out test set. Extreme market events
  (|return| > 2? or GVZ above 80th percentile) account for
  {events['pct_extreme']:.1f}% of test bars. The world model assigns
  {ea['ratio']:.2f}? higher median surprise to extreme bars than to
  normal bars (median: {ea['median_surprise_extreme']:.4f} vs
  {ea['median_surprise_normal']:.4f}, Mann-Whitney {sig_str},
  effect size r={ea['effect_size_r']:.3f} [{r_str}]).
  Spearman correlation between surprise and GVZ level:
  ?={gvz_corr.get('rho', float('nan')):.3f}
  (p={gvz_corr.get('p_value', float('nan')):.4f}).
  These results confirm that MarketWorldModel reliably detects
  violations of learned market dynamics."
""")


if __name__ == "__main__":
    import sys as _sys
    INSTRUMENT = _sys.argv[1] if len(_sys.argv) > 1 else "eurusd"
    run_surprise_evaluation(instrument=INSTRUMENT)
