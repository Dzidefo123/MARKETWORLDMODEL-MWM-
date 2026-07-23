"""
probe_forward_regime.py -- Does JEPA's forward-vol edge appear in HIGH-VOL slices?
=================================================================================

The conditional-thesis test. probe_forward.py showed the trained encoder beats a
random projection at predicting forward realized vol for USD/JPY, but not gold or
EUR/USD. The proposed explanation was "JEPA helps in high-volatility regimes."
A within-asset local check (surprise-vol coupling) partly contradicted it: gold
stayed flat even in gold's volatile bars. This runs the decisive version --
re-run the FORWARD probe restricted to each asset's high-vol vs low-vol samples.

Regime is defined by the encoder's own current volatility (rv_32 at anchor bar t),
so it is known at t -- no lookahead. Both encoders see rv_32, so trained-vs-random
within a slice is a fair test. Tercile thresholds are fit on the TRAIN slice only.

Thesis PASSES if gold/eurusd show trained>random (Δ>0.02) in their high-vol slice
while low-vol stays flat. FAILS if the high-vol slice is also null -> USD/JPY is
idiosyncratic, not a regime law.

Outputs -> probe_forward_regime_{inst}.json

USAGE (Colab, after mount/unzip/deps/symlink)
  !python experiments/probe_forward_regime.py all
  !python experiments/probe_forward_regime.py gold --horizons 12,24
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from data.long_history import build_long
from data.pipeline import DataPipeline
import experiments.retrain_long as R
from models.encoder import MarketEncoder
from evaluation.probing import extract_embeddings, run_single_probe, IDX

DRIVE_RESULTS = "/content/drive/MyDrive/MWM_run/results"
RET1_IDX = IDX["ret_1"]
RV32_IDX = IDX["rv_32"]   # current (backward) realized vol -> regime label, known at t


def _encoder(cfg, device):
    return MarketEncoder(
        lookback=cfg.get("lookback", 48), n_features=cfg.get("n_features", 52),
        patch_size=4, d_model=cfg.get("enc_d_model", 128),
        n_heads=cfg.get("enc_n_heads", 4), n_layers=cfg.get("enc_n_layers", 4),
        dropout=0.0, proj_hidden=cfg.get("enc_proj_hidden", 256),
        z_dim=cfg.get("z_dim", 128),
    ).to(device)


def sample_arrays(ds, horizon):
    """Per sample: forward vol over (t, t+horizon], backward vol rv_32 at t, valid mask."""
    feat = ds.features.numpy()
    lb, n = ds.lookback, ds.features.shape[0]
    fwd = np.full(len(ds.indices), np.nan)
    bwd = np.full(len(ds.indices), np.nan)
    for j, i in enumerate(ds.indices):
        t = i + lb - 1
        a, b = t + 1, t + 1 + horizon
        if b <= n:
            r = feat[a:b, RET1_IDX]
            fwd[j] = np.sqrt(np.mean(r * r))
            bwd[j] = feat[t, RV32_IDX]
    return fwd, bwd, ~np.isnan(fwd)


def _probe_slice(Ztr_t, Zvl_t, Ztr_r, Zvl_r, ytr, yvl, probe):
    """trained & random pearson_r on one regime slice (train+val concatenated)."""
    y = np.concatenate([ytr, yvl])
    tgt = {"y": y, "task": "regression", "classes": 1, "label": "fwd_vol"}
    m_t = run_single_probe(Ztr_t, Zvl_t, tgt, probe)
    m_r = run_single_probe(Ztr_r, Zvl_r, tgt, probe)
    return m_t.get("pearson_r", float("nan")), m_r.get("pearson_r", float("nan"))


def run(instrument, horizons, results_dir=None):
    print("\n" + "#" * 70)
    print(f"# FORWARD-VOL PROBE x REGIME  [{instrument.upper()}]  horizons={horizons}")
    print("#" * 70)

    device = torch.device("cpu")
    ckpt_path = Path(f"{R.CKPT_ROOT}/{instrument}/best_model.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    print(f"  checkpoint epoch {ckpt.get('epoch')}  ret_1 idx={RET1_IDX}  rv_32 idx={RV32_IDX}")

    data = build_long(instrument)
    splits = R.REGIME_SPLITS[instrument]
    pipe = DataPipeline(instrument=instrument, lookback=48, norm_window=500)
    result = pipe.build_from_frames(data["price"], data["macro"],
                                    split_dates=splits, history_len=3, stride=1)
    train_ds, val_ds = result["train"], result["val"]

    trained = _encoder(cfg, device); trained.load_state_dict(ckpt["encoder"]); trained.eval()
    random_ = _encoder(cfg, device); random_.eval()
    Ztr_t, _, _ = extract_embeddings(trained, train_ds)
    Zvl_t, _, _ = extract_embeddings(trained, val_ds)
    Ztr_r, _, _ = extract_embeddings(random_, train_ds)
    Zvl_r, _, _ = extract_embeddings(random_, val_ds)

    rows = []
    print(f"\n  {'horizon':>7} {'probe':>6} {'slice':>8} "
          f"{'trained_r':>10} {'random_r':>10} {'T-R':>8}  {'n':>7}")
    for k in horizons:
        ftr, btr, mtr = sample_arrays(train_ds, k)
        fvl, bvl, mvl = sample_arrays(val_ds, k)
        # tercile thresholds on TRAIN backward-vol only, applied to both splits
        lo_thr, hi_thr = np.nanquantile(btr[mtr], [1/3, 2/3])

        for probe in ["linear", "mlp"]:
            for slabel, sel in [("high_vol", lambda b: b >= hi_thr),
                                ("low_vol",  lambda b: b <= lo_thr)]:
                str_ = mtr & sel(btr)
                svl_ = mvl & sel(bvl)
                rt, rr = _probe_slice(Ztr_t[str_], Zvl_t[svl_], Ztr_r[str_], Zvl_r[svl_],
                                      ftr[str_], fvl[svl_], probe)
                d = rt - rr
                nn = int(str_.sum() + svl_.sum())
                print(f"  {k:>7} {probe:>6} {slabel:>8} {rt:>10.3f} {rr:>10.3f} {d:>+8.3f}  {nn:>7}")
                rows.append({"instrument": instrument, "horizon": k, "probe": probe,
                             "slice": slabel, "trained_r": rt, "random_r": rr,
                             "delta_TR": d, "n": nn})

    out_json = f"./experiments/probe_forward_regime_{instrument}.json"
    json.dump(rows, open(out_json, "w"), indent=2)
    dest = results_dir or (DRIVE_RESULTS if Path("/content/drive").is_dir() else None)
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(out_json, dest)
        print(f"  saved -> {dest}/probe_forward_regime_{instrument}.json")
    else:
        print(f"  wrote {out_json}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", default="all")
    ap.add_argument("--horizons", default="12,24")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    targets = ["gold", "eurusd", "usdjpy"] if args.instrument == "all" else [args.instrument]
    allrows = []
    for t in targets:
        allrows += run(t, horizons, args.results_dir)

    print("\n" + "=" * 70)
    print("THESIS TEST -- max delta(T-R) in HIGH-VOL vs LOW-VOL slice, per instrument:")
    df = pd.DataFrame(allrows)
    for inst in df["instrument"].unique():
        di = df[df["instrument"] == inst]
        hi = di[di["slice"] == "high_vol"]["delta_TR"].max()
        lo = di[di["slice"] == "low_vol"]["delta_TR"].max()
        verdict = "regime lift" if (hi > 0.02 and hi - lo > 0.02) else "no regime lift"
        print(f"  {inst:8s} high_vol best={hi:+.3f}  low_vol best={lo:+.3f}  -> {verdict}")
    print("\nThesis holds ONLY if gold & eurusd show 'regime lift' (high-vol beats random")
    print("AND beats their own low-vol slice). If not, USD/JPY is idiosyncratic.")
