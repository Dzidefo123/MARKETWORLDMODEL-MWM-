"""
scripts/export_m15_for_runpod.py

Run this locally (Windows + MT5 open) to export the pre-computed M15 feature
splits to disk. The output goes to runpod_upload/ and is then rsync'd to the
RunPod pod — no MT5 needed on the pod side.

Usage:
    python -m scripts.export_m15_for_runpod
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import logging
from pathlib import Path

from experiments.phase0_m15_analysis import (
    fetch_m15_dataset,
    build_m15_pipeline,
    M15_CFG,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = Path("./runpod_upload/data")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching M15 data from MT5...")
    m15_ohlcv, macro_m15 = fetch_m15_dataset(n_bars=M15_CFG["n_m15_bars"])

    logger.info("Building feature splits...")
    splits = build_m15_pipeline(
        m15_ohlcv, macro_m15,
        split_dates=M15_CFG["split_dates"],
        norm_window=M15_CFG["norm_window"],
    )

    logger.info("Saving splits to %s ...", OUT_DIR)
    np.save(OUT_DIR / "train_feat.npy", splits["train_feat"])
    np.save(OUT_DIR / "train_act.npy",  splits["train_act"])
    np.save(OUT_DIR / "val_feat.npy",   splits["val_feat"])
    np.save(OUT_DIR / "val_act.npy",    splits["val_act"])

    sizes = {k: v.shape for k, v in splits.items() if hasattr(v, "shape")}
    logger.info("Exported: %s", sizes)
    logger.info("Done. Upload runpod_upload/ to the pod.")

if __name__ == "__main__":
    main()
