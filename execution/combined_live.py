"""
execution/combined_live.py — JS breakout signal + MWM S_t-zone LOGGER (dry-run data collector).
================================================================================================

IMPORTANT: the MWM "Q5 gate" is NOT a validated edge. The attribution that suggested it had a
one-bar lookahead; corrected causally, MWM's S_t does not improve Jane Street (see
experiments/COMBINED_SYSTEM_RATIONALE.md correction). The real trading edge is JS-ALONE.

This tool therefore runs purely as a DRY-RUN LOGGER. On each completed H1 bar where JS fires a
breakout, it records the signal + the CAUSAL S_t-rank/zone + what a hypothetical Q5 gate WOULD do.
Acts on nothing. Purpose: accumulate honest forward data to test whether ANY regime signal (e.g.
the faint "calm/Q1 entries do better" hint, n=13 in backtest) holds prospectively, before ever
gating live. The `gated_action` column is a HYPOTHESIS to evaluate, not a recommendation.

Runs in the MWM env (torch + JS via sys.path). Bars from MWM's MT5Connector. Encoder: best_model.pt.
S_t-rank uses S[-2] (the latest CLOSED bar's realized-surprise rank — causal, no lookahead).

Live order execution (--live) is intentionally disabled.

Usage:
    python -m execution.combined_live                 # dry-run, default settings
    python -m execution.combined_live --gate 0.90 --poll 60
"""

from __future__ import annotations

import sys, os, argparse, logging, csv, json, time, contextlib
sys.path.insert(0, ".")
_JS_SRC = r"C:/Users/kalom/Downloads/janestreet/janestreet/src"
_JS_ROOT = r"C:/Users/kalom/Downloads/janestreet/janestreet"
sys.path.insert(0, _JS_SRC)

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Jane Street (signal + sizing + config)
from janestreet_mvp.config import load_config
from janestreet_mvp.models import Side
from janestreet_mvp.strategy import SessionBreakoutStrategy
from janestreet_mvp.risk import RiskEngine

# MWM (bars + S_t gate)
from execution.mt5_connector import MT5Connector
from models.encoder import MarketEncoder
from models.predictor import CausalPredictor
from data.pipeline import DataPipeline
from execution.heads import encode_split, compute_surprise_features
from scripts.retrain_gold_h1_mt5 import fetch_h1_macro

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("combined_live")

_LOOKBACK, _HISTORY_LEN = 48, 3


def _zone(sr: float) -> str:
    if sr < 0.20: return "Q1"
    if sr < 0.60: return "Q2"
    if sr < 0.90: return "Q4"
    return "Q5"


class MwmGate:
    """Computes the MWM S_t-rank (causal surprise percentile) for the latest bar."""

    def __init__(self, checkpoint: str):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.enc = MarketEncoder(); self.enc.load_state_dict(ckpt["encoder"]); self.enc.eval()
        self.pred = CausalPredictor(); self.pred.load_state_dict(ckpt["predictor"]); self.pred.eval()
        for p in list(self.enc.parameters()) + list(self.pred.parameters()):
            p.requires_grad_(False)
        self.pipe = DataPipeline(instrument="gold", lookback=_LOOKBACK, norm_window=500)
        logger.info("MWM gate encoder loaded: %s (epoch %s)", checkpoint, ckpt.get("epoch", "?"))

    @torch.no_grad()
    def s_rank(self, price_df: pd.DataFrame) -> float:
        """price_df: OHLCV indexed by tz-aware UTC timestamp, >= ~800 H1 bars."""
        macro_df = fetch_h1_macro(price_df.index)
        with contextlib.redirect_stdout(open(os.devnull, "w")):   # silence pipeline prints
            result = self.pipe.build_from_frames(price_df, macro_df, stride=1,
                                                 history_len=_HISTORY_LEN, split_dates=None)
        meta = result["meta"]
        feats = torch.tensor(meta["norm_features"], dtype=torch.float32)
        Z = encode_split(self.enc, feats, lookback=_LOOKBACK)
        S = compute_surprise_features(Z, meta["macro_vecs"], self.pred,
                                      lookback=_LOOKBACK, history_len=_HISTORY_LEN)
        # compute_surprise_features stores surp[i] = surprise REALIZED at bar i+1, and pads the
        # last row with 0. So the latest CLOSED bar's realized-surprise rank is S[-2], not S[-1]
        # (S[-1] is the pad). Using S[-1] would always read ~0. Causal: known at this bar's close.
        return float(S[-2, 1])


def _ensure_log(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "logged_at", "bar_ts", "side", "entry", "stop", "take_profit", "lots",
                "s_rank", "zone", "js_alone", "gated_action",
            ])


def main():
    ap = argparse.ArgumentParser(description="JS breakout + MWM Q5 gate (dry-run)")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--checkpoint", default="./experiments/checkpoints/best_model.pt")
    ap.add_argument("--js-config", default=f"{_JS_ROOT}/config/config.yaml")
    ap.add_argument("--gate", type=float, default=0.90, help="veto entry if S_t-rank >= this (Q5)")
    ap.add_argument("--poll", type=int, default=60, help="poll seconds")
    ap.add_argument("--history", type=int, default=1000, help="H1 bars fetched per poll")
    ap.add_argument("--signal-log", default="./experiments/combined_signals.csv")
    ap.add_argument("--status-json", default="./experiments/combined_status.json")
    ap.add_argument("--live", action="store_true",
                    help="NOT IMPLEMENTED — placing real orders is intentionally disabled in v1")
    args = ap.parse_args()

    if args.live:
        raise SystemExit("Live execution is not wired in v1. Validate in dry-run first; "
                         "then add JS MT5Broker.submit_order at the TODO.")

    log_path = Path(args.signal_log)
    status_path = Path(args.status_json)
    _ensure_log(log_path)

    cfg = load_config(args.js_config)
    strat = SessionBreakoutStrategy(cfg.strategy)
    risk = RiskEngine(cfg.trading.risk_per_trade, cfg.trading.max_notional_exposure)
    gate = MwmGate(args.checkpoint)

    conn = MT5Connector(symbol=args.symbol)
    conn.connect()
    try:
        equity0 = conn.get_equity()
    except Exception:
        equity0 = cfg.trading.initial_cash

    logger.info("Combined live (DRY-RUN). symbol=%s  gate=Q5(>=%.2f)  poll=%ds  encoder=%s",
                args.symbol, args.gate, args.poll, Path(args.checkpoint).name)
    logger.info("Signal log: %s", log_path)

    counts = {"Q1": 0, "Q2": 0, "Q4": 0, "Q5": 0}
    n_signals = n_vetoed = 0
    last_bar_ts = None

    try:
        while True:
            try:
                bars = conn.fetch_ohlcv_bars(n=args.history, timeframe="H1")  # OHLCV indexed by UTC
                df = bars.reset_index().rename(columns={"time": "ts"})        # JS expects a 'ts' column
                bar_ts = pd.Timestamp(df["ts"].iloc[-1])

                if bar_ts == last_bar_ts:
                    time.sleep(args.poll); continue
                last_bar_ts = bar_ts

                # --- JS signal on the just-closed bar ---
                df_ind = strat.attach_indicators(df)
                sig = strat.generate_signal(df_ind, len(df_ind) - 1)

                if sig.side == Side.FLAT or sig.stop_loss is None or sig.take_profit is None:
                    _write_status(status_path, args, bar_ts, None, counts, n_signals, n_vetoed)
                    time.sleep(args.poll); continue

                # --- MWM gate: S_t-rank at this bar ---
                t0 = time.time()
                sr = gate.s_rank(bars[["open", "high", "low", "close", "volume"]])
                zone = _zone(sr)
                vetoed = sr >= args.gate

                entry = float(df["close"].iloc[-1])
                try:
                    equity = conn.get_equity()
                except Exception:
                    equity = equity0
                lots = round(risk.calc_position_size(equity, entry, sig.stop_loss), 2)

                n_signals += 1
                counts[zone] += 1
                if vetoed:
                    n_vetoed += 1

                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        str(bar_ts), sig.side.value, round(entry, 2),
                        round(sig.stop_loss, 2), round(sig.take_profit, 2), lots,
                        round(sr, 4), zone, "take", ("veto" if vetoed else "take"),
                    ])
                logger.info("SIGNAL %s @ %.2f  S_t-rank=%.3f [%s]  -> JS-alone=TAKE  gated=%s  "
                            "(lots~%.2f, %.1fs)", sig.side.value, entry, sr, zone,
                            "VETO" if vetoed else "TAKE", lots, time.time() - t0)

                # TODO(live): if not vetoed: broker.submit_order(Order(..., sl, tp))  — JS MT5Broker

                _write_status(status_path, args, bar_ts, {"side": sig.side.value, "s_rank": sr,
                              "zone": zone, "vetoed": vetoed, "entry": entry}, counts,
                              n_signals, n_vetoed)

            except Exception as exc:   # transient data/macro hiccup — log and retry next poll
                logger.warning("poll error (retrying): %s", exc)
            time.sleep(args.poll)

    except KeyboardInterrupt:
        logger.info("Stopped. signals=%d vetoed=%d zone_counts=%s", n_signals, n_vetoed, counts)
    finally:
        conn.disconnect()


def _write_status(path, args, bar_ts, last_sig, counts, n_signals, n_vetoed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "dry-run", "symbol": args.symbol, "gate_threshold": args.gate,
            "last_bar": str(bar_ts), "last_signal": last_sig,
            "n_signals": n_signals, "n_vetoed_Q5": n_vetoed, "zone_counts": counts,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
