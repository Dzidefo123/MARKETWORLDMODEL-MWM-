# Directional Head Improvement Plan

_Created 2026-06-22. Goal: turn a validated-but-flat system into one that takes **profitable** trades. The plumbing is fixed (see `VALIDATION_NARRATIVE.md`); profitability is now gated by the directional signal._

## The problem, stated precisely

- Deployed `heads_gold`: **dir AUC 0.543**, `structured_features=false`, raw z-history only.
- Magnitude head: **pearson_r = −0.057** — no skill (slightly negative). The "magnitude > 1.5×spread" cost filter is effectively noise.
- Live: `dir_prob` sat in **0.477–0.490 for 22h**, never crossing the 0.53/0.47 thresholds → **0 trades**. The head is both weak (AUC ~0.54) and low-variance (rarely expresses conviction).
- Encoder directional ceiling (linear probe on frozen z): **~0.558**. The head currently extracts *less* directional signal (0.543) than a linear probe does — so there is headroom even without touching the encoder.

## Evaluation protocol (apply to every experiment)

Do **not** judge on AUC alone — a 0.55 head can still lose money. For each variant measure:
1. **Dir AUC** (val, walk-forward) — quick screen.
2. **Backtest on gold test set** (`scripts/backtest_june_mt5.py` / `run_backtest`): Sharpe, max DD, win rate, **# trade entries** (>0, ideally >50 for significance), profit factor.
3. **Trade frequency** at `dir_threshold=0.53` — a head that never fires is useless regardless of AUC.
Keep the encoder **frozen** throughout ([[project_encoder_aux_loss]] proved retraining it doesn't help direction).

---

## Tier 0 — RESULT (2026-06-22): negative, do not deploy

Ran structured-feature head vs matched raw baseline (seed 42, same data window):

| variant | dir AUC | vol AUC | mag r |
|---|---|---|---|
| raw z-history (matched) | 0.5230 | 0.614 | −0.030 |
| structured (probes + S_t) | 0.5281 | 0.610 | −0.074 |
| deployed heads_gold (diff window) | 0.5434 | 0.641 | −0.057 |

**Conclusion:** structured features give +0.005 AUC (noise). The head's input *context* is NOT the bottleneck — the directional signal itself is ~0.52 AUC (barely above 0.50). Magnitude head has negative skill everywhere (cost filter is noise — separate bug to fix). **Do not deploy structured features.** Artifacts: `experiments/heads_gold_structured`, `experiments/heads_gold_raw_seed42`. Pivot to Tier 2 (labels/horizon) and weight Tier 5 (reframe) — the lever is the signal/label, not the head wrapper.

## Tier 0 — Deploy what's already built (original hypothesis, now disproven)

The structured-feature augmentation from the parked plan is **fully coded in `train_heads.py`** (gated on `predictor is not None`) but was never deployed. Retrain heads passing the predictor so the directional head gets:
`z_t (128) + probes [session 3, vol 1, rv 1] + S_t [s_norm, s_rank] = 135-dim` per frame.

- **Effort:** ~minutes; one retrain + one backtest. No new code.
- **Hypothesis:** gives the head structural context (session/vol/surprise) it currently must rediscover from raw z → AUC > 0.543 and, more importantly, more PnL/trades.
- **Also fix the magnitude head here** — its −0.057 r means the cost filter is broken. Inspect the label (`|fwd return|`, horizon 4) and target; consider predicting realized vol instead, or dropping the filter until it has skill.
- **Decision gate:** if structured features beat raw on backtest PnL, deploy and re-run live. If not, the context isn't the bottleneck → Tier 1+.

## Tier 1 — Vol-conditioned directional head

Separate low-vol / high-vol branches, mixed by the frozen vol probe (plan item #2). Rationale: gold momentum is reliable in low vol, mean-reverts in high vol — a single linear head averages these out. Implement as two `DirectionalHead`s blended by `vol_prob`, or a FiLM-style conditioning of the existing head on the vol probe.

## Tier 2 — RESULT (2026-06-22): AUC rises with horizon but is overlap-inflation, NOT tradeable

Horizon sweep (raw head, seed 42), dir AUC: H1 0.502 → H4 0.523 → H8 0.543 → H12 0.548 → **H24 0.589**.
Looked promising (monotonic, H24 above the 0.558 probe baseline). **But the backtest disproved it:** H24 head on the test split (max_holding=24) → return −9.66%, Sharpe −2.70, **win rate 49.89% (coinflip)**, profit factor 0.867. The AUC lift is an artifact of overlapping labels (at H24 consecutive labels share 23/24 bars → autocorrelation inflates AUC on a contiguous val block; evaporates when traded bar-by-bar). **No tradeable directional edge at any horizon.**

CUMULATIVE EVIDENCE (Tier 0 + Tier 2): the directional signal in this encoder is not a profitable alpha source out-of-sample — head input context doesn't help (Tier 0), and no horizon yields a tradeable hit-rate (Tier 2). The honest conclusion points to **Tier 5 (strategic reframe)**: use the validated S_t/vol risk machinery as an overlay and source direction elsewhere. If pursuing labels further, use purged/embargoed CV and non-overlapping samples so AUC isn't overlap-inflated again.

## Tier 2 — Label / target engineering (original plan; revisit only with leak-free CV)

The label is the lever most likely to matter, and the magnitude head's failure suggests the current targets are weak.
- **Horizon sweep:** currently 4 bars (4h). Try {1, 2, 4, 8, 12, 24}. Short horizons are noise; longer may carry trend.
- **Trend-adjusted vs raw vs triple-barrier:** current is `fwd > rolling_median_200`. Try a triple-barrier (TP/SL/time) label that matches how trades actually exit — this aligns the training target with the backtest objective.
- **Regression head:** predict signed forward return and threshold on expected value, instead of a binary classifier — lets sizing scale with conviction.

## Tier 3 — Conviction calibration & thresholds (gets trades *now*)

The user's near-term goal is *some* trades. The head's output is pinned ~0.48 (low variance):
- **Temperature-scale / recalibrate** `dir_prob` so its distribution actually spans the thresholds.
- **Lower `dir_threshold`** (0.53 → 0.51) to generate entries — but only as a diagnostic to see *whether trades are profitable at all*, not as a fix. Pair with strict S_t/vol filtering so quality stays controlled.
- Risk: more trades, lower per-trade edge. Use it to learn, not to ship.

## Tier 4 — Architecture & ensembling

Only if Tiers 0–2 show signal worth amplifying: deeper MLP, attention over the z-history (instead of flatten), or an ensemble across horizons/seeds. Diminishing returns vs Tiers 0–2; defer.

## Tier 5 — RESOLVED (2026-06-22): reframe validated, architecture chosen

The reframe is no longer hypothetical. The Jane Street breakout supplies the directional edge the encoder lacks (walk-forward verified, mean OOS Sharpe 1.81), and MWM's S_t adds measured value **as a Q5-only circuit breaker** (skip top-decile-surprise breakouts: win 49%→54.5%, Sharpe 2.10→2.51). The full four-zone gate would hurt (Q4 is JS's best zone). Build scope: `experiments/COMBINED_SYSTEM_SCOPE.md`. The directional-head effort (Tiers 0–4) is closed — direction isn't the encoder's job.

## Tier 5 — Ceiling check & strategic reframe (original notes)

If the head plateaus below tradeable after Tiers 0–3, the binding constraint is the encoder's **0.558 directional content**, and options become:
- Change the encoder's pretraining objective to inject directional structure (hard; prior aux-loss attempts hurt vol/surprise — see [[project_encoder_aux_loss]]).
- Add features the encoder never saw (order-flow, COT, options skew, cross-asset).
- **Reframe success:** the backtested Sharpe 2.458 came largely from the S_t filter + vol sizing + session selection, not directional alpha. The system may be a strong **risk/vol overlay** rather than a directional predictor — i.e. it's good at *when not to trade*. Consider pairing it with a simple, separately-validated directional rule rather than asking the encoder to be the alpha source.

---

## Recommended sequence

1. **Tier 0** (retrain with structured features + fix/inspect magnitude head) → backtest. _This is the immediate next action._
2. If promising → deploy, re-run live, watch for first real entries.
3. In parallel, **Tier 2 horizon/label sweep** (cheap, high-leverage).
4. Tier 1 vol-conditioning if context helps but isn't enough.
5. Tier 3 only to *diagnose* whether any trades are profitable while the above runs.
6. Re-evaluate against Tier 5 if everything plateaus.

**North-star metric:** not AUC — **live profitable entries**. Every tier is judged by backtest PnL + trade count, then live.
