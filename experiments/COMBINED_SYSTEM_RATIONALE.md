# Combined System — Full Rationale & Decision Log

_Jane Street breakout × MWM regime gate. Written 2026-06-22. This is the "why": the reasoning chain, the experiments, and the numbers behind every decision, from the moment the Jane Street code entered the picture. For the live build design see `COMBINED_SYSTEM_SCOPE.md`._

> ## ⚠️ CORRECTION (2026-06-22) — the MWM gate result was a lookahead artifact
> While building `combined_live.py` the smoke test exposed a one-bar lookahead in the attribution
> (`s_rank[i]` is the surprise realized at bar **i+1**; it was mapped to a trade at bar i, leaking
> whether the breakout had already reversed). **Re-run causally, the Q5 gate finding vanishes:** Q5 is
> not a losing zone (best_model: 43.8% win, +177), and skip-Q5 *reduces* PnL (1289→1112) and Sharpe
> (2.10→1.92). **MWM's S_t does NOT improve Jane Street causally.** Decision: **ship JS-alone** (its
> walk-forward edge, §2, is real); MWM is not in the trade path. `combined_live.py` is repurposed as a
> dry-run logger to prospectively test the one faint hint (calm/Q1 entries did best, 61.5% win, but n=13
> — not actionable). **Sections 3–7 below describe the leaked (incorrect) gate and are kept for the
> record / lesson, NOT as the conclusion.** Section 2 (JS edge) stands.

---

## 0. How we got here (the problem that forced the reframe)

MWM (Market World Model) is a JEPA-style encoder that maps a 48-bar window of 52 features into a latent state `z_t`, with a predictor that forecasts the next state. The prediction error `S_t = ‖ẑ_t − z_t‖²` ("surprise") is the system's signature signal.

Two things were established earlier this session, both with evidence:

1. **The live S_t "spikes" were a bug, not signal.** Morning S_t≈1.9 spikes that made the circuit breaker refuse to trade for ~2 weeks were a normalization artifact (live normalized features over ~48 bars instead of the trained 500; restart mismatch + macro zero-fill). Fixed and re-validated live: a full clean day at S_t≈0.0004, no spikes. _(See `VALIDATION_NARRATIVE.md`.)_

2. **The encoder cannot predict direction.** Measured exhaustively:
   - Deployed directional head: **AUC 0.543** (≈ coinflip). Linear-probe ceiling on the embeddings: ~0.558.
   - Structured-feature augmentation (probes + S_t → 135-dim): **+0.005 AUC** — noise. Head *input* is not the bottleneck.
   - Horizon sweep {1,2,4,8,12,24}: AUC rose to **0.589 @ H24**, but that was overlapping-label inflation — backtested out-of-sample it was a **49.9% win rate, −2.70 Sharpe**. No tradeable directional edge at any horizon.

**Conclusion that forced the reframe:** the encoder's value is *regime detection and risk* (S_t, vol classification, session structure), **not directional alpha**. A risk overlay needs something to overlay. That "something" is a separately-validated directional strategy — which already existed in this project.

---

## 1. The Jane Street system — what it is and why it fits

Location: `C:/Users/kalom/Downloads/janestreet/janestreet`. A self-contained, leak-aware momentum strategy (`src/janestreet_mvp/strategy.py`, ~150 lines).

**Strategy: Asian-range breakout on XAUUSD H1.**
- Compute the high/low of the Asian session (00:00–07:00 UTC).
- Go **long** on a close above `asian_high + 5bps`, **short** below `asian_low − 5bps`.
- **Gates** (all must pass): prior-day **daily ADX ≥ 25** (trend regime), price on the right side of the **MA200** (trend filter), and **ATR-expansion ≥ 1.1** (fast/slow ATR — volatility is expanding).
- **Exits:** ATR stop (×1.5) and a **2R** take-profit, emitted with the signal.
- Costs modeled: 2bps spread, 3-pip entry slippage, −$3.50/lot/day gold-long swap.

**Why it complements MWM precisely:**
- It supplies exactly what MWM lacks — a **validated directional + timing signal with explicit risk levels**.
- It is **gold-specific** and leak-aware (`shift(1)` on the daily ADX so no intraday lookahead; causal rolling indicators).
- Its weakness (below) is exactly the kind of thing a regime detector can address — making the pairing theoretically motivated, not a fallback.

---

## 2. Verifying the Jane Street edge (don't trust, measure)

The README claimed strong numbers but documented none. After being burned twice this session by unverified backtests (the MWM 2.458 Sharpe; the H24 AUC), we re-ran the strategy's own walk-forward harness on **2.5 years of real XAUUSD H1 data** (2023-11 → 2026-06, 15,000 bars).

**Single held-out walk-forward** (grid search on in-sample only, params frozen for OOS):
- In-sample selected `ma=200, exp=1.1` (a methodologically clean choice).
- **OOS @2bps: Sharpe 2.17, PF 2.03, +8.26%, DD −2.47%, 34 trades** (1.91 Sharpe at 4× spread).
- OOS > in-sample, cost-robust — the *opposite* of an overfit signature. And it earned that through 2025-09→2026-06, **the exact 2026 regime MWM's encoder cannot predict.**

**5-fold anchored expanding walk-forward** (the robustness check):
- OOS Sharpe by fold: **[−0.26, 1.55, 4.24, 2.72, 0.82]**, **mean 1.81**, but only **2/5 clear the gate**.
- Verdict: **REGIME-DEPENDENT** — a trend strategy. It prints when gold trends (folds 3–5), struggles/loses when it chops (fold 1).
- `ma=200` is a genuine **interior optimum** (Sharpe peaks there: 100→1.34, 200→1.82, 300→1.46), so the parameter isn't edge-overfit.

**Verdict on JS:** a **real, leak-free, cost-robust directional edge — but regime-dependent**, not all-weather. The single-fold 2.17 was a favorable fold; the honest expectation is mean ~1.8 with a clear failure mode in choppy regimes. **That failure mode is the opening for MWM.**

---

## 3. The core experiment — does the MWM S_t gate add value?

This is the hypothesis the whole architecture rests on, so it was *measured*, not assumed (`scripts/js_mwm_attribution.py`):

**Method:** run JS on the real H1 data → 98 trades with entry timestamps and net PnL. Compute MWM's **S_t-rank** (causal rolling percentile of surprise — the live four-zone signal) at each entry bar using the production encoder. Join, then compare JS-alone vs JS gated at various S_t thresholds. (A small additive `trade_records` field was added to JS's `backtest.py` so PnL stays exactly the validated numbers.)

**Result — the relationship is NON-MONOTONIC** (best_model.pt encoder):

| S_t zone at entry | Trades | Win rate | Avg PnL | Total |
|---|---|---|---|---|
| Q1 (calm, <0.2) | 16 | 50.0% | +21.2 | +339 |
| Q2 (0.2–0.6) | 28 | 46.4% | +6.1 | +170 |
| **Q4 (0.6–0.9, elevated)** | 33 | **63.6%** | **+27.8** | **+918** |
| **Q5 (≥0.9, extreme)** | 21 | **28.6%** | **−6.6** | **−138** |

**Interpretation:** breakouts *are* surprising moves, so **moderate surprise (Q4) = genuine momentum** (JS's best zone), while **extreme surprise (Q5) = chaos where breakouts fail** (the only losing zone). Surprise helps — up to a point.

**Gate comparison:**

| Variant | Trades | Total PnL | Win | Sharpe(per-trade) | PF |
|---|---|---|---|---|---|
| JS-ALL (no gate) | 98 | 1288.8 | 49.0% | 2.10 | 1.64 |
| **skip Q5 only (≤0.9)** | 77 | **1426.7** | **54.5%** | **2.51** | **1.94** |
| skip Q4+Q5 (≤0.6) | 44 | 508.9 | 47.7% | 1.13 | 1.51 |
| Q1 only (≤0.2) | 16 | 339.0 | 50.0% | 1.00 | 1.91 |

**Decision:** MWM adds value as a **Q5-only circuit breaker** — skipping it lifts every metric at once (more PnL on fewer trades). The **full four-zone filter would HURT**, because it penalizes Q4 (JS's best zone). This directly corrected the original write-up's "skip Q2 / half Q4 / flat Q5" proposal.

---

## 4. Which encoder for the gate? (best_model.pt vs gold_mt5_h1)

Earlier we fine-tuned an XAUUSD-specific encoder (`gold_mt5_h1`) to "fix the spikes" — but the spikes were a normalization bug, so it was never needed for that. The intuition remained that a symbol-matched encoder might gate cleaner. We **measured it** by re-running the attribution on both:

| | **best_model.pt** (GC=F) | **gold_mt5_h1** (XAUUSD-FT) |
|---|---|---|
| Q5 zone (to veto) | 21 trades, 28.6% win, **−138** | 14 trades, 35.7% win, **+34** |
| skip-Q5 total PnL | **1426.7** (↑ vs 1289) | 1255.0 (↓ vs 1289) |
| skip-Q5 Sharpe / PF | **2.51** / 1.94 | 2.18 / 1.75 |

**The GC=F `best_model.pt` gives the sharper gate.** With it, Q5 clearly loses, so vetoing is strictly better. With the fine-tuned encoder, Q5 is ~breakeven, so the gate is nearly worthless (it even lowers total PnL).

**Why:** the fine-tune was trained to *minimize* surprise on recent XAUUSD, so it finds fewer states anomalous → calmer, **less-discriminating** S_t. Being *too* fitted to the instrument blunts the regime signal. The "less-adapted" encoder flags genuinely anomalous (failing) regimes more sharply.

**Decision: use `best_model.pt` for the gate. No swap. `gold_mt5_h1` is needed for nothing** (not the spikes, not the gate). _(Caveat: small N — Q5 is 21 vs 14 trades.)_

---

## 5. The validated architecture

| Component | Source | Why |
|---|---|---|
| Entry + timing | JS Asian-range breakout (ADX≥25, MA200, ATR-exp≥1.1) | the only validated directional edge in the project |
| Stop / target | JS ATR stop ×1.5 + 2R target | part of the validated edge |
| Position sizing | JS risk 1%/ATR | part of the validated edge |
| **Regime gate** | **MWM: veto entry if S_t-rank ≥ 0.90 (Q5)**, using `best_model.pt` | measured: win 49%→54.5%, Sharpe 2.10→2.51, PnL 1289→1427 |
| Execution | JS `MT5Broker` (market + SL/TP, demo-safe) | JS's validated live path |

**One sentence:** a validated trend-breakout strategy, with MWM vetoing the small minority of breakouts that occur in market states the world model finds *structurally anomalous* (top-decile surprise), where they fail.

---

## 6. What is explicitly EXCLUDED, and why

Each of these was considered and rejected on evidence — including any of them risks repeating the assume-don't-measure mistakes this investigation exists to correct:

- ❌ **Full four-zone filter** (skip Q2 / half Q4) — Q4 is JS's *best* zone; penalizing it cut Sharpe to 1.13.
- ❌ **MWM directional head** — AUC 0.543, coinflip; direction is not the encoder's job.
- ❌ **MWM vol-head sizing** — never A/B'd against JS's ATR sizing; don't replace a validated component with an untested one.
- ❌ **Mid-trade Q5 force-flat** — attribution tested *entry* gating only, not exits. Optional later, after v1.
- ❌ **`gold_mt5_h1` encoder** — measured worse for the gate.

---

## 7. Build & paper-week plan

- **Build v1** (`execution/combined_live.py`): mirror JS's `poll()` loop, import JS strategy + broker via `sys.path` (proven in the attribution script), and insert one gate call — `mwm_s_rank(df) ≥ 0.90 → veto` — between `generate_signal` and `submit_order`. Run in the MWM env (has torch + JS deps). Dry-run first.
- **Paper week = A/B, not faith:** run `JS-ALONE` and `JS+Q5GATE` in parallel on the demo (distinct magic numbers), same bars. Measure whether the gate skips the Q5 losers live. With ~1–2 trades/week this is a *mechanism* smoke-test; the backtest carries the evidentiary weight.

---

## 8. Honest caveats (the things that could still be wrong)

1. **Small samples.** The Q5-loses result rests on 21 trades (28.6% win is ~1.8 SE below baseline — suggestive, not conclusive). Q4-is-best (33 trades) is firmer. Keep collecting.
2. **JS is regime-dependent.** The gate removes bad-regime *breakouts*; it does **not** make choppy regimes profitable. The one losing fold (F1) stayed negative even gated.
3. **Encoder train/serve.** S_t uses a GC=F-trained encoder on XAUUSD; it's the better gate empirically, but it's a mismatch worth keeping in mind.
4. **Backtest ≠ live.** Per-trade Sharpe is a relative quality metric, not annualized live Sharpe. The paper week tests the mechanism, not the magnitude.

**Bottom line:** for the first time this project, there is a path to profitable trades that is *both* validated and pointing up — a real directional edge (JS) plus a *measured* risk gate (MWM Q5). Keep v1 minimal, measure it live, and resist bolting on the untested.

---

### Related documents
- `VALIDATION_NARRATIVE.md` — why the prior MWM paper-trading results were artifacts.
- `DIRECTIONAL_HEAD_PLAN.md` — the directional-head experiments (Tiers 0/2) and why direction was abandoned.
- `COMBINED_SYSTEM_SCOPE.md` — the live build design (architecture, gate spec, A/B plan).
- `scripts/js_mwm_attribution.py` — the attribution backtest (`--checkpoint` to switch encoders).
- Reproduce JS edge: `PYTHONPATH=src .venv/Scripts/python.exe src/run_walkforward_multifold.py --config config/config.yaml --csv data/real_xauusd_h1.csv`.
