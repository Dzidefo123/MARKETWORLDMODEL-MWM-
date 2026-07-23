# Combined System Scope — Jane Street breakout + MWM Q5 regime gate

_Created 2026-06-22._

> ## ⚠️ SUPERSEDED (2026-06-22) — the gate was a lookahead artifact
> The attribution that motivated this gate had a one-bar lookahead (see `COMBINED_SYSTEM_RATIONALE.md`
> correction). Causally, **MWM's S_t does not improve Jane Street** — skip-Q5 reduces PnL and Sharpe.
> **Do not build the gated system.** Plan instead: **paper-trade JS-alone** (real walk-forward edge), and
> run `execution/combined_live.py` as a **dry-run logger** (no gating) that records each JS signal's
> causal S_t-zone to prospectively test whether any regime signal emerges. The design below is retained
> only as a reference for that logger and for the (now-rejected) gate hypothesis.

## 1. The validated design (do exactly this — no more)

| Component | Source | Status |
|---|---|---|
| Entry signal + timing | JS Asian-range breakout (ADX≥25, MA200, ATR-exp≥1.1) | walk-forward validated (OOS Sharpe ~1.8, regime-dependent) |
| **Regime gate** | **MWM: veto entry if S_t-rank ≥ 0.90 (Q5)** | **measured: lifts win 49%→54.5%, Sharpe 2.10→2.51** |
| Stop / target | JS ATR stop + 2R target (from the Signal) | part of JS validated edge |
| Position sizing | JS risk_per_trade / ATR (`risk.calc_position_size`) | part of JS validated edge |
| Execution | JS `MT5Broker.submit_order` (market + SL/TP, demo-safe) | JS live path |

**Explicitly NOT in v1** (untested — do not bolt on):
- ❌ The full four-zone filter (skip Q2 / half Q4). The attribution showed **Q4 is JS's best zone** — penalizing it *hurts*. Only Q5 is vetoed.
- ❌ MWM vol-head sizing (never A/B'd against JS's ATR sizing).
- ❌ Mid-trade Q5 force-flat (attribution only tested *entry* gating, not exits). Optional later.
Adding any of these without measuring repeats the mistake this whole investigation corrected.

## 2. Integration architecture

Run **one combined process in the MWM env** (it has torch for MWM + pandas/numpy for JS; JS modules import fine via `sys.path`, proven in `js_mwm_attribution.py`). Separation of concerns:
- **JS owns** signal + execution (its validated live loop and broker, untouched).
- **MWM owns** only the gate: a read-only `s_rank` computation.

```
new file: execution/combined_live.py   (runs in MWM env)
  imports JS:  SessionBreakoutStrategy, load_config, MT5DataFeed, MT5Broker, Order, Side
  imports MWM: MarketEncoder, CausalPredictor, GoldFeatureEngineer, compute_surprise_features
  loop (mirrors JS run_mt5_live.poll, ~40 lines):
    df = feed.fetch_candles(symbol, "H1", history_count=1000)   # JS fetch
    if new H1 bar:
        signal = strategy.generate_signal(df, len(df)-1)         # JS entry
        if signal.side != FLAT:
            s_rank = mwm_s_rank(df)                               # MWM gate (see §3)
            if s_rank >= 0.90:
                log "VETO: Q5 regime (s_rank=%.2f)"; skip
            else:
                size = risk.calc_position_size(...); broker.submit_order(Order(... sl, tp))
    write combined status json (incl. s_rank, veto count)
```

Injection point in JS's own loop is line ~167–173 of `run_mt5_live.py` (between `generate_signal` and `submit_order`) — the combined runner replicates that loop and inserts the gate there.

## 3. The gate: `mwm_s_rank(df) -> float`

Stateless per bar (matches the attribution exactly):
1. `price_df = df[[open,high,low,close,volume]]` (last ~1000 H1 bars JS already fetched — ≥ the 500 z-score window after the 250 feature-warm trim, so normalization is full-window per the live-spike fix).
2. Fetch macro (dxy/gvz/tlt/silver) via MWM `fetch_h1_macro` (DXY fallback + forward-fill already hardened).
3. `DataPipeline.build_from_frames` → norm_features, macro_vecs.
4. Encode (prod `best_model.pt`) → Z; `compute_surprise_features(Z, acts, predictor)` → s_rank series.
5. Return s_rank of the **latest** bar.

Threshold: **0.90** (Q5). Single knob. ~3–5 s/bar on CPU — negligible at H1 cadence.

## 4. Paper-week validation plan (A/B, not faith)

The gate's benefit was measured on **98 backtest trades (Q5=21)** — suggestive, not conclusive. The paper week must *measure* it live, not assume it:
- Run **two instances in parallel on the demo account, different magic numbers**:
  - `JS-ALONE` (existing `run_mt5_live.py`), and
  - `JS+Q5GATE` (combined runner).
- Both see identical bars/signals; the only difference is the gate. Log every signal, the s_rank, and whether each instance entered.
- Success = the gate vetoes trades that JS-alone loses on (Q5), and the gated equity curve ≥ JS-alone. Even a few shared signals where the gate correctly skips a Q5 loser is direct live evidence.
- Low trade frequency (≈1–2/week) means one week won't be statistically decisive — treat it as a live smoke test of the *mechanism*, with the backtest as the weight of evidence.

## 5. Risks & open questions

- **Small-sample edge.** Q5-loses rests on 21 trades. The live A/B is a sanity check, not proof; keep collecting.
- **Encoder choice — SETTLED (2026-06-22): use `best_model.pt` (GC=F).** Re-ran the attribution on the XAUUSD-fine-tuned `gold_mt5_h1` encoder: its gate is WEAKER — Q5 is ~breakeven (+34, 35.7% win) vs clearly-losing on best_model.pt (−138, 28.6% win), and skip-Q5 PnL drops below baseline (1255 < 1289) vs rising (1427) on best_model.pt. The fine-tuned encoder is *too* adapted to XAUUSD → calmer, less-discriminating S_t. No swap; the deployed encoder is the right gate. (Caveat: small N.)
- **Two MT5 connections / sync.** Both instances + the demo terminal; ensure distinct magics and that the gate's macro fetch doesn't stall the loop (MWM's macro hardening covers outages).
- **Bar-timing.** Reuse JS's bar-close detection (it already worked); don't import MWM's wall-clock detector.
- **Regime gate ≠ regime fix.** The gate removes bad-regime *breakouts*; it does not make the choppy regime profitable. F1 (losing fold) stayed negative even gated. The system is still fundamentally a trend strategy.

## 6. Build steps (when greenlit)

1. `execution/combined_live.py` — combined runner (§2) in MWM env, importing JS via `sys.path`.
2. `mwm_s_rank(df)` helper (§3) — factor from `js_mwm_attribution.py`.
3. Dry-run/demo both instances (JS-alone + JS+gate), confirm signals match and gate vetoes only Q5.
4. Paper-trade the week; compare equity curves + veto log.
5. Decide: if the gate demonstrably skips Q5 losers live → keep. If not → ship JS-alone (it has the edge; MWM becomes monitoring).

**Bottom line:** this is the first build all project that is *both* validated and points toward profitable trades. Keep v1 minimal (JS + Q5 veto only), measure live, resist adding untested overlays.
