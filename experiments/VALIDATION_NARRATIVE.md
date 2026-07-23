# MWM Paper-Trading Validation — Corrected Narrative

_Last updated: 2026-06-22. Supersedes the pre-fix interpretation of the May–June 2026 paper-trading session._

## Summary

The May–June 2026 paper-trading month was dominated by a **measurement artifact**, not market behavior. The S_t "surprise" spikes that caused the circuit breaker to fire — and that made the system refuse to trade for roughly two weeks — were caused by a normalization bug in the live feature pipeline, not by the encoder failing to recognize the current gold regime. After fixing it, a full-day live dry-run (2026-06-21/22) produced calm, well-behaved S_t with zero spikes. **The earlier "validation" must be rewritten: the headline circuit-breaker "save" was an artifact coincidence, and the directional edge remains unproven.**

## What we previously believed (and why it was wrong)

- **Believed:** The encoder (trained on GC=F through Oct 2025) couldn't handle the 2026 regime (ATH, January crash, Iran conflict, post-NFP June), so every morning it hit market states with no analog and produced genuine high surprise.
- **Reality:** On properly normalized data, both the old and a freshly retrained encoder produce S_t in the ~0.0005 range on the same June bars — three orders of magnitude below the live ~1.9 spikes. The encoder was never the bottleneck.

## Actual root cause (three compounding bugs in `execution/live_trader.py`)

1. **Restart normalization mismatch (primary).** The 500-bar rolling z-score that every feature depends on was being computed over far fewer bars live than in training. `warm_up` normalized over ~298 rows and `_on_bar` over ~48 rows (the feature engineer trims 250 warm-up bars, which the original fetch sizes didn't account for). After each restart, the first bar compared a `z_hat` built in one normalization regime against a `z_t` in another → an encoding gap of ~1.0–1.5 in latent space = the spike. The logs show frequent restarts, so the "persistent daily spikes" were repeated first-bar-after-restart events.
2. **Macro outages.** When Yahoo macro fetches failed (e.g. June 5 NFP, all four tickers empty), `_align_macro` zero-filled them → degenerate features → additional spikes (the largest, 6.5–17.96, stack this with cold-start effects).
3. **Shallow steady-state normalization.** A ~30× baseline elevation from the same under-fetch.

The M15 scanning path had the identical bug and fix.

## What this means for the prior results

- **The May 27 Iran circuit-breaker "save" (S_t=1.449, ~$756 avoided) does not survive scrutiny.** Under correct normalization that bar produces S_t ≈ 0.0001 — *calmer than median*. The breaker fired because of the restart/normalization artifact coinciding with the Iran drop, not because the surprise signal detected a regime break. It was luck, not skill.
- The same holds for the NFP and June-17 spikes: all collapse to ~0.0001 under correct normalization.
- **Net:** the paper-trading month validated *that the system was broken in a way that masqueraded as caution*, not that the S_t signal works. The S_t signal had never actually been validated live.

## The fixes (all verified)

- All H1 + M15 fetch sites deepened so every live bar normalizes over a full 500-bar window (`warm_up` and `_on_bar` now share one normalization space → no restart mismatch).
- `_align_macro` forward-fills the last-known macro value instead of zero-filling; DXY uses an ordered ticker fallback `["DX-Y.NYB","DX=F","UUP"]`.
- Four-zone circuit-breaker floors recalibrated to the post-fix S_t distribution (`0.00008 / 0.00028 / 0.0013`) so the adaptive percentiles drive zone assignment.

## Live re-validation (2026-06-21/22, dry-run)

A full ~22h session (23 H1 closes, `experiments/live_20260621_234720.log`):

| Metric | Result |
|---|---|
| S_t mean / median / max | 0.00038 / 0.00023 / **0.00203** |
| Readings ≥ 1.0 (old spike level) | **0** |
| Q5 circuit-breaker trips | **0** (no false trips) |
| Macro outage at 02:00 (DXY failed) | absorbed at S_t=0.0001 — **hardening proven under live failure** |
| Trades taken | **0** (dir_prob pinned 0.477–0.490, never crossed 0.53/0.47) |

**Plumbing: validated.** The system is now clean and trustworthy across a full 24h cycle, including a real macro outage.

## Honest status

- ✅ **The instrument is fixed.** S_t is a trustworthy signal; the circuit breaker behaves; the macro feed is robust.
- ❓ **The edge is unproven.** A full quiet day produced no entries because the directional head never reached conviction (AUC 0.543, output pinned ~0.48). Profitability now depends entirely on the directional signal, which has not yet been tested live and is weak in-sample.
- **Implication for FTMO / external claims:** do not present the prior month as validation of the strategy. Present it accurately: a measurement bug was found and fixed; live re-validation of the *plumbing* passed; live validation of the *edge* is the next milestone and has not yet occurred.

## Next milestone

The first time `dir_prob` crosses 0.53 (long) or 0.47 (short) and takes a position — that trade, and the ones after it, are the first real test of whether there is tradeable edge. If entries stay absent or unprofitable, the work is on the directional head (see `DIRECTIONAL_HEAD_PLAN.md`), not the plumbing.
