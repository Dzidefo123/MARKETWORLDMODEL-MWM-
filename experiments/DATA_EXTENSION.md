# Data extension — lifting the 730-day Yahoo cap

The paper's binding limitation ("579 training bars for USD/JPY; 48/136-sample
probe val sets ... read as indicative"). Fixed by `data/long_history.py`.

## Architecture
- **Price (H1 OHLCV):** Dukascopy, free, tick-derived, back to ~2003 for
  EUR/USD, USD/JPY, XAU/USD. Pulled in 90-day chunks, each cached to
  `data/cache/dukascopy/` (resumable).
- **Macro (daily):** Yahoo `interval='1d', period='max'` (no 730-day cap on
  daily), forward-filled onto the H1 price grid via backward `merge_asof`
  (lookahead-safe; pipeline re-z-scores with shift(1) anyway).
- **Output:** `data/cache/long/{inst}_h1_long.parquet` — OHLCV + one column per
  macro key, identical shape to `DataFetcher.fetch_all()`.

## Honest start dates (gated by macro availability, verified 2026-07 via Yahoo)
| Asset  | Start   | Gating covariate            | Regimes now in-sample |
|--------|---------|-----------------------------|-----------------------|
| Gold   | 2008-07 | GVZ starts 2008-06-03       | GFC tail, 2013 taper, 2020, 2022 |
| EURUSD | 2004-01 | GBP/USD 2003-12, TLT 2002   | GFC, 2011 EU crisis, 2015 CHF, 2020, 2022 |
| USDJPY | 2006-06 | AUD/USD 2006-05 (carry proxy)| GFC carry unwind, 2013 Abenomics, 2020, 2022 |

~18–22 years each vs the current ~2 — the severe VoE regimes the paper wants.

## Proven
EUR/USD 2015-01→2016-06 slice: **8,817 clean H1 bars**, 5 macro cols, 0 NaNs,
correct values across the Jan-2015 CHF shock. (Full pull running via
`python -m data.long_history all`; progress in `experiments/long_history_pull.log`.)

## Integration (no fetcher plumbing changes needed)
`DataPipeline` already exposes `build_from_frames(price_df, macro_df)`, which
runs feature-engineering → normalization → splits on pre-fetched frames. So:

```python
from data.long_history import build_long
from data.pipeline import DataPipeline

d = build_long("eurusd")                       # {"price","macro"}, long history
pipe = DataPipeline(instrument="eurusd", lookback=48, norm_window=500)
result = pipe.build_from_frames(d["price"], d["macro"],
                                split_dates=(...), history_len=3, stride=1)
```

Macro columns already match each instrument's `INSTRUMENT_CONFIGS` keys
(gold: dxy/gvz/tlt/silver; eurusd: dxy/evz/tlt/shy/gbpusd; usdjpy:
dxy/tlt/spy/audusd), so `features.py` is unchanged.

## End-to-end retrain — WIRED (`experiments/retrain_long.py`)
Single script does per instrument: `build_long()` → `DataPipeline.build_from_frames`
with regime-spanning splits → `train_o1.train()` (pure JEPA: MSE+0.1·SIGReg, from
scratch, no dir-aux) → O3 surprise on the long test set → precision/recall.
Checkpoints: `experiments/checkpoints_long/{inst}/best_model.pt`.
Surprise JSONs: `experiments/surprise_timeseries_long_{inst}.json`.

Regime-spanning splits (test held out, multi-year, calendar-overlapping):
  gold   train→2021-12-31 | val 2022 (rate shock)  | test 2023→now
  eurusd train→2022-12-31 | val 2023               | test 2024→now
  usdjpy train→2022-12-31 | val 2023               | test 2024→now
Val windows are now thousands of bars (vs 48/136) → reliable probes.

Run:
  python experiments/retrain_long.py all --start 2015-01-01 --epochs 40   # in-session demo
  python experiments/retrain_long.py all --start 2004-01-01 --epochs 200  # full-fidelity (paper)

Validated end-to-end (2-epoch smoke test on cached 2015-16 slice: data→train→
checkpoint→surprise→JSON all run). A 40-epoch 2015-start run over all three is
currently executing in the background (`experiments/retrain_long_all.log`).

## Remaining for paper-final numbers (user-run; CPU-hours)
1. Let the retrain finish; sanity-check O3 significance + probe deltas.
2. Bump to `--epochs 200 --start 2004-01-01` for full fidelity.
3. Extend `macro_calendar.json` back to the long test start (currently 2024-08+)
   so precision/recall covers the whole multi-year test window.
4. Watch the SIGReg divergence diagnostic across the real regime breaks now
   in-sample — the strongest test yet of the paper's own contribution.

## Caveats
- Dukascopy bars are bid-side; spreads/rollovers differ slightly from Yahoo —
  fine for a self-supervised model on z-scored features, but note it.
- Weekend/holiday gaps and DST: Dukascopy timestamps are UTC; existing session
  features (hour_sin/cos, London/NY flags) already handle this.
- Daily macro ffill means no intraday macro path — matches the current design
  (macro covariates are daily instruments), but worth a sentence in Methods.
