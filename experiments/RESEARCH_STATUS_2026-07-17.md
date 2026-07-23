# MWM Research Status — 2026-07-17 (updated 2026-07-19)

An honest accounting of where the MarketWorldModel paper stands after this
work session. Written so you can decide the next direction with the full
picture. Numbers here are all from verified runs on the retrained
long-history checkpoints unless marked otherwise.

**2026-07-19 update:** the O2-rescue arc (§8) was run to completion. Short
version: O2 could not be rescued, the conditional "high-vol regime" thesis was
refuted, and the one aggregate positive (USD/JPY forward vol) dissolved under
the regime control. The evidence now points to a **negative-results /
methodology paper**. See §8 for the full arc and §5/§7 for the revised
conclusion.

---

## 1. What triggered this session

A Colab long-history retrain appeared "lost" (empty checkpoint). Investigating
that turned into a chain of findings that reach the paper's core claims.

---

## 2. Bugs found and fixed (engineering)

**Bug A — Colab checkpoints written to ephemeral disk.**
`MWM_colab_retrain.ipynb` cell 6 guarded the Google Drive symlink with
`if not (islink or exists)`. The uploaded repo zip already contained
`experiments/checkpoints_long/`, so the guard skipped linking and training
wrote to the container's throwaway disk — while printing
`cache + checkpoints -> Drive` unconditionally. On disconnect, the run was gone.
*Fixed:* the cell now force-relinks (moving any real dir into Drive first) and
asserts `islink` on all three paths before proceeding.

**Bug B — `best_model.pt` held the FINAL epoch, not the best.**
`training/train_o1.py` saved on `ckpt_score < best_val_loss OR is_last_epoch`,
so the last epoch always overwrote the best checkpoint. Every checkpoint the
trainer ever produced was the final-epoch model, typically worse:

| checkpoint | best val | what was actually saved |
|---|---|---|
| checkpoints/eurusd | 0.0151 | 0.0254 |
| checkpoints/gold | 0.0175 | 0.0188 |
| checkpoints/usdjpy | 0.0232 | 0.0272 |
| checkpoints_long/gold | 0.0145 | 0.0156 |
| checkpoints_long/eurusd | 0.0157 | 0.0171 |
| checkpoints_long/usdjpy | 0.0157 | 0.0178 |
| gold_mt5_h1 | 0.0077 | 0.0079 |

*Fixed:* best goes to `best_model.pt` (only on improvement), final to
`last_model.pt`. **Consequence: every pre-fix result in the paper was computed
on final-epoch weights — a different, worse model than "best".**
Unaffected: m15 / m15_warmstart (the deployed live checkpoint) — saved by
runpod_warmstart.py, which only saves on improvement.

**Bug C — stale zip / BadZipFile.**
The 267 MB repo zip (234 MB of it `.pt` files) truncated on upload, and cell 3
trusted the first glob hit. *Fixed:* a 3.8 MB slim zip (code only), a
staleness-guard cell that patches an old `train_o1.py` in place, and a cell 3
that validates every candidate zip.

---

## 3. Retrained checkpoints (the good news)

All three instruments retrained from scratch, 60 epochs, pure JEPA objective
(MSE + 0.1·SIGReg, no directional aux), verified: `best_model.pt` = best epoch,
`last_model.pt` = final epoch, distinct weights.

| instrument | July 1 (shipped, stale) | new best val | honest read |
|---|---|---|---|
| gold | 0.0156 | 0.0110 @ ep51 | **~0.0117 plateau** |
| eurusd | 0.0171 | 0.0122 @ ep48 | **~0.0126 plateau** |
| usdjpy | 0.0178 | 0.0074 @ ep50 | **0.0075** (still descending at 60) |

**Important caveat on reading these:** val plateaus around epoch 40 then
wanders (gold plateau sd ≈ 0.0004). The argmin epoch is partly selection on
noise — quote the plateau mean, not the lucky minimum. Improvement over the
July 1 baselines is real (well outside the noise band); usdjpy improved >2×.

**Do not shorten n_epochs to "stop at the best epoch":** `get_lr` uses
`progress = (epoch-warmup)/(n_epochs-warmup)`, so n_epochs shapes the whole
cosine LR curve. A 40-epoch run is a different optimisation path, not a
truncation. Reproduce a result by rerunning its exact schedule.

---

## 4. Objective-by-objective status

### O1 — Anti-collapse / embedding spread — **SURVIVES**
Trained embedding variance ≫ random. SIGReg mechanically prevents collapse.
This is real and defensible. (Exact long-history spread ratios were computed
by `regen_probing.py` into `embedding_stats_long_*.json` but that file was not
pulled down locally — re-fetch from Drive `MWM_run/results` to quote them.)

### O1b — SIGReg train/val divergence as a regime-change diagnostic — **REFUTED (batching artifact) — 2026-07-22, see §9**
The abstract's last sentence: when a training window spans a structural regime
change, val SIGReg diverges from train SIGReg. Was the paper's "strongest
surviving claim." **It does not survive.** The apparent 4× gap (eurusd train_sig
0.029 vs val_sig 0.123) is an artifact of the training loop measuring train
SIGReg on **shuffled** batches (`train_loader` shuffle=True → batches mix all
years → ~Gaussian) and val SIGReg on **contiguous** batches (`val_loader`
shuffle=False → temporally-adjacent 48-bar windows lie on a slow-drifting
low-dim manifold → non-Gaussian). On the SAME held-out embeddings, contiguous
batching scores 2–5× higher than random subsampling of those identical vectors
(eurusd bs128: 0.110 contiguous vs 0.032 random; usdjpy 0.061 vs 0.030). Measured
cleanly (random subsample, fixed N), a fully-collapsed encoder assigns ~equal
SIGReg to its train span, its dated-val span, and genuinely out-of-sample data,
and val SIGReg does NOT track an encoder-free distributional-shift metric
(ρ≈0, both instruments). Full arc + reproduction in §9.

### O2 — Linear probes recover latent structure — **DOES NOT SURVIVE AS FRAMED**
The paper argues "linear probes recover session/vol/macro, so JEPA learned it."
The correct test of what *training* added is trained-vs-random encoder, not
trained-vs-chance. On trained−random, the trained encoder does **not beat a
random encoder** — it usually loses:

Long checkpoints, Δ(trained − random):

| target | gold | eurusd | usdjpy |
|---|---|---|---|
| session | +0.00 | −0.01 | +0.00 |
| high_vol | −0.04 | −0.03 | −0.17 |
| ret_direction | −0.04 | +0.01 | +0.00 |
| realized_vol | −0.34 | −0.38 | −0.30 |
| dxy_direction | +0.00 | −0.02 | +0.02 |

The May short-history files show the same pattern, so this is not caused by the
retrain — O2 never demonstrated the claim. **Why:** the probe targets are
essentially input features (session flags = idx 38–39, realized vol = idx 10),
and a random projection preserves their linear decodability (Johnson–
Lindenstrauss). "Recover session 17pp above chance" is a property of the inputs
surviving *any* projection, not of JEPA. The trained encoder does *worse* on
realized_vol because prediction-driven compression discards input detail.
The paper's own code has the Δ(T−R) column; its docstring picked Δ(T−C) as "the
honest measure," which was the wrong choice.

### O3 — Surprise = Violation-of-Expectation / tracks uncertainty — **REFUTED**
Verified on long test sets (gold 2023+, fx 2024+, n = 16–21k), via
`regen_surprise.py`. **Read `effect_size_r` as |r| — the sign is inverted vs
its own docstring** (`r = 1 − 2U/(n1n2)` with `alternative="greater"`, so
negative r = the hypothesized direction). Ignore p-values at this n.

- **Surprise does NOT track market uncertainty:** vs external implied vol,
  gold/GVZ Spearman ρ = 0.030, eurusd/EVZ ρ = −0.036. Both zero, one wrong sign.
- **No violation-of-expectation on extreme returns** (the flagship analogy):
  ratio 0.93 / 0.85 / 1.11, all null; on gold & eurusd surprise is mildly
  *lower* on extreme-return bars. Proxy-independent (keys off idx 0).
- **What survives:** surprise weakly tracks *contemporaneous realized* vol
  (partly mechanical). Holding vol_idx=10 fixed across all three:
  gold |r|=0.134 ρ=0.116, eurusd |r|=0.152 ρ=0.085, usdjpy |r|=0.433 ρ=0.353.
- The `--vol-idx` control showed the per-instrument proxy (gold/eurusd→implied,
  usdjpy→realized) explained most, but not all, of usdjpy's apparent
  exceptionalism. usdjpy stays ~3× stronger even on the identical metric —
  likely real: its 2024–26 window is dominated by unscheduled shocks
  (carry unwind, BoJ), consistent with the earlier event-detection audit.
- Precision/recall (low, unchanged): gold 10/42 & 4/15, eurusd 12/52 & 8/15,
  usdjpy 9/50 & 5/15. Consistent with "tracks regime change, not calendar."

---

## 5. What the research honestly shows (revised 2026-07-19)

A label-free JEPA + SIGReg model on H1 market data:
- **produces non-collapsed, high-variance embeddings** (O1 — solid);
- whose training signal was thought to **diagnose structural regime change**
  (O1b) but does NOT: that train/val SIGReg gap is a shuffled-vs-contiguous
  batching artifact, refuted 2026-07-22 (see §9);
- but whose frozen embeddings **do not beat a random encoder** on linear
  probes of session/vol/macro (O2 — does not support the paper's claim), and
  **still do not beat random once probed on strictly-future targets** (the O2
  rescue, §8);
- and whose prediction-error "surprise" **does not exhibit violation-of-
  expectation** to discrete events and **does not track market uncertainty**;
  it weakly tracks realized-volatility regimes (O3 — refuted as framed).

After the full O2-rescue arc (§8): **no robust, mechanistically-coherent
positive representation-learning result survives.** The conditional "JEPA helps
in high-vol regimes" thesis was tested and refuted — high-vol slices are where
a random projection wins most (it exploits vol persistence). Two of three
headline objectives fail their stated claims, and the rescue attempt for the
third did not produce a defensible positive.

---

## 6. Tooling now in the repo (all reuse the training code paths)

- `training/train_o1.py` — checkpoint bug fixed (best vs last).
- `experiments/regen_surprise.py` — recompute O3 from any checkpoint, no
  retraining; `--vol-idx N` forces one vol definition across instruments,
  writes `*_vol{N}.json` (never overwrites canonical).
- `experiments/regen_probing.py` — recompute O1/O2 on the long checkpoints via
  build_long → build_from_frames; writes `probing_results_long_*.json` and
  `embedding_stats_long_*.json` (never overwrites the May files).
- `experiments/probe_forward.py` (07-19) — O2 rescue: probes strictly-future
  realized vol over [t+1..t+k] (not in the input window), trained vs random.
- `experiments/probe_forward_regime.py` (07-19) — the same forward probe sliced
  by the encoder's current vol (rv_32 tercile at t); the decisive test of the
  high-vol-regime thesis.
- `experiments/o1b_boundary_slide.py` (07-22) — O1b within-instrument boundary
  slide: retrains a fresh encoder at each train/val boundary (same init seed),
  measures the SIGReg gap vs an encoder-free shift metric. (Diagnostic; the
  frozen-slide + artifact-check below are the decisive/cheap versions.)
- `experiments/o1b_frozen_slide.py` (07-22) — trains ONE encoder to collapse on
  an early window, freezes it, slides the val window across later history;
  val SIGReg(t) vs encoder-free shift.
- `experiments/o1b_artifact_check.py` (07-22) — the O1b verdict, reuses the long
  checkpoints (no training): Test 1 = contiguous-vs-random batching on one
  held-out embedding set; Test 2 = clean sliding SIGReg + segment means + shift
  correlation. Writes `o1b_artifact_check_{inst}.json`.
- `experiments/make_o1b_figure.py` → `o1b_artifact_figure.png` (the two-panel
  figure for the negative result).
- `MWM_colab_retrain.ipynb` — Drive-symlink assert, stale-zip guard cell,
  hardened unzip, per-instrument (60-epoch) config, results persisted to Drive.
- Slim Colab zip builder in scratchpad (`make_slim_zip.py`), 3.8 MB code-only.

Verified checkpoints (best + last, distinct) for gold/eurusd/usdjpy are on
Drive at `MyDrive/MWM_run/checkpoints_long/` and locally at
`experiments/checkpoints_long/`.

---

## 7. Possible directions (revised 2026-07-19, after §8)

Direction 3 below was pursued to completion (§8) and closed: O2 could not be
rescued. That removes the "conditional / when-does-JEPA-help" option and leaves:

1. **Negative-results / methodology paper (now the leading option).** Report the
   full investigation honestly. The methodological contribution — random-encoder
   baseline + forward-vs-backward vol design + regime slicing + vol-proxy control
   + shuffled-vs-contiguous batching control (§9) — is a reusable toolkit that
   caught five separate over-claims. USD/JPY's
   aggregate forward-vol edge is reported as a documented open anomaly, not a
   law. Fits reproducibility / critique venues.

2. ~~**SIGReg regime-diagnostic paper (O1b).**~~ **DONE and CLOSED (§9, 07-22) —
   O1b is a batching artifact; the within-instrument boundary slide found no
   regime tracking on eurusd or usdjpy.** This removes the last candidate
   positive headline and makes direction 1 the only defensible framing.

3. ~~Rescue O2 with honest targets.~~ **DONE and CLOSED (§8) — O2 did not
   survive; the high-vol-regime thesis was refuted.**

4. **Keep O1 (+ O1b if re-verified), drop O2/O3 to a candid limitations
   section.** A smaller representation-learning paper stating plainly what the
   probes and surprise analysis do and don't show.

Current recommendation: **(1), the negative-results / methodology paper.** Do
NOT chase the low-vol MLP wins in §8 — they are the same fragile,
one-probe-one-slice pattern that has collapsed under every added control.

**2026-07-22 — DECISION TAKEN + DRAFT WRITTEN.** User chose (1). A full rewrite
now exists at repo root `mwm_audit_paper.tex` (the old positive-framing draft
`mwm_revised_full_corrected.tex` is preserved untouched). New framing: an
**evaluation protocol** (C1 random-encoder baseline / C2 forward-vs-backward
targets / C3 regime stratification / C4 matched-sampling geometry metrics) is
the contribution; the Findings section applies it and every MWM positive claim
dissolves except O1 anti-collapse (now 35–43× spread on the long checkpoints,
Table `tab:spread`). Uses figures `figure1.png` (arch) + `o1b_artifact_figure.png`.
Static-checked (envs/refs/cites/braces balanced, figs resolve via
`\graphicspath{{experiments/}{./}}`); NOT yet compiled (no local LaTeX) — needs
a pdflatex pass + author review. New refs added: JL 1984, Saxe 2011, Rahimi–Recht
2007, Hewitt–Liang 2019 (probing control tasks), Kapoor–Narayanan 2023 (leakage).

---

## 8. The O2-rescue arc (2026-07-19) — pursued to completion

Goal: give the "JEPA learns structure" claim its fairest possible test before
deciding a paper framing. Three steps, each a cleaner control than the last.

**Step 1 — forward-vol probe (`probe_forward.py`).** Probe strictly-future
realized vol over [t+1..t+k] (k=1/4/12/24), which is NOT in the input window, so
a random projection cannot trivially preserve it. Δ(trained−random) pearson_r,
8 cells/instrument:
- gold: 0/8 trained-wins, best Δ = −0.014 → dead.
- eurusd: 1/8 (isolated), best Δ = +0.032 → dead.
- usdjpy: 6/8, **6/6 positive at k≥4**, Δ up to +0.094 (n=110k) → looked alive.
This convergently matched O3 (usdjpy was the O3 outlier too), suggesting a
"JEPA helps in high-vol-dynamics regimes" thesis.

**Step 2 — local within-asset check (surprise-vol coupling).** Spearman(surprise,
realized-vol) in high-vol vs low-vol terciles: eurusd 0.099→0.247, usdjpy
0.090→0.251, but **gold 0.001→−0.048 (flat even in gold's volatile bars)**. First
crack: gold does not fit a pure regime story.

**Step 3 — regime-sliced forward probe (`probe_forward_regime.py`), decisive.**
Re-ran the forward probe separately in each asset's high-vol vs low-vol samples
(sliced by rv_32 tercile at t, thresholds fit on train). Thesis predicted
trained>random in HIGH-vol slices. Result was the **opposite**:
- HIGH-vol slices favor the RANDOM encoder (gold random wins all cells, Δ to
  −0.32; usdjpy random wins 3/4). Mechanical: in volatile periods forward vol ≈
  backward-vol persistence, which a random projection preserving rv_32 nails,
  while the JEPA-compressed encoder does worse.
- The only trained>random wins were in LOW-vol slices, MLP-only, and
  linear-inconsistent (gold low-vol mlp +0.055/+0.132 but low-vol linear −0.12;
  usdjpy similar; eurusd nothing). No regime lift for any asset.

**Conclusions:**
- The conditional "high-vol regime" thesis is **refuted**.
- The usdjpy aggregate win from Step 1 **dissolves**: it was not high-vol regime
  tracking but a low-vol, MLP-only, linear-inconsistent effect — real in
  aggregate, no coherent mechanism, gone under the regime control.
- No robust positive representation-learning result survives. The durable
  outputs are the anti-collapse mechanics (O1) and the negative-result
  methodology itself.

Artifacts: `probe_forward_{inst}.json`, `probe_forward_regime_{inst}.json`
(local + Drive `MWM_run/results`).

---

## 9. The O1b arc (2026-07-22) — the "regime-change diagnostic" is a batching artifact

Goal: run the "real within-instrument boundary-slide" §7 asked for, to decide
whether O1b (val SIGReg diverges from train SIGReg when the training window spans
a regime change) could carry a paper. Run entirely on this CPU box (Colab sub
expired); eurusd 2015→2026 and usdjpy 2020→2026 have offline combined caches.

**Step 1 — retrain-per-boundary slide (`o1b_boundary_slide.py`).** Slide the
train/val boundary; at each, train a fresh encoder (same init seed) and measure
the final train/val SIGReg gap vs an encoder-free shift metric (return vol-ratio,
two-sample KS between the train and val windows). On CPU a full collapse costs
~50 min/boundary, so a 3-boundary contrast (calm-2019 / COVID-2020 /
rate-shock-2022, 26 epochs) was run first. Only rate-shock showed any gap
(ratio 1.16); COVID stayed flat despite the biggest vol jump. Cause: at 26
epochs train SIGReg had not collapsed (0.07–0.11 vs 0.03 in the full run), and
the gap cannot open until train Gaussianizes.

**Step 2 — frozen-slide (`o1b_frozen_slide.py`), cheap high-res version.** Pay
the collapse cost ONCE: train one encoder on eurusd 2015→mid-2019, freeze it,
slide the val window across 2019→2026 (106 windows). Validate val SIGReg against
the encoder-free shift metric, which is non-monotonic so it separates "different
regime" from "later in time." Result: val SIGReg sits in a tight 0.060–0.081
band (train baseline 0.060) and does NOT correlate with vol-ratio (ρ=−0.01) or
KS (ρ=−0.05); if anything it drifts slightly DOWN with calendar time. The
biggest realized-vol window (2022-H2 gilt crisis) shows the LOWEST val SIGReg —
because the 52 features are rolling-z-scored (window 500), so a *sustained* vol
regime is normalized away; only transitions could show. But transition metrics
(within-window max|ret|, vol-of-vol, first-half-vs-second-half KS) are null too.

**Step 3 — the artifact, decisive (`o1b_artifact_check.py`, no training).** The
long checkpoints' dramatic gap (eurusd train_sig 0.029 vs val_sig 0.123) is a
measurement artifact of `training/train_o1.py`: `train_loader` uses
**shuffle=True** (batches mix all years → ~Gaussian → low SIGReg) and
`val_loader` uses **shuffle=False** (batches are 128 temporally-adjacent 48-bar
windows, which overlap and drift slowly → lie on a low-dim manifold →
non-Gaussian → high SIGReg). Proof, on ONE fixed held-out embedding set from the
fully-trained encoder:

| | contiguous (shuffle=False) | random (shuffle=True) | ratio |
|---|---|---|---|
| eurusd bs64  | 0.164 | 0.035 | 4.7× |
| eurusd bs128 | 0.110 | 0.032 | 3.4× |
| usdjpy bs64  | 0.101 | 0.033 | 3.0× |
| usdjpy bs128 | 0.061 | 0.030 | 2.1× |
| gold   bs64  | 0.181 | 0.043 | 4.2× |
| gold   bs128 | 0.107 | 0.040 | 2.7× |

(bs128 contiguous 0.110 ≈ the training-logged val_sigreg 0.123 — same effect.)
Measured cleanly (random subsample, fixed N) the fully-collapsed encoder gives
~equal mean SIGReg on train / dated-val / genuine-OOS spans, on all three:
- eurusd: 0.0401 / 0.0394 / 0.0426  (baseline 0.028)
- usdjpy: 0.0324 / 0.0313 / 0.0333  (baseline 0.027)
- gold:   0.0494 / 0.0525 / 0.0492  (baseline 0.035)
and no correlation with the encoder-free shift (eurusd ρ=+0.11 p=0.28; usdjpy
ρ=−0.25 p=0.07 wrong sign; gold ρ=+0.02 p=0.78). SIGReg's finite-sample floor is
negligible here (0.0013 at N=400), so the flatness is real, not small-sample
noise. All three instruments tell the same story.

**Conclusions:**
- O1b is **refuted**: the train/val SIGReg "divergence" is a shuffled-vs-
  contiguous batching artifact, not a regime signal. A frozen encoder Gaussianizes
  in-regime, dated-val, and out-of-sample windows equally, and clean SIGReg does
  not track any encoder-free measure of distributional shift or transition.
- This removes the paper's last candidate positive headline. All three of
  O1b/O2/O3 now fail their stated claims; only O1 (anti-collapse) stands. It
  firmly confirms direction 1 (negative-results / methodology paper) and adds a
  fifth, very teachable over-claim: a "generalization gap" that was pure DataLoader
  batching. (Note for that paper: report train/val SIGReg with matched sampling.)

Artifacts: `o1b_artifact_check_{eurusd,usdjpy}.json`, `o1b_frozen_slide_eurusd.json`,
`o1b_artifact_figure.png`; scripts `o1b_boundary_slide.py`, `o1b_frozen_slide.py`,
`o1b_artifact_check.py`, `make_o1b_figure.py`.

**2026-07-22 — matched-range re-run (for the paper).** The checkpoints are the
compute-constrained demonstration models (`RETRAIN_LONG_RESULTS.md`): trained on
gold 2015–21 (40ep), EUR/USD & USD/JPY 2020–22 (20ep) — NOT 2004/2008. The first
artifact pass evaluated on wider ranges than training (e.g. eurusd on 2015-cache
while trained on 2020), so the "train" segment leaked pre-training data. Re-ran
`o1b_artifact_check.py all` with `CACHED_RANGE` matched to each checkpoint's
training range (gold 2015-, FX 2020-). Results essentially unchanged and now
range-consistent: batching artifact 2.1–4.7× (bs128 2.1–3.5×); segment means
flat (gold 0.049/0.052/0.049, eurusd 0.041/0.040/0.041, usdjpy 0.032/0.031/0.033
train/val/oos); O1 spread now 37–46× (gold 40.6, eurusd 46.0, usdjpy 36.7). Shift
correlations small and sign-inconsistent (vol_ratio ρ = −0.15/+0.26/−0.25
gold/eurusd/usdjpy, none sig; KS all null) — so the paper leads the null on the
flat segment means + the artifact, not on "ρ≈0". Paper (`mwm_audit_paper.tex`)
Method + all three tables (`tab:spread`/`tab:artifact`/`tab:segments`) updated to
these; figure Panel B switched to gold (cleanly null, longest span); Discussion
gained an anti-"under-training" paragraph.

Artifacts: `probe_forward_{inst}.json`, `probe_forward_regime_{inst}.json`
(local + Drive `MWM_run/results`).
