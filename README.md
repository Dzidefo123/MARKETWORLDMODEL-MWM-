# MarketWorldModel (MWM)

**When a Learned Encoder Does Not Beat a Random One: An Evaluation Protocol for
Self-Supervised World Models on Financial Markets**

This repository accompanies a paper currently under review (ICAIF '26 format).
It contains the model under audit, the four controls the paper proposes, and
the scripts that regenerate every table and figure.

> **Manuscript withheld.** The paper source (`mwm_audit_paper.tex` / `.pdf`) is
> not included here until it is approved for release. All code, checkpoints, and
> result files needed to reproduce the findings are present; the sections below
> map each finding to the script that produces it.

## What the paper says

We built a JEPA–SIGReg world model (MWM) over hourly bars for gold, EUR/USD, and
USD/JPY, and it produced four results that each looked like learned market
structure. We then subjected each to a control that self-supervised
representation-learning studies often skip. **Only one of the four survives.**

| Claim | Control | Verdict |
|---|---|---|
| **O1** SIGReg prevents representation collapse | C1 random-encoder baseline | **Holds** — embeddings 34–43× more spread out than random |
| **O2** Linear probes recover market structure | C1 random-encoder baseline | Fails — trained loses to a random encoder in 27 of 30 probe cells |
| **O3** Prediction "surprise" flags violations of expectation | C2 forward targets, C3 stratification | Fails — no elevation on extreme returns, ρ≈0.03 vs implied vol; the vol coupling is mostly regime drift |
| **O4** SIGReg train/val gap diagnoses regime change | C4 matched-sampling geometry | Fails — the gap is a `shuffle=True` (train) vs `shuffle=False` (val) batching artifact |

The durable output is a four-part **evaluation protocol** (C1–C4) we recommend
as a precondition for structure claims in self-supervised financial
representation learning.

## The four controls

- **C1 — Random-encoder baseline.** Report Δ(trained − random), not
  Δ(trained − chance). A random projection preserves the linear decodability of
  the inputs (Johnson–Lindenstrauss), so beating chance is a property of the
  features, not of training.
- **C2 — Forward- vs backward-looking targets.** Only targets outside the input
  window test predictive structure.
- **C3 — Regime-stratified evaluation.** Recompute every effect within
  volatility strata, so "the model tracks X" is not confused with "X and the
  model both drift with the regime."
- **C4 — Matched-sampling geometry metrics.** Any batch-distribution metric
  (SIGReg, covariance) must use identical sampling on train and validation, or
  it measures the sampler.

## Repository layout

```
PROVENANCE.md                 every paper number → its artifact → its script
models/                       encoder.py, predictor.py, sigreg.py, mwm_loss.py
training/train_o1.py          the JEPA–SIGReg training loop
data/                         feature engineering + long-history data assembly
evaluation/                   probing.py (O2), surprise.py (O3)
experiments/                  result JSONs, figure scripts, the audit scripts,
                              and checkpoints_long/ (the checkpoints under audit)
execution/, scripts/          live-trading path (not used by the paper)
```

## Reproducing the findings

Every number derives from the 60-epoch checkpoints in
`experiments/checkpoints_long/` (trained on a Colab GPU; see
[`PROVENANCE.md`](PROVENANCE.md)). The **audit itself runs on CPU** — each
control is a forward pass over frozen weights plus a small probe fit.

```bash
pip install -r requirements.txt        # Python 3.12; torch, numpy, pandas, scikit-learn, scipy, matplotlib

python experiments/embedding_spread.py all          # Table 1  (O1 anti-collapse)
python experiments/probe_absolute.py all            # Tables 2 & 3  (O2 probes)
python experiments/o3_stratified.py all --dump-arrays  # Table 4 + Figure 2  (O3)
python experiments/regen_surprise_matched.py all    # calendar recall/precision (O3)
python experiments/o1b_artifact_check.py gold       # Tables 5 & 6 + Figure 3  (O4)
python experiments/o1b_artifact_check.py eurusd
python experiments/o1b_artifact_check.py usdjpy

python experiments/make_o3_figure.py                # Figure 2
python experiments/make_o1b_figure.py               # Figure 3
```

### Verifying the numbers

`check_paper_numbers.py` re-derives every quantitative claim from the stored
result JSONs and exits non-zero on any mismatch. When the manuscript source is
present (the author's working copy) it also cross-checks each value against the
`.tex`; when it is absent (this public repo) it validates the result files for
internal consistency and skips the `.tex` cross-references.

```bash
python experiments/check_paper_numbers.py
```

## Retraining from scratch (optional, needs a GPU)

The checkpoints were produced by `experiments/MWM_colab_retrain.ipynb`
(60 epochs, one instrument per session). `experiments/retrain_long.py` is the
same pipeline as a script. The audit does not require this step — it runs from
the released checkpoints.

## Notes

- `experiments/checkpoints/` (superseded short-history checkpoints) and
  `last_model.pt` files are gitignored; the paper uses `checkpoints_long/*/best_model.pt` only.
- The `execution/` and `scripts/` trees are a separate live-trading line of work
  and are not part of the paper's claims.
- See [`PROVENANCE.md`](PROVENANCE.md) for the full artifact-to-claim mapping and
  the four places the evidence base is explicitly weaker.

## Citation

Anonymous. *When a Learned Encoder Does Not Beat a Random One: An Evaluation
Protocol for Self-Supervised World Models on Financial Markets.* ICAIF '26
(under review).
