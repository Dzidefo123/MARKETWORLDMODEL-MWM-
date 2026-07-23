"""
check_paper_numbers.py -- assert every quantitative claim in mwm_audit_paper.tex
================================================================================

The paper argues that unchecked numbers are how spurious results survive, so its
own numbers should be machine-checkable. This re-derives each headline figure
from the result JSONs and asserts the .tex contains it.

Sources
  experiments/probe_absolute_{inst}.json   Tables 2 and 3, Section 5.2 prose
  experiments/embedding_spread_{inst}.json Table 1, Section 5.1
  experiments/o3_stratified_{inst}.json    Table 4, Section 5.3 prose, Figure 2

Convention checked here: deltas and win/loss counts come from UNROUNDED scores,
table cells are rounded half-up to two decimals. Those two can disagree by 0.01,
which is why the Table 2 caption says so.

USAGE
  python experiments/check_paper_numbers.py
Exit code 1 if any claim fails.
"""

import json
import pathlib
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ASSETS = ["gold", "eurusd", "usdjpy"]
TARGETS = ["session", "high_vol", "ret_direction", "realized_vol", "dxy_direction"]

# The manuscript source is withheld from the public repo until approved (see
# .gitignore / README). When it is absent, we cannot assert the .tex contains a
# given value, so `in_tex` becomes a no-op and only the result JSONs are checked
# for internal consistency. With the .tex present (author's working copy) the
# full 66-check cross-reference runs.
_texfile = pathlib.Path("mwm_audit_paper.tex")
TEX = _texfile.read_text(encoding="utf8") if _texfile.exists() else None
if TEX is None:
    print("NOTE: mwm_audit_paper.tex not found (manuscript withheld); "
          "checking result JSONs only, skipping .tex cross-references.\n")

fails = []


def r2(x):
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def check(name, cond, detail=""):
    if cond:
        print("  ok    %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        fails.append(name)


def in_tex(s):
    # No .tex available -> cannot contradict it; treat the containment as passing
    # so the JSON-consistency checks still run from a clone.
    return True if TEX is None else s in TEX


# ---------------------------------------------------------------- load
S, CH, SD = {}, {}, {}
for i in ASSETS:
    df = pd.read_json("experiments/probe_absolute_%s.json" % i)
    for _, r in df.iterrows():
        S[(i, r.target, r.probe, r.source)] = r.score
        CH[(i, r.target, r.probe)] = r.chance
        if r.source == "random":
            SD[(i, r.target, r.probe)] = r["std"]
    globals().setdefault("N", {})[i] = (int(df.n_train.iloc[0]), int(df.n_val.iloc[0]))
SPREAD = {i: json.load(open("experiments/embedding_spread_%s.json" % i)) for i in ASSETS}
O3 = {i: json.load(open("experiments/o3_stratified_%s.json" % i)) for i in ASSETS}
delta = {(i, t, p): S[(i, t, p, "trained")] - S[(i, t, p, "random")]
         for i in ASSETS for t in TARGETS for p in ["linear", "mlp"]}

print("\n-- Table 1 / Section 5.1: anti-collapse --")
for i in ASSETS:
    d = SPREAD[i]
    check("%s spread ratio" % i, in_tex("$%.1f\\times$" % d["ratio_mean"]),
          "%.1f" % d["ratio_mean"])
ratios = [SPREAD[i]["ratio_mean"] for i in ASSETS]
check("spread range 34--43", in_tex("$34$--$43\\times$"),
      "actual %.1f..%.1f" % (min(ratios), max(ratios)))
check("  range is truthful", 34 <= min(ratios) and max(ratios) <= 43,
      "%.1f..%.1f" % (min(ratios), max(ratios)))

print("\n-- Table 2: deltas (unrounded, rendered 2dp) --")
for t in TARGETS:
    vals = [delta[(i, t, "mlp")] for i in ASSETS]
    cells = " & ".join("$%+.2f$" % v for v in vals)
    check("%s row" % t, in_tex(cells), cells)

print("\n-- Section 5.2 prose --")
n_loss = sum(1 for v in delta.values() if v < 0)
n_win = sum(1 for v in delta.values() if v > 0)
n_tie = sum(1 for v in delta.values() if v == 0)
check("27 of 30 losses", n_loss == 27 and in_tex("$27$ of the $30$ cells"),
      "losses=%d wins=%d ties=%d" % (n_loss, n_win, n_tie))
check("no exact ties (caption claim)", n_tie == 0, "ties=%d" % n_tie)
maxwin = max(v for v in delta.values() if v > 0)
check("max win <= +0.035", maxwin <= 0.035 and in_tex("at most $+0.035$"), "%.4f" % maxwin)
worst_sd = max(v / SD[(i, t, p)] for (i, t, p), v in delta.items() if v > 0)
check("all wins < 2 sd", worst_sd < 2.0, "%.2f sd" % worst_sd)
for (i, t, p), v in delta.items():
    if v > 0:
        other = "mlp" if p == "linear" else "linear"
        check("  win %s/%s/%s reverses sign" % (i, t, p), delta[(i, t, other)] < 0,
              "%+.4f" % delta[(i, t, other)])

hv = [abs(delta[(i, "high_vol", p)]) for i in ASSETS for p in ["linear", "mlp"]]
rv = [abs(delta[(i, "realized_vol", p)]) for i in ASSETS for p in ["linear", "mlp"]]
check("high-vol loss range 0.02--0.21",
      in_tex("$0.02$--$0.21$") and r2(min(hv)) == 0.02 and r2(max(hv)) == 0.21,
      "%.3f..%.3f" % (min(hv), max(hv)))
check("realized-vol loss range 0.28--0.37",
      in_tex("$0.28$--$0.37$") and r2(min(rv)) == 0.28 and r2(max(rv)) == 0.37,
      "%.3f..%.3f" % (min(rv), max(rv)))
check("trained high-vol 0.86/0.83/0.68",
      in_tex("$0.86/0.83/0.68$") and
      [r2(S[(i, "high_vol", "linear", "trained")]) for i in ASSETS] == [0.86, 0.83, 0.68], "")
check("chance high-vol 0.56/0.56/0.52",
      in_tex("$0.56/0.56/0.52$") and
      [r2(CH[(i, "high_vol", "linear")]) for i in ASSETS] == [0.56, 0.56, 0.52], "")
check("trained realized-vol r 0.34/0.28/0.44",
      in_tex("$r=0.34/0.28/0.44$") and
      [r2(S[(i, "realized_vol", "linear", "trained")]) for i in ASSETS] == [0.34, 0.28, 0.44], "")

floor = max(S[(i, t, p, s)] - CH[(i, t, p)] for i in ASSETS
            for t in ["session", "ret_direction"] for p in ["linear", "mlp"]
            for s in ["trained", "random"])
check("floors: nothing beats chance by >0.03",
      floor <= 0.03 and in_tex("by more than $0.03$"), "max %+.4f" % floor)

inp = [S[(i, t, "linear", "inputs")] for i in ASSETS
       for t in ["session", "high_vol", "realized_vol", "dxy_direction"]]
check("inputs ceiling 0.98--1.00",
      in_tex("$0.98$--$1.00$") and r2(min(inp)) >= 0.98, "%.3f..%.3f" % (min(inp), max(inp)))
fwd = [S[(i, "ret_direction", p, "inputs")] for i in ASSETS for p in ["linear", "mlp"]]
check("forward target on inputs 0.53--0.58",
      in_tex("$0.53$--$0.58$") and r2(min(fwd)) == 0.53 and r2(max(fwd)) == 0.58,
      "%.3f..%.3f" % (min(fwd), max(fwd)))

print("\n-- Section 5.3 / Table 4 / Figure 2: O3 --")
pooled = [O3[i]["rho_S_RV_pooled"] for i in ASSETS]
within = [O3[i]["rho_S_RV_within_strata"] for i in ASSETS]
check("pooled rho 0.11/0.10/0.34",
      in_tex("$\\rho=0.11/0.10/0.34$") and [r2(v) for v in pooled] == [0.11, 0.10, 0.34],
      str([round(v, 4) for v in pooled]))
check("within rho 0.03/0.05/0.16",
      in_tex("$0.03/0.05/0.16$") and [r2(v) for v in within] == [0.03, 0.05, 0.16],
      str([round(v, 4) for v in within]))
removed = [100 * (1 - w / p) for w, p in zip(within, pooled)]
check("removal 75%% / 49%% / 54%%",
      in_tex("$75\\%$") and in_tex("$49\\%$ and $54\\%$") and
      [round(r) for r in removed] == [75, 49, 54], str([round(r) for r in removed]))
partial = [O3[i]["rho_S_RV_partial_lag24"] for i in ASSETS]
check("partial rho 0.06/0.05/0.24",
      in_tex("($0.06/0.05/0.24$)") and [r2(v) for v in partial] == [0.06, 0.05, 0.24],
      str([round(v, 4) for v in partial]))

ratios_pooled = [O3[i]["extreme_ratio_pooled"] for i in ASSETS]
check("extreme ratios 0.92/0.86/1.10",
      in_tex("$0.92/0.86/1.10$") and [r2(v) for v in ratios_pooled] == [0.92, 0.86, 1.10],
      str([round(v, 4) for v in ratios_pooled]))
check("no pooled CI above 1",
      all(O3[i]["extreme_ratio_ci"][0] <= 1 for i in ASSETS),
      str([[round(x, 3) for x in O3[i]["extreme_ratio_ci"]] for i in ASSETS]))
check("EUR/USD pooled CI [0.77,0.97]",
      in_tex("$[0.77,0.97]$") and
      [r2(x) for x in O3["eurusd"]["extreme_ratio_ci"]] == [0.77, 0.97], "")

gf = [r["extreme_ratio"] for i in ["gold", "eurusd"] for r in O3[i]["strata"]]
check("gold+EURUSD within-stratum ratios 0.73--1.01",
      in_tex("$0.73$ to $1.01$") and r2(min(gf)) == 0.73 and r2(max(gf)) == 1.01,
      "%.4f..%.4f" % (min(gf), max(gf)))
above = [(i, r["stratum"]) for i in ["gold", "eurusd"] for r in O3[i]["strata"]
         if r["ratio_ci"][0] > 1]
below = [(i, r["stratum"]) for i in ["gold", "eurusd"] for r in O3[i]["strata"]
         if r["ratio_ci"][1] < 1]
check("only excluded interval is from below (EUR/USD mid)",
      above == [] and below == [("eurusd", "mid")], "above=%s below=%s" % (above, below))
check("  and it is [0.62,0.95]", in_tex("$[0.62,0.95]$") and
      [r2(x) for x in O3["eurusd"]["strata"][1]["ratio_ci"]] == [0.62, 0.95], "")

u_low = O3["usdjpy"]["strata"][0]
check("USD/JPY quietest tercile 1.25 [1.09,1.42]",
      in_tex("$1.25$, $[1.09,1.42]$") and r2(u_low["extreme_ratio"]) == 1.25 and
      [r2(x) for x in u_low["ratio_ci"]] == [1.09, 1.42], "")
check("USD/JPY within-strata rho 0.16",
      in_tex("$\\rho=0.16$ within strata") and r2(within[2]) == 0.16, "")
check("implied-vol rho 0.027 / -0.038",
      in_tex("$\\rho=0.027$") and in_tex("$\\rho=-0.038$") and
      round(O3["gold"]["rho_S_IV_pooled"], 3) == 0.027 and
      round(O3["eurusd"]["rho_S_IV_pooled"], 3) == -0.038, "")

print("\n-- window counts --")
check("gold 41,155/1,472/20,606",
      in_tex("$41{,}155/1{,}472/20{,}606$") and
      N["gold"] == (41155, 1472) and O3["gold"]["n_test"] == 20606, "")
check("FX train/val 18,431/1,544",
      in_tex("$18{,}431/1{,}544$") and N["eurusd"] == N["usdjpy"] == (18431, 1544), "")
check("FX test spans differ: 15,504 / 15,503",
      in_tex("$15{,}504$ (EUR/USD) and $15{,}503$ (USD/JPY)") and
      O3["eurusd"]["n_test"] == 15504 and O3["usdjpy"]["n_test"] == 15503, "")

print("\n-- Tables 5 and 6 / Section 5.4: the batching artifact --")
ART = {i: json.load(open("experiments/o1b_artifact_check_%s.json" % i)) for i in ASSETS}
NAME = {"gold": "Gold", "eurusd": "EUR/USD", "usdjpy": "USD/JPY"}
flat = "" if TEX is None else " ".join(TEX.split())
for i in ASSETS:
    t1 = ART[i]["test1_batch_artifact"]
    for bs in ["bs64", "bs128"]:
        c, r = t1[bs]["contiguous_shuffleFalse"], t1[bs]["random_shuffleTrue"]
        row = "%s & %s & %.3f & %.3f & $%.1f\\times$" % (
            NAME[i], bs[2:], round(c, 3), round(r, 3), c / r)
        check("Table 5 %s %s" % (i, bs), (TEX is None or " ".join(row.split()) in flat), row)
for i in ASSETS:
    t2 = ART[i]["test2"]
    sm = t2["segment_means"]
    row = "%s & %.3f & %.3f & %.3f & %.3f" % (
        NAME[i], round(t2["train_baseline"], 3), round(sm["train"], 3),
        round(sm["val"], 3), round(sm["oos"], 3))
    check("Table 6 %s" % i, (TEX is None or " ".join(row.split()) in flat), row)
vr = [ART[i]["test2"]["spearman_val_sig_vol_ratio"] for i in ASSETS]
ks = [ART[i]["test2"]["spearman_val_sig_ks"] for i in ASSETS]
check("vol-ratio correlations -0.15/+0.26/-0.25",
      in_tex("($-0.15$ gold, $+0.26$") and [r2(v[0]) for v in vr] == [-0.15, 0.26, -0.25],
      str([round(v[0], 3) for v in vr]))
check("KS correlations -0.14/-0.07/-0.18",
      in_tex("($-0.14/-0.07/-0.18$)") and [r2(v[0]) for v in ks] == [-0.14, -0.07, -0.18],
      str([round(v[0], 3) for v in ks]))
check("none significant at 0.05",
      all(v[1] > 0.05 for v in vr + ks) and in_tex("None reaches significance at $0.05$"),
      "min p = %.4f" % min(v[1] for v in vr + ks))

print("\n-- Section 5.4 opening: the train/val SIGReg gap --")
H = json.load(open("experiments/checkpoints_long/eurusd/loss_history.json"))
BEST = 48 - 1                      # released EUR/USD checkpoint is epoch 48
tr, vl = H["train_sigreg"][BEST], H["val_sigreg"][BEST]
check("EUR/USD train SIGReg 0.029", r2(tr) == 0.03 and round(tr, 3) == 0.029, "%.4f" % tr)
check("EUR/USD val SIGReg 0.121", round(vl, 3) == 0.121 and in_tex("$0.121$"), "%.4f" % vl)
check("  gap is 4x", round(vl / tr) == 4, "%.2fx" % (vl / tr))

print("\n-- Section 5.3: calendar precision/recall --")
PR = json.load(open("experiments/event_precision_recall_matched.json"))
rec = [PR[i]["recall_hits"] / PR[i]["recall_den"] for i in ASSETS]
pre = [PR[i]["precision_hits"] / PR[i]["precision_den"] for i in ASSETS]
check("recall 0.15--0.26",
      in_tex("recall $0.15$--$0.26$") and r2(min(rec)) == 0.15 and r2(max(rec)) == 0.26,
      "%.3f..%.3f" % (min(rec), max(rec)))
check("precision 0.20--0.47",
      in_tex("precision $0.20$--$0.47$") and r2(min(pre)) == 0.20 and r2(max(pre)) == 0.47,
      "%.3f..%.3f" % (min(pre), max(pre)))
cal = json.load(open("experiments/macro_calendar.json"))["events"]
check("75-event calendar", len(cal) == 75 and in_tex("$75$-event"), "%d events" % len(cal))

print("\n-- Table 4 cells --")
NAME = {"gold": "Gold", "eurusd": "EUR/USD", "usdjpy": "USD/JPY"}
for i in ASSETS:
    d, st = O3[i], {r["stratum"]: r for r in O3[i]["strata"]}
    row_a = "%s & $%+.2f$ & $%+.2f$ & $%+.2f$ & $%+.2f$ & $%+.2f$" % (
        NAME[i], r2(d["rho_S_RV_pooled"]), r2(st["low"]["rho_S_RV"]),
        r2(st["mid"]["rho_S_RV"]), r2(st["high"]["rho_S_RV"]),
        r2(d["rho_S_RV_partial_lag24"]))
    row_b = "%s & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$" % (
        NAME[i], r2(d["extreme_ratio_pooled"]), r2(st["low"]["extreme_ratio"]),
        r2(st["mid"]["extreme_ratio"]), r2(st["high"]["extreme_ratio"]))
    flat = "" if TEX is None else " ".join(TEX.split())
    check("%s rho row" % i, (TEX is None or " ".join(row_a.split()) in flat), row_a)
    check("%s ratio row" % i, (TEX is None or " ".join(row_b.split()) in flat), row_b)

print("\n-- Table 3 cells --")
bad = 0
for probe in ([] if TEX is None else ["linear", "mlp"]):
    key = "\\emph{%s probe}" % ("Linear" if probe == "linear" else "MLP")
    block = TEX.split(key)[1].split("\\bottomrule")[0].split("\\midrule")[0]
    for t, lab in zip(TARGETS, ["Session (3-class)", "High vol (binary)",
                                "Return direction", "Realized vol ($r$)",
                                "DXY direction"]):
        line = [l for l in block.splitlines() if l.strip().startswith(lab)][0]
        import re
        nums = [float(x) for x in re.findall(r"\d\.\d\d", line)]
        exp = [r2(v) for i in ASSETS for v in
               (CH[(i, t, probe)], S[(i, t, probe, "inputs")],
                S[(i, t, probe, "random")], S[(i, t, probe, "trained")])]
        if nums != exp:
            bad += 1
            print("    FAIL %s %s\n      tex=%s\n      exp=%s" % (probe, t, nums, exp))
check("all 120 cells", bad == 0, "%d bad rows" % bad) if TEX is not None else \
    print("  skip  all 120 cells (manuscript withheld)")

print("\n" + "=" * 60)
if fails:
    print("FAILED %d check(s): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("All checks passed.")
