"""
event_precision_recall.py -- Precision/Recall audit of the MWM surprise signal
==============================================================================

WHY THIS EXISTS
---------------
The paper (O3) currently narrates the top 1-2 surprise clusters per asset and
matches them post-hoc to a known macro event ("FOMC", "Jackson Hole", ...).
A reviewer's obvious attack: that is selection. It reports the *numerator*
(spikes that matched an event) and hides the *denominator* -- how many big
spikes matched NOTHING, and how many real events the model was SILENT on.

This script computes the two-directional, defensible version instead:

  PRECISION  Of the model's top-N surprise episodes, how many sit on a known
             scheduled macro event (within +/- WINDOW days)?  The rest are
             "unexplained" and are listed explicitly.

  RECALL     Of the scheduled events that ACTUALLY MOVED THE MARKET (a >2sigma
             bar occurred in the event window -- a "reacted" event), how many
             did the model spike on?  Non-reacting events (a BOJ hold that did
             nothing) are excluded from the denominator on purpose: a regime
             detector *should* stay quiet when price does not move.

INPUTS (already produced by evaluation/surprise.py -- no model rerun needed)
  experiments/surprise_timeseries*.json   per-bar smoothed S_t + ret_z + ts
  experiments/macro_calendar.json         editable scheduled-event calendar

OUTPUT
  Prints a per-asset precision/recall table + the honest reframing sentence,
  and writes experiments/event_precision_recall_results.json.

PARAMETERS (all exposed; defaults chosen to be conservative)
  --window   +/- days around an event that count as "on the event"   (default 1)
  --spike-pct  smoothed-S_t percentile that counts as a spike         (default 95)
  --react-sigma  |ret| z-score marking an event as "market reacted"   (default 2.0)
  --top-n    number of top surprise episodes to audit for precision   (default 15)
  --merge-days  merge spikes within this many days into one episode   (default 3)

USAGE
  python -m experiments.event_precision_recall
  python experiments/event_precision_recall.py --spike-pct 99 --window 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

ASSETS = {
    "gold":   HERE / "surprise_timeseries.json",
    "eurusd": HERE / "surprise_timeseries_eurusd.json",
    "usdjpy": HERE / "surprise_timeseries_usdjpy.json",
}

# Which calendar categories are the fair "recall denominator" for each asset.
# We only hold an asset accountable for events that plausibly drive it.
RELEVANT_CATS = {
    "gold":   {"FOMC", "ECB", "CPI", "NFP", "OTHER"},   # gold: real yields / USD / risk
    "eurusd": {"FOMC", "ECB", "CPI", "NFP", "OTHER"},   # EUR/USD: Fed-ECB divergence + US data
    "usdjpy": {"FOMC", "BOJ", "CPI", "NFP", "OTHER"},   # USD/JPY: Fed-BOJ / carry + US data
}


def load_series(path):
    d = json.load(open(path))
    df = pd.DataFrame({
        "ts":     pd.to_datetime(d["timestamps"]),
        "smooth": np.asarray(d["surprise_smooth"], float),
        "raw":    np.asarray(d["surprise_raw"], float),
        "ret_z":  np.asarray(d["ret_z"], float),
    })
    return df


def load_calendar():
    cal = json.load(open(HERE / "macro_calendar.json"))
    ev = pd.DataFrame(cal["events"])
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    return ev


def merge_episodes(spike_days, merge_days):
    """Collapse a sorted list of spike dates into episodes (a spike day within
    `merge_days` of the previous is the same episode)."""
    episodes = []
    cur = []
    prev = None
    for day in sorted(spike_days):
        if prev is not None and (day - prev).days > merge_days:
            episodes.append(cur)
            cur = []
        cur.append(day)
        prev = day
    if cur:
        episodes.append(cur)
    return episodes


def nearest_event(day, events, window):
    """Return (name, category, delta_days) of the nearest calendar event within
    +/- window days of `day`, else None."""
    best = None
    for _, e in events.iterrows():
        delta = abs((day - e["date"]).days)
        if delta <= window and (best is None or delta < best[2]):
            best = (e["name"], e["category"], delta)
    return best


def analyse(asset, path, cal, window, spike_pct, react_sigma, top_n, merge_days):
    df = load_series(path)
    df["day"] = df["ts"].dt.date

    thr = np.percentile(df["smooth"], spike_pct)
    per_day = df.groupby("day").agg(
        peak_smooth=("smooth", "max"),
        peak_raw=("raw", "max"),
        peak_retz=("ret_z", "max"),
    ).reset_index()

    win_start, win_end = df["day"].min(), df["day"].max()
    rel = cal[cal["category"].isin(RELEVANT_CATS[asset])].copy()
    rel = rel[(rel["date"] >= win_start) & (rel["date"] <= win_end)]

    # ---- reaction filter: did the market actually move around the event? ----
    def event_reaction(edate):
        m = (df["day"] >= edate - pd.Timedelta(days=window)) & \
            (df["day"] <= edate + pd.Timedelta(days=window))
        sub = df[m]
        if len(sub) == 0:
            return np.nan, np.nan
        return float(sub["ret_z"].max()), float(sub["smooth"].max())

    if len(rel) == 0:
        # No calendar events overlap this test window (e.g. window predates the
        # calendar). Return an empty-but-valid result rather than crashing.
        rel["max_retz"] = pd.Series(dtype=float)
        rel["max_smooth"] = pd.Series(dtype=float)
    else:
        rel[["max_retz", "max_smooth"]] = rel["date"].apply(
            lambda d: pd.Series(event_reaction(d)))
    rel["reacted"] = rel["max_retz"] >= react_sigma
    rel["model_spiked"] = rel["max_smooth"] >= thr

    reacted = rel[rel["reacted"]]
    recall_hits = int(reacted["model_spiked"].sum())
    recall_den = int(len(reacted))
    recall = recall_hits / recall_den if recall_den else float("nan")

    # ---- precision: top-N surprise episodes, explained vs unexplained -------
    spike_days = per_day.loc[per_day["peak_smooth"] >= thr, "day"].tolist()
    episodes = merge_episodes(spike_days, merge_days)
    # rank episodes by their peak smoothed surprise
    ep_rows = []
    for ep in episodes:
        sub = per_day[per_day["day"].isin(ep)]
        peak_day = sub.loc[sub["peak_smooth"].idxmax(), "day"]
        peak_val = float(sub["peak_smooth"].max())
        ne = nearest_event(peak_day, rel, window)
        # also check against ALL events (not just relevant cats), for context
        ne_all = nearest_event(peak_day, cal[(cal["date"] >= win_start) &
                                             (cal["date"] <= win_end)], window)
        ep_rows.append({
            "peak_day": str(peak_day),
            "peak_smooth": peak_val,
            "peak_x_mean": peak_val / df["raw"].mean(),
            "span_days": (max(ep) - min(ep)).days + 1,
            "event": ne_all[0] if ne_all else None,
            "event_cat": ne_all[1] if ne_all else None,
            "delta_days": ne_all[2] if ne_all else None,
        })
    ep_rows.sort(key=lambda r: r["peak_smooth"], reverse=True)
    top = ep_rows[:top_n]
    prec_hits = sum(1 for r in top if r["event"] is not None)
    precision = prec_hits / len(top) if top else float("nan")

    return {
        "asset": asset,
        "window_start": str(win_start),
        "window_end": str(win_end),
        "spike_threshold_smooth": float(thr),
        "spike_pct": spike_pct,
        "precision": precision,
        "precision_hits": prec_hits,
        "precision_den": len(top),
        "recall": recall,
        "recall_hits": recall_hits,
        "recall_den": recall_den,
        "n_relevant_events": int(len(rel)),
        "n_reacted_events": recall_den,
        "top_episodes": top,
        "events_detail": rel[["date", "name", "category", "max_retz",
                              "reacted", "model_spiked"]].astype(str).to_dict("records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--spike-pct", type=float, default=95.0)
    ap.add_argument("--react-sigma", type=float, default=2.0)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--merge-days", type=int, default=3)
    args = ap.parse_args()

    cal = load_calendar()
    results = {}
    print("=" * 78)
    print(f"MWM surprise precision/recall  |  window=+/-{args.window}d  "
          f"spike>={args.spike_pct}pct  react>={args.react_sigma}sigma  "
          f"top_n={args.top_n}")
    print("=" * 78)

    for asset, path in ASSETS.items():
        if not Path(path).exists():
            print(f"[skip] {asset}: {path} missing")
            continue
        r = analyse(asset, path, cal, args.window, args.spike_pct,
                    args.react_sigma, args.top_n, args.merge_days)
        results[asset] = r

        print(f"\n##### {asset.upper()}  ({r['window_start']} -> {r['window_end']})")
        print(f"  relevant scheduled events in window: {r['n_relevant_events']}"
              f"   | of which market reacted (>{args.react_sigma}sig): {r['n_reacted_events']}")
        print(f"  RECALL   = {r['recall_hits']}/{r['recall_den']} "
              f"= {r['recall']:.2f}  (reacted events the model spiked on)")
        print(f"  PRECISION= {r['precision_hits']}/{r['precision_den']} "
              f"= {r['precision']:.2f}  (top surprise episodes on a known event)")

        print(f"\n  Top {args.top_n} surprise episodes (precision view):")
        print(f"    {'peak day':<12}{'xmean':>7}  {'event (<= window)'}")
        for e in r["top_episodes"]:
            tag = f"{e['event']} (d{e['delta_days']:+d})" if e["event"] else "-- UNEXPLAINED --"
            print(f"    {e['peak_day']:<12}{e['peak_x_mean']:>6.1f}x  {tag}")

        print(f"\n  Reacted events (recall view):")
        for e in r["events_detail"]:
            if e["reacted"] == "True":
                hit = "SPIKE" if e["model_spiked"] == "True" else "  .  (MISS)"
                print(f"    {e['date']}  {hit}  retz={float(e['max_retz']):.1f}  {e['name']}")

    out = HERE / "event_precision_recall_results.json"
    json.dump({"params": vars(args), "results": results}, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")

    # ---- one-line honest summary per asset ----
    print("\n" + "=" * 78)
    print("HONEST REFRAMING (paste-ready, fill asset):")
    print("=" * 78)
    for asset, r in results.items():
        print(f"  {asset}: model spiked on {r['recall_hits']}/{r['recall_den']} "
              f"market-moving scheduled events (recall {r['recall']:.0%}); "
              f"{r['precision_hits']}/{r['precision_den']} of its top surprise "
              f"episodes sit on a known event (precision {r['precision']:.0%}).")


if __name__ == "__main__":
    main()
