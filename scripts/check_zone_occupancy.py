"""
scripts/check_zone_occupancy.py — Four-zone circuit-breaker occupancy on replay S_t.
====================================================================================

Replays the live circuit-breaker zone assignment over the post-fix S_t series
(experiments/st_replay_results.json -> s_new), exactly as live_trader does:
rolling causal percentiles p20/p60/p90 over the surprise history so far, each
floored, classified after `surprise_warmup` bars. Reports zone occupancy under the
NEW floors vs the OLD floors so the recalibration can be verified before any live run.

Intended design (percentiles drive assignment): ~Q1 20% / Q2-skip 40% / Q4-half 30%
/ Q5-CB 10%  (the boundaries are p20/p60/p90, so the partition is 20/40/30/10 by
construction when the floors don't bind).

Usage:  python -m scripts.check_zone_occupancy
"""

import sys, json
sys.path.insert(0, ".")
import numpy as np

from execution.live_trader import _CB_FLOOR_P20, _CB_FLOOR_P60, _CB_FLOOR_P90

OLD_FLOORS = (0.0005, 0.003, 0.010)
NEW_FLOORS = (_CB_FLOOR_P20, _CB_FLOOR_P60, _CB_FLOOR_P90)
SURPRISE_WARMUP = 100   # matches LiveTrader default


def occupancy(s_new, floors, warmup=SURPRISE_WARMUP):
    f20, f60, f90 = floors
    zones = {"Q1": 0, "Q2-skip": 0, "Q4-half": 0, "Q5-CB": 0}
    floor_binds = {"p20": 0, "p60": 0, "p90": 0}
    n = 0
    for i in range(len(s_new)):
        hist = s_new[: i + 1]                       # includes current bar, as in live
        if len(hist) < warmup:
            continue
        p20 = max(float(np.percentile(hist, 20)), f20)
        p60 = max(float(np.percentile(hist, 60)), f60)
        p90 = max(float(np.percentile(hist, 90)), f90)
        floor_binds["p20"] += (np.percentile(hist, 20) < f20)
        floor_binds["p60"] += (np.percentile(hist, 60) < f60)
        floor_binds["p90"] += (np.percentile(hist, 90) < f90)
        s = s_new[i]
        if   s < p20: zones["Q1"]      += 1
        elif s < p60: zones["Q2-skip"] += 1
        elif s < p90: zones["Q4-half"] += 1
        else:         zones["Q5-CB"]   += 1
        n += 1
    return zones, floor_binds, n


def _print(label, floors, s_new):
    zones, fb, n = occupancy(s_new, floors)
    print(f"\n  {label}  floors={floors}  (classified {n} bars after {SURPRISE_WARMUP}-bar warmup)")
    print(f"  {'zone':<10} {'count':>6} {'occupancy':>10}   target")
    print(f"  {'-'*10} {'-'*6} {'-'*10}   {'-'*8}")
    targets = {"Q1": "20%", "Q2-skip": "40%", "Q4-half": "30%", "Q5-CB": "10%"}
    for z in ("Q1", "Q2-skip", "Q4-half", "Q5-CB"):
        pct = 100 * zones[z] / n if n else 0
        print(f"  {z:<10} {zones[z]:>6} {pct:>9.1f}%   {targets[z]:>6}")
    print(f"  floor-binds: p20 {100*fb['p20']/n:.0f}%  p60 {100*fb['p60']/n:.0f}%  "
          f"p90 {100*fb['p90']/n:.0f}%  of classified bars")


def main():
    with open("./experiments/st_replay_results.json") as f:
        data = json.load(f)
    s_new = np.array([r["s_new"] for r in data["series"]])
    print("=" * 70)
    print(f"ZONE OCCUPANCY on post-fix S_t  ({data['window'][0][:16]} -> "
          f"{data['window'][1][:16]}, {len(s_new)} bars)")
    print("=" * 70)
    print(f"  S_t_new percentiles:  p20={np.percentile(s_new,20):.5f}  "
          f"p60={np.percentile(s_new,60):.5f}  p90={np.percentile(s_new,90):.5f}")

    _print("OLD floors (pre-fix calibration)", OLD_FLOORS, s_new)
    _print("NEW floors (recalibrated)",        NEW_FLOORS, s_new)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
