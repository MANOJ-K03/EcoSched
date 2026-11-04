"""Decision advisor and automatic interpretation. Every sentence is computed from results."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

TOL = 1e-9

# key: (label, better direction, unit)
METRICS: Dict[str, tuple] = {
    "avg_wait": ("Average waiting time", "min", "ticks"),
    "avg_turnaround": ("Average turnaround time", "min", "ticks"),
    "avg_response": ("Average response time", "min", "ticks"),
    "max_wait": ("Maximum waiting time", "min", "ticks"),
    "throughput": ("Throughput", "max", "jobs/tick"),
    "utilization": ("CPU utilization", "max", "fraction"),
    "context_switches": ("Context switches", "min", "count"),
    "fairness": ("Jain fairness index", "max", "0-1"),
    "energy_j": ("Estimated energy (simulated)", "min", "J"),
}

WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "Balanced (all metrics equal)": {k: 1 for k in METRICS},
    "Responsiveness first": {"avg_response": 3, "avg_wait": 2, "max_wait": 1, "fairness": 1},
    "Energy first": {"energy_j": 4, "context_switches": 2, "utilization": 1},
    "Fairness first": {"fairness": 4, "max_wait": 2, "avg_response": 1},
    "Throughput first": {"throughput": 3, "avg_turnaround": 2, "utilization": 2},
}


def ranks(values: Dict[str, float], direction: str) -> Dict[str, int]:
    """Competition ranking with ties (1 = best)."""
    out = {}
    for name, v in values.items():
        better = sum((o < v - TOL) if direction == "min" else (o > v + TOL) for o in values.values())
        out[name] = 1 + better
    return out


def composite_scores(summary: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Weighted 0-1 score. Each metric is min-max normalised ACROSS the compared algorithms
    (1 = best of the group, 0 = worst), so it measures relative standing, not absolute quality."""
    total_w = sum(w for k, w in weights.items() if k in METRICS) or 1.0
    score = pd.Series(0.0, index=summary.index)
    for key, w in weights.items():
        if key not in METRICS or w <= 0:
            continue
        col = summary[key].astype(float)
        lo, hi = col.min(), col.max()
        norm = pd.Series(1.0, index=summary.index) if hi - lo < TOL else (
            (hi - col) / (hi - lo) if METRICS[key][1] == "min" else (col - lo) / (hi - lo))
        score += w * norm
    return (score / total_w).rename("score")


def interpret(summary: pd.DataFrame, focus: str = "EMI", counterpart: str = "Round Robin") -> List[str]:
    """Plain-English findings. Reports measured outcomes only, never causes."""
    names = list(summary["algorithm"])
    if focus not in names or len(names) < 2:
        return ["Not enough results to interpret."]
    lines: List[str] = []
    best_on, worst_on = [], []
    detail = []
    for key, (label, direction, unit) in METRICS.items():
        vals = dict(zip(summary["algorithm"], summary[key].astype(float)))
        rk = ranks(vals, direction)
        if all(r == 1 for r in rk.values()):
            continue  # everyone tied on this metric
        leaders = [a for a, r in rk.items() if r == 1]
        if rk[focus] == 1:
            best_on.append(label.lower())
        if rk[focus] == max(rk.values()):
            worst_on.append(label.lower())
        detail.append(f"{label}: {focus} = {vals[focus]:.3g} (rank {rk[focus]} of {len(names)}); "
                      f"best = {', '.join(leaders)} ({vals[leaders[0]]:.3g}).")
    lines.append(f"{focus} was best or tied-best on {len(best_on)} metric(s)"
                 + (f": {', '.join(best_on)}." if best_on else "."))
    lines.append(f"{focus} was worst or tied-worst on {len(worst_on)} metric(s)"
                 + (f": {', '.join(worst_on)}." if worst_on else "."))
    if counterpart in names:
        a = summary.set_index("algorithm")
        parts = []
        for key in ("avg_response", "avg_wait", "context_switches", "energy_j"):
            label, direction, _ = METRICS[key]
            x, y = float(a.loc[focus, key]), float(a.loc[counterpart, key])
            if abs(x - y) < TOL:
                parts.append(f"{label.lower()} equal ({x:.3g})")
            else:
                pct = f" ({(x - y) / y * 100:+.1f}%)" if y else ""
                verdict = "lower" if x < y else "higher"
                parts.append(f"{label.lower()} {x:.3g} vs {y:.3g}{pct}, {verdict}")
        lines.append(f"Against {counterpart} on this workload: " + "; ".join(parts) + ".")
    lines.extend(detail)
    lines.append("These statements describe measured outcomes for this one workload and these settings only; "
                 "they do not explain why a result occurred and may change with other workloads or parameters.")
    return lines
