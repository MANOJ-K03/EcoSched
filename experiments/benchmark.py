"""Benchmarks: single runs, profile comparison, Monte-Carlo robustness, sensitivity analyses."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from scheduler.adaptive import EMIConfig, EMIDecision, emi_schedule
from scheduler.algorithms import fcfs, priority_preemptive, round_robin, sjf, srtf
from scheduler.energy import EnergyModel
from scheduler.insights import METRICS, TOL
from scheduler.metrics import process_metrics, summarize
from scheduler.models import Process, SimulationResult
from .workloads import PROFILE_SEEDS, generate

BASELINES = ["FCFS", "SJF", "Round Robin", "Priority"]
ALGORITHMS = BASELINES + ["EMI"]


@dataclass(frozen=True)
class SimConfig:
    quantum: int = 4
    cs_cost: int = 1
    energy: EnergyModel = field(default_factory=EnergyModel)
    emi: EMIConfig = field(default_factory=EMIConfig)


def run_all(processes: Sequence[Process], cfg: SimConfig, profile_hint: Optional[str] = None,
            include_reference: bool = False) -> Tuple[Dict[str, SimulationResult], EMIDecision]:
    """Run every algorithm on the SAME processes with the SAME cost parameters."""
    results = {
        "FCFS": fcfs(processes, cfg.cs_cost),
        "SJF": sjf(processes, cfg.cs_cost),
        "Round Robin": round_robin(processes, cfg.quantum, cfg.cs_cost),
        "Priority": priority_preemptive(processes, cfg.cs_cost),
    }
    emi_result, decision = emi_schedule(processes, cfg.cs_cost, cfg.emi, profile_hint)
    results["EMI"] = emi_result
    if include_reference:
        results["SRTF (reference)"] = srtf(processes, cfg.cs_cost)
    return results, decision


def summary_table(results: Dict[str, SimulationResult], energy: EnergyModel) -> pd.DataFrame:
    return pd.DataFrame([summarize(r, energy) for r in results.values()])


def process_table(results: Dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows += [{"algorithm": name, **row} for row in process_metrics(r)]
    return pd.DataFrame(rows)


def run_profiles(cfg: SimConfig) -> pd.DataFrame:
    """All built-in profiles x all algorithms (each at its fixed default seed)."""
    frames = []
    for profile in PROFILE_SEEDS:
        res, _ = run_all(generate(profile), cfg, profile)
        df = summary_table(res, cfg.energy)
        df.insert(0, "profile", profile)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def winners(df: pd.DataFrame, group: str = "profile") -> pd.DataFrame:
    """Best algorithm(s) per group and metric (ties listed together)."""
    rows = []
    for g, sub in df.groupby(group, sort=False):
        row = {group: g}
        for key, (label, direction, _) in METRICS.items():
            target = sub[key].min() if direction == "min" else sub[key].max()
            row[label] = ", ".join(sub.loc[(sub[key] - target).abs() < TOL, "algorithm"])
        rows.append(row)
    return pd.DataFrame(rows)


def monte_carlo(profile: str, n_seeds: int, cfg: SimConfig, base_seed: int = 1000) -> pd.DataFrame:
    """Re-generate the profile with many seeds to test whether a result is robust."""
    frames = []
    for seed in range(base_seed, base_seed + n_seeds):
        res, _ = run_all(generate(profile, seed), cfg, profile)
        df = summary_table(res, cfg.energy)
        df.insert(0, "seed", seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def win_rates(mc: pd.DataFrame) -> pd.DataFrame:
    """How often EMI beats Round Robin (its closest relative) and every other baseline."""
    rows = []
    for key, (label, direction, _) in METRICS.items():
        piv = mc.pivot(index="seed", columns="algorithm", values=key)
        emi, rr, others = piv["EMI"], piv["Round Robin"], piv.drop(columns="EMI")
        if direction == "min":
            beats, loses = emi < rr - TOL, emi > rr + TOL
            best = emi <= others.min(axis=1) + TOL
            rank = (others.lt(emi - TOL, axis=0)).sum(axis=1) + 1
        else:
            beats, loses = emi > rr + TOL, emi < rr - TOL
            best = emi >= others.max(axis=1) - TOL
            rank = (others.gt(emi + TOL, axis=0)).sum(axis=1) + 1
        rows.append({
            "metric": label, "EMI mean": emi.mean(), "RR mean": rr.mean(),
            "EMI beats RR (% seeds)": 100 * beats.mean(), "EMI loses to RR (% seeds)": 100 * loses.mean(),
            "EMI best or tied (% seeds)": 100 * best.mean(), "EMI mean rank": rank.mean(),
        })
    return pd.DataFrame(rows)


def energy_sensitivity(processes: Sequence[Process], cfg: SimConfig, switch_js: Iterable[float],
                       profile_hint: Optional[str] = None) -> pd.DataFrame:
    """Re-price the same traces under different energy-per-switch assumptions."""
    res, _ = run_all(processes, cfg, profile_hint)
    rows = []
    for sj in switch_js:
        model = replace(cfg.energy, switch_j=float(sj))
        for name, r in res.items():
            rows.append({"switch_j": float(sj), "algorithm": name, "energy_j": summarize(r, model)["energy_j"]})
    return pd.DataFrame(rows)


def overhead_grid(processes: Sequence[Process], cfg: SimConfig, cs_costs: Iterable[int],
                  switch_js: Iterable[float], profile_hint: Optional[str] = None) -> pd.DataFrame:
    """% energy difference EMI vs Round Robin over (switch time cost x switch energy)."""
    rows = []
    for cs in cs_costs:
        res, _ = run_all(processes, replace(cfg, cs_cost=int(cs)), profile_hint)
        for sj in switch_js:
            model = replace(cfg.energy, switch_j=float(sj))
            e_emi = summarize(res["EMI"], model)["energy_j"]
            e_rr = summarize(res["Round Robin"], model)["energy_j"]
            rows.append({"cs_cost": int(cs), "switch_j": float(sj),
                         "diff_pct": (e_emi - e_rr) / e_rr * 100 if e_rr else 0.0})
    return pd.DataFrame(rows)


def quantum_sweep(processes: Sequence[Process], cfg: SimConfig, max_q: int) -> pd.DataFrame:
    rows = []
    for q in range(1, max(1, max_q) + 1):
        s = summarize(round_robin(processes, q, cfg.cs_cost), cfg.energy)
        s["quantum"] = q
        rows.append(s)
    return pd.DataFrame(rows)
