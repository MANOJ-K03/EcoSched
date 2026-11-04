"""EcoMode Intelligence (EMI): explainable, rule-based, workload-aware Round Robin.

EMI is deliberately NOT machine learning. It
  1. measures the workload (category mix, burst distribution, switch cost),
  2. lets transparent rules vote for a mode (latency / balanced / efficiency),
  3. converts the winning mode into a time quantum, and
  4. runs Round Robin with two runtime refinements:
       * completion grace  - a job within `completion_grace` x quantum of finishing is allowed
                             to finish instead of paying a switch for a tiny remainder;
       * queue pressure    - if many jobs are waiting, the quantum is halved to keep response low.
Every threshold is configurable in `EMIConfig`. The policy is never assumed to be optimal; the
benchmark module measures when it wins and when it loses.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .algorithms import _prepare, rr_engine
from .models import Process, SimulationResult, ValidationError

MODES = ("latency", "balanced", "efficiency")


@dataclass(frozen=True)
class EMIConfig:
    short_burst_threshold: int = 5          # a job is "short" if burst <= this
    interactive_share_latency: float = 0.5  # >= this share of interactive jobs -> favour latency
    short_share_latency: float = 0.6        # >= this share of short jobs -> favour latency
    cpu_share_efficiency: float = 0.6       # >= this share of batch/cpu jobs -> favour efficiency
    spread_cv_balanced: float = 1.0         # burst coefficient of variation >= this -> favour balanced
    switch_cost_ratio: float = 2.0          # median burst < ratio x switch cost -> favour efficiency
    factor_latency: float = 0.5             # quantum = factor x median burst
    factor_balanced: float = 1.0
    factor_efficiency: float = 1.5
    min_quantum: int = 1
    max_quantum: int = 40
    switch_amortization: int = 4            # quantum >= this x switch cost (caps overhead at ~1/(k+1))
    completion_grace: float = 1.5           # 1.0 disables the grace rule
    queue_pressure_threshold: int = 8       # 0 disables queue-pressure shrinking
    mode_override: Optional[str] = None     # force a mode (still reports what rules said)

    def factor(self, mode: str) -> float:
        return {"latency": self.factor_latency, "balanced": self.factor_balanced,
                "efficiency": self.factor_efficiency}[mode]


@dataclass
class EMIDecision:
    mode: str
    quantum: int
    raw_quantum: int
    quantum_floor: int
    features: Dict[str, float]
    rules: List[Dict]          # evidence table: rule, fired, votes_for, points, detail
    scores: Dict[str, int]
    explanation: str
    overridden: bool = False
    config: EMIConfig = field(default_factory=EMIConfig)


def analyze_workload(processes: Iterable[Process], cfg: EMIConfig) -> Dict[str, float]:
    procs = list(processes)
    n = len(procs)
    if n == 0:
        return {"processes": 0}
    bursts = [p.burst for p in procs]
    mean = statistics.fmean(bursts)
    cv = (statistics.pstdev(bursts) / mean) if mean else 0.0
    span = max(p.arrival for p in procs) - min(p.arrival for p in procs)
    return {
        "processes": n,
        "mean_burst": mean,
        "median_burst": float(statistics.median(bursts)),
        "burst_cv": cv,
        "short_share": sum(b <= cfg.short_burst_threshold for b in bursts) / n,
        "interactive_share": sum(p.category == "interactive" for p in procs) / n,
        "cpu_share": sum(p.category != "interactive" for p in procs) / n,
        "arrival_span": span,
        "load_ratio": sum(bursts) / max(span, 1),   # demand per tick of arrival window (>1 = backlog)
    }


def decide(processes: Iterable[Process], cs_cost: int = 0, cfg: Optional[EMIConfig] = None,
           profile_hint: Optional[str] = None) -> EMIDecision:
    """Pick a mode and quantum. `profile_hint` is the workload profile name the user selected."""
    cfg = cfg or EMIConfig()
    if cfg.mode_override is not None and cfg.mode_override not in MODES:
        raise ValidationError(f"mode_override must be one of {MODES} or None.")
    f = analyze_workload(processes, cfg)
    if f["processes"] == 0:
        return EMIDecision("balanced", max(1, cfg.min_quantum), 1, 1, f, [], {m: 0 for m in MODES},
                           "Empty workload: defaulting to balanced mode.", config=cfg)

    scores = {m: 0 for m in MODES}
    rules: List[Dict] = []

    def rule(name: str, fired: bool, mode: str, points: int, detail: str) -> None:
        if fired:
            scores[mode] += points
        rules.append({"rule": name, "fired": fired, "votes_for": mode if fired else "-",
                      "points": points if fired else 0, "detail": detail})

    rule("Mostly interactive jobs", f["interactive_share"] >= cfg.interactive_share_latency, "latency", 2,
         f"interactive share {f['interactive_share']:.0%} vs threshold {cfg.interactive_share_latency:.0%}")
    rule("Mostly short bursts", f["short_share"] >= cfg.short_share_latency, "latency", 2,
         f"jobs with burst <= {cfg.short_burst_threshold}: {f['short_share']:.0%} vs threshold {cfg.short_share_latency:.0%}")
    rule("Mostly batch / CPU-bound jobs", f["cpu_share"] >= cfg.cpu_share_efficiency, "efficiency", 2,
         f"batch + CPU-intensive share {f['cpu_share']:.0%} vs threshold {cfg.cpu_share_efficiency:.0%}")
    rule("Wide spread of burst lengths", f["burst_cv"] >= cfg.spread_cv_balanced, "balanced", 1,
         f"burst coefficient of variation {f['burst_cv']:.2f} vs threshold {cfg.spread_cv_balanced:.2f}")
    rule("Switches are expensive vs job size", cs_cost > 0 and f["median_burst"] < cfg.switch_cost_ratio * cs_cost,
         "efficiency", 1, f"median burst {f['median_burst']:.1f} vs {cfg.switch_cost_ratio:g} x switch cost {cs_cost}")
    hint_mode = {"interactive": "latency", "batch": "efficiency", "mixed": "balanced",
                 "high load": "balanced"}.get((profile_hint or "").strip().lower())
    rule("Selected workload profile", hint_mode is not None, hint_mode or "balanced", 1,
         f"profile '{profile_hint}' suggests {hint_mode}" if hint_mode else "no built-in profile selected (custom/demo)")

    # Tie-break (deterministic, documented): balanced wins any tie or zero-evidence case,
    # otherwise latency beats efficiency.
    best = max(scores.values())
    leaders = [m for m in MODES if scores[m] == best]
    if len(leaders) == 1:
        mode = leaders[0]
    elif best == 0 or "balanced" in leaders:
        mode = "balanced"
    else:
        mode = "latency"
    overridden = cfg.mode_override is not None
    if overridden:
        mode = cfg.mode_override

    floor = max(cfg.min_quantum, cfg.switch_amortization * cs_cost)
    raw = max(1, math.ceil(cfg.factor(mode) * f["median_burst"]))
    quantum = max(floor, min(cfg.max_quantum, raw))

    fired = [r for r in rules if r["fired"]]
    why = "; ".join(r["rule"].lower() + f" ({r['detail']})" for r in fired) or "no rule fired strongly"
    explanation = (
        f"EMI chose {mode.upper()} mode{' (manual override)' if overridden else ''} with a quantum of {quantum} ticks. "
        f"Evidence: {why}. The quantum is {cfg.factor(mode):g} x the median burst ({f['median_burst']:.1f}) = {raw}, "
        f"kept between the switch-cost floor ({floor}) and the maximum ({cfg.max_quantum})."
    )
    return EMIDecision(mode, quantum, raw, floor, f, rules, scores, explanation, overridden, cfg)


def emi_schedule(processes: Iterable[Process], cs_cost: int = 0, cfg: Optional[EMIConfig] = None,
                 profile_hint: Optional[str] = None) -> Tuple[SimulationResult, EMIDecision]:
    """Run EMI: choose the quantum, then Round Robin with completion grace + queue pressure."""
    cfg = cfg or EMIConfig()
    procs, cs = _prepare(processes, cs_cost)
    d = decide(procs, cs, cfg, profile_hint)
    floor = max(cfg.min_quantum, cfg.switch_amortization * cs)

    def slice_fn(remaining: int, waiting: int) -> int:
        q = d.quantum
        if cfg.queue_pressure_threshold > 0 and waiting >= cfg.queue_pressure_threshold:
            q = max(floor, q // 2)
        if remaining <= q * cfg.completion_grace:
            return remaining
        return q

    params = {"mode": d.mode, "quantum": d.quantum}
    return rr_engine(procs, cs, slice_fn, "EMI", params), d
