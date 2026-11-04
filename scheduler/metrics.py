"""Metrics computed from the execution trace (never from algorithm internals).

Definitions (identical for every algorithm):
  completion = end of the process's last run segment
  turnaround = completion - arrival
  waiting    = turnaround - burst            (no I/O blocking is modelled)
  response   = first CPU start - arrival
  makespan   = trace end - trace start        (clock starts at the earliest arrival)
  utilization= run time / makespan            (idle and switch overhead count as elapsed time)
  throughput = completed processes / makespan
  fairness   = Jain's index over x_i = burst_i / (burst_i + waiting_i)
"""
from __future__ import annotations

from typing import Dict, Iterable, List

from .energy import EnergyModel, estimate_energy
from .models import SimulationResult


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def jain_index(values: Iterable[float]) -> float:
    """Jain's fairness index (sum x)^2 / (n * sum x^2): 1 = perfectly equal, 1/n = one hog.

    Here x_i is each process's CPU share while in the system. It treats *equal slowdown* as
    fair, which may not match every notion of fairness (e.g. priority-based service)."""
    xs = [float(v) for v in values]
    return safe_div(sum(xs) ** 2, len(xs) * sum(x * x for x in xs))


def count_context_switches(result: SimulationResult) -> int:
    """A switch = consecutive run segments of different processes with no idle gap between."""
    switches = 0
    prev_pid = None
    for seg in result.trace:
        if seg.is_idle:
            prev_pid = None
        elif seg.is_run:
            if prev_pid is not None and prev_pid != seg.pid:
                switches += 1
            prev_pid = seg.pid
    return switches


def process_metrics(result: SimulationResult) -> List[Dict]:
    first: Dict[int, int] = {}
    last: Dict[int, int] = {}
    for seg in result.run_segments:
        first.setdefault(seg.pid, seg.start)
        last[seg.pid] = seg.end
    rows = []
    for p in sorted(result.processes, key=lambda q: q.pid):
        if p.pid not in last:
            continue
        completion = last[p.pid]
        turnaround = completion - p.arrival
        rows.append({
            "pid": p.pid, "process": p.label, "category": p.category, "arrival": p.arrival,
            "burst": p.burst, "priority": p.priority, "first_start": first[p.pid],
            "completion": completion, "turnaround": turnaround,
            "waiting": turnaround - p.burst, "response": first[p.pid] - p.arrival,
        })
    return rows


def summarize(result: SimulationResult, energy: EnergyModel) -> Dict:
    rows = process_metrics(result)
    n = len(rows)
    run_t = sum(s.duration for s in result.trace if s.is_run)
    idle_t = sum(s.duration for s in result.trace if s.is_idle)
    switch_t = sum(s.duration for s in result.trace if s.is_switch)
    makespan = (result.trace[-1].end - result.trace[0].start) if result.trace else 0
    switches = count_context_switches(result)
    e = estimate_energy(energy, run_t, idle_t, switches)
    return {
        "algorithm": result.algorithm,
        "processes": n,
        "avg_wait": safe_div(sum(r["waiting"] for r in rows), n),
        "avg_turnaround": safe_div(sum(r["turnaround"] for r in rows), n),
        "avg_response": safe_div(sum(r["response"] for r in rows), n),
        "max_wait": max((r["waiting"] for r in rows), default=0),
        "utilization": safe_div(run_t, makespan),
        "throughput": safe_div(n, makespan),
        "context_switches": switches,
        "fairness": jain_index(safe_div(r["burst"], r["burst"] + r["waiting"]) for r in rows),
        "energy_j": e.total_j,
        "energy_active_j": e.active_j,
        "energy_idle_j": e.idle_j,
        "energy_switch_j": e.switch_j,
        "busy_time": run_t, "idle_time": idle_t, "switch_time": switch_t, "makespan": makespan,
    }


def verify_trace(result: SimulationResult) -> List[str]:
    """Independent consistency check. Returns a list of problems (empty = trace is valid)."""
    problems: List[str] = []
    procs = {p.pid: p for p in result.processes}
    if not result.trace:
        return [] if not procs else ["empty trace for a non-empty workload"]
    executed = {pid: 0 for pid in procs}
    completes = {pid: 0 for pid in procs}
    cursor = result.trace[0].start
    for seg in result.trace:
        if seg.duration <= 0:
            problems.append(f"non-positive duration at t={seg.start}")
        if seg.start != cursor:
            problems.append(f"gap or overlap: expected start {cursor}, got {seg.start}")
        cursor = seg.end
        if seg.is_switch and seg.duration != result.cs_cost:
            problems.append(f"switch at t={seg.start} lasts {seg.duration}, expected {result.cs_cost}")
        if seg.is_run:
            p = procs.get(seg.pid)
            if p is None:
                problems.append(f"unknown process {seg.pid}")
                continue
            if seg.start < p.arrival:
                problems.append(f"P{p.pid} runs at t={seg.start} before arrival {p.arrival}")
            executed[p.pid] += seg.duration
            completes[p.pid] += int(seg.completes)
    for pid, p in procs.items():
        if executed[pid] != p.burst:
            problems.append(f"P{pid} executed {executed[pid]} of {p.burst}")
        if completes[pid] != 1:
            problems.append(f"P{pid} completed {completes[pid]} times")
    return problems
