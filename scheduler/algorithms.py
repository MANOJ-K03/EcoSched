"""Baseline CPU scheduling algorithms: FCFS, SJF, Round Robin, Priority (+SRTF reference).

Shared rules (documented, tested):
* Timeline starts at the earliest arrival (the simulation clock starts when work first exists).
* Tie-breaking is always (arrival time, process ID) after the algorithm's own key.
* Context switch: counted when the CPU goes from running process A straight to running a
  different process B with no idle gap in between. The first dispatch and a dispatch after
  an idle gap are NOT switches. If cs_cost > 0 each switch inserts a "switch" segment.
* The dispatch decision is made BEFORE the switch overhead; arrivals during the overhead are
  queued and considered at the next scheduling point.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .models import Process, Segment, SimulationResult, ValidationError, as_int, validate_processes

SliceFn = Callable[[int, int], int]  # (remaining_burst, waiting_count) -> slice length


class _TraceBuilder:
    """Builds a gap-free trace; merges back-to-back runs of the same process."""

    def __init__(self, cs_cost: int, start: int) -> None:
        self.t = start
        self.cs = cs_cost
        self.segs: List[Segment] = []
        self.last_pid: Optional[int] = None  # None after idle / at start

    def idle_until(self, t_new: int) -> None:
        if t_new > self.t:  # only a REAL idle gap breaks continuity (next dispatch is not a switch)
            self.segs.append(Segment("idle", self.t, t_new))
            self.t = t_new
            self.last_pid = None

    def dispatch(self, pid: int) -> None:
        if self.last_pid is not None and self.last_pid != pid and self.cs > 0:
            self.segs.append(Segment("switch", self.t, self.t + self.cs, pid))
            self.t += self.cs

    def run(self, pid: int, duration: int, remaining_after: int) -> None:
        start, end = self.t, self.t + duration
        pre = remaining_after > 0
        prev = self.segs[-1] if self.segs else None
        if prev is not None and prev.is_run and prev.pid == pid and prev.end == start:
            self.segs[-1] = replace(prev, end=end, preempted=pre, completes=not pre)
        else:
            self.segs.append(Segment("run", start, end, pid, preempted=pre, completes=not pre))
        self.t = end
        self.last_pid = pid


def _prepare(processes: Iterable[Process], cs_cost: int) -> Tuple[List[Process], int]:
    procs = validate_processes(processes)
    cs = as_int(cs_cost, "Context-switch cost")
    if cs < 0:
        raise ValidationError("Context-switch cost must be >= 0.")
    return procs, cs


def _by_arrival(procs: List[Process]) -> List[Process]:
    return sorted(procs, key=lambda p: (p.arrival, p.pid))


def fcfs(processes: Iterable[Process], cs_cost: int = 0) -> SimulationResult:
    """Non-preemptive First Come First Served: run in arrival order (ties by PID)."""
    procs, cs = _prepare(processes, cs_cost)
    if not procs:
        return SimulationResult("FCFS", procs, [], cs)
    order = _by_arrival(procs)
    b = _TraceBuilder(cs, order[0].arrival)
    for p in order:
        b.idle_until(p.arrival)
        b.dispatch(p.pid)
        b.run(p.pid, p.burst, 0)
    return SimulationResult("FCFS", procs, b.segs, cs)


def sjf(processes: Iterable[Process], cs_cost: int = 0) -> SimulationResult:
    """Non-preemptive Shortest Job First among the processes that have already arrived."""
    procs, cs = _prepare(processes, cs_cost)
    if not procs:
        return SimulationResult("SJF", procs, [], cs)
    pending = _by_arrival(procs)
    b = _TraceBuilder(cs, pending[0].arrival)
    while pending:
        ready = [p for p in pending if p.arrival <= b.t]
        if not ready:
            b.idle_until(min(p.arrival for p in pending))
            continue
        p = min(ready, key=lambda q: (q.burst, q.arrival, q.pid))
        pending.remove(p)
        b.dispatch(p.pid)
        b.run(p.pid, p.burst, 0)
    return SimulationResult("SJF", procs, b.segs, cs)


def rr_engine(procs: List[Process], cs: int, slice_fn: SliceFn, name: str, params: dict) -> SimulationResult:
    """Generic Round Robin loop; `slice_fn` decides how long each dispatch may run.

    Queue rule: processes that arrive at or before the moment a slice ends are placed in the
    ready queue BEFORE the pre-empted process is re-queued (standard textbook convention).
    """
    if not procs:
        return SimulationResult(name, procs, [], cs, params)
    order = _by_arrival(procs)
    remaining: Dict[int, int] = {p.pid: p.burst for p in order}
    b = _TraceBuilder(cs, order[0].arrival)
    queue: deque = deque()
    idx = 0
    done = 0

    def admit() -> None:
        nonlocal idx
        while idx < len(order) and order[idx].arrival <= b.t:
            queue.append(order[idx])
            idx += 1

    while done < len(order):
        admit()
        if not queue:
            b.idle_until(order[idx].arrival)
            continue
        p = queue.popleft()
        b.dispatch(p.pid)
        admit()  # arrivals during switch overhead queue up ahead of p's re-queue
        rem = remaining[p.pid]
        run = int(slice_fn(rem, len(queue)))
        if not 1 <= run <= rem:
            raise ValidationError(f"Slice function returned {run}, expected 1..{rem}.")
        remaining[p.pid] = rem - run
        b.run(p.pid, run, remaining[p.pid])
        admit()
        if remaining[p.pid] > 0:
            queue.append(p)
        else:
            done += 1
    return SimulationResult(name, procs, b.segs, cs, params)


def round_robin(processes: Iterable[Process], quantum: int = 4, cs_cost: int = 0) -> SimulationResult:
    """Preemptive Round Robin with a fixed time quantum."""
    procs, cs = _prepare(processes, cs_cost)
    q = as_int(quantum, "Time quantum")
    if q <= 0:
        raise ValidationError("Time quantum must be a positive integer.")
    return rr_engine(procs, cs, lambda rem, waiting: min(q, rem), "Round Robin", {"quantum": q})


def _preemptive_by_key(procs: List[Process], cs: int, key, name: str) -> SimulationResult:
    """Preemptive scheduler: always run the ready process with the smallest `key`.

    Preemption can only happen when a new process arrives, so a run lasts until completion or
    the next arrival. After a context-switch overhead that swallowed an arrival, the process is
    guaranteed 1 tick of progress and then the scheduler re-evaluates.
    """
    if not procs:
        return SimulationResult(name, procs, [], cs)
    order = _by_arrival(procs)
    remaining = {p.pid: p.burst for p in order}
    b = _TraceBuilder(cs, order[0].arrival)
    while any(r > 0 for r in remaining.values()):
        ready = [p for p in order if p.arrival <= b.t and remaining[p.pid] > 0]
        if not ready:
            b.idle_until(min(p.arrival for p in order if remaining[p.pid] > 0))
            continue
        decided_at = b.t
        p = min(ready, key=lambda q: key(q, remaining[q.pid]))
        b.dispatch(p.pid)
        future = [q.arrival for q in order if q.arrival > decided_at]
        rem = remaining[p.pid]
        if not future:
            run = rem
        elif min(future) <= b.t:
            run = 1
        else:
            run = min(rem, min(future) - b.t)
        remaining[p.pid] = rem - run
        b.run(p.pid, run, remaining[p.pid])
    return SimulationResult(name, procs, b.segs, cs)


def priority_preemptive(processes: Iterable[Process], cs_cost: int = 0) -> SimulationResult:
    """Preemptive Priority: LOWER number = HIGHER priority; ties by arrival then PID."""
    procs, cs = _prepare(processes, cs_cost)
    return _preemptive_by_key(procs, cs, lambda p, rem: (p.priority, p.arrival, p.pid), "Priority")


def srtf(processes: Iterable[Process], cs_cost: int = 0) -> SimulationResult:
    """Shortest Remaining Time First - a reference baseline, optimal for average waiting time
    when switching is free. Not one of the five required policies."""
    procs, cs = _prepare(processes, cs_cost)
    return _preemptive_by_key(procs, cs, lambda p, rem: (rem, p.arrival, p.pid), "SRTF (reference)")
