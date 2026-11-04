"""Process model, execution trace and input validation.

Time is measured in integer "time units" (ticks). A *trace* is the ordered list of
`Segment`s the CPU went through; every metric in EcoSched is computed from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

CATEGORIES = ("interactive", "batch", "cpu-intensive")
PRIORITY_MIN, PRIORITY_MAX = 0, 99  # lower number = higher priority (Unix "nice" style)


class ValidationError(ValueError):
    """Raised when a process, workload or simulation parameter is invalid."""


def as_int(value: Any, name: str) -> int:
    """Convert numbers (incl. 3.0 or numpy ints) to int; reject 1.5, NaN, text, bools."""
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a whole number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a whole number, got {value!r}.") from None
    if number != number or not number.is_integer():  # NaN or fractional
        raise ValidationError(f"{name} must be a whole number, got {value!r}.")
    return int(number)


def normalize_category(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if text in ("cpu", "cpu-bound", "cpu-intensive"):
        text = "cpu-intensive"
    if text not in CATEGORIES:
        raise ValidationError(f"Category must be one of {', '.join(CATEGORIES)}; got {value!r}.")
    return text


@dataclass(frozen=True)
class Process:
    """One schedulable job. Validation runs on construction."""

    pid: int
    arrival: int
    burst: int
    priority: int = 5
    category: str = "batch"

    def __post_init__(self) -> None:
        pid = as_int(self.pid, "Process ID")
        arrival = as_int(self.arrival, f"Arrival time of P{pid}")
        burst = as_int(self.burst, f"Burst time of P{pid}")
        priority = as_int(self.priority, f"Priority of P{pid}")
        if pid < 0:
            raise ValidationError(f"Process ID must be non-negative, got {pid}.")
        if arrival < 0:
            raise ValidationError(f"Arrival time of P{pid} must be >= 0, got {arrival}.")
        if burst <= 0:
            raise ValidationError(f"Burst time of P{pid} must be > 0, got {burst}.")
        if not PRIORITY_MIN <= priority <= PRIORITY_MAX:
            raise ValidationError(
                f"Priority of P{pid} must be between {PRIORITY_MIN} and {PRIORITY_MAX}, got {priority}."
            )
        for field_name, val in (("pid", pid), ("arrival", arrival), ("burst", burst), ("priority", priority)):
            object.__setattr__(self, field_name, val)
        object.__setattr__(self, "category", normalize_category(self.category))

    @property
    def label(self) -> str:
        return f"P{self.pid}"


def validate_processes(processes: Iterable[Process]) -> List[Process]:
    """Return a list of processes after checking types and unique IDs."""
    procs = list(processes)
    seen = set()
    for p in procs:
        if not isinstance(p, Process):
            raise ValidationError(f"Expected Process objects, got {type(p).__name__}.")
        if p.pid in seen:
            raise ValidationError(f"Duplicate process ID: {p.pid}. IDs must be unique.")
        seen.add(p.pid)
    return procs


@dataclass(frozen=True)
class Segment:
    """One contiguous CPU interval.

    kind: "run"    - a process executes
          "idle"   - no process is ready; CPU idles
          "switch" - context-switch overhead (pid = the process being dispatched)
    preempted: the run ended while the process still had work left (quantum expiry or
               a better process arrived). completes: the run finished the process.
    """

    kind: str
    start: int
    end: int
    pid: Optional[int] = None
    preempted: bool = False
    completes: bool = False

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def is_idle(self) -> bool:
        return self.kind == "idle"

    @property
    def is_run(self) -> bool:
        return self.kind == "run"

    @property
    def is_switch(self) -> bool:
        return self.kind == "switch"


@dataclass
class SimulationResult:
    algorithm: str
    processes: List[Process]
    trace: List[Segment]
    cs_cost: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_segments(self) -> List[Segment]:
        return [s for s in self.trace if s.is_run]
