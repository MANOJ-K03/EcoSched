"""Reproducible workload profiles (fixed seeds) and DataFrame <-> Process helpers."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import pandas as pd

from scheduler.models import Process, ValidationError, as_int

COLUMNS = ["pid", "arrival", "burst", "priority", "category"]

# Priorities are drawn per category: interactive jobs get urgent (low) numbers, batch jobs middling,
# CPU-intensive jobs lowest urgency. This is an assumption of the generator, stated in the README.
_PRIO = {"interactive": (1, 3), "batch": (4, 6), "cpu-intensive": (6, 9)}

PROFILE_SEEDS: Dict[str, int] = {"Interactive": 11, "Batch": 22, "Mixed": 33, "High load": 44}


def classic_demo() -> List[Process]:
    """Fixed, hand-readable 6-process demo: short interactive jobs mixed with long batch jobs."""
    rows = [(1, 0, 8, 4, "batch"), (2, 1, 2, 1, "interactive"), (3, 2, 14, 6, "cpu-intensive"),
            (4, 3, 3, 2, "interactive"), (5, 6, 5, 3, "batch"), (6, 8, 2, 1, "interactive")]
    return [Process(*r) for r in rows]


def _mk(rng: random.Random, specs: List[tuple]) -> List[Process]:
    specs = sorted(specs, key=lambda s: (s[0], s[1]))  # by arrival, burst -> tidy PIDs
    out = []
    for i, (arrival, burst, cat) in enumerate(specs, start=1):
        lo, hi = _PRIO[cat]
        out.append(Process(i, arrival, burst, rng.randint(lo, hi), cat))
    return out


def generate(profile: str, seed: Optional[int] = None) -> List[Process]:
    """Build a profile. Same (profile, seed) always yields the same workload."""
    if profile not in PROFILE_SEEDS:
        raise ValidationError(f"Unknown profile {profile!r}. Choose from {list(PROFILE_SEEDS)}.")
    rng = random.Random(PROFILE_SEEDS[profile] if seed is None else seed)
    specs: List[tuple] = []
    if profile == "Interactive":      # many short jobs, spread-out arrivals
        for _ in range(12):
            specs.append((rng.randint(0, 24), rng.randint(1, 4), "interactive"))
    elif profile == "Batch":          # few long CPU-bound jobs, arriving together
        for _ in range(6):
            specs.append((rng.randint(0, 6), rng.randint(15, 30), rng.choice(["batch", "cpu-intensive"])))
    elif profile == "Mixed":          # ~60% short interactive, ~40% long batch/CPU jobs
        for _ in range(10):
            if rng.random() < 0.6:
                specs.append((rng.randint(0, 30), rng.randint(1, 5), "interactive"))
            else:
                specs.append((rng.randint(0, 15), rng.randint(12, 28), rng.choice(["batch", "cpu-intensive"])))
    else:                             # High load: many overlapping jobs, burst decides the category
        for _ in range(20):
            b = rng.randint(3, 15)
            specs.append((rng.randint(0, 10), b, "interactive" if b <= 5 else "batch" if b <= 10 else "cpu-intensive"))
    return _mk(rng, specs)


PRESETS = ["Classic demo"] + list(PROFILE_SEEDS)


def load_preset(name: str, seed: Optional[int] = None) -> List[Process]:
    return classic_demo() if name == "Classic demo" else generate(name, seed)


def processes_to_df(processes: List[Process]) -> pd.DataFrame:
    return pd.DataFrame([{c: getattr(p, c) for c in COLUMNS} for p in processes], columns=COLUMNS)


def df_to_processes(df: pd.DataFrame) -> List[Process]:
    """Convert the editable table to validated processes with row-numbered error messages."""
    out = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        try:
            out.append(Process(row["pid"], row["arrival"], row["burst"], row["priority"], row["category"]))
        except ValidationError as exc:
            raise ValidationError(f"Row {i}: {exc}") from None
    seen = set()
    for p in out:
        if p.pid in seen:
            raise ValidationError(f"Duplicate process ID {p.pid}. IDs must be unique.")
        seen.add(p.pid)
    return out


def next_pid(df: pd.DataFrame) -> int:
    ids = [as_int(v, "pid") for v in df["pid"] if pd.notna(v)]
    return (max(ids) + 1) if ids else 1
