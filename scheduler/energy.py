"""Configurable, educational energy model.

    E = active_power * run_time + idle_power * idle_time + switch_energy * n_switches

Run/idle durations are converted to seconds with `time_unit_s` (seconds per tick), so with
power in watts the result is in joules. The per-switch energy term stands for the whole
context-switch cost (state save/restore, cache and TLB refill); the overhead interval itself
is not billed again. Values are assumptions to experiment with - NOT hardware measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

ENERGY_DISCLAIMER = "SIMULATED ENERGY ESTIMATE — NOT MEASURED HARDWARE POWER"


@dataclass(frozen=True)
class EnergyModel:
    active_w: float = 10.0       # CPU power while executing a process (watts)
    idle_w: float = 2.0          # CPU power while idle (watts)
    switch_j: float = 0.05       # energy per context switch (joules)
    time_unit_s: float = 0.01    # one simulation tick in seconds (10 ms)

    def __post_init__(self) -> None:
        for name in ("active_w", "idle_w", "switch_j", "time_unit_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class EnergyBreakdown:
    active_j: float
    idle_j: float
    switch_j: float

    @property
    def total_j(self) -> float:
        return self.active_j + self.idle_j + self.switch_j


def estimate_energy(model: EnergyModel, run_time: int, idle_time: int, switches: int) -> EnergyBreakdown:
    return EnergyBreakdown(
        active_j=model.active_w * run_time * model.time_unit_s,
        idle_j=model.idle_w * idle_time * model.time_unit_s,
        switch_j=model.switch_j * switches,
    )
