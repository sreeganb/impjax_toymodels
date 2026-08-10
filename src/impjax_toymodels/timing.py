"""Timing utilities for benchmarking IMP and JAX/BlackJAX workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingSnapshot:
    """Timing values in seconds."""

    wall_time: float
    cpu_time: float


def start_timing() -> TimingSnapshot:
    """Capture the current timing baseline."""
    return TimingSnapshot(wall_time=time.perf_counter(), cpu_time=time.process_time())


def elapsed_timing(start: TimingSnapshot) -> TimingSnapshot:
    """Compute elapsed wall and CPU time from a baseline."""
    return TimingSnapshot(
        wall_time=time.perf_counter() - start.wall_time,
        cpu_time=time.process_time() - start.cpu_time,
    )
