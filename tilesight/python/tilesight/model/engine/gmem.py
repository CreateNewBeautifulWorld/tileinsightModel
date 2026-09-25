"""gmem — the fabric between the shader slices and the memory slices.

It owns one question: when N cores are pulling on a shared level, how much of it does one core
get? That is the only place the whole-GPU bandwidths turn into a per-core rate:

    per-core rate = min( total rate / active cores ,      # fair share of the level
                         per-core ceiling ,               # what one core can physically pull
                         outstanding lines x line / latency )   # Little's law (in the ceiling)

Upstream (towards the shader slices) and downstream (towards the memory slices) can be
configured separately, read and write apart, via `memory.gmem.*`; when they are absent the
level's own bandwidth is used, which is what every datasheet-form preset does.
"""
from __future__ import annotations

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import Lane


def per_core_rate(lane: Lane, active_cores: int) -> float:
    """Bandwidth one shader core sees on a shared lane while `active_cores` are running."""
    return min(lane.total_rate / max(1, active_cores), lane.per_sm_cap)


def lane_time(work: float, lane: Lane, active_cores: int) -> float:
    """Seconds this much work occupies the lane, from one core's point of view."""
    if work == 0.0:
        return 0.0
    if not lane.shared:
        return work                       # per-core lanes already carry seconds
    return work / per_core_rate(lane, active_cores)


def direction_of(lane_name: str) -> str:
    """Which side of the fabric a lane sits on (used for reporting)."""
    if lane_name in ("ddr", "sram"):
        return "downstream"
    if lane_name in ("l2",):
        return "downstream/upstream boundary"
    return "upstream"
