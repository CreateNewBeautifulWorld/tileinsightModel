"""shader slice — one core modelled, a slice composed of copies of it.

Following the paper: only ONE core is ever evaluated. A slice is `cores_per_slice` copies of
that core sharing an L1, and the GPU is `slices` copies of that slice; the grid is decomposed
into waves and the copies are staggered in time rather than simulated separately. What varies
between waves is only how many cores are active, because that changes each core's share of the
shared levels (gmem).

The three pieces here:
  `round_time`    one steady-state K-loop round on one core: resource bound vs the two
                  recurrences (buffer recycling / stages, loop-carried compute / consumers)
  `phase_time`    a non-looped phase (prologue, epilogue)
  `wave_plan`     grid -> waves, and for each wave how many cores are active and how many
                  blocks each of them holds
"""
from __future__ import annotations

import math

from ..hw.spec import Lane
from ..ir.kernel import Action
from .gmem import lane_time


def wave_plan(num_blocks: int, cores: int, resident: int) -> list[tuple[int, int, int, int]]:
    """[(blocks in the wave, how many such waves, active cores, blocks per active core)]."""
    resident = max(1, resident)
    conc = cores * resident
    full, tail = divmod(num_blocks, conc)
    out = []
    for blocks, count in ((conc, full), (tail, 1)):
        if not blocks or not count:
            continue
        active = min(cores, math.ceil(blocks / resident))
        out.append((blocks, count, active, math.ceil(blocks / active)))
    return out


def lane_sums(actions: list[Action], lanes: list[Lane], active: int, bps: int) -> dict[str, float]:
    """Seconds each lane is occupied on one core, for one round of all its resident blocks."""
    return {l.name: bps * sum(lane_time(a.work.get(l.name, 0.0), l, active) for a in actions)
            for l in lanes}


def node_weight(a: Action, lanes: list[Lane], active: int, qf: float = 1.0) -> float:
    return max((lane_time(a.work.get(l.name, 0.0), l, active) for l in lanes), default=0.0) \
        + a.latency_s * qf


def critical_path(actions: list[Action], weights: list[float]) -> tuple[float, list[int]]:
    """Longest path (the actions are in topological order) and the node indices on it."""
    n = len(actions)
    best, pred = [0.0] * n, [-1] * n
    for i, a in enumerate(actions):
        start, p = 0.0, -1
        for d in a.deps:
            if best[d] > start:
                start, p = best[d], d
        best[i], pred[i] = start + weights[i], p
    if n == 0:
        return 0.0, []
    end = max(range(n), key=lambda i: best[i])
    path = []
    while end >= 0:
        path.append(end)
        end = pred[end]
    return best[path[0]], path[::-1]
