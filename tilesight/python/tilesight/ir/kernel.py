"""Engine IR — the contract between kernel lowerings and the engine (Python or C++).

A `Kernel` is a *tile execution plan* already lowered to numbers:
  * num_blocks thread blocks (tiles), each running `iters` iterations of `body`
  * `prologue` runs once per block before the loop, `epilogue` once after
  * `resident` = blocks co-resident per SM (from the occupancy solver)
  * `stages`   = software-pipeline depth (multi-buffered loads)

Each `Action` carries a work vector over lanes (see gpuTilingHWModel/spec.py for units) and DAG
dependencies on earlier actions *of the same list* (listed in topological order).
`recurrent=True` marks loop-carried compute (e.g. online-softmax state, attention
accumulator) whose chain can only be overlapped by `consumers` ping-pong warpgroups
(docs/DESIGN.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    name: str
    work: dict[str, float]
    deps: list[int] = field(default_factory=list)
    recurrent: bool = False
    latency_s: float = 0.0


@dataclass
class Kernel:
    name: str
    kind: str                       # gemm | grouped_gemm | attention | elementwise | comm
    num_blocks: int
    iters: int
    stages: int
    resident: int
    body: list[Action]
    prologue: list[Action] = field(default_factory=list)
    epilogue: list[Action] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)   # flops, bytes, tile, cache stats ...
    fixed_time_s: float | None = None   # comm kernels: time computed analytically (alpha-beta)
    consumers: int = 1                  # ping-pong consumer warpgroups (warp specialization)


@dataclass
class KernelResult:
    name: str
    kind: str
    time_s: float
    bottleneck: str                 # dominant limiter (lane name | latency | launch | net)
    util: dict[str, float]          # busy fraction per lane over the kernel time
    breakdown: dict[str, float]     # prologue/fill/steady/epilogue/launch seconds
    limiter_time: dict[str, float]  # seconds attributed to each limiter
    waves: int
    meta: dict[str, Any] = field(default_factory=dict)
    # fine-grained attribution: "lane:action" (e.g. ddr:load:weight, smem:mma, tc:gemm_qk)
    # or "latency:action(lane|mem-lat)" (critical-path node) -> seconds
    limiter_detail: dict[str, float] = field(default_factory=dict)
