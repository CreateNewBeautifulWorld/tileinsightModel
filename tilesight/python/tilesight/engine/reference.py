"""Pure-Python reference engine (the C++ core in cpp/ must match it to 1e-9 rel).

Structured like the hardware it models — three blocks, one module each:
  shader.py        one core, composed into slices of copies, then into waves
  gmem.py          the fabric: how much of a shared level one active core gets
  memory_model.py  the memory slices (tile-level L2 per slice, HBM) and the on-chip buffer
This file is the scheduler that ties them together into a kernel time.

Implements docs/DESIGN.md §3:

  wave decomposition : concurrent = SMs * resident; full waves + one tail wave
  per-SM share        : shared lanes get rate = min(total/active_SMs, per_sm_cap)
  steady round        : R = max( bps * max_r sum_o u_r(o) ,  CP / stages ,  CPrec / consumers )  (Eq. 4')
                        unit latency (tensor core, SFU, SMEM ...) counts in the steady bound only
                        for `recurrent` actions: a loop-carried result must land before the next
                        iteration can use it, while independent operations (a tile MMA, a load)
                        keep several in flight and their latency only shows up in the fill
                        CP    = longest path through the iteration DAG, node weight
                                w(o) = max_r u_r(o) + lat(o)   (SMEM-buffer recycling recurrence:
                                load_i waits for the consumer of iteration i-stages)
                        CPrec = longest path through `recurrent` actions only
                                (loop-carried compute, overlapped by ping-pong consumers)
  wave time           : T = T_pro + T_fill + iters * R + T_epi             (Eq. 2')
                        T_fill = max(0, CP - R)  (first iteration's exposed latency)
  kernel time         : sum over waves + launch overhead
"""
from __future__ import annotations

import math

from ..gpuTilingPerfHWModel.spec import HardwareSpec, Lane
from ..ir.kernel import Action, Kernel, KernelResult
from .gmem import lane_time as _u_lane
from .shader import critical_path as _critical_path
from .shader import node_weight as _node_w
from .shader import wave_plan


def _u(a: Action, lane: Lane, active: int) -> float:
    return _u_lane(a.work.get(lane.name, 0.0), lane, active)


def _longest_path(actions: list[Action], weights: list[float]) -> float:
    return _critical_path(actions, weights)[0]


def _lane_detail(actions: list[Action], lane: Lane, active: int) -> str:
    """'lane:action' for the action putting the most work on `lane` (first max)."""
    best, arg = -1.0, 0
    for i, a in enumerate(actions):
        v = _u(a, lane, active)
        if v > best:
            best, arg = v, i
    return f"{lane.name}:{actions[arg].name}"


def _latency_detail(actions: list[Action], weights: list[float], lanes: list[Lane], active: int) -> str:
    """'latency:action(lane)' for the heaviest node on the critical path."""
    _, path = _critical_path(actions, weights)
    if not path:
        return "latency:none"
    j = max(path, key=lambda i: weights[i])          # first max in path order
    a = actions[j]
    best, lane = -1.0, "none"
    for l in lanes:
        v = _u(a, l, active)
        if v > best:
            best, lane = v, l.name
    if a.latency_s > best:
        lane = "mem-lat"
    return f"latency:{a.name}({lane})"


def _phase(actions: list[Action], lanes: list[Lane], active: int, bps: int,
           qf: float = 1.0) -> tuple[float, str, str]:
    """Non-looped phase (prologue/epilogue): max(resource bound, dependency chain)."""
    if not actions:
        return 0.0, "none", "none"
    sums = {l.name: bps * sum(_u(a, l, active) for a in actions) for l in lanes}
    lane, rb = max(sums.items(), key=lambda kv: kv[1])
    w = [_node_w(a, lanes, active, qf) for a in actions]
    cp = _longest_path(actions, w)
    if rb >= cp:
        return rb, lane, _lane_detail(actions, next(l for l in lanes if l.name == lane), active)
    return cp, "latency", _latency_detail(actions, w, lanes, active)


def evaluate(k: Kernel, cur_gpu_config: HardwareSpec) -> KernelResult:
    lanes = cur_gpu_config.lanes()
    launch = cur_gpu_config.launch_s
    if k.fixed_time_s is not None:          # communication kernels
        return KernelResult(k.name, k.kind, k.fixed_time_s + launch, "net",
                            {"net": 1.0}, {"comm": k.fixed_time_s, "launch": launch},
                            {"net": k.fixed_time_s, "launch": launch}, 0, k.meta,
                            {f"net:{k.meta.get('algo', 'comm')}": k.fixed_time_s, "launch": launch})

    sms = cur_gpu_config.sms
    resident = max(1, k.resident)
    conc = sms * resident
    full, tail = divmod(k.num_blocks, conc)
    # the shader block decides the wave structure: one core modelled, slices are copies
    waves = [(blocks, count) for blocks, count, _a, _b in wave_plan(k.num_blocks, sms, resident)]

    total = 0.0
    bd = {"prologue": 0.0, "fill": 0.0, "steady": 0.0, "epilogue": 0.0, "launch": launch}
    lim: dict[str, float] = {"launch": launch}
    det: dict[str, float] = {"launch": launch}

    for (blocks, count, active, bps) in wave_plan(k.num_blocks, sms, resident):
        # steady-state round (one iteration of every resident block on an SM)
        sums = {l.name: bps * sum(_u(a, l, active) for a in k.body) for l in lanes}
        lane, rb = max(sums.items(), key=lambda kv: kv[1])
        # queueing: a shared lane close to saturation does not just run out of bandwidth, it
        # also makes every access wait. M/D/1 waiting time W = u/(2(1-u)) * S, applied as a
        # latency multiplier (throughput is already capped by the resource bound).
        qc = float(cur_gpu_config.get("memory.queueing.coef") or 0.0)
        qmax = float(cur_gpu_config.get("memory.queueing.max_factor") or 3.0)
        qf = 1.0
        if qc > 0:
            base = max(rb, 1e-18)
            u = max((sums[l.name] / base for l in lanes if l.shared), default=0.0)
            u = min(u, 0.995)
            qf = min(qmax, 1.0 + qc * u / (1.0 - u))
        w = [_node_w(a, lanes, active, qf) for a in k.body]      # full weights: pipeline fill
        cp = _longest_path(k.body, w)
        # steady state: only loop-carried latency is exposed
        w_st = [_node_w(a, lanes, active, qf) - (0.0 if a.recurrent else a.latency_s * qf)
                for a in k.body]
        cp_st = _longest_path(k.body, w_st)
        w_rec = [wi if a.recurrent else 0.0 for a, wi in zip(k.body, w_st)]
        cp_rec = _longest_path(k.body, w_rec)
        lat_bound = max(cp_st / max(1, k.stages), cp_rec / max(1, k.consumers))
        if rb >= lat_bound:
            rnd, limiter = rb, lane
            d_steady = _lane_detail(k.body, next(l for l in lanes if l.name == lane), active)
        else:
            rnd, limiter = lat_bound, "latency"
            use_rec = cp_rec / max(1, k.consumers) > cp_st / max(1, k.stages)
            d_steady = _latency_detail(k.body, w_rec if use_rec else w_st, lanes, active)
        t_pro, l_pro, d_pro = _phase(k.prologue, lanes, active, bps, qf)
        t_epi, l_epi, d_epi = _phase(k.epilogue, lanes, active, bps, qf)
        t_fill = max(0.0, cp - rnd) if k.iters > 0 else 0.0
        t_steady = k.iters * rnd
        t_wave = t_pro + t_fill + t_steady + t_epi
        total += count * t_wave
        bd["prologue"] += count * t_pro
        bd["fill"] += count * t_fill
        bd["steady"] += count * t_steady
        bd["epilogue"] += count * t_epi
        for name, t in ((limiter, t_steady), ("latency", t_fill), (l_pro, t_pro), (l_epi, t_epi)):
            if t > 0:
                lim[name] = lim.get(name, 0.0) + count * t
        for name, t in ((d_steady, t_steady), ("latency:fill", t_fill), (d_pro, t_pro), (d_epi, t_epi)):
            if t > 0:
                det[name] = det.get(name, 0.0) + count * t

    total += launch

    # utilization: whole-machine busy fraction per lane
    util: dict[str, float] = {}
    for l in lanes:
        per_block = sum(a.work.get(l.name, 0.0) for a in k.prologue) + \
            k.iters * sum(a.work.get(l.name, 0.0) for a in k.body) + \
            sum(a.work.get(l.name, 0.0) for a in k.epilogue)
        busy = k.num_blocks * per_block
        busy = busy / l.total_rate if l.shared else busy / sms
        if busy > 0:
            util[l.name] = busy / total
    bottleneck = max(lim.items(), key=lambda kv: kv[1])[0]
    return KernelResult(k.name, k.kind, total, bottleneck, util, bd, lim,
                        len(waves) and (full + (1 if tail else 0)), k.meta, det)
