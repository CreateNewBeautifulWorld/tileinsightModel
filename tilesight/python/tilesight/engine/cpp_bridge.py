"""Pack Python IR into the C++ core's lane-indexed structs and unpack results."""
from __future__ import annotations

from ..ir.kernel import Kernel, KernelResult

_LANE_CACHE: dict = {}


def _lanes(core, hw):
    key = hw.fingerprint
    if key not in _LANE_CACHE:
        py = hw.lanes()
        cl = []
        for l in py:
            c = core.Lane()
            c.name, c.shared, c.total_rate, c.per_sm_cap = l.name, l.shared, float(l.total_rate), float(min(l.per_sm_cap, 1e300))
            cl.append(c)
        _LANE_CACHE[key] = ([l.name for l in py], cl)
    return _LANE_CACHE[key]


def _acts(core, acts, names):
    out = []
    for a in acts:
        c = core.Action()
        c.name = a.name
        c.work = [float(a.work.get(n, 0.0)) for n in names]
        c.deps = list(a.deps)
        c.recurrent = a.recurrent
        c.latency = float(a.latency_s)
        out.append(c)
    return out


def pack(core, k: Kernel, names, hw_queue=(0.0, 3.0)):
    c = core.Kernel()
    c.name, c.num_blocks, c.iters = k.name, int(k.num_blocks), int(k.iters)
    c.stages, c.resident, c.consumers = int(k.stages), int(k.resident), int(k.consumers)
    c.body, c.prologue, c.epilogue = _acts(core, k.body, names), _acts(core, k.prologue, names), _acts(core, k.epilogue, names)
    if k.fixed_time_s is not None:
        c.fixed, c.fixed_time = True, float(k.fixed_time_s)
    c.queue_coef = float(hw_queue[0])
    c.queue_max = float(hw_queue[1])
    return c


def unpack(r, k: Kernel) -> KernelResult:
    det = dict(r.limiter_detail)
    if k.fixed_time_s is not None:                      # comm label uses the algorithm like Python
        det = {f"net:{k.meta.get('algo', 'comm')}": k.fixed_time_s, "launch": det.get("launch", 0.0)}
    return KernelResult(k.name, k.kind, r.time, r.bottleneck, dict(r.util), dict(r.breakdown),
                        dict(r.limiter_time), int(r.waves), k.meta, det)


def evaluate_cpp(core, k: Kernel, hw) -> KernelResult:
    names, lanes = _lanes(core, hw)
    q = (float(hw.get("memory.queueing.coef") or 0.0), float(hw.get("memory.queueing.max_factor") or 3.0))
    return unpack(core.evaluate(pack(core, k, names, q), lanes, hw.sms, hw.launch_s), k)


def evaluate_batch_cpp(core, ks: list[Kernel], hw, threads: int = 8) -> list[KernelResult]:
    names, lanes = _lanes(core, hw)
    q = (float(hw.get("memory.queueing.coef") or 0.0), float(hw.get("memory.queueing.max_factor") or 3.0))
    res = core.evaluate_batch([pack(core, k, names, q) for k in ks], lanes, hw.sms, hw.launch_s, threads)
    return [unpack(r, k) for r, k in zip(res, ks)]
