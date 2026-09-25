"""Sizing and *using* a large on-chip buffer.

Once the buffer is a multiple of L2 (say 4x or more), what you put in it matters more than
how big it is. `memory.sram.policy` picks the rule:

  cache : one shared pool, every tensor class may occupy it (default)
  pin   : fixed shares per class, `memory.sram.pin: {weight: .., kv: .., act: ..}`

Two more knobs matter once the buffer is large:
  bypass_l2 : the buffer feeds SMEM directly, so resident bytes stop occupying the L2
              datapath (otherwise L2 bandwidth becomes the next wall)
  prefetch  : resident bytes are staged ahead of use, so they cost no exposed memory latency
  costream  : keep HBM busy in parallel instead of serving everything from the buffer — the
              two levels are independent datapaths, so their bandwidths add

`optimize_buffer` searches the share simplex for the split that minimises step time at a
given capacity, and `capacity_curve` shows what each capacity buys. Both re-lower the
kernels per point, so tile choices adapt to the new memory hierarchy.
"""
from __future__ import annotations

from itertools import product

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import HardwareSpec
from tilesight.interfaceAndModelRun.run_config import RunConfig
from tilesight.interfaceAndModelRun.runner import run_model
from tilesight.interfaceAndModelRun.spec import ModelSpec


def with_buffer(cur_gpu_config: HardwareSpec, capacity_MB: float, *, bandwidth_TBps: float | None = None,
                policy: str = "cache", pin: dict | None = None, latency_ns: float = 400,
                assoc: int = 16, efficiency: float = 0.9, bypass_l2: bool = False,
                prefetch: bool = False, costream: bool = False) -> HardwareSpec:
    """Return `cur_gpu_config` with an extra on-chip shared buffer configured."""
    if capacity_MB <= 0:
        raw = {k: v for k, v in cur_gpu_config.raw.items()}
        mem = dict(raw.get("memory", {}))
        mem.pop("sram", None)
        raw["memory"] = mem
        return HardwareSpec(raw)
    bw = bandwidth_TBps if bandwidth_TBps is not None else cur_gpu_config.get("memory.l2.bandwidth_TBps", 15.0)
    sram = {"capacity_MB": capacity_MB, "effective_capacity_MB": capacity_MB,
            "bandwidth_TBps": bw, "latency_ns": latency_ns, "assoc": assoc,
            "per_sm_max_GBps": cur_gpu_config.get("memory.l2.per_sm_max_GBps", 180), "policy": policy,
            "bypass_l2": bypass_l2, "prefetch": prefetch, "costream": costream}
    if pin:
        sram["pin"] = pin
    return cur_gpu_config.override({"memory.sram": sram, "efficiency.sram": efficiency})


def capacity_curve(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig,
                   capacities_MB=(0, 32, 64, 128, 256, 512, 1024, 2048, 4096), **kw):
    rows = []
    l2 = cur_gpu_config.get("memory.l2.capacity_MB", 1)
    for cap in capacities_MB:
        h = with_buffer(cur_gpu_config, cap, **kw) if cap else with_buffer(cur_gpu_config, 0)
        rep = run_model(model, h, rc)
        rows.append({"capacity_MB": cap, "x_L2": round(cap / l2, 2), "step_ms": rep.step_time_s * 1e3,
                     "tok_s_gpu": rep.tokens_per_s_per_gpu,
                     "top_bound": next(iter(rep.detail_breakdown()), "-")})
    base = rows[0]["step_ms"]
    for r in rows:
        r["speedup"] = round(base / r["step_ms"], 3)
    return rows


def optimize_buffer(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig, capacity_MB: float,
                    steps: int = 4, classes=("weight", "kv", "act"), progress=None, **kw):
    """Grid-search the pin shares (plus the shared-cache policy) at a fixed capacity.

    Returns (best, rows). `steps` controls the grid: shares are multiples of 1/steps that
    sum to 1, so steps=4 gives 15 splits for three classes."""
    splits = [("cache", None)]
    grid = [g for g in product(range(steps + 1), repeat=len(classes)) if sum(g) == steps]
    for g in grid:
        splits.append(("pin", {c: v / steps for c, v in zip(classes, g)}))
    cands = [(p, pin, byp, pre, cos) for (p, pin) in splits for byp in (False, True)
             for pre in (False, True) for cos in (False, True)]
    rows = []
    for i, (policy, pin, byp, pre, cos) in enumerate(cands):
        if progress:
            progress(i, len(cands), f"{policy} {pin or ''} bypass={byp} prefetch={pre} costream={cos}")
        h = with_buffer(cur_gpu_config, capacity_MB, policy=policy, pin=pin, bypass_l2=byp, prefetch=pre,
                        costream=cos, **kw)
        rep = run_model(model, h, rc)
        rows.append({"policy": policy, **{f"pin_{c}": (pin or {}).get(c, "") for c in classes},
                     "bypass_l2": byp, "prefetch": pre, "costream": cos,
                     "step_ms": rep.step_time_s * 1e3, "tok_s_gpu": rep.tokens_per_s_per_gpu,
                     "top_bound": next(iter(rep.detail_breakdown()), "-")})
    none_ms = run_model(model, with_buffer(cur_gpu_config, 0), rc).step_time_s * 1e3
    for r in rows:
        r["speedup_vs_no_buffer"] = round(none_ms / r["step_ms"], 3)
    rows.sort(key=lambda r: r["step_ms"])
    return rows[0], rows


# --------------------------------------------------------------------------- closed form
CLASSES = ("weight", "kv", "act")


def _class_of(op) -> str:
    if op.op.kind in ("attn_decode", "attn_prefill"):
        return "kv"
    if op.op.weight_bytes > 0:
        return "weight"
    return "act"


def profile_classes(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig) -> dict:
    """Per-class HBM traffic per step and footprint per GPU — the closed form's inputs."""
    rep = run_model(model, with_buffer(cur_gpu_config, 0), rc)
    ddr_rate = cur_gpu_config.get("memory.ddr.bandwidth_TBps") * 1e12 * cur_gpu_config.eff("ddr")
    traffic = dict.fromkeys(CLASSES, 0.0)
    for o in rep.ops:
        for k in o.kernels:
            traffic[_class_of(o)] += k.util.get("ddr", 0.0) * ddr_rate * k.time_s * o.repeat
    m = rep.memory
    foot = {"weight": m.weights_GB * 1e9, "kv": max(1.0, m.kv_cache_GB * 1e9),
            "act": max(1.0, m.activations_GB * 1e9)}
    return {"traffic_bytes": traffic, "footprint_bytes": foot,
            "baseline_step_s": rep.step_time_s,
            "value_per_byte": {c: traffic[c] / foot[c] for c in CLASSES}}


def best_alloc(prof: dict, capacity_MB: float) -> dict:
    """Optimal pin shares without re-running the model.

    Pinning x_c bytes of class c removes T_c * min(1, x_c/F_c) bytes of HBM traffic, so the
    saving per byte of capacity is the constant T_c/F_c until the class is fully pinned.
    Sorting by that ratio and filling greedily is therefore exactly optimal for the traffic
    objective (it is a fractional knapsack with linear, capped value per class)."""
    cap = capacity_MB * 1024 * 1024
    left, take = cap, dict.fromkeys(CLASSES, 0.0)
    saved = 0.0
    for c in sorted(CLASSES, key=lambda c: -prof["value_per_byte"][c]):
        t = min(left, prof["footprint_bytes"][c])
        take[c], saved, left = t, saved + prof["value_per_byte"][c] * t, left - t
        if left <= 0:
            break
    return {"pin": {c: (take[c] / cap if cap else 0.0) for c in CLASSES},
            "bytes": take, "hbm_saved_bytes_per_step": saved, "unused_bytes": max(0.0, left),
            "order": sorted(CLASSES, key=lambda c: -prof["value_per_byte"][c])}


def l2_tradeoff(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig, total_MB: float,
                buffer_shares=(0.0, 0.25, 0.5, 0.75, 0.9, 0.97), bw_per_MB: float | None = None,
                progress=None, **kw):
    """Spend a fixed on-chip SRAM budget on L2 vs the shared buffer.

    Same silicon, two ways: a small fast L2 or a big slower staging buffer. Bandwidth is
    scaled with capacity (`bw_per_MB`, default: keep the part's L2 bandwidth density), so a
    smaller L2 is also proportionally narrower — which is what makes this a real trade.
    """
    l2_mb = cur_gpu_config.get("memory.l2.capacity_MB")
    l2_bw = cur_gpu_config.get("memory.l2.bandwidth_TBps")
    dens = bw_per_MB if bw_per_MB is not None else l2_bw / l2_mb
    rows = []
    for i, share in enumerate(buffer_shares):
        if progress:
            progress(i, len(buffer_shares), f"buffer share {share:.0%}")
        buf_mb = total_MB * share
        new_l2 = max(1.0, total_MB - buf_mb)
        h = cur_gpu_config.override({"memory.l2.capacity_MB": new_l2,
                         "memory.l2.effective_capacity_MB": new_l2 * 0.66,
                         "memory.l2.bandwidth_TBps": max(0.5, dens * new_l2)})
        if buf_mb > 0:
            h = with_buffer(h, buf_mb, bandwidth_TBps=max(0.5, dens * buf_mb * 0.5), **kw)
        rep = run_model(model, h, rc)
        rows.append({"buffer_share": share, "l2_MB": round(new_l2, 1), "buffer_MB": round(buf_mb, 1),
                     "l2_TBps": round(max(0.5, dens * new_l2), 1),
                     "step_ms": rep.step_time_s * 1e3,
                     "top_bound": next(iter(rep.detail_breakdown()), "-")})
    base = rows[0]["step_ms"]
    for r in rows:
        r["speedup"] = round(base / r["step_ms"], 3)
    return rows
