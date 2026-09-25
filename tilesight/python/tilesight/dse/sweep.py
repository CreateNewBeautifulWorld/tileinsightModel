"""Design-space exploration: vary hardware fields, re-run the model, find requirements.

    sweep(model, cur_gpu_config, rc, "memory.ddr.bandwidth_TBps", [4, 6, 8, 12, 16])
    required_value(model, cur_gpu_config, rc, "memory.ddr.bandwidth_TBps", target_step_ms=15, lo=2, hi=64)

Any dotted YAML path works (tc peak, L2 BW, SMEM size, SM count, NVLink BW ...).
`linked` lets one knob drive others, e.g. keep L2 BW = 2.5x DDR BW.
"""
from __future__ import annotations

import csv
import io
from typing import Callable

from ..gpuTilingPerfHWModel.spec import HardwareSpec
from ..model.run_config import RunConfig
from ..model.runner import run_model
from ..model.spec import ModelSpec


def default_links(cur_gpu_config: HardwareSpec, path: str):
    """Knobs that cannot move alone.

    Compute peaks in the YAML are whole-GPU numbers, so the engine derives the per-SM rate as
    peak / sms. Sweeping `sms` by itself therefore keeps total FLOPS fixed and makes each SM
    weaker — never what anyone means. This returns the companion overrides that keep a sweep
    physical; pass `linked=None` explicitly to opt out."""
    if path != "sms":
        return None
    base_sms = cur_gpu_config.sms
    tc = dict(cur_gpu_config.get("compute.tc_dense_tflops") or {})
    cuda = cur_gpu_config.get("compute.cuda_fp32_tflops")
    sfu = cur_gpu_config.get("compute.sfu_tops")

    def f(v: float) -> dict:
        k = v / base_sms
        out = {f"compute.tc_dense_tflops.{d}": p * k for d, p in tc.items()}
        out["compute.cuda_fp32_tflops"] = cuda * k
        out["compute.sfu_tops"] = sfu * k
        return out
    return f


def sweep(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig, path: str, values: list[float],
          linked: Callable[[float], dict] | None = None, progress=None, auto_link: bool = True):
    if linked is None and auto_link:
        linked = default_links(cur_gpu_config, path)
    rows = []
    for i, v in enumerate(values):
        if progress:
            progress(i, len(values), f"{path} = {v}")
        ch = {path: v}
        if linked:
            ch.update(linked(v))
        h = cur_gpu_config.override(ch)
        rep = run_model(model, h, rc)
        mix = rep.limiter_breakdown()
        tot = rep.step_time_s
        rows.append(dict(value=v, step_ms=tot * 1e3, tok_s_gpu=rep.tokens_per_s_per_gpu,
                         top_limiter=next(iter(mix)), **{f"share_{k}": t / tot for k, t in mix.items()}))
    return rows


def rows_to_csv(rows) -> str:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, restval=0.0)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    return buf.getvalue()


def required_value(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig, path: str, target_step_ms: float,
                   lo: float, hi: float, iters: int = 18, linked=None, auto_link: bool = True) -> float | None:
    """Smallest value of `path` achieving step time <= target (assumes monotone). None if unreachable."""
    if linked is None and auto_link:
        linked = default_links(cur_gpu_config, path)

    def t(v):
        ch = {path: v}
        if linked:
            ch.update(linked(v))
        return run_model(model, cur_gpu_config.override(ch), rc).step_time_s * 1e3
    if t(hi) > target_step_ms:
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        if t(mid) <= target_step_ms:
            hi = mid
        else:
            lo = mid
    return hi
