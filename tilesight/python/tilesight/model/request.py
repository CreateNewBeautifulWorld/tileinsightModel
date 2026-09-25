"""Request-level simulation: prompt_len + output_len -> TTFT, TPOT curve, peak memory.

Static (no model is executed). For a batch of identical requests:

  prefill : one step over prompt_len tokens                      -> TTFT
  decode  : step t (1..output_len) reads KV of length L_t = prompt_len + t
            we evaluate the full model at `decode_samples` KV lengths and integrate the
            piecewise-linear TPOT(L) curve (attention scales with L, weights don't)
  memory  : weights + KV(pages) + activations + reserve, tracked along the generation
            peak KV = ceil((prompt+output)/page)*page tokens per request ("peak" reserve)
            peak total = weights + peak KV + max(prefill act, decode act) + reserve
            (conservative: assumes a prefill can run while all slots hold full KV;
             shrink prefill activations with `prefill_batch`)

Sliding-window layers cap their KV at the window automatically (kv_bytes_per_seq).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..hw.spec import HardwareSpec
from .memory import GB, kv_bytes_per_seq_all
from .run_config import RunConfig
from .runner import ModelReport, run_model
from .spec import ModelSpec


@dataclass
class DecodePoint:
    step: int                 # 1-based decode step
    kv_len: int
    tpot_s: float
    kv_GB: float              # KV resident per GPU at this step (current allocation)
    total_GB: float           # weights + KV + act + reserve at this step
    top_bound: str            # finest limiter (resource:tensor)
    report: ModelReport


@dataclass
class RequestReport:
    model: str
    gpu_name: str
    rc: RunConfig
    prefill: ModelReport
    points: list[DecodePoint]
    ttft_s: float
    tpot_avg_s: float
    e2e_s: float
    peak_total_GB: float
    peak_kv_GB: float
    capacity_GB: float
    max_requests_per_rank: int          # concurrent requests that fit with the chosen reserve policy

    @property
    def fits(self) -> bool:
        return self.peak_total_GB <= self.capacity_GB


def _pages(tokens: int, page: int) -> int:
    return math.ceil(tokens / page) * page if page > 0 else tokens


def run_request(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig, progress=None) -> RequestReport:
    P = rc.prompt_len or rc.seq_len
    O = max(1, rc.output_len)
    nseq = rc.seqs_per_rank
    # ---- prefill -----------------------------------------------------------------------
    pb = rc.prefill_batch or rc.batch
    if progress:
        progress(0, 1 + max(2, rc.decode_samples), "prefill")
    pre = run_model(model, cur_gpu_config, replace(rc, phase="prefill", seq_len=P, batch=pb))
    ttft = pre.step_time_s
    # ---- decode samples along the generation --------------------------------------------
    n = max(2, rc.decode_samples)
    steps = sorted({max(1, round(1 + (O - 1) * i / (n - 1))) for i in range(n)})
    pts: list[DecodePoint] = []
    base = None
    for i, t in enumerate(steps):
        L = P + t
        if progress:
            progress(1 + i, 1 + len(steps), f"decode step {t} (kv {L})")
        rep = run_model(model, cur_gpu_config, replace(rc, phase="decode", seq_len=L))
        base = base or rep
        kv_now = kv_bytes_per_seq_all(model, rc, _pages(L, rc.page_size)) * nseq / GB
        m = rep.memory
        total = m.weights_GB + kv_now + m.activations_GB + m.reserve_GB
        top = next(iter(rep.detail_breakdown()), "-")
        pts.append(DecodePoint(t, L, rep.step_time_s, kv_now, total, top, rep))
    # trapezoid over steps (TPOT is ~linear in L between samples)
    tot = 0.0
    for a, b in zip(pts, pts[1:]):
        tot += (a.tpot_s + b.tpot_s) / 2 * (b.step - a.step)
    tot += pts[0].tpot_s                          # step 1 itself
    tpot_avg = tot / O if O > 1 else pts[0].tpot_s
    # ---- memory: peak & capacity -----------------------------------------------------------
    peak_tokens = _pages(P + O, rc.page_size)
    kv_req_peak = kv_bytes_per_seq_all(model, rc, peak_tokens)
    m = base.memory
    act = max(pre.memory.activations_GB, m.activations_GB)
    peak_kv = kv_req_peak * nseq / GB
    peak_total = m.weights_GB + peak_kv + act + m.reserve_GB
    room = (m.capacity_GB - m.weights_GB - act - m.reserve_GB) * GB
    per_req = kv_req_peak if rc.kv_reserve == "peak" else \
        kv_bytes_per_seq_all(model, rc, _pages(P + O // 2, rc.page_size))
    max_req = max(0, math.floor(room / per_req)) if per_req > 0 else 10 ** 9
    return RequestReport(model.name, cur_gpu_config.name, rc, pre, pts, ttft, tpot_avg, ttft + tpot_avg * O,
                         peak_total, peak_kv, m.capacity_GB, max_req)


def request_summary(r: RequestReport) -> str:
    from ..report.table import bound_report
    rc = r.rc
    P, O = rc.prompt_len or rc.seq_len, rc.output_len
    L = [f"== {r.model} on {r.gpu_name}  request: prompt={P} output={O}  batch={rc.batch} "
         f"(seqs/rank {rc.seqs_per_rank}) tp={rc.tp} dp={rc.dp} ep={rc.ep_size} page={rc.page_size}"]
    L.append(f"TTFT (prefill {rc.prefill_batch or rc.batch} seq x {P})  : {r.ttft_s * 1e3:9.2f} ms")
    L.append(f"TPOT first / last / avg          : {r.points[0].tpot_s * 1e3:.2f} / "
             f"{r.points[-1].tpot_s * 1e3:.2f} / {r.tpot_avg_s * 1e3:.2f} ms")
    L.append(f"end-to-end latency               : {r.e2e_s:9.3f} s   "
             f"decode throughput {rc.batch / r.tpot_avg_s / rc.world:,.0f} tok/s/GPU")
    L.append(f"PEAK memory / GPU                : {r.peak_total_GB:.1f} / {r.capacity_GB:.0f} GB "
             f"(KV peak {r.peak_kv_GB:.1f} GB)  -> {'OK' if r.fits else 'OVERFLOW'}")
    L.append(f"max concurrent requests / rank   : {r.max_requests_per_rank} "
             f"(reserve={rc.kv_reserve}, {P}+{O} tokens, page {rc.page_size})")
    L.append("")
    L.append(f"{'step':>6s} {'kv_len':>8s} {'TPOT ms':>9s} {'KV GB':>8s} {'total GB':>9s}  top bound")
    L.append(f"{'prefill':>6s} {P:8d} {r.ttft_s * 1e3:9.2f} {r.prefill.memory.kv_cache_GB:8.2f} "
             f"{r.prefill.memory.total_GB:9.1f}  {next(iter(r.prefill.detail_breakdown()), '-')}")
    for p in r.points:
        L.append(f"{p.step:6d} {p.kv_len:8d} {p.tpot_s * 1e3:9.2f} {p.kv_GB:8.2f} {p.total_GB:9.1f}  {p.top_bound}")
    L.append("")
    L.append("-- where the last decode step is bound --")
    L.append(bound_report(r.points[-1].report))
    L.append("-- where prefill is bound --")
    L.append(bound_report(r.prefill))
    return "\n".join(L)
