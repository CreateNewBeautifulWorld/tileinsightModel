"""Plain-text / CSV reports."""
from __future__ import annotations

import csv
import io


def _fmt_t(s: float) -> str:
    return f"{s * 1e6:9.1f}us" if s < 1e-2 else f"{s * 1e3:9.2f}ms"


RESOURCE_CLASS = {
    "sram": "on-chip buffer", "switch": "on-chip buffer:switch",
    "tc": "compute:tensor-core", "cuda": "compute:cuda-core", "sfu": "compute:sfu",
    "smem": "on-chip:smem", "tmem": "on-chip:tmem", "l2": "cache:L2", "ddr": "memory:DDR/HBM",
    "net": "interconnect", "launch": "launch-overhead",
}


def domain_of(detail: str) -> str:
    """Which of the model's three blocks a limiter belongs to."""
    from ..hw.spec import lane_domain
    head = detail.split(":", 1)[0]
    if head == "latency":
        inner = detail[detail.rfind("(") + 1: -1] if detail.endswith(")") else ""
        return lane_domain(inner) if inner and inner != "mem-lat" else "shader_slice"
    if head in ("launch", "net"):
        return "runtime" if head == "launch" else "memory"
    return lane_domain(head)


def resource_class(detail: str) -> str:
    """Map 'ddr:load:weight' -> 'memory:DDR/HBM', 'latency:softmax(sfu)' -> 'latency<-sfu'."""
    head = detail.split(":", 1)[0]
    if head == "latency":
        inner = detail[detail.rfind("(") + 1: -1] if detail.endswith(")") else detail.split(":", 1)[1]
        return f"latency<-{inner}"
    if head.startswith("path"):
        return f"load-path:{detail.split(':')[1]}"
    if "spill:regs" in detail:
        return "on-chip:regs(spill)"
    return RESOURCE_CLASS.get(head, head)


def bound_report(rep, top: int = 12) -> str:
    """Where the step is bound: by resource class, and by resource+tensor."""
    tot = rep.step_time_s
    det = rep.detail_breakdown()
    cls: dict[str, float] = {}
    for k, v in det.items():
        c = resource_class(k)
        cls[c] = cls.get(c, 0.0) + v
    dom: dict[str, float] = {}
    for k, v in det.items():
        d = domain_of(k)
        dom[d] = dom.get(d, 0.0) + v
    L = ["bound by block    : " + ", ".join(f"{k} {v / tot:.0%}" for k, v in
                                            sorted(dom.items(), key=lambda kv: -kv[1])),
         "activity by block : " + ", ".join(f"{k} {v:.0%}" for k, v in
                                            rep.activity_by_domain().items())
         if hasattr(rep, "activity_by_domain") else "",
         "bound by resource : " + ", ".join(f"{k} {v / tot:.0%}" for k, v in
                                             sorted(cls.items(), key=lambda kv: -kv[1]) if v / tot >= 0.005)]
    L = [x for x in L if x]
    L.append("bound by tensor   :")
    for k, v in list(det.items())[:top]:
        if v / tot >= 0.005:
            L.append(f"    {v / tot:6.1%}  {k}")
    occ: dict[str, float] = {}
    for o in rep.ops:
        oc = o.occupancy
        if oc != "-":
            key = oc.split("/", 1)[1]
            occ[key] = occ.get(key, 0.0) + o.total_s
    if occ:
        L.append("occupancy limited by (time-weighted): " +
                 ", ".join(f"{k} {v / tot:.0%}" for k, v in sorted(occ.items(), key=lambda kv: -kv[1])))
    return "\n".join(L)


def model_summary(rep, top: int = 25) -> str:
    rc = rep.rc
    L = []
    L.append(f"== {rep.model} on {rep.gpu_name}  [{rc.phase}] batch={rc.batch} seq={rc.seq_len} "
             f"tp={rc.tp} dp={rc.dp} ep={rc.ep_size} (backend={rep.backend})")
    L.append(f"step time      : {_fmt_t(rep.step_time_s).strip()}")
    if rc.phase == "decode":
        L.append(f"TPOT           : {rep.step_time_s * 1e3:.2f} ms   throughput {rep.tokens_per_s_per_gpu:,.0f} tok/s/GPU")
    else:
        L.append(f"prefill        : {rep.tokens_per_s_per_gpu:,.0f} tok/s/GPU")
    m = rep.memory
    L.append(f"memory / GPU   : weights {m.weights_GB:.1f} GB + KV {m.kv_cache_GB:.1f} GB + act {m.activations_GB:.2f} GB"
             f" + reserve {m.reserve_GB:.0f} GB = {m.total_GB:.1f} / {m.capacity_GB:.0f} GB"
             f"  -> {'OK' if m.fits else 'OVERFLOW'} (max seqs/rank @seq={rc.seq_len}: {m.max_seqs_per_rank(rc.seq_len)})")
    tot = rep.step_time_s
    L.append("bottleneck mix : " + ", ".join(f"{k} {v / tot:.0%}" for k, v in rep.limiter_breakdown().items() if v / tot >= 0.005))
    L.append(bound_report(rep))
    L.append("")
    L.append(f"{'op':40s} {'x':>3s} {'tile':>20s} {'time/layer':>11s} {'share':>6s} {'occ':>12s}  {'bound (resource:tensor)':34s} util")
    rows = sorted(rep.ops, key=lambda o: -o.total_s)[:top]
    for o in rows:
        u = {}
        for k in o.kernels:
            for n, v in k.util.items():
                u[n] = max(u.get(n, 0), v)
        us = " ".join(f"{n}={v:.0%}" for n, v in sorted(u.items(), key=lambda kv: -kv[1])[:3])
        L.append(f"{o.op.name:40s} {o.repeat:3d} {o.tile:>20s} {_fmt_t(o.time_s)} {o.total_s / tot:6.1%} "
                 f"{o.occupancy:>12s}  {o.bottleneck_detail:34s} {us}")
    for n in rep.notes:
        L.append("NOTE: " + n)
    return "\n".join(L)


def ops_csv(rep) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["op", "group", "repeat", "kind", "tile", "time_us", "total_us", "limiter", "bound_detail",
                "occupancy", "flops", "weight_MB"])
    for o in rep.ops:
        fl = sum(k.meta.get("flops", 0) for k in o.kernels)
        w.writerow([o.op.name, o.group, o.repeat, o.op.kind, o.tile, f"{o.time_s * 1e6:.3f}",
                    f"{o.total_s * 1e6:.3f}", o.bottleneck, o.bottleneck_detail, o.occupancy,
                    f"{fl:.4g}", f"{o.op.weight_bytes / 1e6:.2f}"])
    return buf.getvalue()
