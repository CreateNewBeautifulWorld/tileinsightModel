"""Collectives via the alpha-beta stage model (paper §3.6, Eq. 11).

Each collective is decomposed into stages; stage time = alpha (per hop) + bytes on the
bottleneck link / link bandwidth. Link = NVLink inside `network.nvlink.domain_size`,
scale-out NIC otherwise (hierarchical collectives are a TODO, docs/TASKS.md E3).
"""
from __future__ import annotations

import math

from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import HardwareSpec
from tilesight.gpuTilingPerfHWModel.model.ir.kernel import Kernel


def _link(cur_gpu_config: HardwareSpec, group: int) -> tuple[float, float]:
    nv = cur_gpu_config.get("network.nvlink")
    if group <= int(nv["domain_size"]):
        return nv["alpha_us"] * 1e-6, nv["bandwidth_GBps"] * 1e9
    so = cur_gpu_config.get("network.scaleout")
    return so["alpha_us"] * 1e-6, so["bandwidth_GBps"] * 1e9


def _flat(a: float, bw: float, nbytes: float, group: int, algo: str = "auto"):
    if group <= 1:
        return 0.0, "none"
    ring = 2 * (group - 1) * a + 2 * (group - 1) / group * nbytes / bw
    rd = math.ceil(math.log2(group)) * (a + nbytes / bw)              # recursive doubling
    if algo == "auto":
        return min((ring, "ring"), (rd, "recursive_doubling"))
    return (ring, "ring") if algo == "ring" else (rd, "recursive_doubling")


def allreduce(cur_gpu_config: HardwareSpec, name: str, nbytes: float, group: int, algo: str = "auto") -> list[Kernel]:
    """Flat inside the fast domain, hierarchical beyond it.

    Once the group leaves the NVLink/UALink domain, a real library does not run one slow ring
    over the scale-out fabric: it reduce-scatters inside each node, all-reduces the 1/d-sized
    shards between nodes, then all-gathers inside the node again. Modelling it flat
    overestimates an ep=16 all-reduce by ~5x."""
    if group <= 1 or nbytes <= 0:
        return []
    d = int(cur_gpu_config.get("network.nvlink.domain_size"))
    if group <= d:
        a, bw = _link(cur_gpu_config, group)
        t, used = _flat(a, bw, nbytes, group, algo)
        return [Kernel(name, "comm", 0, 0, 1, 1, [], meta=dict(bytes=nbytes, group=group, algo=used),
                       fixed_time_s=t)]
    nodes = math.ceil(group / d)
    ai, bwi = _link(cur_gpu_config, d)                     # intra-node (fast domain)
    ao, bwo = _link(cur_gpu_config, group)                 # inter-node (scale-out)
    rs = (d - 1) * ai + (d - 1) / d * nbytes / bwi          # reduce-scatter inside the node
    inter, _ = _flat(ao, bwo, nbytes / d, nodes)            # all-reduce the shard between nodes
    ag = rs                                                 # all-gather inside the node
    return [Kernel(name, "comm", 0, 0, 1, 1, [],
                   meta=dict(bytes=nbytes, group=group, algo=f"hierarchical({d}x{nodes})",
                             intra_s=rs + ag, inter_s=inter),
                   fixed_time_s=rs + inter + ag)]


def all_to_all(cur_gpu_config: HardwareSpec, name: str, send_bytes: float, group: int) -> list[Kernel]:
    """send_bytes = bytes this GPU sends in total (to all peers)."""
    if group <= 1 or send_bytes <= 0:
        return []
    d = int(cur_gpu_config.get("network.nvlink.domain_size"))
    if group <= d:
        a, bw = _link(cur_gpu_config, group)
        t = a * math.ceil(math.log2(group)) + send_bytes * (group - 1) / group / bw
        return [Kernel(name, "comm", 0, 0, 1, 1, [],
                       meta=dict(bytes=send_bytes, group=group, algo="a2a"), fixed_time_s=t)]
    # hierarchical: only the fraction leaving the node crosses the slow fabric
    nodes = math.ceil(group / d)
    ai, bwi = _link(cur_gpu_config, d)
    ao, bwo = _link(cur_gpu_config, group)
    intra = ai * math.ceil(math.log2(d)) + send_bytes * (d - 1) / group / bwi
    inter = ao + send_bytes * (group - d) / group / bwo
    return [Kernel(name, "comm", 0, 0, 1, 1, [],
                   meta=dict(bytes=send_bytes, group=group, algo=f"a2a-hierarchical({d}x{nodes})",
                             intra_s=intra, inter_s=inter),
                   fixed_time_s=intra + inter)]
