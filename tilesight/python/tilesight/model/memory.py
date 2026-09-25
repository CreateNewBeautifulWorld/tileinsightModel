"""Per-GPU HBM accounting: weights + KV cache + activations/workspace + runtime reserve."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..gpuTilingPerfHWModel.spec import DTYPE_BYTES, HardwareSpec
from .attention_blocks import kv_bytes_per_seq
from .run_config import RunConfig
from .spec import ModelSpec

GB = 1e9
RUNTIME_RESERVE_GB = 4.0     # CUDA context, NCCL/NVSHMEM buffers, allocator slack  [calib]


@dataclass
class MemoryReport:
    weights_GB: float
    kv_cache_GB: float
    activations_GB: float
    reserve_GB: float
    capacity_GB: float
    kv_bytes_per_token: float        # per sequence-token on this GPU, all layers (at the run's seq_len)
    weights_detail_GB: dict[str, float]
    kv_seq_fn: object = None         # seq_len -> KV bytes per sequence (window-aware)

    @property
    def total_GB(self) -> float:
        return self.weights_GB + self.kv_cache_GB + self.activations_GB + self.reserve_GB

    @property
    def fits(self) -> bool:
        return self.total_GB <= self.capacity_GB

    @property
    def free_GB(self) -> float:
        return self.capacity_GB - self.total_GB

    def max_seqs_per_rank(self, seq_len: int) -> int:
        """Largest per-rank batch at this seq_len whose KV cache fits the remaining HBM."""
        room = (self.capacity_GB - self.weights_GB - self.reserve_GB - self.activations_GB) * GB
        per = self.kv_seq_fn(seq_len) if self.kv_seq_fn else self.kv_bytes_per_token * seq_len
        return max(0, math.floor(room / per)) if per > 0 else 10 ** 9


def kv_bytes_per_seq_all(m: ModelSpec, rc: RunConfig, seq_len: int) -> float:
    return sum(g.repeat * kv_bytes_per_seq(blk, rc, seq_len) for g in m.layers for blk in g.blocks)


def kv_bytes_per_token(m: ModelSpec, rc: RunConfig) -> float:
    """Per-token KV bytes ignoring sliding windows (full-attention equivalent)."""
    return kv_bytes_per_seq_all(m, rc, 1)


def memory_report(m: ModelSpec, rc: RunConfig, groups, cur_gpu_config: HardwareSpec) -> MemoryReport:
    detail: dict[str, float] = {}
    act_peak = 0.0
    ab = DTYPE_BYTES[rc.act_dtype]
    for gname, rep, ops in groups:
        for op in ops:
            cat = op.name.split(".")[-1]
            detail[cat] = detail.get(cat, 0.0) + op.weight_bytes * rep / GB
            p = op.p
            if op.kind == "gemm":
                a = (p["M"] * p["K"] * ab + p["M"] * p["N"] * DTYPE_BYTES[p["c_dtype"]]) * p["batch"]
            elif op.kind == "elementwise":
                a = p["bytes_in"] + p["bytes_out"]
            else:
                a = 0.0
            act_peak = max(act_peak, a)
    weights = sum(detail.values())
    kvt = kv_bytes_per_token(m, rc)
    kv = kv_bytes_per_seq_all(m, rc, rc.seq_len) * rc.seqs_per_rank / GB
    acts = 2 * act_peak / GB            # double-buffered live activations
    return MemoryReport(weights, kv, acts, RUNTIME_RESERVE_GB, cur_gpu_config.ddr_capacity_bytes / GB, kvt,
                        dict(sorted(detail.items(), key=lambda kv: -kv[1])),
                        lambda s: kv_bytes_per_seq_all(m, rc, s))
