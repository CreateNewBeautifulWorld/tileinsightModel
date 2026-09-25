"""Run configuration: phase, batch, sequence, parallelism, dtypes.

Tile policy (how the GPU tiles a GEMM/attention op) is NOT here: it's a GPU-side modelling
choice, not a property of the workload, and lives in the hardware config instead
(hw/schema.py's compute.tile_policy.*, see model/runner.py)."""
from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class RunConfig:
    phase: str = "decode"            # decode | prefill
    batch: int = 64                  # global number of sequences
    seq_len: int = 8192              # decode: KV length; prefill: prompt length (single-point runs)
    # ---- request-level runs (model/request.py): KV grows from prompt_len to prompt_len+output_len
    prompt_len: int = 0              # 0 -> use seq_len
    output_len: int = 0              # generated tokens per request
    page_size: int = 64              # paged-KV block size (tokens); memory is reserved in whole pages
    kv_reserve: str = "peak"         # peak: reserve prompt+output per request up front | current
    decode_samples: int = 5          # KV lengths sampled along the generation (>=2)
    prefill_batch: int = 0           # sequences per prefill step (0 -> batch)
    tp: int = 1                      # tensor parallel (attention heads, dense MLP, shared expert, lm_head)
    dp: int = 8                      # data-parallel attention groups
    ep: int = 0                      # expert parallel; 0 -> tp*dp (all GPUs)
    # dtypes
    weight_dtype: str = "fp8"        # dense / attention weights
    expert_dtype: str = "fp8"        # routed expert weights (e.g. fp4 / int4 for Kimi-K2 W4A16)
    act_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    compute_dtype: str = "fp8"       # tensor-core datapath for linears
    attn_compute_dtype: str = "bf16"
    attn_impl: str = "flash"         # flash (fused) | naive (scores GEMM -> softmax -> PV GEMM); per-block `impl` overrides
    attn_scores_dtype: str = "fp32"  # naive attention: dtype of the materialized S matrix
    dispatch_dtype: str = "fp8"      # MoE all-to-all dispatch payload
    include_lm_head: bool = True
    comm_overlap: float = 0.0        # only used when overlap_mode == "manual"
    overlap_mode: str = "none"       # none | stream | two_batch | manual
    overlap_efficiency: float = 0.8  # how much of the overlappable compute actually hides comm
    mla_absorb: bool | None = None   # None -> per-block `absorb` (default auto: absorb in decode, not in prefill)

    @property
    def world(self) -> int:
        return self.tp * self.dp

    @property
    def ep_size(self) -> int:
        return self.ep or self.world

    @property
    def attn_tokens(self) -> int:
        """Tokens processed by one attention rank (replicated across its tp group)."""
        per = self.batch // self.dp if self.batch >= self.dp else 1
        return per if self.phase == "decode" else per * self.seq_len

    @property
    def seqs_per_rank(self) -> int:
        return max(1, self.batch // self.dp)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
