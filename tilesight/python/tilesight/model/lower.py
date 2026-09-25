"""Model blocks -> Ops (one GPU's view of one layer).

An Op is still shape-level (M,N,K / attention dims); `runner.py` turns each Op into
kernels with a tile config (fixed or auto-searched) and evaluates them.

Attention families (mha / gqa / mla, flash or naive) live in attention_blocks.py.
Parallelism semantics (docs/DESIGN.md §6):
  * attention: DP groups of size tp; heads sharded by tp; tokens per rank = attn_tokens
  * dense MLP / shared expert: column(N)/row(K) split by tp + all-reduce over tp
  * routed experts: EP over ep GPUs; tokens deduplicated across the tp group
  * lm_head: vocab sharded by tp
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import DTYPE_BYTES, WEIGHT_ONLY
from tilesight.model.attention_blocks import lower_mha_gqa, lower_mla
from tilesight.interfaceAndModelRun.run_config import RunConfig
from tilesight.interfaceAndModelRun.spec import ModelSpec


@dataclass
class Op:
    name: str
    kind: str                      # gemm | attn_decode | attn_prefill | elementwise | allreduce | a2a
    p: dict[str, Any] = field(default_factory=dict)
    weight_bytes: float = 0.0      # resident weight bytes on this GPU for this op


def _gemm(name, M, N, K, rc: RunConfig, wdt, batch=1, weight_batch=None, names=("act", "weight", "out")):
    wb = K * N * DTYPE_BYTES[wdt] * (weight_batch if weight_batch is not None else batch)
    comp = _comp(rc, wdt)
    # low-precision datapath => activations are quantized to it (per-token/block quant, fused upstream)
    a_dt = comp if DTYPE_BYTES[comp] < DTYPE_BYTES[rc.act_dtype] else rc.act_dtype
    return Op(name, "gemm", dict(M=M, N=N, K=K, batch=batch, a_dtype=a_dt, b_dtype=wdt,
                                 c_dtype=rc.act_dtype, compute_dtype=comp, names=names), wb)


def _act_gemm(name, M, N, K, rc: RunConfig, *, b_dtype, c_dtype, batch=1, names=("act", "act", "out")):
    """GEMM whose B operand is an activation/KV tensor (no resident weights)."""
    comp = rc.attn_compute_dtype
    a_dt = comp if DTYPE_BYTES[comp] < DTYPE_BYTES[rc.act_dtype] else rc.act_dtype
    return Op(name, "gemm", dict(M=M, N=N, K=K, batch=batch, a_dtype=a_dt, b_dtype=b_dtype,
                                 c_dtype=c_dtype, compute_dtype=comp, names=names), 0.0)


def _comp(rc: RunConfig, wdt: str) -> str:
    """Datapath for a GEMM whose weights are stored as `wdt`."""
    if wdt in WEIGHT_ONLY:            # int4 weights, activations stay wide -> dequant then MMA
        return rc.act_dtype
    return rc.compute_dtype if DTYPE_BYTES.get(wdt, 2) <= DTYPE_BYTES[rc.compute_dtype] else "bf16"


def _ew(name, T, hidden, rc, reads=1, writes=1, flops_per=5, sfu_per=0):
    ab = DTYPE_BYTES[rc.act_dtype]
    return Op(name, "elementwise", dict(bytes_in=T * hidden * ab * reads, bytes_out=T * hidden * ab * writes,
                                        flops=T * hidden * flops_per, sfu_ops=T * hidden * sfu_per))


def lower_block(blk: dict, prefix: str, m: ModelSpec, rc: RunConfig) -> list[Op]:
    t = blk["type"]
    D = m.hidden
    T = rc.attn_tokens
    tp = rc.tp
    ab = DTYPE_BYTES[rc.act_dtype]
    wdt = rc.weight_dtype
    ops: list[Op] = []

    if t == "norm":
        ops.append(_ew(f"{prefix}.rmsnorm", T, D, rc, flops_per=4))

    elif t in ("mha", "gqa"):
        ops += lower_mha_gqa(blk, prefix, m, rc)

    elif t == "mla":
        ops += lower_mla(blk, prefix, m, rc)

    elif t == "mlp":
        f = math.ceil(blk["d_ff"] / tp)
        gated = blk.get("gated", True)
        ops.append(_gemm(f"{prefix}.gate_up", T, f * (2 if gated else 1), D, rc, wdt))
        ops.append(_ew(f"{prefix}.act", T, f, rc, reads=2 if gated else 1, flops_per=6, sfu_per=1))
        ops.append(_gemm(f"{prefix}.down", T, D, f, rc, wdt))
        if tp > 1:
            ops.append(Op(f"{prefix}.mlp_allreduce", "allreduce", dict(bytes=T * D * ab, group=tp)))

    elif t == "moe":
        E, k, f = blk["experts"], blk["topk"], blk["d_ff"]
        ep = rc.ep_size
        El = max(1, E // ep)
        Tu = max(1, T // tp)                               # unique tokens this GPU dispatches
        pairs_group = Tu * ep * k                          # token-expert pairs landing in the EP group
        p_active = 1.0 - (1.0 - k / E) ** (Tu * ep)
        active = max(1, round(El * p_active))
        m_exp = max(1, math.ceil(pairs_group / E / p_active))
        edt = rc.expert_dtype
        ops.append(_gemm(f"{prefix}.router", T, E, D, rc, "bf16"))
        ops.append(Op(f"{prefix}.topk", "elementwise", dict(bytes_in=T * E * 4, bytes_out=T * k * 8,
                                                             flops=T * E * 8, sfu_ops=T * E)))
        disp_b = Tu * k * D * DTYPE_BYTES[rc.dispatch_dtype]
        if ep > 1:
            ops.append(Op(f"{prefix}.dispatch", "a2a", dict(bytes=disp_b, group=ep)))
        g = _gemm(f"{prefix}.experts_gate_up", m_exp, 2 * f, D, rc, edt, batch=active, weight_batch=El,
                  names=("act", "expert_weight", "out"))
        g.p["experts_active"] = active
        g.p["experts_local"] = El
        ops.append(g)
        ops.append(Op(f"{prefix}.experts_act", "elementwise",
                      dict(bytes_in=active * m_exp * 2 * f * ab, bytes_out=active * m_exp * f * ab,
                           flops=active * m_exp * f * 6, sfu_ops=active * m_exp * f)))
        d = _gemm(f"{prefix}.experts_down", m_exp, D, f, rc, edt, batch=active, weight_batch=El,
                  names=("act", "expert_weight", "out"))
        ops.append(d)
        if ep > 1:
            ops.append(Op(f"{prefix}.combine", "a2a", dict(bytes=Tu * k * D * ab, group=ep)))
        sh = blk.get("shared_experts", 0)
        if sh:
            fs = math.ceil(blk.get("shared_d_ff", f * sh) / tp)
            ops.append(_gemm(f"{prefix}.shared_gate_up", T, 2 * fs, D, rc, wdt))
            ops.append(_ew(f"{prefix}.shared_act", T, fs, rc, reads=2, flops_per=6, sfu_per=1))
            ops.append(_gemm(f"{prefix}.shared_down", T, D, fs, rc, wdt))
            if tp > 1:
                ops.append(Op(f"{prefix}.shared_allreduce", "allreduce", dict(bytes=T * D * ab, group=tp)))
        ops.append(_ew(f"{prefix}.moe_reduce", T, D, rc, reads=k + 1, flops_per=2 * k))

    elif t == "gemm":                                       # raw user op
        N, K = blk["N"], blk["K"]
        shard = blk.get("shard", "none")
        if shard == "col":
            N = math.ceil(N / tp)
        elif shard == "row":
            K = math.ceil(K / tp)
        ops.append(_gemm(f"{prefix}.{blk.get('name', 'gemm')}", T, N, K, rc, blk.get("dtype", wdt)))
        if shard == "row" and tp > 1:
            ops.append(Op(f"{prefix}.{blk.get('name', 'gemm')}_allreduce", "allreduce",
                          dict(bytes=T * N * ab, group=tp)))

    elif t == "elementwise":
        ops.append(_ew(f"{prefix}.{blk.get('name', 'ew')}", T, blk.get("width", D), rc,
                       reads=blk.get("reads", 1), writes=blk.get("writes", 1), flops_per=blk.get("flops_per", 4)))
    else:
        raise ValueError(f"unknown block type {t!r}")
    return ops


def lower_model(m: ModelSpec, rc: RunConfig) -> list[tuple[str, int, list[Op]]]:
    """Returns [(group_name, repeat, ops_of_one_layer)], plus a final 'head' group."""
    out = []
    for g in m.layers:
        ops = []
        for i, blk in enumerate(g.blocks):
            ops += lower_block(blk, f"{g.name}.{i}.{blk['type']}", m, rc)
        out.append((g.name, g.repeat, ops))
    head: list[Op] = []
    emb_rows = math.ceil(m.vocab / rc.tp)
    wb = DTYPE_BYTES[rc.weight_dtype]
    head.append(Op("head.embed", "elementwise", dict(bytes_in=rc.attn_tokens * m.hidden * 2,
                                                     bytes_out=rc.attn_tokens * m.hidden * 2),
                   weight_bytes=emb_rows * m.hidden * 2))
    if rc.include_lm_head:
        head.append(_ew("head.final_norm", rc.attn_tokens, m.hidden, rc, flops_per=4))
        Tl = rc.seqs_per_rank                                 # logits only for the last token
        lm = _gemm("head.lm_head", Tl, emb_rows, m.hidden, rc, "bf16")
        lm.weight_bytes = 0.0 if m.tie_embeddings else emb_rows * m.hidden * 2
        head.append(lm)
        head.append(Op("head.sample", "elementwise", dict(bytes_in=Tl * emb_rows * 4, bytes_out=Tl * 8,
                                                          flops=Tl * emb_rows * 4, sfu_ops=Tl * emb_rows)))
    del wb
    out.append(("head", 1, head))
    return out
