"""Attention blocks -> Ops.  Three configurable families, each with flash or naive core.

Block YAML (all fields except the shape ones are optional):

  # plain multi-head attention (kv_heads == heads)
  {type: mha, heads: 32, head_dim: 128}

  # grouped-query attention (kv_heads < heads; kv_heads: 1 == MQA)
  {type: gqa, heads: 64, kv_heads: 8, head_dim: 128}

  # multi-head latent attention (DeepSeek-V3 / Kimi-K2)
  {type: mla, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope: 128, qk_rope: 64, v_head: 128}

Common options (mha/gqa/mla):
  impl: flash | naive          # default RunConfig.attn_impl
  causal: true
  sliding_window: 0            # >0 limits keys per query (and the KV cache) to the window
Options for mha/gqa:
  v_head_dim: <head_dim>       # if V width differs from Q/K width
  fused_qkv: true              # one QKV GEMM vs separate Q, K, V GEMMs
  qk_norm: false               # per-head RMSNorm on q,k (Qwen3 / Gemma-style)
  rope_dim: <head_dim>         # partial rotary
Options for mla:
  absorb: auto | true | false  # auto = absorbed for decode, expanded (kv_b) for prefill

Core implementations (one GPU, tokens T = RunConfig.attn_tokens):
  flash : one fused kernel (kernels/attention.py); S/P stay on chip
  naive : scores GEMM -> softmax -> PV GEMM, S (fp32) and P round-trip through HBM,
          causal masking does NOT skip work. Query heads sharing a KV head are folded
          into M so K/V are read once per KV head: M = (H/KVH)*Sq, batch = B*KVH.
"""
from __future__ import annotations

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import DTYPE_BYTES
from tilesight.interfaceAndModelRun.run_config import RunConfig


def _impl(blk, rc: RunConfig) -> str:
    impl = blk.get("impl", rc.attn_impl)
    if impl not in ("flash", "naive"):
        raise ValueError(f"attention impl must be flash|naive, got {impl!r}")
    return impl


def _kv_len(blk, rc: RunConfig) -> int:
    w = int(blk.get("sliding_window", 0) or 0)
    return min(rc.seq_len, w) if w > 0 else rc.seq_len


def attn_core(prefix, *, B, H, KVH, Sq, Skv, d_qk, d_v, v_in_k, causal, impl, rc: RunConfig, window=0):
    """Emit the attention core (flash kernel op, or the 3 naive ops)."""
    from tilesight.model.lower import Op, _act_gemm
    if impl == "flash":
        kind = "attn_decode" if rc.phase == "decode" else "attn_prefill"
        return [Op(f"{prefix}.attn", kind, dict(B=B, H=H, kv_heads=KVH, S=Skv, d_qk=d_qk, d_v=d_v,
                                                v_in_k=v_in_k, causal=causal, window=window))]
    qpk = H // KVH
    M = qpk * Sq
    batch = B * KVH
    sd = rc.attn_scores_dtype
    rows = batch * M
    return [
        _act_gemm(f"{prefix}.attn_scores", M, Skv, d_qk, rc, b_dtype=rc.kv_dtype, c_dtype=sd, batch=batch,
                  names=("q", "kv_cache(K)", "scores(S)")),
        Op(f"{prefix}.attn_softmax", "elementwise",
           dict(bytes_in=rows * Skv * DTYPE_BYTES[sd], bytes_out=rows * Skv * DTYPE_BYTES[rc.act_dtype],
                flops=5 * rows * Skv, sfu_ops=rows * Skv, names=("scores(S)", "probs(P)"))),
        _act_gemm(f"{prefix}.attn_pv", M, d_v, Skv, rc, b_dtype=rc.kv_dtype, c_dtype=rc.act_dtype, batch=batch,
                  names=("probs(P)", "kv_cache(V)", "out")),
    ]


def _sq(rc: RunConfig) -> int:
    return 1 if rc.phase == "decode" else rc.seq_len


def lower_mha_gqa(blk, prefix, m, rc: RunConfig):
    from tilesight.model.lower import Op, _ew, _gemm
    t = blk["type"]
    tp, T, D = rc.tp, rc.attn_tokens, m.hidden
    H = blk["heads"]
    KV = H if t == "mha" else blk.get("kv_heads", H)
    if H % KV:
        raise ValueError(f"{prefix}: heads ({H}) must be a multiple of kv_heads ({KV})")
    Hl, KVl = max(1, H // tp), max(1, KV // tp)          # kv heads replicate when tp > kv_heads
    hd = blk["head_dim"]
    vd = blk.get("v_head_dim", hd)
    wdt = rc.weight_dtype
    ops = []
    if blk.get("fused_qkv", True):
        ops.append(_gemm(f"{prefix}.qkv", T, Hl * hd + KVl * (hd + vd), D, rc, wdt))
    else:
        ops.append(_gemm(f"{prefix}.q", T, Hl * hd, D, rc, wdt))
        ops.append(_gemm(f"{prefix}.k", T, KVl * hd, D, rc, wdt))
        ops.append(_gemm(f"{prefix}.v", T, KVl * vd, D, rc, wdt))
    if blk.get("qk_norm", False):
        ops.append(_ew(f"{prefix}.qk_norm", T, (Hl + KVl) * hd, rc, flops_per=4))
    rd = blk.get("rope_dim", hd)
    if rd:
        ops.append(_ew(f"{prefix}.rope", T, (Hl + KVl) * rd, rc, flops_per=6))
    win = int(blk.get("sliding_window", 0) or 0)
    ops += attn_core(prefix, B=rc.seqs_per_rank, H=Hl, KVH=KVl, Sq=_sq(rc), Skv=_kv_len(blk, rc),
                     d_qk=hd, d_v=vd, v_in_k=False, causal=blk.get("causal", True),
                     impl=_impl(blk, rc), rc=rc, window=win)
    ops.append(_gemm(f"{prefix}.o_proj", T, D, Hl * vd, rc, wdt))
    if tp > 1:
        ops.append(Op(f"{prefix}.attn_allreduce", "allreduce", dict(bytes=T * D * DTYPE_BYTES[rc.act_dtype], group=tp)))
    return ops


def lower_mla(blk, prefix, m, rc: RunConfig):
    from tilesight.model.lower import Op, _ew, _gemm
    tp, T, D = rc.tp, rc.attn_tokens, m.hidden
    Hl = max(1, blk["heads"] // tp)
    qr, kvr, nope, rope, vh = (blk.get("q_lora_rank", 0), blk["kv_lora_rank"], blk["qk_nope"],
                               blk["qk_rope"], blk["v_head"])
    wdt = rc.weight_dtype
    a = blk.get("absorb", "auto")
    if rc.mla_absorb is not None and a == "auto":
        a = rc.mla_absorb
    absorb = (rc.phase == "decode") if a == "auto" else bool(a)
    ops = []
    if qr:
        ops.append(_gemm(f"{prefix}.qkv_a", T, qr + kvr + rope, D, rc, wdt))
        ops.append(_ew(f"{prefix}.q_kv_norm", T, qr + kvr, rc, flops_per=4))
        ops.append(_gemm(f"{prefix}.q_b", T, Hl * (nope + rope), qr, rc, wdt))
    else:
        ops.append(_gemm(f"{prefix}.q", T, Hl * (nope + rope), D, rc, wdt))
        ops.append(_gemm(f"{prefix}.kv_a", T, kvr + rope, D, rc, wdt))
        ops.append(_ew(f"{prefix}.kv_norm", T, kvr, rc, flops_per=4))
    ops.append(_ew(f"{prefix}.rope", T, Hl * rope + rope, rc, flops_per=6))
    impl, causal, Skv = _impl(blk, rc), blk.get("causal", True), _kv_len(blk, rc)
    win = int(blk.get("sliding_window", 0) or 0)
    kvb_w = kvr * Hl * (nope + vh)                  # W_UK + W_UV elements (= kv_b)
    if absorb:
        # q_nope @ W_UK per head, attention over the shared latent (MQA-like, 1 kv head)
        uk = _gemm(f"{prefix}.absorb_uk", T, kvr, nope, rc, "bf16", batch=Hl,
                   names=("q_nope", "weight(W_UK)", "q_latent"))
        uk.weight_bytes = 0.0
        ops.append(uk)
        ops += attn_core(prefix, B=rc.seqs_per_rank, H=Hl, KVH=1, Sq=_sq(rc), Skv=Skv,
                         d_qk=kvr + rope, d_v=kvr, v_in_k=True, causal=causal, impl=impl, rc=rc, window=win)
        uv = _gemm(f"{prefix}.absorb_uv", T, vh, kvr, rc, "bf16", batch=Hl,
                   names=("o_latent", "weight(W_UV)", "out"))
        uv.weight_bytes = kvb_w * DTYPE_BYTES["bf16"]
        ops.append(uv)
    else:
        # expand the latent to per-head K/V (prefill); attention is plain MHA with d_qk != d_v
        ops.append(_gemm(f"{prefix}.kv_b", T, Hl * (nope + vh), kvr, rc, wdt))
        ops += attn_core(prefix, B=rc.seqs_per_rank, H=Hl, KVH=Hl, Sq=_sq(rc), Skv=Skv,
                         d_qk=nope + rope, d_v=vh, v_in_k=False, causal=causal, impl=impl, rc=rc, window=win)
    ops.append(_gemm(f"{prefix}.o_proj", T, D, Hl * vh, rc, wdt))
    if tp > 1:
        ops.append(Op(f"{prefix}.attn_allreduce", "allreduce", dict(bytes=T * D * DTYPE_BYTES[rc.act_dtype], group=tp)))
    return ops


def kv_bytes_per_seq(blk, rc: RunConfig, seq_len: int) -> float:
    """KV-cache bytes one sequence of `seq_len` needs for ONE layer of this block on one GPU."""
    t = blk["type"]
    kvb = DTYPE_BYTES[rc.kv_dtype]
    w = int(blk.get("sliding_window", 0) or 0)
    toks = min(seq_len, w) if w > 0 else seq_len
    if t == "mla":                                   # latent + rope, replicated across tp
        return toks * (blk["kv_lora_rank"] + blk["qk_rope"]) * kvb
    if t in ("mha", "gqa"):
        H = blk["heads"]
        KV = H if t == "mha" else blk.get("kv_heads", H)
        kvl = max(1, KV // rc.tp)
        hd = blk["head_dim"]
        return toks * kvl * (hd + blk.get("v_head_dim", hd)) * kvb
    return 0.0
