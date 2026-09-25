"""The workload configuration — the second input, next to the GPU config.

The model says what runs on ONE GPU: one transformer layer (attention + FFN/MoE), its
dimensions and its datatypes. Multi-GPU mapping is deliberately out of scope here; this is the
single-device view the paper models. Tile policy is NOT here: how the GPU tiles a GEMM/attention
op is a GPU-side modelling choice, not a property of the workload — it lives in the GPU config
(gpuTilingPerfHWModel/schema.py's compute.tile_policy.*).

Like gpuTilingPerfHWModel/schema.py, this file is the single source of what a workload may contain. Fields:
  attention : type (mha | gqa | mla), head counts and dims, sequence, batch, causal/window, impl
  ffn       : dense MLP or MoE (experts, top-k, shared experts, expert FFN width — sparsity is
              topk/experts: ALL experts are resident weight-wise, only topk are on the compute path)
  dtypes    : weight / activation / KV / compute / expert
  run       : phase (decode | prefill) for a single-phase run, batch, and the three sequence
              lengths a request-serving GPU actually cares about — prefill_seq_len (prompt),
              cur_decoding_seq_len (KV length right now) and max_seq_len (the longest this GPU
              must ever hold KV for — sizes the KV cache, run_workload_both_phases() enforces
              max_seq_len > prefill_seq_len + cur_decoding_seq_len)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import DTYPE_BYTES


@dataclass(frozen=True)
class WField:
    path: str
    kind: str
    unit: str
    default: object
    section: str
    doc: str


W = WField
WORKLOAD_FIELDS: tuple[WField, ...] = (
    W("name", "str", "", "layer", "identity", "Name of this workload"),
    W("hidden", "int", "elements", 7168, "identity", "Model hidden size (d_model)"),
    W("layers", "int", "count", 1, "identity",
      "Layers of this kind on this GPU; the report multiplies by it"),

    W("attention.type", "str", "", "mla", "attention", "mha | gqa | mla (mqa = gqa with kv_heads 1)"),
    W("attention.impl", "str", "", "flash", "attention", "flash (fused) | naive (3 kernels through HBM)"),
    W("attention.heads", "int", "count", 64, "attention", "Query heads on this GPU"),
    W("attention.kv_heads", "int", "count", 0, "attention", "KV heads (0 = same as query heads; mla ignores it)"),
    W("attention.head_dim", "int", "elements", 128, "attention", "Q/K head dim (mha/gqa)"),
    W("attention.v_head_dim", "int", "elements", 0, "attention", "V head dim (0 = head_dim)"),
    W("attention.q_lora_rank", "int", "elements", 1536, "attention", "MLA: Q down-projection rank (0 = none)"),
    W("attention.kv_lora_rank", "int", "elements", 512, "attention", "MLA: latent KV rank"),
    W("attention.qk_nope", "int", "elements", 128, "attention", "MLA: non-positional Q/K dim per head"),
    W("attention.qk_rope", "int", "elements", 64, "attention", "MLA: rotary Q/K dim per head"),
    W("attention.v_head", "int", "elements", 128, "attention", "MLA: V dim per head"),
    W("attention.absorb", "str", "", "auto", "attention", "MLA weight absorption: auto | true | false"),
    W("attention.causal", "bool", "", True, "attention", "Causal masking in prefill"),
    W("attention.sliding_window", "int", "tokens", 0, "attention", "0 = full attention"),
    W("attention.qk_norm", "bool", "", False, "attention", "Per-head RMSNorm on q,k"),

    W("ffn.type", "str", "", "moe", "ffn", "mlp | moe | none"),
    W("ffn.d_ff", "int", "elements", 2048, "ffn", "FFN width (per expert when type = moe)"),
    W("ffn.experts", "int", "count", 384, "ffn", "Routed experts held on this GPU"),
    W("ffn.topk", "int", "count", 8, "ffn", "Experts per token"),
    W("ffn.shared_experts", "int", "count", 1, "ffn", "Always-on experts"),
    W("ffn.gated", "bool", "", True, "ffn", "Gated FFN (gate+up then down)"),

    W("dtypes.weight", "str", "", "fp8", "dtypes", "Attention / dense weights"),
    W("dtypes.expert", "str", "", "fp8", "dtypes", "Routed expert weights (int4 = weight-only)"),
    W("dtypes.activation", "str", "", "bf16", "dtypes", "Activations"),
    W("dtypes.kv", "str", "", "bf16", "dtypes", "KV cache"),
    W("dtypes.compute", "str", "", "fp8", "dtypes", "Tensor-core datapath for the linears"),
    W("dtypes.attn_compute", "str", "", "bf16", "dtypes", "Tensor-core datapath inside attention"),

    W("run.phase", "str", "", "decode", "run",
      "decode (1 token/sequence) | prefill (seq tokens); only used for a single-phase run "
      "(run_workload()) — run_workload_both_phases() ignores this and runs both"),
    W("run.batch", "int", "sequences", 32, "run", "Sequences resident on this GPU"),
    W("run.seq_len", "int", "tokens", 8192, "run",
      "KV length (decode) or prompt length (prefill); only used for a single-phase run"),
    W("run.prefill_seq_len", "int", "tokens", 4096, "run",
      "Prompt length for the prefill-phase run in run_workload_both_phases()"),
    W("run.cur_decoding_seq_len", "int", "tokens", 8192, "run",
      "KV length for the decode-phase run in run_workload_both_phases() — decode always models "
      "ONE token-generation step at a given KV length, so this is where that step is taken"),
    W("run.max_seq_len", "int", "tokens", 16384, "run",
      "The longest sequence this GPU must hold KV for — sizes the KV cache (see memory_breakdown()"
      " and the run.max_seq_len > run.prefill_seq_len + run.cur_decoding_seq_len check in "
      "validate_workload()). Headroom beyond prefill+cur_decoding: room for the sequence to keep "
      "generating past where cur_decoding_seq_len currently samples it."),
)

W_BY_PATH = {f.path: f for f in WORKLOAD_FIELDS}
W_SECTIONS = tuple(dict.fromkeys(f.section for f in WORKLOAD_FIELDS))


def w_default(path: str):
    f = W_BY_PATH.get(path)
    return f.default if f else None


def get(cfg: dict, path: str):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return w_default(path)
        node = node[k]
    return node


def validate_workload(cfg: dict) -> list[str]:
    problems: list[str] = []

    def walk(node, prefix=""):
        for k, v in (node or {}).items():
            p = f"{prefix}{k}"
            if isinstance(v, dict):
                if not any(x.startswith(p + ".") for x in W_BY_PATH):
                    problems.append(f"unknown section: {p}")
                walk(v, p + ".")
                continue
            if p not in W_BY_PATH:
                problems.append(f"unknown field: {p}")
    walk(cfg)
    for name, key in (("attention", "attention.type"), ("ffn", "ffn.type")):
        v = get(cfg, key)
        allowed = {"attention": {"mha", "gqa", "mla"}, "ffn": {"mlp", "moe", "none"}}[name]
        if v not in allowed:
            problems.append(f"{key}: expected one of {sorted(allowed)}, got {v!r}")
    for p in ("dtypes.weight", "dtypes.expert", "dtypes.activation", "dtypes.kv",
              "dtypes.compute", "dtypes.attn_compute"):
        if get(cfg, p) not in DTYPE_BYTES:
            problems.append(f"{p}: unknown datatype {get(cfg, p)!r}")
    if get(cfg, "attention.type") == "gqa":
        h, kv = get(cfg, "attention.heads"), get(cfg, "attention.kv_heads") or get(cfg, "attention.heads")
        if h % max(1, kv):
            problems.append(f"attention.heads ({h}) must be a multiple of kv_heads ({kv})")
    pfx = int(get(cfg, "run.prefill_seq_len"))
    cds = int(get(cfg, "run.cur_decoding_seq_len"))
    mx = int(get(cfg, "run.max_seq_len"))
    if mx <= pfx + cds:
        problems.append(f"run.max_seq_len ({mx}) must be greater than run.prefill_seq_len + "
                        f"run.cur_decoding_seq_len ({pfx} + {cds} = {pfx + cds})")
    return problems


def to_model_spec(cfg: dict):
    """Turn the workload config into the internal block-level ModelSpec (one GPU, one layer kind)."""
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.model_spec import ModelSpec
    at = get(cfg, "attention.type")
    if at == "mla":
        attn = {"type": "mla", "heads": get(cfg, "attention.heads"),
                "q_lora_rank": get(cfg, "attention.q_lora_rank"),
                "kv_lora_rank": get(cfg, "attention.kv_lora_rank"),
                "qk_nope": get(cfg, "attention.qk_nope"), "qk_rope": get(cfg, "attention.qk_rope"),
                "v_head": get(cfg, "attention.v_head"), "absorb": get(cfg, "attention.absorb")}
    else:
        hd = get(cfg, "attention.head_dim")
        attn = {"type": at, "heads": get(cfg, "attention.heads"), "head_dim": hd,
                "v_head_dim": get(cfg, "attention.v_head_dim") or hd,
                "qk_norm": get(cfg, "attention.qk_norm")}
        if at == "gqa":
            attn["kv_heads"] = get(cfg, "attention.kv_heads") or get(cfg, "attention.heads")
    attn["impl"] = get(cfg, "attention.impl")
    attn["causal"] = get(cfg, "attention.causal")
    if get(cfg, "attention.sliding_window"):
        attn["sliding_window"] = get(cfg, "attention.sliding_window")

    ff = get(cfg, "ffn.type")
    blocks = [{"type": "norm"}, attn, {"type": "norm"}]
    if ff == "moe":
        blocks.append({"type": "moe", "experts": get(cfg, "ffn.experts"), "topk": get(cfg, "ffn.topk"),
                       "d_ff": get(cfg, "ffn.d_ff"), "shared_experts": get(cfg, "ffn.shared_experts")})
    elif ff == "mlp":
        blocks.append({"type": "mlp", "d_ff": get(cfg, "ffn.d_ff"), "gated": get(cfg, "ffn.gated")})
    raw = {"name": get(cfg, "name"), "hidden": get(cfg, "hidden"), "vocab": 0,
           "layers": [{"name": "layer", "repeat": int(get(cfg, "layers")), "blocks": blocks}]}
    return ModelSpec.from_dict(raw)


def to_run_config(cfg: dict):
    """RunConfig for a single GPU: no TP/DP/EP, everything resident here."""
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig
    return RunConfig(phase=get(cfg, "run.phase"), batch=int(get(cfg, "run.batch")),
                     seq_len=int(get(cfg, "run.seq_len")), tp=1, dp=1, ep=1,
                     weight_dtype=get(cfg, "dtypes.weight"), expert_dtype=get(cfg, "dtypes.expert"),
                     act_dtype=get(cfg, "dtypes.activation"), kv_dtype=get(cfg, "dtypes.kv"),
                     compute_dtype=get(cfg, "dtypes.compute"),
                     attn_compute_dtype=get(cfg, "dtypes.attn_compute"),
                     attn_impl=get(cfg, "attention.impl"),
                     include_lm_head=False)


def run_workload(cfg: dict, cur_gpu_config, progress=None):
    """Evaluate the workload on one GPU. Returns the standard ModelReport.

    `cfg` (the workload dict) only ever generates a `CurModelConfig` here — the model layer
    never sees "workload" itself. `cur_gpu_config` carries the tile policy
    (compute.tile_policy.*); that's GPU-side, not part of `cfg`."""
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.runner import CurModelConfig, run
    cur_model_config = CurModelConfig(spec=to_model_spec(cfg), run=to_run_config(cfg))
    return run(cur_gpu_config, cur_model_config, progress=progress)


def run_workload_both_phases(cfg: dict, cur_gpu_config, progress=None) -> dict:
    """Evaluate the workload's prefill and decode phases in one call.

    prefill runs at run.prefill_seq_len (the prompt); decode runs at run.cur_decoding_seq_len
    (one token-generation step at that KV length — decode is always a single step, so this is
    where that step is taken). Ignores run.phase/run.seq_len entirely — those are for
    run_workload()'s single-phase path. Raises ValueError (via validate_workload(), which
    includes the run.max_seq_len > prefill_seq_len + cur_decoding_seq_len check) if cfg is
    invalid. Returns {"prefill": ModelReport, "decode": ModelReport}."""
    problems = validate_workload(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.runner import CurModelConfig, run
    model_spec = to_model_spec(cfg)
    base_rc = to_run_config(cfg)
    prefill_rc = replace(base_rc, phase="prefill", seq_len=int(get(cfg, "run.prefill_seq_len")))
    decode_rc = replace(base_rc, phase="decode", seq_len=int(get(cfg, "run.cur_decoding_seq_len")))
    rep_prefill = run(cur_gpu_config, CurModelConfig(spec=model_spec, run=prefill_rc), progress=progress)
    rep_decode = run(cur_gpu_config, CurModelConfig(spec=model_spec, run=decode_rc), progress=progress)
    return {"prefill": rep_prefill, "decode": rep_decode}


def memory_breakdown(cfg: dict, cur_gpu_config=None) -> dict:
    """What the dimensions and datatypes imply for memory, before anything is simulated.

    weights: attention (q_a/q_b or q+kv_a, kv_b, o) + routed experts (ALL of them are resident;
    which ones are active is not known ahead of time — sparsity, run.ffn.topk / run.ffn.experts,
    only narrows what's on the compute path, never what's resident) + shared experts, x layers.
    KV cache: MLA stores the latent (kv_lora + rope) per token per layer; GQA/MHA store
    kv_heads x (head_dim + v_head_dim). Sized by run.max_seq_len (the "typical max" a request
    might reach), not run.seq_len — that's the whole point of keeping it separate from wherever
    the run currently samples a request (run.cur_decoding_seq_len).

    Pass `cur_gpu_config` to also check whether that KV cache actually fits in the on-chip
    buffer (compute.tile_policy sizing assumes it does — every KV read hits the buffer,
    0 misses to DDR — see kernels/attention.py's resident_frac()); the model always simulates
    correctly either way, this is just a heads-up when the assumption doesn't hold."""
    D = get(cfg, "hidden")
    L = int(get(cfg, "layers"))
    wb, eb, kvb = (DTYPE_BYTES[get(cfg, f"dtypes.{k}")] for k in ("weight", "expert", "kv"))
    at = get(cfg, "attention.type")
    if at == "mla":
        H = get(cfg, "attention.heads")
        qr, kvr = get(cfg, "attention.q_lora_rank"), get(cfg, "attention.kv_lora_rank")
        nope, rope, vh = (get(cfg, f"attention.{k}") for k in ("qk_nope", "qk_rope", "v_head"))
        attn_params = ((D * (qr + kvr + rope) if qr else D * (H * (nope + rope) + kvr + rope))
                       + (qr * H * (nope + rope) if qr else 0)
                       + kvr * H * (nope + vh) + H * vh * D)
        kv_per_token = (kvr + rope) * kvb
    else:
        H = get(cfg, "attention.heads")
        KV = get(cfg, "attention.kv_heads") or H
        hd = get(cfg, "attention.head_dim")
        vd = get(cfg, "attention.v_head_dim") or hd
        attn_params = D * (H * hd + KV * (hd + vd)) + H * vd * D
        kv_per_token = KV * (hd + vd) * kvb

    ff = get(cfg, "ffn.type")
    d_ff, experts, topk = get(cfg, "ffn.d_ff"), get(cfg, "ffn.experts"), get(cfg, "ffn.topk")
    shared = get(cfg, "ffn.shared_experts")
    per_expert = 3 * D * d_ff if get(cfg, "ffn.gated") else 2 * D * d_ff
    expert_bytes = per_expert * experts * eb if ff == "moe" else 0
    dense_bytes = (per_expert * shared * wb if ff == "moe" else
                   (per_expert * wb if ff == "mlp" else 0))
    router = D * experts * 2 if ff == "moe" else 0

    seq = int(get(cfg, "run.max_seq_len") or get(cfg, "run.seq_len"))
    batch = int(get(cfg, "run.batch"))
    kv_cache_GB = kv_per_token * L * seq * batch / 1e9
    out = {
        "layers": L,
        "attention_weights_GB": attn_params * wb * L / 1e9,
        "expert_weights_GB": expert_bytes * L / 1e9,
        "dense_weights_GB": (dense_bytes + router) * L / 1e9,
        "weights_GB": (attn_params * wb + expert_bytes + dense_bytes + router) * L / 1e9,
        "kv_bytes_per_token_per_layer": kv_per_token,
        "kv_cache_GB": kv_cache_GB,
        "kv_seq_len": seq, "batch": batch,
        "sparsity": f"{topk}/{experts} experts per token" if ff == "moe" else "dense",
        "active_expert_bytes_per_token_GB": per_expert * topk * eb / 1e9 if ff == "moe" else 0.0,
        "total_GB": ((attn_params * wb + expert_bytes + dense_bytes + router) * L
                     + kv_per_token * L * seq * batch) / 1e9,
    }
    if cur_gpu_config is not None:
        kv_buf_GB = cur_gpu_config.sram_capacity_for("kv") / 1e9
        out["on_chip_buffer_kv_capacity_GB"] = kv_buf_GB
        out["kv_fits_on_chip_buffer"] = kv_buf_GB >= kv_cache_GB
    return out


def compare_attention_impl(cfg: dict, cur_gpu_config) -> dict:
    """Run the same workload with and without flash attention.

    Flash keeps S and P on chip; the naive path writes the score matrix to HBM and reads it
    back, so the two differ in time, in what binds, and in how much activation memory the layer
    needs — which is why the choice is part of the workload config, not an internal assumption.
    """
    out = {}
    for impl in ("flash", "naive"):
        c = {**cfg, "attention": {**(cfg.get("attention") or {}), "impl": impl}}
        rep = run_workload(c, cur_gpu_config)
        attn = sum(o.total_s for o in rep.ops
                   if ".attn" in o.op.name and "allreduce" not in o.op.name)
        out[impl] = {"step_ms": rep.step_time_s * 1e3, "attention_ms": attn * 1e3,
                     "activations_GB": rep.memory.activations_GB,
                     "top_bound": next(iter(rep.detail_breakdown()), "-"),
                     "by_block": {k: v / rep.step_time_s for k, v in rep.by_domain().items()}}
    f, n = out["flash"], out["naive"]
    out["speedup_of_flash"] = n["step_ms"] / f["step_ms"] if f["step_ms"] else 0.0
    out["extra_activation_GB_without_flash"] = n["activations_GB"] - f["activations_GB"]
    return out


def as_markdown() -> str:
    L = ["# Workload configuration reference", "",
         "What runs on one GPU: one layer's attention + FFN, its dimensions and datatypes.", ""]
    for sec in W_SECTIONS:
        L += [f"## {sec}", "", "| field | type | unit | default | meaning |", "|---|---|---|---|---|"]
        for f in WORKLOAD_FIELDS:
            if f.section == sec:
                L.append(f"| `{f.path}` | {f.kind} | {f.unit or '—'} | `{f.default}` | {f.doc} |")
        L.append("")
    L.append(f"Total: {len(WORKLOAD_FIELDS)} fields in {len(W_SECTIONS)} sections.")
    return "\n".join(L)
