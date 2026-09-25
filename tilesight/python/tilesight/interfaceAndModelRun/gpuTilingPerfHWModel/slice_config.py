"""The slice-based GPU description.

This is the config a hardware architect fills in. It talks about the things you actually
choose when you design a part — how many tensor cores sit in a shader core, how big one MMA
tile is and how many cycles it takes, how many shader cores share an L1, how many memory
slices there are and what each one owns — and it *derives* the numbers the model needs
(per-core TFLOPS, whole-GPU PFLOPS, aggregate bandwidths, capacities).

Structure

    shader slice   : N shader cores sharing one L1
      shader core  : T tensor cores + W wave32 lanes, its own shared memory and GPR file,
                     each either shared by the tensor cores or split evenly between them
    memory slice   : one HBM channel + one L2 port + one DMA port; its own L2 (may be 0)
    on-chip buffer : one piece sits next to each shader slice and they are interconnected by a
                     switch, so logically it is one buffer. It sits BETWEEN the shader slices
                     and L2 — a request may also bypass it and go straight to the memory slice,
                     which is not automatically faster because HBM latency is hundreds of cycles.
                     Pre-allocated at tile granularity; contents chosen by policy
    gmem           : upstream (to the shader slices) and downstream (to the memory slices)
                     bandwidth, read and write separately

`to_hardware_spec()` translates all of it into the flat hardware config the model consumes, so
the model still only ever sees `gpuTilingPerfHWModel/schema.py` fields.
"""
from __future__ import annotations

from dataclasses import dataclass

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import DTYPE_BYTES, HardwareSpec


@dataclass(frozen=True)
class SField:
    path: str
    kind: str
    unit: str
    default: object
    section: str
    doc: str


S = SField
SLICE_FIELDS: tuple[SField, ...] = (
    S("name", "str", "", "custom-gpu", "identity", "Name of the part"),
    S("core.freq_ghz", "float", "GHz", 2.0, "shader core", "Core clock — everything in cycles scales with it"),
    S("core.wave32_per_core", "int", "count", 4, "shader core",
      "wave32 SIMD units per shader core (SIMD width is fixed at 32 lanes)"),
    S("core.tensor_cores", "int", "count", 4, "shader core", "Tensor cores inside one shader core"),
    S("core.resident_waves_per_simd", "int", "count", 8, "shader core",
      "wave32 contexts a SIMD can keep resident; sets how many blocks fit on a core"),

    S("tensor_core.tile_m", "int", "rows", 64, "tensor core", "MMA tile M"),
    S("tensor_core.tile_n", "int", "cols", 64, "tensor core", "MMA tile N"),
    S("tensor_core.tile_k", "int", "depth", 32, "tensor core", "MMA tile K"),
    S("tensor_core.tile_latency_cycles", "float", "cycles", 16, "tensor core",
      "Cycles one tensor core needs for one MMA tile (issue-to-issue, i.e. the throughput cost)"),
    S("tensor_core.dtype", "str", "", "bf16", "tensor core",
      "Datatype of the MMA datapath. Simplification: all tensors use the KQV datatype"),
    S("attention_tile.m", "int", "rows", 64, "tensor core", "Attention tile M (query rows / heads per block)"),
    S("attention_tile.n", "int", "cols", 64, "tensor core", "Attention tile N (KV rows per step)"),

    S("shader_core.shared_mem_KB", "float", "KB", 228, "shader core", "Shared memory per shader core"),
    S("shader_core.shared_mem_shared_by_tc", "bool", "", True, "shader core",
      "True: the tensor cores share it; False: it is split evenly between them"),
    S("shader_core.gpr_KB", "float", "KB", 256, "shader core", "Register file per shader core"),
    S("shader_core.gpr_shared_by_tc", "bool", "", True, "shader core", "Shared or split evenly"),

    S("shader_slice.cores_per_slice", "int", "count", 8, "shader slice",
      "Shader cores in one slice — they share the slice's L1"),
    S("shader_slice.count", "int", "count", 20, "shader slice",
      "Shader slices. Slices are copies of each other; the model staggers them in time"),
    S("shader_slice.l1_KB", "float", "KB", 256, "shader slice",
      "L1 per slice, shared by its cores. Tile-granular and deterministic: the model knows "
      "exactly which tile sits in it"),
    S("shader_slice.l1_latency_cycles", "float", "cycles", 20, "shader slice", "L1 hit latency"),
    S("shader_slice.l1_bytes_per_clk", "float", "B/clk", 128, "shader slice", "L1 bandwidth per slice"),

    S("memory_slice.count", "int", "count", 8, "memory slice",
      "Memory slices; each owns one HBM channel, one L2 port and one DMA port"),
    S("memory_slice.l2_MB", "float", "MB", 16, "memory slice",
      "L2 per memory slice. 0 = no L2: the port and the DMA talk straight to the on-chip buffer"),
    S("memory_slice.l2_bytes_per_clk", "float", "B/clk", 512, "memory slice", "L2 port width"),
    S("memory_slice.l2_latency_cycles", "float", "cycles", 200, "memory slice", "L2 hit latency"),
    S("memory_slice.hbm_GB", "float", "GB", 24, "memory slice", "HBM capacity attached to this slice"),
    S("memory_slice.hbm_latency_cycles", "float", "cycles", 1200, "memory slice", "HBM latency"),
    S("memory_slice.dma_ports", "int", "count", 1, "memory slice", "DMA ports per memory slice"),

    S("addressing.mode", "str", "", "interleave", "addressing",
      "linear (one slice owns a contiguous range) | interleave (stripe across slices)"),
    S("addressing.interleave_KB", "float", "KB", 1, "addressing", "Stripe size when mode = interleave"),

    S("onchip_buffer.capacity_MB", "float", "MB", 256, "on-chip buffer",
      "Logically one buffer (physically a piece per slice). Not part of a memory slice"),
    S("onchip_buffer.bytes_per_clk", "float", "B/clk", 4096, "on-chip buffer",
      "Aggregate bandwidth of the buffer pieces themselves"),
    S("onchip_buffer.switch_bytes_per_clk", "float", "B/clk", 0, "on-chip buffer",
      "Aggregate bandwidth of the switch between the shader slices and the buffer pieces "
      "(0 = same as the buffer bandwidth). One slice may draw its 1/slices share"),
    S("onchip_buffer.latency_cycles", "float", "cycles", 60, "on-chip buffer", "Buffer latency"),
    S("onchip_buffer.contents", "str", "", "ab", "on-chip buffer",
      "What is pre-allocated on it, at tile granularity: "
      "ab (A and B) | abc (A, B and C) | abc_kv (plus the KV cache) | abc_kv_moe (plus every "
      "expert, because which ones are active is not known ahead of time) | none"),

    S("gmem.upstream_read_GBps", "float", "GB/s", 0, "gmem",
      "Read bandwidth towards the shader slices (0 = derive it from the L1/L2 port widths)"),
    S("gmem.upstream_write_GBps", "float", "GB/s", 0, "gmem", "Write bandwidth towards the shader slices"),
    S("gmem.downstream_read_GBps", "float", "GB/s", 0, "gmem",
      "Read bandwidth towards the memory slices (0 = derive it from the HBM channels)"),
    S("gmem.downstream_write_GBps", "float", "GB/s", 0, "gmem", "Write bandwidth towards the memory slices"),
    S("gmem.hbm_GBps_per_slice", "float", "GB/s", 1000, "gmem", "HBM bandwidth of one memory slice"),

    S("network.domain_size", "int", "GPUs", 8, "network", "Scale-up domain (single-GPU runs ignore it)"),
    S("network.bandwidth_GBps", "float", "GB/s", 900, "network", "Per-GPU scale-up bandwidth"),
    S("runtime.launch_overhead_us", "float", "us", 2.0, "runtime", "Per-kernel launch cost"),
)

S_BY_PATH = {f.path: f for f in SLICE_FIELDS}
S_SECTIONS = tuple(dict.fromkeys(f.section for f in SLICE_FIELDS))
BUFFER_CONTENTS = {
    "none": "nothing is pinned",
    "ab": "the A and B operands of every GEMM",
    "abc": "A, B and C",
    "abc_kv": "A, B, C and the KV cache",
    "abc_kv_moe": "A, B, C, the KV cache and every expert weight (active or not)",
}


def sget(cfg: dict, path: str):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            f = S_BY_PATH.get(path)
            return f.default if f else None
        node = node[k]
    return node


def validate_slice_config(cfg: dict) -> list[str]:
    problems: list[str] = []

    def walk(node, prefix=""):
        for k, v in (node or {}).items():
            p = f"{prefix}{k}"
            if isinstance(v, dict):
                if not any(x.startswith(p + ".") for x in S_BY_PATH):
                    problems.append(f"unknown section: {p}")
                walk(v, p + ".")
                continue
            if p not in S_BY_PATH:
                problems.append(f"unknown field: {p}")
    walk(cfg)
    if sget(cfg, "onchip_buffer.contents") not in BUFFER_CONTENTS:
        problems.append(f"onchip_buffer.contents: expected one of {sorted(BUFFER_CONTENTS)}")
    if sget(cfg, "addressing.mode") not in ("linear", "interleave"):
        problems.append("addressing.mode: expected linear | interleave")
    if sget(cfg, "tensor_core.dtype") not in DTYPE_BYTES:
        problems.append(f"tensor_core.dtype: unknown datatype {sget(cfg, 'tensor_core.dtype')!r}")
    return problems


def derive(cfg: dict) -> dict:
    """Everything the config implies — shown in the UI so the numbers are never a surprise."""
    f = sget(cfg, "core.freq_ghz") * 1e9
    tm, tn, tk = (sget(cfg, f"tensor_core.tile_{x}") for x in "mnk")
    lat = max(1e-9, sget(cfg, "tensor_core.tile_latency_cycles"))
    tc_per_core = sget(cfg, "core.tensor_cores")
    cores = sget(cfg, "shader_slice.cores_per_slice") * sget(cfg, "shader_slice.count")
    mslices = sget(cfg, "memory_slice.count")

    flops_per_tile = 2.0 * tm * tn * tk
    tc_flops = flops_per_tile / lat * f                      # one tensor core
    core_flops = tc_flops * tc_per_core                      # one shader core
    total_flops = core_flops * cores

    waves = sget(cfg, "core.wave32_per_core")
    vector_flops = waves * 32 * 2 * f * cores                # FMA per lane per clock

    hbm = sget(cfg, "gmem.hbm_GBps_per_slice") * mslices * 1e9
    down_r = (sget(cfg, "gmem.downstream_read_GBps") or 0) * 1e9 or hbm
    down_w = (sget(cfg, "gmem.downstream_write_GBps") or 0) * 1e9 or hbm
    l2_bw = sget(cfg, "memory_slice.l2_bytes_per_clk") * f * mslices
    l1_bw = sget(cfg, "shader_slice.l1_bytes_per_clk") * f * sget(cfg, "shader_slice.count")
    up_r = (sget(cfg, "gmem.upstream_read_GBps") or 0) * 1e9 or min(l1_bw, l2_bw or l1_bw)
    up_w = (sget(cfg, "gmem.upstream_write_GBps") or 0) * 1e9 or up_r

    smem = sget(cfg, "shader_core.shared_mem_KB")
    gpr = sget(cfg, "shader_core.gpr_KB")
    per_tc_smem = smem if sget(cfg, "shader_core.shared_mem_shared_by_tc") else smem / tc_per_core
    per_tc_gpr = gpr if sget(cfg, "shader_core.gpr_shared_by_tc") else gpr / tc_per_core

    return {
        "shader_cores": cores,
        "tensor_cores": cores * tc_per_core,
        "tflops_per_tensor_core": tc_flops / 1e12,
        "tflops_per_shader_core": core_flops / 1e12,
        "pflops_total": total_flops / 1e15,
        "vector_tflops": vector_flops / 1e12,
        "mma_tile": f"{tm}x{tn}x{tk} {sget(cfg, 'tensor_core.dtype')} in {lat:g} cycles",
        "attention_tile": f"{sget(cfg, 'attention_tile.m')}x{sget(cfg, 'attention_tile.n')}",
        "hbm_capacity_GB": sget(cfg, "memory_slice.hbm_GB") * mslices,
        "hbm_TBps": hbm / 1e12,
        "downstream_read_TBps": down_r / 1e12, "downstream_write_TBps": down_w / 1e12,
        "upstream_read_TBps": up_r / 1e12, "upstream_write_TBps": up_w / 1e12,
        "l2_total_MB": sget(cfg, "memory_slice.l2_MB") * mslices,
        "l2_TBps": l2_bw / 1e12,
        "l1_total_MB": sget(cfg, "shader_slice.l1_KB") * sget(cfg, "shader_slice.count") / 1024,
        "l1_TBps": l1_bw / 1e12,
        "buffer_MB": sget(cfg, "onchip_buffer.capacity_MB"),
        "buffer_TBps": sget(cfg, "onchip_buffer.bytes_per_clk") * f / 1e12,
        "switch_TBps": ((sget(cfg, "onchip_buffer.switch_bytes_per_clk")
                         or sget(cfg, "onchip_buffer.bytes_per_clk")) * f / 1e12),
        "switch_TBps_per_slice": ((sget(cfg, "onchip_buffer.switch_bytes_per_clk")
                                   or sget(cfg, "onchip_buffer.bytes_per_clk")) * f
                                  / max(1, sget(cfg, "shader_slice.count")) / 1e12),
        "buffer_MB_per_slice": (sget(cfg, "onchip_buffer.capacity_MB")
                                / max(1, sget(cfg, "shader_slice.count"))),
        "buffer_contents": BUFFER_CONTENTS[sget(cfg, "onchip_buffer.contents")],
        "smem_per_tensor_core_KB": per_tc_smem,
        "gpr_per_tensor_core_KB": per_tc_gpr,
        "memory_slices": mslices,
        "addressing": (f"{sget(cfg, 'addressing.mode')}"
                       + (f" @ {sget(cfg, 'addressing.interleave_KB')} KB"
                          if sget(cfg, "addressing.mode") == "interleave" else "")),
    }


_PIN = {"none": {}, "ab": {"weight": 0.5, "act": 0.5}, "abc": {"weight": 0.45, "act": 0.55},
        "abc_kv": {"weight": 0.35, "act": 0.3, "kv": 0.35},
        "abc_kv_moe": {"weight": 0.6, "act": 0.15, "kv": 0.25}}


def to_hardware_spec(cfg: dict) -> HardwareSpec:
    """Translate the slice description into the flat config the model consumes."""
    d = derive(cfg)
    f_ghz = sget(cfg, "core.freq_ghz")
    dt = sget(cfg, "tensor_core.dtype")
    tc_peak = d["pflops_total"] * 1000                       # TFLOP/s
    mslices = sget(cfg, "memory_slice.count")
    l2_mb = sget(cfg, "memory_slice.l2_MB") * mslices
    contents = sget(cfg, "onchip_buffer.contents")

    raw = {
        "name": sget(cfg, "name"),
        "arch": "slice-config",
        "sms": d["shader_cores"],
        "clock_ghz": f_ghz,
        "dies": 1,
        "compute": {
            "tc_dense_tflops": {dt: tc_peak},
            "cuda_fp32_tflops": d["vector_tflops"],
            "sfu_tops": d["vector_tflops"] / 4,               # one transcendental per 4 lanes-clk
            "tc_min_m": sget(cfg, "tensor_core.tile_m"),
            "cta_pair": False,
            "cluster_multicast": False,
            "mma_latency_cycles": sget(cfg, "tensor_core.tile_latency_cycles"),
            "sfu_latency_cycles": 16,
            # the attention tile is a property of the part in this description
            "attention_tile_m": sget(cfg, "attention_tile.m"),
            "attention_tile_n": sget(cfg, "attention_tile.n"),
        },
        "memory": {
            "interleave_KB": sget(cfg, "addressing.interleave_KB"),
            "addressing": {
                "addr_bits": 48,
                "l2": {"ports": mslices,
                       "mode": "range" if sget(cfg, "addressing.mode") == "linear" else "interleave",
                       "granularity_KB": sget(cfg, "addressing.interleave_KB")},
                "ddr": {"ports": mslices,
                        "mode": "range" if sget(cfg, "addressing.mode") == "linear" else "interleave",
                        "granularity_KB": sget(cfg, "addressing.interleave_KB")},
            },
            "ddr": {"capacity_GB": d["hbm_capacity_GB"], "bandwidth_TBps": d["downstream_read_TBps"],
                    "latency_ns": sget(cfg, "memory_slice.hbm_latency_cycles") / f_ghz,
                    "ports": mslices},
            "l2": {"capacity_MB": max(0.001, l2_mb), "capacity_derate": 1.0,
                   "bandwidth_TBps": max(d["l2_TBps"], d["upstream_read_TBps"]),
                   "latency_ns": sget(cfg, "memory_slice.l2_latency_cycles") / f_ghz,
                   "partitions": mslices, "policy": "lru", "waves_simulated": 2,
                   "blocks": mslices, "ports_per_block": 1,
                   "bytes_per_clk_per_port": sget(cfg, "memory_slice.l2_bytes_per_clk"),
                   "line_bytes": 128, "sector_bytes": 32,
                   "per_sm_max_GBps": d["upstream_read_TBps"] * 1e3 / max(1, d["shader_cores"]) * 4},
            "l1": {"owner": "cluster", "capacity_KB": sget(cfg, "shader_slice.l1_KB"),
                   "smem_carveout_KB": 0, "cluster_size": sget(cfg, "shader_slice.cores_per_slice"),
                   "cache_global_loads": True, "capacity_derate": 1.0,
                   "bytes_per_clk": sget(cfg, "shader_slice.l1_bytes_per_clk"),
                   "latency_cycles": sget(cfg, "shader_slice.l1_latency_cycles"), "policy": "lru"},
            "onchip": {"smem": {"capacity_KB": d["smem_per_tensor_core_KB"],
                                "bytes_per_clk": 128, "latency_cycles": 30}},
            "dma": {"engines": mslices * sget(cfg, "memory_slice.dma_ports"),
                    "per_l2_block": True,
                    "destination": "bypass" if l2_mb <= 0 else "smem"},
            "outstanding": {"per_sm_lines": 0},
            "queueing": {"coef": 0.0, "max_factor": 3.0},
        },
        "load_paths": {
            "tma": {"engine": "dma", "per_sm_GBps": d["upstream_read_TBps"] * 1e3 / max(1, d["shader_cores"]),
                    "latency_ns": 100, "smem_direct": True, "multicast": False,
                    "regs_per_thread": 0, "issue_bytes_per_clk": 0},
            "lsu": {"engine": "lsu", "per_sm_GBps": d["upstream_read_TBps"] * 1e3 / max(1, d["shader_cores"]) * 0.6,
                    "latency_ns": 50, "smem_direct": False, "multicast": False,
                    "regs_per_thread": 16, "issue_bytes_per_clk": 64},
        },
        "load_paths_default": "tma",
        "occupancy": {"max_blocks_per_sm": 32,
                      "max_threads_per_sm": (sget(cfg, "core.wave32_per_core") * 32
                                             * sget(cfg, "core.resident_waves_per_simd")),
                      "regs_per_sm": int(sget(cfg, "shader_core.gpr_KB") * 1024 / 4),
                      "max_regs_per_thread": 255},
        "network": {"nvlink": {"domain_size": sget(cfg, "network.domain_size"),
                               "bandwidth_GBps": sget(cfg, "network.bandwidth_GBps"), "alpha_us": 3.0},
                    "scaleout": {"bandwidth_GBps": 50, "alpha_us": 8.0}},
        "runtime": {"launch_overhead_us": sget(cfg, "runtime.launch_overhead_us")},
    }
    if contents != "none":
        raw["memory"]["sram"] = {
            "capacity_MB": sget(cfg, "onchip_buffer.capacity_MB"), "capacity_derate": 1.0,
            "bandwidth_TBps": d["buffer_TBps"], "switch_TBps": d["switch_TBps"],
            "latency_ns": sget(cfg, "onchip_buffer.latency_cycles") / f_ghz,
            "per_sm_max_GBps": 1e9, "policy": "pin", "pin": _PIN[contents],
            "keep_intermediates": contents in ("abc", "abc_kv", "abc_kv_moe"),
            "bypass_l2": l2_mb <= 0, "prefetch": True,
        }
    return HardwareSpec(raw)


def as_markdown() -> str:
    L = ["# Slice-based GPU configuration", "",
         "The architect's view: shader slices and memory slices, what sits inside them, and the "
         "on-chip buffer that is shared by both. `to_hardware_spec()` derives the flat config "
         "the model consumes.", ""]
    for sec in S_SECTIONS:
        L += [f"## {sec}", "", "| field | type | unit | default | meaning |", "|---|---|---|---|---|"]
        for f in SLICE_FIELDS:
            if f.section == sec:
                L.append(f"| `{f.path}` | {f.kind} | {f.unit or '—'} | `{f.default}` | {f.doc} |")
        L.append("")
    return "\n".join(L)
