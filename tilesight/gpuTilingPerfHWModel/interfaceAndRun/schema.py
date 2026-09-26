"""The hardware configuration schema — the interface between a GPU and the model.

This file is the ONLY place that knows what a GPU description contains. The model (engine,
kernel lowerings, model layer) never hard-codes a device, a vendor or a default: it asks
`HardwareSpec.get(path)` and the value comes from the user's YAML, or from the default
declared here. Adding a GPU means writing a YAML; adding a hardware *concept* means adding a
field here and using it in the model.

Each field carries:
  path        dotted location in the YAML
  kind        float | int | bool | str | map      (map = free-form sub-tree, e.g. per-dtype peaks)
  unit        physical unit, "" when dimensionless
  default     what the model uses when the field is absent; REQUIRED = the config must give it,
              None = optional with no default (the model checks for its presence)
  tag         spec   — a vendor/datasheet number, copy it, do not invent
              calib  — must be measured on the part (microbenchmark); defaults are placeholders
              policy — a modelling choice, not a property of the silicon
              loss   — a derating knob; DEFAULT IS ALWAYS "no loss"
  section     grouping for the reference doc
  doc         one-line meaning
"""
from __future__ import annotations

from dataclasses import dataclass


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()


@dataclass(frozen=True)
class Field:
    path: str
    kind: str
    unit: str
    default: object
    tag: str
    section: str
    doc: str


F = Field
FIELDS: tuple[Field, ...] = (
    # ---------------------------------------------------------------- identity
    F("name", "str", "", REQUIRED, "spec", "identity", "Display name of the part"),
    F("arch", "str", "", "", "spec", "identity", "Architecture tag (informational)"),
    F("sms", "int", "count", REQUIRED, "spec", "identity",
      "Streaming multiprocessors / compute units. Compute peaks are whole-GPU, so the model "
      "derives the per-SM rate as peak / sms"),
    F("clock_ghz", "float", "GHz", 1.8, "spec", "identity", "Clock used for every cycle <-> second conversion"),
    F("dies", "int", "count", 1, "spec", "identity", "Dies/XCDs; default number of L2 slices is 8 per die"),
    F("wavefront", "int", "threads", 32, "spec", "identity", "Warp / wavefront width (informational today)"),

    # ---------------------------------------------------------------- compute
    F("compute.tc_dense_tflops", "map", "TFLOP/s", REQUIRED, "spec", "compute",
      "Dense tensor-core peak per datapath: fp4 / fp6 / fp8 / int8 / bf16 / fp16 / tf32. "
      "A dtype the part lacks widens to the next one present"),
    F("compute.cuda_fp32_tflops", "float", "TFLOP/s", REQUIRED, "spec", "compute", "Vector FP32 peak"),
    F("compute.sfu_tops", "float", "TOP/s", REQUIRED, "spec", "compute",
      "Transcendental (exp) throughput; drives softmax. Measured is often far below spec"),
    F("compute.tc_min_m", "int", "rows", 64, "spec", "compute",
      "Smallest M one MMA instruction covers; smaller tiles are padded to it"),
    F("compute.attention_tile_m", "int", "rows", 0, "spec", "compute",
      "Attention tile M fixed by the part (0 = let the tile search choose)"),
    F("compute.attention_tile_n", "int", "cols", 0, "spec", "compute",
      "Attention tile N fixed by the part (0 = let the tile search choose)"),
    F("compute.cta_pair", "bool", "", False, "spec", "compute", "2-CTA MMA (Blackwell tcgen05 cta_group::2)"),
    F("compute.cluster_multicast", "bool", "", False, "spec", "compute",
      "Thread-block clusters with TMA multicast; required for cluster_m > 1 tiles"),
    F("compute.mma_latency_cycles", "float", "cycles", 64, "calib", "compute",
      "Issue-to-result of one tile MMA. Independent MMAs overlap, so this costs pipeline fill, "
      "not throughput"),
    F("compute.cuda_latency_cycles", "float", "cycles", 4, "calib", "compute", "Vector op latency"),
    F("compute.sfu_latency_cycles", "float", "cycles", 16, "calib", "compute",
      "Transcendental latency; loop-carried in online softmax, so it does bound attention"),

    # ---------------------------------------------------------------- losses
    F("efficiency.tc", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak tensor core"),
    F("efficiency.cuda", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak vector"),
    F("efficiency.sfu", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak SFU"),
    F("efficiency.smem", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak SMEM or LDS"),
    F("efficiency.l2", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak L2"),
    F("efficiency.ddr", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak HBM"),
    F("efficiency.sram", "float", "ratio", 1.0, "loss", "losses", "Achieved / peak on-chip buffer"),

    # ---------------------------------------------------------------- HBM
    F("memory.ddr.capacity_GB", "float", "GB", REQUIRED, "spec", "memory.ddr", "HBM capacity per GPU"),
    F("memory.ddr.bandwidth_TBps", "float", "TB/s", REQUIRED, "spec", "memory.ddr", "HBM peak bandwidth"),
    F("memory.ddr.latency_ns", "float", "ns", 800, "calib", "memory.ddr", "Idle load-to-use latency from HBM"),
    F("memory.ddr.ports", "int", "count", 8, "spec", "memory.ddr",
      "HBM ports/stacks; used by the address map (aggregated for bandwidth)"),
    F("memory.ddr.blocks", "int", "count", 0, "calib", "memory.ddr", "Independent controllers (0 = do not port-cap)"),
    F("memory.ddr.ports_per_block", "int", "count", 1, "calib", "memory.ddr", "Ports per controller"),
    F("memory.ddr.bytes_per_clk_per_port", "float", "B/clk", 0, "calib", "memory.ddr",
      "Port width; blocks x ports x width x clock caps the level"),

    # ---------------------------------------------------------------- L2
    F("memory.l2.capacity_MB", "float", "MB", REQUIRED, "spec", "memory.l2", "Physical L2 (or MALL) capacity"),
    F("memory.l2.capacity_derate", "float", "ratio", 1.0, "loss", "memory.l2",
      "Fraction of the capacity the simulation may use. DEFAULT 1.0 = no loss"),
    F("memory.l2.effective_capacity_MB", "float", "MB", None, "calib", "memory.l2",
      "Direct override of the usable capacity (a measured bandwidth-vs-working-set cliff)"),
    F("memory.l2.bandwidth_TBps", "float", "TB/s", REQUIRED, "calib", "memory.l2", "L2 peak bandwidth"),
    F("memory.l2.latency_ns", "float", "ns", 300, "calib", "memory.l2", "L2 hit latency"),
    F("memory.l2.per_sm_max_GBps", "float", "GB/s", 1e9, "calib", "memory.l2",
      "Most one SM can pull from the shared levels; binds when the grid is small"),
    F("memory.l2.partitions", "int", "count", 1, "spec", "memory.l2",
      "Independent partitions, each a tile-level cache; a tile goes to the one its address maps to"),
    F("memory.l2.policy", "str", "", "lru", "policy", "memory.l2", "Replacement: lru | fifo | mru"),
    F("memory.l2.model", "str", "", "lru", "policy", "memory.l2",
      "lru = deterministic tile simulation (default) | sdcm = the paper's probabilistic model"),
    F("memory.l2.waves_simulated", "int", "count", 2, "policy", "memory.l2",
      "Waves replayed in the simulation; >1 makes cross-wave reuse visible"),
    F("memory.l2.assoc", "int", "ways", 16, "calib", "memory.l2", "Associativity (only used by the sdcm model)"),
    F("memory.l2.slices", "int", "count", 0, "calib", "memory.l2", "Slices for the address map (0 = 8 per die)"),
    F("memory.l2.blocks", "int", "count", 0, "calib", "memory.l2",
      "Independent L2 blocks, each with its own load/store port (0 = do not port-cap)"),
    F("memory.l2.ports_per_block", "int", "count", 1, "calib", "memory.l2", "Ports per block"),
    F("memory.l2.bytes_per_clk_per_port", "float", "B/clk", 0, "calib", "memory.l2",
      "Port width; blocks x ports x width x clock caps the level"),
    F("memory.l2.line_bytes", "int", "B", 128, "spec", "memory.l2", "Cache line (used by the Little's-law cap)"),
    F("memory.l2.sector_bytes", "int", "B", 32, "spec", "memory.l2", "Sector a miss actually fetches"),
    F("memory.l2.apply_conflict_penalty", "bool", "", False, "policy", "memory.l2",
      "Fold the measured address-spread imbalance into efficiency.l2"),

    # ---------------------------------------------------------------- L1
    F("memory.l1.owner", "str", "", "sm", "spec", "memory.l1", "sm | cluster (a cluster shares its L1s)"),
    F("memory.l1.capacity_KB", "float", "KB", 0, "spec", "memory.l1", "Unified L1 + scratchpad per SM"),
    F("memory.l1.smem_carveout_KB", "float", "KB", 0, "spec", "memory.l1",
      "What the tile plan takes as scratchpad; only the remainder caches global loads"),
    F("memory.l1.cluster_size", "int", "SMs", 1, "spec", "memory.l1", "SMs per cluster when owner = cluster"),
    F("memory.l1.cache_global_loads", "bool", "", True, "policy", "memory.l1",
      "False = global loads bypass L1"),
    F("memory.l1.capacity_derate", "float", "ratio", 1.0, "loss", "memory.l1", "DEFAULT 1.0 = no loss"),
    F("memory.l1.bytes_per_clk", "float", "B/clk", 128, "calib", "memory.l1", "L1 bandwidth per SM"),
    F("memory.l1.latency_cycles", "float", "cycles", 20, "calib", "memory.l1", "L1 hit latency"),
    F("memory.l1.policy", "str", "", "lru", "policy", "memory.l1", "Replacement policy"),

    # ---------------------------------------------------------------- on-chip scratchpads
    F("memory.onchip.smem.capacity_KB", "float", "KB", REQUIRED, "spec", "memory.onchip",
      "Shared memory / LDS per SM available to a tile plan"),
    F("memory.onchip.smem.bytes_per_clk", "float", "B/clk", 128, "spec", "memory.onchip", "SMEM bandwidth per SM"),
    F("memory.onchip.smem.latency_cycles", "float", "cycles", 30, "calib", "memory.onchip", "SMEM latency"),
    F("memory.onchip.tmem.capacity_KB", "float", "KB", 0, "spec", "memory.onchip",
      "Tensor memory per SM (Blackwell); its presence also moves accumulators out of registers"),
    F("memory.onchip.tmem.read_TBps", "float", "TB/s", 0, "calib", "memory.onchip", "TMEM read bandwidth per SM"),
    F("memory.onchip.tmem.write_TBps", "float", "TB/s", 0, "calib", "memory.onchip", "TMEM write bandwidth per SM"),
    F("memory.onchip.tmem.latency_cycles", "float", "cycles", 16, "calib", "memory.onchip", "TMEM latency"),

    # ---------------------------------------------------------------- optional shared buffer
    F("memory.sram.capacity_MB", "float", "MB", 0, "spec", "memory.sram",
      "Extra on-chip shared buffer between L2 and HBM (absent = no such buffer)"),
    F("memory.sram.capacity_derate", "float", "ratio", 1.0, "loss", "memory.sram", "DEFAULT 1.0 = no loss"),
    F("memory.sram.bandwidth_TBps", "float", "TB/s", 0, "calib", "memory.sram", "Buffer bandwidth"),
    F("memory.sram.switch_TBps", "float", "TB/s", 0, "calib", "memory.sram",
      "Aggregate bandwidth of the switch between the shader slices and the buffer pieces "
      "(0 = same as the buffer bandwidth). A slice draws its 1/slices share of it"),
    F("memory.sram.latency_ns", "float", "ns", 400, "calib", "memory.sram", "Buffer latency"),
    F("memory.sram.per_sm_max_GBps", "float", "GB/s", 1e9, "calib", "memory.sram", "Per-SM ceiling"),
    F("memory.sram.policy", "str", "", "cache", "policy", "memory.sram", "cache | pin"),
    F("memory.sram.pin", "map", "share", None, "policy", "memory.sram",
      "Capacity shares per tensor class when policy = pin: {weight, kv, act}"),
    F("memory.sram.alloc", "map", "share", None, "policy", "memory.sram", "Alias of pin"),
    F("memory.sram.keep_intermediates", "bool", "", False, "policy", "memory.sram",
      "Elementwise intermediates stay on chip instead of round-tripping HBM"),
    F("memory.sram.bypass_l2", "bool", "", False, "policy", "memory.sram",
      "Resident bytes skip the L2 datapath (otherwise L2 becomes the next wall)"),
    F("memory.sram.prefetch", "bool", "", False, "policy", "memory.sram", "Resident bytes cost no exposed latency"),
    F("memory.sram.costream", "bool", "", False, "policy", "memory.sram", "Buffer and HBM serve in parallel"),
    F("memory.sram.stage", "map", "", None, "policy", "memory.sram",
      "Mega-tile staging: {share_blocks: N} — one DMA fills a panel that N blocks consume"),


    # ---------------------------------------------------------------- tensor-core geometry
    F("compute.tensor_core.tile_m", "int", "rows", 64, "spec", "tensor_core",
      "M of one tensor-core tile instruction (all tensor datatypes use the KQV dtype here)"),
    F("compute.tensor_core.tile_n", "int", "cols", 64, "spec", "tensor_core", "N of one tensor-core tile"),
    F("compute.tensor_core.tile_k", "int", "depth", 32, "spec", "tensor_core", "K of one tensor-core tile"),
    F("compute.tensor_core.tile_latency_cycles", "float", "cycles", 16, "calib", "tensor_core",
      "Issue-to-result of ONE tile instruction"),
    F("compute.tensor_core.tile_interval_cycles", "float", "cycles", 4, "calib", "tensor_core",
      "Issue interval between back-to-back tile instructions on one tensor core (throughput)"),
    F("compute.tensor_core.per_shader_core", "int", "count", 4, "spec", "tensor_core",
      "Tensor cores inside one shader core (2, 4, 16 ...)"),

    # ---------------------------------------------------------------- tile policy
    # How the GPU tiles a GEMM/attention op — a modelling choice about *this part's* run,
    # not a property of the model/workload. kernels/tiles.py owns the search space and the
    # TileConfig/AttnTileConfig shapes; this is just which point in that space to use.
    F("compute.tile_policy.gemm", "str", "", "auto", "policy", "tile_policy",
      'GEMM block tile: "auto" searches the space per op shape, or a JSON TileConfig '
      'override, e.g. {"bm":128,"bn":256,"bk":64}'),
    F("compute.tile_policy.attn", "str", "", "auto", "policy", "tile_policy",
      '"auto" or a JSON AttnTileConfig override. Ignored on parts where compute.'
      'attention_tile_m/n force a fixed attention tile.'),
    F("compute.tile_policy.overrides", "map", "", {}, "policy", "tile_policy",
      "fnmatch pattern on op name -> partial tile fields, applied before gemm/attn above"),

    # ---------------------------------------------------------------- shader slice
    F("shader.cores_per_slice", "int", "count", 1, "spec", "shader",
      "Shader cores in one shader slice; a slice shares one L1"),
    F("shader.slices", "int", "count", 0, "spec", "shader",
      "Shader slices (0 = sms / cores_per_slice). Slices are copies that run staggered in time"),
    F("shader.wave32_per_core", "int", "count", 4, "spec", "shader",
      "wave32 (SIMD-32) issue slots per shader core; with the clock this gives the vector rate"),
    F("shader.clock_ghz", "float", "GHz", 0, "spec", "shader", "Core clock (0 = use the top-level clock_ghz)"),
    F("shader.smem_per_core_KB", "float", "KB", 0, "spec", "shader",
      "Scratchpad per shader core (0 = use memory.onchip.smem.capacity_KB)"),
    F("shader.smem_sharing", "str", "", "shared", "spec", "shader",
      "How the tensor cores inside a shader core see the scratchpad: shared | split"),
    F("shader.gpr_total_KB", "float", "KB", 0, "spec", "shader", "Register file per shader core (0 = from occupancy.regs_per_sm)"),
    F("shader.gpr_sharing", "str", "", "split", "spec", "shader", "Register file: shared | split across the tensor cores"),
    F("shader.l1_per_slice_KB", "float", "KB", 0, "spec", "shader",
      "L1 per shader slice (0 = memory.l1.capacity_KB). Tile-granular, deterministic residency"),

    # ---------------------------------------------------------------- memory slices
    F("memory.slices.count", "int", "count", 0, "spec", "memory.slices",
      "Memory slices; each owns its HBM stack, its L2 port and its DMA port (0 = ddr.ports)"),
    F("memory.slices.l2_per_slice_MB", "float", "MB", 0, "spec", "memory.slices",
      "L2 inside one memory slice. 0 = no L2 at all: the port and the DMA talk to the on-chip buffer"),
    F("memory.slices.hbm_per_slice_GB", "float", "GB", 0, "spec", "memory.slices",
      "HBM capacity per slice (0 = ddr.capacity_GB / slices)"),
    F("memory.slices.address_mode", "str", "", "interleave", "spec", "memory.slices",
      "How a tensor is spread over the slices: linear | interleave"),
    F("memory.slices.interleave_KB", "float", "KB", 1, "spec", "memory.slices",
      "Interleave granularity when address_mode = interleave (1, 4, 16 ...)"),

    # ---------------------------------------------------------------- gmem bandwidth
    F("memory.gmem.downstream_read_GBps_per_slice", "float", "GB/s", 0, "spec", "memory.gmem",
      "Read bandwidth from one memory slice towards the fabric"),
    F("memory.gmem.downstream_write_GBps_per_slice", "float", "GB/s", 0, "spec", "memory.gmem",
      "Write bandwidth into one memory slice"),
    F("memory.gmem.upstream_read_GBps_per_slice", "float", "GB/s", 0, "spec", "memory.gmem",
      "Read bandwidth delivered to one shader slice"),
    F("memory.gmem.upstream_write_GBps_per_slice", "float", "GB/s", 0, "spec", "memory.gmem",
      "Write bandwidth out of one shader slice"),

    # ---------------------------------------------------------------- on-chip buffer placement
    F("memory.sram.placement", "str", "", "none", "policy", "memory.sram",
      "What is pre-allocated in the buffer, at tile granularity: none | ab | abc | abc_kv | abc_kv_moe "
      "(1: A and B, 2: A B C, 3: + KV cache, 4: + every expert)"),
    F("memory.sram.unified", "bool", "", True, "policy", "memory.sram",
      "Treat the per-slice buffers as one pool (physically each slice holds a piece)"),
    # ---------------------------------------------------------------- addressing
    F("memory.interleave_KB", "float", "KB", 2, "calib", "memory.addressing",
      "Default interleave granularity when a side does not set its own"),
    F("memory.addressing.addr_bits", "int", "bits", 48, "spec", "memory.addressing", "Physical address width"),
    F("memory.addressing.l2", "map", "", None, "calib", "memory.addressing",
      "{ports, mode: interleave|range|hash, granularity_KB} for L2 slices / load ports"),
    F("memory.addressing.ddr", "map", "", None, "calib", "memory.addressing",
      "{ports, mode, granularity_KB} for the HBM ports the DMA engines target"),

    # ---------------------------------------------------------------- memory-level parallelism
    F("memory.outstanding.per_sm_lines", "int", "lines", 0, "calib", "memory.parallelism",
      "Cache lines in flight per SM (MSHR-style). Little's law: BW_per_SM <= lines x line / latency. "
      "0 = unlimited"),
    F("memory.outstanding.dma_per_engine_lines", "int", "lines", 0, "calib", "memory.parallelism",
      "In-flight lines per DMA engine (recorded; not yet a cap)"),
    F("memory.queueing.coef", "float", "", 0.0, "loss", "memory.parallelism",
      "M/D/1-style latency inflation 1 + coef*u/(1-u) as a lane saturates. DEFAULT 0 = no loss"),
    F("memory.queueing.max_factor", "float", "", 3.0, "loss", "memory.parallelism", "Cap on that inflation"),

    # ---------------------------------------------------------------- DMA
    F("memory.dma.engines", "int", "count", 1, "spec", "memory.dma", "Copy engines"),
    F("memory.dma.per_l2_block", "bool", "", False, "spec", "memory.dma",
      "One engine per L2 block (false = a single shared engine)"),
    F("memory.dma.destination", "str", "", "smem", "policy", "memory.dma",
      "Where a DMA drops data: smem (through the L2 datapath) | l2 (fills L2) | bypass"),
    F("memory.dma.hugepage_KB", "float", "KB", 0, "calib", "memory.dma",
      "Fixed size of one DMA operation: a DMA moves exactly one hugepage, never less (0 = unset: "
      "fall back to an occupancy proxy for L2/DDR slice spread; NVIDIA DMA/TMA moves 2048). Must be "
      "a whole multiple of memory.addressing.l2's granularity x port count, so a hugepage always "
      "splits evenly across every memory slice by construction. Also changes the L2 simulation's "
      "cache atom (for GEMM A/B and attention K/V) to hugepage granularity, so real reuse across "
      "loop iterations shows up as a hit instead of a fresh miss every call"),

    # ---------------------------------------------------------------- load paths
    F("load_paths", "map", "", REQUIRED, "spec", "load_paths",
      "One entry per path (tma, lsu, buffer_lds, tdm ...), each with: engine (dma|lsu), "
      "per_sm_bytes_per_clk (this path's own clock domain, same as every other on-chip lane — "
      "scales with core.freq_ghz), latency_ns (issue overhead), smem_direct, multicast, "
      "regs_per_thread, issue_bytes_per_clk"),
    F("load_paths_default", "str", "", "", "spec", "load_paths",
      "Path a request resolves to when the named one does not exist on this part"),

    # ---------------------------------------------------------------- occupancy
    F("occupancy.max_blocks_per_sm", "int", "count", 32, "spec", "occupancy", "Hardware limit on resident blocks"),
    F("occupancy.max_threads_per_sm", "int", "threads", 2048, "spec", "occupancy", "Thread limit per SM"),
    F("occupancy.regs_per_sm", "int", "registers", 65536, "spec", "occupancy", "Register file per SM, in 4-byte registers"),
    F("occupancy.max_regs_per_thread", "int", "registers", 255, "spec", "occupancy",
      "Above this the accumulator spills to local memory"),

    # ---------------------------------------------------------------- network
    F("network.nvlink.domain_size", "int", "GPUs", 8, "spec", "network",
      "Size of the fast scale-up domain; collectives go hierarchical beyond it"),
    F("network.nvlink.bandwidth_GBps", "float", "GB/s", REQUIRED, "spec", "network", "Per-GPU scale-up bandwidth (one direction)"),
    F("network.nvlink.alpha_us", "float", "us", 3.0, "calib", "network", "Scale-up startup latency"),
    F("network.scaleout.bandwidth_GBps", "float", "GB/s", REQUIRED, "spec", "network", "Per-GPU scale-out bandwidth"),
    F("network.scaleout.alpha_us", "float", "us", 8.0, "calib", "network", "Scale-out startup latency"),

    # ---------------------------------------------------------------- runtime
    F("runtime.launch_overhead_us", "float", "us", 2.0, "calib", "runtime",
      "Per-kernel launch cost; ~0.5-1 with CUDA graphs"),
)

BY_PATH: dict[str, Field] = {f.path: f for f in FIELDS}
SECTIONS = tuple(dict.fromkeys(f.section for f in FIELDS))


def default_for(path: str):
    """Default the model uses when a config leaves the field out."""
    f = BY_PATH.get(path)
    if f is None or isinstance(f.default, _Required):
        return None
    return f.default


def validate(raw: dict) -> list[str]:
    """Check a hardware config against the schema. Returns a list of problems (empty = ok)."""
    problems: list[str] = []

    def walk(node, prefix=""):
        for k, v in (node or {}).items():
            path = f"{prefix}{k}"
            f = BY_PATH.get(path)
            if isinstance(v, dict) and (f is None or f.kind != "map"):
                if f is None and not any(p.startswith(path + ".") for p in BY_PATH):
                    problems.append(f"unknown section: {path}")
                walk(v, path + ".")
                continue
            if f is None:
                problems.append(f"unknown field: {path}")
                continue
            if f.kind in ("float", "int") and not isinstance(v, (int, float)) and v is not None:
                problems.append(f"{path}: expected a number, got {type(v).__name__}")
            elif f.kind == "bool" and not isinstance(v, bool):
                problems.append(f"{path}: expected true/false")
            elif f.kind == "str" and not isinstance(v, str):
                problems.append(f"{path}: expected a string")
    walk(raw)
    for f in FIELDS:
        if isinstance(f.default, _Required):
            node, ok = raw, True
            for part in f.path.split("."):
                if not isinstance(node, dict) or part not in node:
                    ok = False
                    break
                node = node[part]
            if not ok:
                problems.append(f"missing required field: {f.path}")
    return problems


def as_markdown() -> str:
    L = ["# Hardware configuration reference", "",
         "The model only ever sees these fields. `spec` = copy it from the vendor, "
         "`calib` = measure it, `policy` = a modelling choice, `loss` = a derating knob whose "
         "default is always no loss.", ""]
    for sec in SECTIONS:
        L.append(f"## {sec}")
        L.append("")
        L.append("| field | type | unit | default | tag | meaning |")
        L.append("|---|---|---|---|---|---|")
        for f in FIELDS:
            if f.section != sec:
                continue
            d = ("**required**" if isinstance(f.default, _Required)
                 else "—" if f.default is None else f"`{f.default}`")
            L.append(f"| `{f.path}` | {f.kind} | {f.unit or '—'} | {d} | {f.tag} | {f.doc} |")
        L.append("")
    L.append(f"Total: {len(FIELDS)} fields "
             f"({sum(1 for f in FIELDS if f.tag == 'spec')} spec, "
             f"{sum(1 for f in FIELDS if f.tag == 'calib')} calib, "
             f"{sum(1 for f in FIELDS if f.tag == 'policy')} policy, "
             f"{sum(1 for f in FIELDS if f.tag == 'loss')} loss).")
    return "\n".join(L)


def as_csv() -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["path", "kind", "unit", "default", "tag", "section", "doc"])
    for f in FIELDS:
        w.writerow([f.path, f.kind, f.unit,
                    "REQUIRED" if isinstance(f.default, _Required) else ("" if f.default is None else f.default),
                    f.tag, f.section, f.doc])
    return buf.getvalue()
