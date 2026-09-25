"""Hardware description.

The hardware is a plain nested dict loaded from YAML (see hw/db/*.yaml) so that
design-space experiments can override ANY field by dotted path, e.g.
    hw.override({"memory.ddr.bandwidth_TBps": 12.0})

`HardwareSpec.lanes()` turns the description into the resource lanes used by the
engine (docs/DESIGN.md §2).  Two kinds of lanes:
  * per-SM lanes   (tc, cuda, sfu, smem, tmem, path:<name>): action work is given in
    *seconds on one SM*; the lowering converts FLOPs/bytes to seconds with the
    helpers below.
  * shared lanes   (l2, ddr): action work is given in *bytes*; the engine divides by
    the per-SM share  min(total_rate / active_SMs, per_sm_cap).
Adding a new on-chip memory or load path to the YAML automatically adds a lane.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DB_DIR = Path(__file__).parent / "db"

DTYPE_BYTES = {
    "fp4": 0.5, "nvfp4": 0.5, "mxfp4": 0.5, "int4": 0.5,
    "fp6": 0.75, "mxfp6": 0.75,
    "fp8": 1.0, "mxfp8": 1.0, "int8": 1.0,
    "bf16": 2.0, "fp16": 2.0,
    "tf32": 4.0, "fp32": 4.0,
}
# name -> the tensor-core datapath it runs on (hardware tables are keyed by these)
DTYPE_ALIAS = {"nvfp4": "fp4", "mxfp4": "fp4", "mxfp6": "fp6", "mxfp8": "fp8", "fp16": "fp16"}
# weight-only formats: weights are stored narrow, the MMA runs at the activation width
WEIGHT_ONLY = {"int4"}
# if a datapath is missing on a part, fall back to the next wider one
WIDEN = ["fp4", "fp6", "fp8", "int8", "bf16", "fp16", "tf32", "fp32"]


# The three blocks the model is organised in, mirroring the slice-based GPU config:
#   shader_slice   : a shader core's units, the L1 its slice shares, and the load paths it issues
#   onchip_buffer  : the buffer that is logically one piece and belongs to neither side
#   memory         : the memory slices — L2 ports and HBM
# (gmem is not a block: it is the upstream/downstream bandwidth *between* these three.)
DOMAINS = ("shader_slice", "onchip_buffer", "memory")
LANE_DOMAIN = {"tc": "shader_slice", "cuda": "shader_slice", "sfu": "shader_slice",
               "smem": "shader_slice", "tmem": "shader_slice", "l1": "shader_slice",
               "sram": "onchip_buffer", "switch": "onchip_buffer",
               "l2": "memory", "ddr": "memory", "net": "memory"}


def lane_domain(name: str) -> str:
    if name.startswith("path"):
        return "shader_slice"          # the core issues the load; the DMA port is on the memory side
    return LANE_DOMAIN.get(name, "memory")


@dataclass(frozen=True)
class Lane:
    name: str
    shared: bool          # True -> work in bytes, bandwidth shared by active SMs
    total_rate: float     # bytes/s for shared lanes (whole GPU); unused for per-SM lanes
    per_sm_cap: float     # bytes/s max one SM can draw from a shared lane (inf if none)
    domain: str = "memory"   # shader_slice | gmem | memory
    scope: str = "global"    # per_core | per_slice | global — where the resource physically is


class HardwareSpec:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self._fp: str | None = None

    @property
    def fingerprint(self) -> str:
        """Content hash; kernels lowered for equal specs are cached across objects."""
        if self._fp is None:
            import hashlib
            import json
            self._fp = hashlib.sha1(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()
        return self._fp

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, name_or_path: str) -> "HardwareSpec":
        p = Path(name_or_path)
        if not p.exists():
            p = DB_DIR / f"{name_or_path.lower()}.yaml"
        with open(p) as f:
            return cls(yaml.safe_load(f))

    def override(self, changes: dict[str, Any]) -> "HardwareSpec":
        """Return a copy with dotted-path fields replaced (used by DSE sweeps)."""
        raw = copy.deepcopy(self.raw)
        for path, value in changes.items():
            node = raw
            keys = path.split(".")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
        return HardwareSpec(raw)

    _MISSING = object()

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Read a config field. An absent field falls back to the schema default, so the model
        never carries hardware knowledge of its own (hw/schema.py is the single source)."""
        node: Any = self.raw
        for k in path.split("."):
            if not isinstance(node, dict) or k not in node:
                if default is not HardwareSpec._MISSING:
                    return default
                from .schema import default_for
                return default_for(path)
            node = node[k]
        return node

    def validate(self) -> list[str]:
        from .schema import validate
        return validate(self.raw)

    # ---------------------------------------------------------------- basics
    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def sms(self) -> int:
        return int(self.raw["sms"])

    @property
    def clock_hz(self) -> float:
        return float(self.raw.get("clock_ghz", 1.8)) * 1e9

    def eff(self, key: str) -> float:
        """Achieved/peak ratio for a lane. Absent = 1.0: every loss must be configured."""
        return float(self.get(f"efficiency.{key}", 1.0))

    def lossless(self) -> "HardwareSpec":
        """A copy with every modelled loss switched off — the theoretical-peak baseline.

        Zeroes: all `efficiency.*` factors (back to 1.0), every cache `capacity_derate` and
        measured-cliff override, the queueing coefficient, and the per-SM / outstanding-request
        ceilings. What remains is pure structure: tile shapes, dependencies, capacities and
        bandwidths as quoted. Useful to see how much of a prediction is the model's losses."""
        raw = copy.deepcopy(self.raw)
        raw["efficiency"] = {k: 1.0 for k in (raw.get("efficiency") or {})}
        mem = raw.get("memory") or {}
        for level in ("l1", "l2", "sram"):
            cfg = mem.get(level)
            if isinstance(cfg, dict):
                cfg["capacity_derate"] = 1.0
                cfg.pop("effective_capacity_MB", None)
        if isinstance(mem.get("queueing"), dict):
            mem["queueing"]["coef"] = 0.0
        if isinstance(mem.get("l2"), dict):
            mem["l2"].pop("per_sm_max_GBps", None)
        if isinstance(mem.get("outstanding"), dict):
            mem["outstanding"]["per_sm_lines"] = 0
        return HardwareSpec(raw)

    @property
    def has_tmem(self) -> bool:
        return self.get("memory.onchip.tmem") is not None

    @property
    def launch_s(self) -> float:
        return float(self.get("runtime.launch_overhead_us")) * 1e-6

    @property
    def ddr_capacity_bytes(self) -> float:
        return float(self.get("memory.ddr.capacity_GB")) * 1e9

    # ---------------------------------------------------------------- per-SM rate helpers
    def tc_datapath(self, dtype: str) -> str | None:
        """Resolve a dtype to a tensor-core datapath this part has, widening if needed."""
        d = DTYPE_ALIAS.get(dtype, dtype)
        table = self.get("compute.tc_dense_tflops") or {}
        if d in table:
            return d
        if d in WIDEN:                       # e.g. fp4 on Hopper -> fp8; int8 -> bf16
            for wider in WIDEN[WIDEN.index(d):]:
                if wider in table:
                    return wider
        return None

    def tc_time_per_sm(self, flops: float, dtype: str) -> float:
        """Seconds one SM needs for `flops` tensor-core FLOPs of `dtype` (bf16 if unknown)."""
        d = self.tc_datapath(dtype) or "bf16"
        table = self.get("compute.tc_dense_tflops")
        return flops / (table[d] * 1e12 * self.eff("tc") / self.sms)

    def mma_cost(self, flops: float, dtype: str) -> tuple[str, float]:
        """(lane, seconds-on-one-SM) for a matmul: the tensor core, or CUDA cores for
        datatypes this part has no tensor-core datapath for (e.g. fp32 on most GPUs)."""
        if self.tc_datapath(dtype) is None:
            return "cuda", self.cuda_time_per_sm(flops)
        return "tc", self.tc_time_per_sm(flops, dtype)

    def resolve_path(self, path: str) -> str:
        """Map a requested load path to one this part actually has (AMD has no TMA)."""
        paths = self.get("load_paths") or {}
        if path in paths:
            return path
        default = self.get("load_paths_default")
        if default in paths:
            return default
        return next(iter(paths), path)

    def cuda_time_per_sm(self, flops: float) -> float:
        return flops / (self.get("compute.cuda_fp32_tflops") * 1e12 * self.eff("cuda") / self.sms)

    def sfu_time_per_sm(self, ops: float) -> float:
        return ops / (self.get("compute.sfu_tops") * 1e12 * self.eff("sfu") / self.sms)

    def l1_time_per_sm(self, nbytes: float) -> float:
        bpc = float(self.get("memory.l1.bytes_per_clk") or 128)
        return nbytes / (bpc * self.clock_hz * self.eff("smem"))

    def smem_time_per_sm(self, nbytes: float) -> float:
        bw = self.get("memory.onchip.smem.bytes_per_clk") * self.clock_hz * self.eff("smem")
        return nbytes / bw

    def tmem_time_per_sm(self, read_bytes: float, write_bytes: float) -> float:
        t = self.get("memory.onchip.tmem")
        if t is None:
            return 0.0
        return read_bytes / (t["read_TBps"] * 1e12) + write_bytes / (t["write_TBps"] * 1e12)

    # unit -> (yaml path, default in cycles).  Latency is issue-to-result for ONE operation on
    # that unit; it does not change throughput, it lengthens the dependency chain (and the
    # pipeline fill), which is what matters for fused kernels and short loops.
    _UNIT_LATENCY = {
        "tc": ("compute.mma_latency_cycles", 64),
        "cuda": ("compute.cuda_latency_cycles", 4),
        "sfu": ("compute.sfu_latency_cycles", 16),
        "l1": ("memory.l1.latency_cycles", 20),
        "smem": ("memory.onchip.smem.latency_cycles", 30),
        "tmem": ("memory.onchip.tmem.latency_cycles", 16),
        "l2": ("memory.l2.latency_cycles", 0),
        "ddr": ("memory.ddr.latency_cycles", 0),
        "sram": ("memory.sram.latency_cycles", 0),
    }

    def unit_latency_s(self, unit: str) -> float:
        """Latency of one operation on `unit`, in seconds.

        Configure it in cycles (e.g. `compute.mma_latency_cycles: 64` — a tensor-core tile
        MMA takes 64 cycles to produce its result) or in ns for the memory levels
        (`memory.ddr.latency_ns`); cycles win if both are given."""
        path, default = self._UNIT_LATENCY.get(unit, (None, 0))
        if path is None:
            return 0.0
        cyc = self.get(path)
        if cyc is None:
            ns = self.get(path.replace("latency_cycles", "latency_ns"))
            if ns is not None:
                return float(ns) * 1e-9
            cyc = default
        return float(cyc) / self.clock_hz

    def path_attr(self, path: str, key: str, default=None):
        return (self.get(f"load_paths.{self.resolve_path(path)}") or {}).get(key, default)

    def path_is_dma(self, path: str) -> bool:
        """DMA engines (TMA, CDNA5 TDM, buffer_load->LDS) vs ordinary vector loads (LSU)."""
        return str(self.path_attr(path, "engine", "dma" if "tma" in path or "tdm" in path
                                  or "buffer" in path else "lsu")) == "dma"

    def path_time_per_sm(self, path: str, nbytes: float) -> float:
        p = self.get(f"load_paths.{self.resolve_path(path)}")
        if p is None:
            return 0.0
        return nbytes / (p["per_sm_GBps"] * 1e9)

    def path_latency_s(self, path: str) -> float:
        return float(self.get(f"load_paths.{self.resolve_path(path)}.latency_ns", 800)) * 1e-9

    # ---------------------------------------------------------------- capacities
    @property
    def smem_per_sm(self) -> int:
        return int(self.get("memory.onchip.smem.capacity_KB") * 1024)

    @property
    def tmem_per_sm(self) -> int:
        return int(self.get("memory.onchip.tmem.capacity_KB", 0) * 1024)

    def usable_capacity_bytes(self, level: str) -> float:
        """Physical capacity x derating factor, in bytes.

        The simulation itself is exact (tile-granular LRU over partitions), so everything it
        does NOT simulate — line granularity, conflict misses inside a partition, instruction
        and other-kernel traffic, streaming/evict-first hints — is folded into one honest knob,
        `capacity_derate`. Derate first, then allocate deterministically: the hit/miss decision
        stays a fact about tiles, and the uncertainty lives in a single calibratable number.
        `effective_capacity_MB` still works as a direct override."""
        cfg = self.get(f"memory.{level}") or {}
        if not cfg:
            return 0.0
        if "effective_capacity_MB" in cfg:
            return float(cfg["effective_capacity_MB"]) * 1024 * 1024
        cap = float(cfg.get("capacity_MB", cfg.get("capacity_KB", 0) / 1024)) * 1024 * 1024
        return cap * float(cfg.get("capacity_derate", 1.0))

    @property
    def l1_capacity_bytes(self) -> float:
        """L1 available to global-load caching, after the SMEM carve-out and derating."""
        l1 = self.get("memory.l1") or {}
        if not l1 or not l1.get("cache_global_loads", True):
            return 0.0
        cap = (float(l1.get("capacity_KB", 0)) - float(l1.get("smem_carveout_KB", 0))) * 1024
        cap = max(0.0, cap) * float(l1.get("capacity_derate", 1.0))
        if str(l1.get("owner", "sm")) == "cluster":
            cap *= float(l1.get("cluster_size", 1))      # a cluster shares its L1s through DSMEM
        return cap

    @property
    def l2_capacity_bytes(self) -> float:
        l2 = self.get("memory.l2")
        return self.usable_capacity_bytes("l2")

    @property
    def sram(self) -> dict | None:
        """Optional extra on-chip shared buffer (e.g. a 64 MB A/B staging SRAM).

        Sits between L2 and HBM: tiles that miss L2 but fit here are served on the `sram`
        lane instead of HBM, which is exactly how such a buffer saves DRAM bandwidth."""
        return self.get("memory.sram")

    @property
    def sram_capacity_bytes(self) -> float:
        s = self.sram
        return self.usable_capacity_bytes("sram")

    def sram_capacity_for(self, klass: str) -> float:
        """Capacity the buffer gives to one tensor class: "weight" | "kv" | "act".

        `memory.sram.alloc` (or the alias `pin`) splits it, e.g. {weight: .6, kv: .35, act: .05};
        shares are normalised. Without a split the buffer behaves as a shared cache and every
        class may use all of it."""
        if self.sram is None:
            return 0.0
        split = self.sram.get("alloc") or self.sram.get("pin")
        if not split:
            # shared cache: classes compete, so split it in proportion to their footprints
            # (a model-level run installs those; a single-kernel study has only one class)
            fp = self.get("memory.sram.footprint")
            if fp:
                tot = sum(max(0.0, float(v)) for v in fp.values()) or 1.0
                return self.sram_capacity_bytes * max(0.0, float(fp.get(klass, 0.0))) / tot
            return self.sram_capacity_bytes
        tot = sum(max(0.0, float(v)) for v in split.values()) or 1.0
        return self.sram_capacity_bytes * max(0.0, float(split.get(klass, 0.0))) / tot

    @property
    def sram_keeps_intermediates(self) -> bool:
        """Policy: elementwise intermediates stay on chip instead of round-tripping HBM."""
        return bool(self.sram and self.sram.get("keep_intermediates", False)
                    and self.sram_capacity_for("act") > 0)

    @property
    def l2_assoc(self) -> int:
        return int(self.get("memory.l2.assoc"))

    @property
    def tc_min_m(self) -> int:
        return int(self.get("compute.tc_min_m", 64))

    # ---------------------------------------------------------------- lanes
    # ---------------------------------------------------------------- memory-system limits
    def block_limited_rate(self, level: str, configured: float) -> float:
        """Cap a level's bandwidth by its blocks x ports x per-port width x clock.

        L2 is physically several blocks, each with its own load/store port; a level can never
        exceed what its ports can move even if the quoted aggregate bandwidth is higher."""
        cfg = self.get(f"memory.{level}") or {}
        blocks = float(cfg.get("blocks", 0) or 0)
        ports = float(cfg.get("ports_per_block", 1) or 1)
        bpc = float(cfg.get("bytes_per_clk_per_port", 0) or 0)
        if blocks and bpc:
            return min(configured, blocks * ports * bpc * self.clock_hz)
        return configured

    def outstanding_cap(self, level: str) -> float:
        """Little's law: one SM cannot pull more than in-flight bytes / latency.

        `memory.outstanding.per_sm_lines` is the MSHR-style limit on cache lines in flight per
        SM; with the level's latency it bounds per-SM bandwidth regardless of how wide the
        memory is. This is why a latency-bound kernel does not speed up when you add HBM."""
        lines = float(self.get("memory.outstanding.per_sm_lines") or 0)
        line_b = float(self.get("memory.l2.line_bytes") or 128)
        lat = self.unit_latency_s(level)
        return lines * line_b / lat if lines and lat > 0 else math.inf

    @property
    def dma_destination(self) -> str:
        """Where a DMA engine drops the data: smem | l2 | bypass (straight to the consumer)."""
        return str(self.get("memory.dma.destination"))

    def lanes(self) -> list[Lane]:
        inf = math.inf
        out = [Lane("tc", False, 0, inf, "shader_slice", "per_core"),
               Lane("cuda", False, 0, inf, "shader_slice", "per_core"),
               Lane("sfu", False, 0, inf, "shader_slice", "per_core")]
        for mem in (self.get("memory.onchip") or {}):
            out.append(Lane(mem, False, 0, inf, "shader_slice", "per_core"))
        if self.l1_capacity_bytes > 0:
            # L1 belongs to the slice: its bandwidth is shared by the cores in that slice, which
            # is exactly a global lane of (slices x per-slice bandwidth) under uniform activity
            slices = max(1, self.sms // max(1, int(self.get("memory.l1.cluster_size", 1))))
            bw = float(self.get("memory.l1.bytes_per_clk")) * self.clock_hz * self.eff("smem")
            out.append(Lane("l1", True, bw * slices, inf, "shader_slice", "per_slice"))
        for path in (self.get("load_paths") or {}):
            out.append(Lane(f"path:{path}", False, 0, inf, "shader_slice", "per_core"))
        l2 = self.get("memory.l2")
        cap = float(l2.get("per_sm_max_GBps", 1e9)) * 1e9
        ddr_rate = self.get("memory.ddr.bandwidth_TBps") * 1e12 * self.eff("ddr")
        if self.usable_capacity_bytes("l2") <= 0:
            # no L2 level: the lane is just the memory-slice ports, so it must not throttle
            # traffic that now goes straight to HBM (which pays the HBM latency instead)
            rate = max(self.block_limited_rate("l2", ddr_rate), ddr_rate)
        else:
            rate = self.block_limited_rate("l2", l2["bandwidth_TBps"] * 1e12 * self.eff("l2"))
        out.append(Lane("l2", True, rate, min(cap, self.outstanding_cap("l2")), "memory", "global"))
        if self.sram:
            out.append(Lane("sram", True, self.sram["bandwidth_TBps"] * 1e12 * self.eff("sram"),
                            float(self.sram.get("per_sm_max_GBps", 1e9)) * 1e9,
                            "onchip_buffer", "global"))
            # the switch between the shader slices and the buffer pieces: one aggregate
            # bandwidth, of which a slice may draw its 1/slices share
            sw = float(self.sram.get("switch_TBps") or self.sram["bandwidth_TBps"])
            slices = max(1, self.sms // max(1, int(self.get("memory.l1.cluster_size", 1))))
            out.append(Lane("switch", True, sw * 1e12 * self.eff("sram"),
                            sw * 1e12 * self.eff("sram") / slices * max(1, int(
                                self.get("memory.l1.cluster_size", 1))),
                            "onchip_buffer", "global"))
        ddr = self.get("memory.ddr")
        out.append(Lane("ddr", True,
                        self.block_limited_rate("ddr", ddr["bandwidth_TBps"] * 1e12 * self.eff("ddr")),
                        min(cap, self.outstanding_cap("ddr")), "memory", "global"))
        return out

    def __repr__(self) -> str:
        return f"HardwareSpec({self.name}, sms={self.sms}, ddr={self.get('memory.ddr.bandwidth_TBps')} TB/s)"
