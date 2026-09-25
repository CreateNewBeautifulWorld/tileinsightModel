"""Per-GPU memory map: where every tensor lives.

The model layer knows how many layers this GPU holds and the shape of every matrix in them,
so addresses are simply an allocation: walk the ops in execution order and hand out aligned
regions from a base offset. That is enough for the tile-granular address questions this
simulator asks (which L2 slice / HBM port a tile lands on, how a swizzle spreads a wave's
tiles) without pretending to model a real allocator's fragmentation.

Layout per region:
  weights      : one region per distinct weight tensor, laid out layer by layer
  kv_cache     : one region per attention layer (paged: `page_size` tokens per page)
  activations  : a double-buffered scratch region reused by every layer
  workspace    : split-K partials, attention split partials, MoE staging

`region.tile_addr(i, j, layout)` gives the byte address of tile (i, j) under a layout, which
is what report/addressing.py and the Excel/CSV/trace exports use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..gpuTilingHWModel.spec import DTYPE_BYTES
from .lower import lower_model
from .memory import kv_bytes_per_seq_all
from .run_config import RunConfig
from .spec import ModelSpec

ALIGN = 2 * 1024 * 1024          # 2 MB, a large-page boundary


def _align(x: int, a: int = ALIGN) -> int:
    return -(-x // a) * a


@dataclass
class Region:
    name: str
    kind: str                     # weight | kv | act | workspace
    base: int
    size: int
    rows: int = 0                 # logical matrix shape (rows x cols), 0 if not a matrix
    cols: int = 0
    elem_bytes: float = 2.0
    layer: str = ""

    def tile_grid(self, bm: int, bn: int) -> tuple[int, int]:
        import math
        return math.ceil(self.rows / bm), math.ceil(self.cols / bn)

    def tile_addr(self, i: int, j: int, bm: int, bn: int, layout: str = "row") -> int:
        """Byte address of tile (i, j) of this matrix under `layout`."""
        from ..report.addressing import tile_index
        ni, nj = self.tile_grid(bm, bn)
        tile_bytes = int(bm * bn * self.elem_bytes)
        return self.base + tile_index(layout, i, j, ni, nj) * tile_bytes

    @property
    def end(self) -> int:
        return self.base + self.size


@dataclass
class MemoryMap:
    regions: list[Region] = field(default_factory=list)
    base: int = 0

    def by_kind(self, kind: str) -> list[Region]:
        return [r for r in self.regions if r.kind == kind]

    def find(self, name_fragment: str) -> Region | None:
        return next((r for r in self.regions if name_fragment in r.name), None)

    @property
    def total_bytes(self) -> int:
        return max((r.end for r in self.regions), default=self.base) - self.base

    def text(self, limit: int = 40) -> str:
        L = [f"{'region':46s} {'kind':10s} {'base':>14s} {'size MB':>10s}  shape"]
        for r in self.regions[:limit]:
            shape = f"{r.rows} x {r.cols} x {r.elem_bytes}B" if r.rows else ""
            L.append(f"{r.name:46s} {r.kind:10s} 0x{r.base:012x} {r.size / 1e6:10.2f}  {shape}")
        if len(self.regions) > limit:
            L.append(f"... {len(self.regions) - limit} more regions")
        tot = {k: sum(r.size for r in self.by_kind(k)) / 1e9 for k in ("weight", "kv", "act", "workspace")}
        L.append("")
        L.append("totals: " + ", ".join(f"{k} {v:.2f} GB" for k, v in tot.items()) +
                 f"  span {self.total_bytes / 1e9:.2f} GB from 0x{self.base:x}")
        return "\n".join(L)


def build_memory_map(model: ModelSpec, rc: RunConfig, base: int = 0) -> MemoryMap:
    """Allocate every tensor this GPU holds, in execution order."""
    mm = MemoryMap(base=base)
    cur = base
    groups = lower_model(model, rc)

    # 1) weights: every layer instance gets its own region (that is what the addresses are for)
    for gname, rep, ops in groups:
        for layer_i in range(rep):
            for op in ops:
                if op.weight_bytes <= 0:
                    continue
                p = op.p or {}
                rows, cols = int(p.get("K", 0)), int(p.get("N", 0))
                eb = DTYPE_BYTES.get(p.get("b_dtype", "bf16"), 2.0)
                size = _align(int(op.weight_bytes))
                short = op.name.split(".", 2)[-1]
                mm.regions.append(Region(f"{gname}[{layer_i}].{short}", "weight", cur, size,
                                         rows, cols, eb, gname))
                cur += size

    # 2) KV cache, one region per attention layer (page-aligned)
    per_seq = kv_bytes_per_seq_all(model, rc, rc.seq_len)
    n_attn = sum(g.repeat for g in model.layers
                 for b in g.blocks if b["type"] in ("mla", "gqa", "mha"))
    if per_seq > 0 and n_attn:
        per_layer = _align(int(per_seq * rc.seqs_per_rank / n_attn))
        for i in range(n_attn):
            mm.regions.append(Region(f"kv_cache.layer{i}", "kv", cur, per_layer,
                                     rc.seqs_per_rank * rc.seq_len, 0,
                                     DTYPE_BYTES[rc.kv_dtype]))
            cur += per_layer

    # 3) activations (double buffered) and workspace
    act = 0
    for _, _, ops in groups:
        for op in ops:
            p = op.p or {}
            if op.kind == "gemm":
                act = max(act, int(p["M"] * p["N"] * p.get("batch", 1) *
                                   DTYPE_BYTES.get(p.get("c_dtype", "bf16"), 2.0)))
            elif op.kind == "elementwise":
                act = max(act, int(p.get("bytes_out", 0)))
    for i in range(2):
        mm.regions.append(Region(f"activations.buf{i}", "act", cur, _align(act), 0, 0, 2.0))
        cur += _align(act)
    mm.regions.append(Region("workspace", "workspace", cur, _align(act * 2), 0, 0, 4.0))
    return mm
