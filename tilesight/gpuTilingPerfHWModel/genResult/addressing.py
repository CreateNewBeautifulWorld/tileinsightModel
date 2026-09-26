"""Tile-granularity addressing: how a layout/swizzle spreads tiles over L2 slices and HBM ports.

Deliberately **tile-granular, not cache-line granular**. A tile is the atom: it has one
address (its first byte), one size, and it lands in one L2 slice and one HBM port group.
That is enough to answer the questions a tile model should answer — does this layout spread
the tiles the memory system sees evenly, or does it pile them onto a few slices/ports — and
it stays consistent with the rest of the simulator, which never models cache lines.

Layouts (per tensor):
  row        : tiles in row-major order                addr = (i*NT_j + j) * tile_bytes
  col        : tiles in column-major order
  swizzle:G  : grouped raster over G tile rows (Triton GROUP_M style)
  xor:B      : row-major with the low B bits of the column index XORed into the row index
               (the classic bank/slice-spreading swizzle)
  zorder     : Morton order

Mapping (both at tile granularity):
  slice = (addr // interleave_bytes) % l2_slices        # which L2 slice owns the tile
  port  = (addr // interleave_bytes) % ddr_ports        # which HBM port serves it

Outputs: per-slice / per-port tile counts, an imbalance factor (max/mean) and the effective
bandwidth that imbalance implies (`1/imbalance` of peak when the hottest unit is the limit).
Set `memory.l2.apply_conflict_penalty: true` to fold that factor into the modelled
efficiency instead of only reporting it.
"""
from __future__ import annotations

from dataclasses import dataclass

from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import HardwareSpec


def tile_index(layout: str, i: int, j: int, ni: int, nj: int) -> int:
    """Linear tile index of tile (i, j) in a grid of ni x nj tiles under `layout`."""
    kind, _, arg = layout.partition(":")
    if kind == "col":
        return j * ni + i
    if kind == "swizzle":
        g = max(1, int(arg or 8))
        group = i // g
        rows = min(g, ni - group * g)
        return group * g * nj + j * rows + (i % g)
    if kind == "xor":
        b = int(arg or 3)
        return (i ^ (j & ((1 << b) - 1))) * nj + j
    if kind == "zorder":
        z = 0
        for bit in range(16):
            z |= ((i >> bit) & 1) << (2 * bit)
            z |= ((j >> bit) & 1) << (2 * bit + 1)
        return z
    return i * nj + j                       # "row"


@dataclass
class AddressMap:
    """Which unit owns which address — configurable, and dumpable.

    Two independent maps, because the L2 side and the memory side are usually different
    hardware: `l2` (slices / load ports) and `ddr` (HBM ports, what the DMA engines target).
    Each is configured under `memory.addressing.<side>`:

      mode: interleave   port = (addr >> log2(granularity)) % ports        (the usual case)
      mode: range        the addr_bits space is split into `ports` equal contiguous ranges
      mode: hash         like interleave, but the higher address bits are XOR-folded in first,
                         which breaks the power-of-two-stride aliasing that makes every tile of
                         a 256 KB-pitch matrix land on one slice

      granularity_KB: 1  (or 2, 4 …) — the interleave stripe
      addr_bits: 48      physical address width, used by `range` mode and the dump
    """
    side: str
    ports: int
    mode: str
    granularity: int
    addr_bits: int

    @classmethod
    def from_hw(cls, cur_gpu_config, side: str) -> "AddressMap":
        cfg = (cur_gpu_config.get(f"memory.addressing.{side}") or {})
        defaults = {"l2": (int(cur_gpu_config.get("memory.l2.slices", max(1, int(cur_gpu_config.get("dies", 1)) * 8)) or 1)),
                    "ddr": int(cur_gpu_config.get("memory.ddr.ports") or 1)}
        gran_kb = float(cfg.get("granularity_KB", cur_gpu_config.get("memory.interleave_KB")))
        return cls(side, int(cfg.get("ports", defaults.get(side, 8))),
                   str(cfg.get("mode", "interleave")), int(gran_kb * 1024),
                   int(cfg.get("addr_bits", cur_gpu_config.get("memory.addressing.addr_bits"))))

    def port_of(self, addr: int) -> int:
        if self.mode == "range":
            span = (1 << self.addr_bits) // self.ports
            return min(self.ports - 1, addr // span)
        idx = addr // self.granularity
        if self.mode == "hash":                       # XOR-fold the higher bits (stride breaking)
            h = idx
            for shift in (4, 8, 16):
                h ^= idx >> shift
            idx = h
        return int(idx % self.ports)

    def spans(self, addr: int, size: int) -> list[tuple[int, float]]:
        """(port, bytes) for one tile: a tile covers several stripes, so it spreads."""
        if self.mode == "range":
            return [(self.port_of(addr), float(size))]
        n = max(1, -(-size // self.granularity))
        per = size / n
        return [(self.port_of(addr + i * self.granularity), per) for i in range(n)]

    def describe(self) -> str:
        if self.mode == "range":
            span = (1 << self.addr_bits) // self.ports
            return (f"{self.side}: {self.ports} ports, {self.addr_bits}-bit space split into equal "
                    f"ranges of {span / 2**30:.0f} GiB")
        return (f"{self.side}: {self.ports} ports, {self.mode} at {self.granularity // 1024} KB "
                f"granularity over a {self.addr_bits}-bit space")

    def dump(self, rows: int = 8, base: int = 0) -> str:
        """Human-readable mapping table: which port owns what."""
        L = [self.describe(), f"{'port':>5s}  owns"]
        if self.mode == "range":
            span = (1 << self.addr_bits) // self.ports
            for p in range(min(rows, self.ports)):
                L.append(f"{p:5d}  0x{p * span:012x} .. 0x{(p + 1) * span - 1:012x}")
        else:
            for p in range(min(rows, self.ports)):
                first = next((base + i * self.granularity for i in range(4 * self.ports)
                              if self.port_of(base + i * self.granularity) == p), None)
                stripes = ", ".join(f"0x{base + i * self.granularity:x}"
                                    for i in range(4 * self.ports)
                                    if self.port_of(base + i * self.granularity) == p)[:60]
                L.append(f"{p:5d}  stripes {stripes}…" if first is not None else f"{p:5d}  (none)")
        if self.ports > rows:
            L.append(f"  … {self.ports - rows} more ports")
        return "\n".join(L)

    def dump_csv(self, limit: int = 4096, base: int = 0) -> str:
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        if self.mode == "range":
            w.writerow(["port", "start", "end"])
            span = (1 << self.addr_bits) // self.ports
            for p in range(self.ports):
                w.writerow([p, f"0x{p * span:012x}", f"0x{(p + 1) * span - 1:012x}"])
        else:
            w.writerow(["stripe", "address", "port"])
            for i in range(limit):
                a = base + i * self.granularity
                w.writerow([i, f"0x{a:012x}", self.port_of(a)])
        return buf.getvalue()


@dataclass
class AddressReport:
    slices: list[float]
    ports: list[float]
    slice_imbalance: float
    port_imbalance: float
    tiles: int
    tile_bytes: int
    interleave_bytes: int

    @property
    def effective_l2_fraction(self) -> float:
        return 1.0 / self.slice_imbalance if self.slice_imbalance else 1.0

    @property
    def effective_ddr_fraction(self) -> float:
        return 1.0 / self.port_imbalance if self.port_imbalance else 1.0

    def text(self) -> str:
        def bar(counts, name):
            m = max(counts) or 1
            return "\n".join(f"  {name} {i:2d} |{'#' * int(30 * c / m):30s}| {c:.0f}"
                              for i, c in enumerate(counts))
        return (f"{self.tiles} concurrently-touched tiles of {self.tile_bytes} B, "
                f"interleave {self.interleave_bytes} B\n"
                f"L2 slices (imbalance {self.slice_imbalance:.2f}x -> "
                f"{self.effective_l2_fraction:.0%} of peak):\n{bar(self.slices, 'slice')}\n"
                f"HBM ports (imbalance {self.port_imbalance:.2f}x -> "
                f"{self.effective_ddr_fraction:.0%} of peak):\n{bar(self.ports, 'port')}")


def spread(cur_gpu_config: HardwareSpec, addrs_bytes: list[tuple[int, int]]) -> AddressReport:
    """Spread of a *concurrently touched* set of tiles over L2 slices and HBM ports.

    `addrs_bytes` is [(byte address of the tile, tile size)]. A tile is not a cache line: it
    covers several interleave chunks, so its bytes are spread over the consecutive slices/ports
    those chunks map to. Which tiles are touched together is what the layout/swizzle decides —
    the full set of tile addresses is the same under any layout.
    """
    lmap, dmap = AddressMap.from_hw(cur_gpu_config, "l2"), AddressMap.from_hw(cur_gpu_config, "ddr")
    sl = [0.0] * lmap.ports
    po = [0.0] * dmap.ports
    inter = lmap.granularity
    tile_bytes = addrs_bytes[0][1] if addrs_bytes else 0
    for addr, size in addrs_bytes:
        for p, b in lmap.spans(addr, size):
            sl[p] += b
        for p, b in dmap.spans(addr, size):
            po[p] += b
    def imb(counts):
        mean = sum(counts) / len(counts) if counts else 0.0
        return (max(counts) / mean) if mean else 1.0
    return AddressReport(sl, po, imb(sl), imb(po), len(addrs_bytes), tile_bytes, inter)


def gemm_wave_tiles(cur_gpu_config: HardwareSpec, M: int, N: int, K: int, tile, a_bytes: float, b_bytes: float,
                    layout_a: str = "row", layout_b: str = "row", resident: int = 1, k_step: int = 0):
    """Addresses of the A and B tiles touched together by one wave at K-step `k_step`."""
    import math

    def grouped_raster(mt, nt, group_m):
        """Triton-style GROUP_M swizzle: yields (m, n) in issue order (mirrors gpu_top's)."""
        group_m = max(1, min(group_m, mt))
        for g0 in range(0, mt, group_m):
            rows = min(group_m, mt - g0)
            for n in range(nt):
                for r in range(rows):
                    yield g0 + r, n

    mt, nt, kt = math.ceil(M / tile.bm), math.ceil(N / tile.bn), math.ceil(K / tile.bk)
    conc = cur_gpu_config.sms * max(1, resident)
    a_tb, b_tb = int(tile.bm * tile.bk * a_bytes), int(tile.bn * tile.bk * b_bytes)
    base_b = int(M * K * a_bytes)
    seen_a, seen_b = {}, {}
    for n_block, (m, n) in enumerate(grouped_raster(mt, nt, tile.swizzle)):
        if n_block >= conc:
            break
        seen_a[m] = tile_index(layout_a, m, k_step, mt, kt) * a_tb
        seen_b[n] = base_b + tile_index(layout_b, k_step, n, kt, nt) * b_tb
    return ([(v, a_tb) for v in seen_a.values()], [(v, b_tb) for v in seen_b.values()])


def analyze_gemm(cur_gpu_config: HardwareSpec, M: int, N: int, K: int, tile, a_bytes: float, b_bytes: float,
                 layout_a: str = "row", layout_b: str = "row", resident: int = 1) -> dict:
    """Spread of the tiles a wave touches together, for both operands."""
    ta, tb = gemm_wave_tiles(cur_gpu_config, M, N, K, tile, a_bytes, b_bytes, layout_a, layout_b, resident)
    a, b = spread(cur_gpu_config, ta), spread(cur_gpu_config, tb)
    both = spread(cur_gpu_config, ta + tb)
    return {"A": a, "B": b, "both": both, "slice_imbalance": both.slice_imbalance,
            "port_imbalance": both.port_imbalance,
            "effective_l2_fraction": both.effective_l2_fraction,
            "effective_ddr_fraction": both.effective_ddr_fraction}


def with_conflict_penalty(cur_gpu_config: HardwareSpec, report: dict) -> HardwareSpec:
    """Fold the measured imbalance into the L2/DDR efficiency factors."""
    return cur_gpu_config.override({
        "efficiency.l2": cur_gpu_config.eff("l2") * report["effective_l2_fraction"],
        "efficiency.ddr": cur_gpu_config.eff("ddr") * report["effective_ddr_fraction"],
    })


def kernel_addr_fn(cur_gpu_config: HardwareSpec, M: int, N: int, K: int, tile, a_bytes: float, b_bytes: float,
                   base: int = 0x80000000, layout_a: str = "row", layout_b: str = "row",
                   c_bytes: float = 2.0, block_m: int = 0, block_n: int = 0):
    """Address of the tile each action touches, for a representative block.

    Tensors are laid out A | B | C from `base` (a model-level study passes real region bases
    from model/memmap.py instead). Returns f(action_name, iteration) -> (addr, size) or None.
    """
    import math
    mt, nt, kt = math.ceil(M / tile.bm), math.ceil(N / tile.bn), math.ceil(K / tile.bk)
    a_tb, b_tb, c_tb = int(tile.bm * tile.bk * a_bytes), int(tile.bn * tile.bk * b_bytes), \
        int(tile.bm * tile.bn * c_bytes)
    base_a = base
    base_b = base_a + int(M * K * a_bytes)
    base_c = base_b + int(K * N * b_bytes)
    lmap, dmap = AddressMap.from_hw(cur_gpu_config, "l2"), AddressMap.from_hw(cur_gpu_config, "ddr")

    def f(action: str, iteration: int):
        it = max(0, int(iteration))
        if action.startswith("load:") and ("weight" in action or "kv" in action):
            addr = base_b + tile_index(layout_b, min(it, kt - 1), block_n, kt, nt) * b_tb
            size = b_tb
        elif action.startswith("load:"):
            addr = base_a + tile_index(layout_a, block_m, min(it, kt - 1), mt, kt) * a_tb
            size = a_tb
        elif action.startswith("store:"):
            addr = base_c + (block_m * nt + block_n) * c_tb
            size = c_tb
        else:
            return None
        return {"addr": addr, "size": size, "slice": lmap.port_of(addr), "port": dmap.port_of(addr)}
    return f
