"""Tile configurations and the auto-tile search space.

A TileConfig is what a DSL (Triton/TileLang/CuTe) exposes: block shape, pipeline
depth, raster swizzle, split-K and which load path to use (extended-HW feature).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product


@dataclass(frozen=True)
class TileConfig:
    bm: int = 128
    bn: int = 128
    bk: int = 64
    stages: int = 0            # 0 -> max that fits in SMEM (capped at 8)
    swizzle: int = 8           # grouped raster: GROUP_M block rows
    split_k: int = 1
    load_path: str = "tma"     # key into cur_gpu_config.load_paths; "split:tma=0.7,lsu=0.3" also allowed
    cluster_m: int = 1         # CTAs along M sharing B via TMA multicast (thread-block cluster)
    cta_pair: bool = False     # 2-CTA MMA (Blackwell tcgen05 cta_group::2); requires cluster_m == 2

    def with_(self, **kw) -> "TileConfig":
        return replace(self, **kw)

    def short(self) -> str:
        s = f"{self.bm}x{self.bn}x{self.bk}/s{self.stages}"
        if self.split_k > 1:
            s += f"/sk{self.split_k}"
        if self.cluster_m > 1:
            s += f"/c{self.cluster_m}" + ("p" if self.cta_pair else "")
        return s


@dataclass(frozen=True)
class AttnTileConfig:
    block_m: int = 64          # query rows (prefill) or heads-per-block (decode, padded to tc_min_m)
    block_n: int = 64          # KV rows per iteration
    stages: int = 2
    num_splits: int = 0        # decode split-KV; 0 -> auto fill the machine
    consumers: int = 2         # ping-pong consumer warpgroups (FA3/FlashMLA style)

    def short(self) -> str:
        return f"{self.block_m}x{self.block_n}/s{self.stages}/sp{self.num_splits}/c{self.consumers}"


def gemm_search_space(M: int, N: int, K: int) -> list[TileConfig]:
    out = []
    for bm, bn, bk, sk in product((64, 128), (64, 128, 256), (64, 128), (1, 2, 4, 8)):
        if sk > 1 and K // sk < 2 * bk:
            continue
        out.append(TileConfig(bm=bm, bn=bn, bk=bk, split_k=sk))
        if sk == 1 and M >= 2 * bm:                       # large-M: clustered variants
            out.append(TileConfig(bm=bm, bn=bn, bk=bk, cluster_m=2))
            out.append(TileConfig(bm=bm, bn=bn, bk=bk, cluster_m=2, cta_pair=True))
    return out


def attn_search_space() -> list[AttnTileConfig]:
    return [AttnTileConfig(block_m=bm, block_n=bn, stages=st)
            for bm, bn, st in product((64, 128), (32, 64, 128), (2, 3))]


def parse_path_split(load_path: str) -> dict[str, float]:
    if not load_path.startswith("split:"):
        return {load_path: 1.0}
    parts = dict(p.split("=") for p in load_path[len("split:"):].split(","))
    fr = {k: float(v) for k, v in parts.items()}
    s = sum(fr.values())
    return {k: v / s for k, v in fr.items()}
