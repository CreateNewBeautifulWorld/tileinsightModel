"""memory slices — L2, HBM and the on-chip buffer, at tile granularity.

A memory slice owns one HBM channel, one L2 port and one DMA port; which slice a tile belongs
to is decided by its ADDRESS through the configured map (linear or interleaved), never by a
hash of convenience. Each slice's L2 is simulated exactly (tile-level replacement), so the
answer to "is this tile in L2" is a fact with a residency list behind it, not a probability.

The on-chip buffer is not part of a memory slice: it is a separate, explicitly managed level
between L2 and HBM whose contents are decided ahead of time (see gpuTilingPerfHWModel/slice_config.py
`onchip_buffer.contents`), so its residency is a deterministic capacity share per tensor class.
"""
from __future__ import annotations

from dataclasses import dataclass

from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import HardwareSpec
from tilesight.gpuTilingPerfHWModel.model.engine.cache_sim import SimResult
from tilesight.gpuTilingPerfHWModel.model.engine.cache_sim import simulate as _simulate


@dataclass
class TileStream:
    """The tiles a kernel touches, in issue order, with their addresses and sizes."""
    keys: list[int]
    addrs: list[int]
    sizes: list[float]
    streams: list[int]
    n_streams: int


def simulate_l2(cur_gpu_config: HardwareSpec, s: TileStream) -> SimResult:
    """Run the stream through the memory slices' L2s (one partition per slice)."""
    from tilesight.gpuTilingPerfHWModel.genResult.addressing import AddressMap
    lmap = AddressMap.from_hw(cur_gpu_config, "l2")
    parts = int(cur_gpu_config.get("memory.l2.partitions") or 1)
    return _simulate(s.keys, s.addrs, s.sizes, s.streams, s.n_streams,
                     cur_gpu_config.l2_capacity_bytes, parts, lmap.port_of,
                     str(cur_gpu_config.get("memory.l2.policy")))


def buffer_residency(cur_gpu_config: HardwareSpec, footprint_bytes: float, klass: str) -> float:
    """Share of a tensor class that is pre-allocated on the on-chip buffer (deterministic)."""
    from tilesight.gpuTilingPerfHWModel.model.kernels.gemm import resident_frac
    return resident_frac(cur_gpu_config, footprint_bytes, klass)


def level_split(bytes_total: float, l2_miss: float, buffer_miss: float | None) -> dict[str, float]:
    """Bytes served by each level: everything crosses L2's datapath, the misses go on."""
    to_hbm = l2_miss if buffer_miss is None else min(l2_miss, buffer_miss)
    out = {"l2": bytes_total, "ddr": bytes_total * to_hbm}
    if buffer_miss is not None:
        out["sram"] = bytes_total * l2_miss
    return out
