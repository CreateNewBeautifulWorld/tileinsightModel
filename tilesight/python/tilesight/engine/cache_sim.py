"""Deterministic tile-level cache simulation.

The probabilistic SDCM model answers "how likely is a hit at this reuse distance". That is the
wrong question for a hardware-managed L2 whose contents are fully determined: which tiles are
resident is decidable, so we decide it.

Model: the tile is the atom (no cache lines). L2 is split into `partitions`, each an independent
set of ways holding whole tiles; a tile goes to the partition its address maps to
(`report.addressing.AddressMap`, so the layout/swizzle decides the balance). Each partition runs
an explicit replacement policy over its resident set — `lru` (default), `fifo`, or `mru`.
Capacity is in bytes, so tiles of different sizes compete for it honestly.

`simulate()` returns per-stream misses AND the final residency: exactly which tiles are in
which partition at the end, plus per-partition occupancy and eviction counts. That is what lets
you answer "is the B panel still in L2 when the next wave starts" instead of guessing.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Partition:
    capacity: int
    resident: "OrderedDict[int, int]" = field(default_factory=OrderedDict)   # key -> bytes
    used: int = 0
    evictions: int = 0

    def access(self, key: int, size: int, policy: str) -> bool:
        if key in self.resident:
            if policy == "lru":
                self.resident.move_to_end(key)
            return True
        while self.used + size > self.capacity and self.resident:
            victim, vsize = self.resident.popitem(last=(policy == "mru"))
            self.used -= vsize
            self.evictions += 1
        if size <= self.capacity:
            self.resident[key] = size
            self.used += size
        return False


@dataclass
class SimResult:
    misses: list[float]                 # per stream
    accesses: list[int]                 # per stream
    partitions: list[Partition]
    hit_rate: float

    @property
    def miss_fraction(self) -> list[float]:
        return [m / a if a else 0.0 for m, a in zip(self.misses, self.accesses)]

    def residency(self, top: int = 20) -> list[tuple[int, int, int]]:
        """(partition, tile key, bytes) of what is still resident — newest first."""
        out = []
        for i, p in enumerate(self.partitions):
            for key, size in reversed(p.resident.items()):
                out.append((i, key, size))
                if len(out) >= top:
                    return out
        return out

    def summary(self) -> str:
        L = [f"hit rate {self.hit_rate:.1%} over {sum(self.accesses)} tile accesses"]
        for i, p in enumerate(self.partitions):
            L.append(f"  partition {i}: {len(p.resident)} tiles, {p.used / 1024:.0f} KB / "
                     f"{p.capacity / 1024:.0f} KB used, {p.evictions} evictions")
        return "\n".join(L)


def simulate(keys, addrs, sizes, streams, n_streams: int, capacity_bytes: float,
             n_partitions: int, part_of, policy: str = "lru") -> SimResult:
    """Run the access sequence through `n_partitions` independent tile caches.

    `part_of(addr)` maps a tile address to its partition (the address map does this, so a
    layout that piles tiles onto one partition shows up as a lower hit rate, not just as an
    imbalance number)."""
    parts = [Partition(int(capacity_bytes // max(1, n_partitions))) for _ in range(n_partitions)]
    miss = [0.0] * n_streams
    acc = [0] * n_streams
    for key, addr, size, st in zip(keys, addrs, sizes, streams):
        acc[st] += 1
        if not parts[part_of(addr) % n_partitions].access(key, int(size), policy):
            miss[st] += 1.0
    total = sum(acc)
    return SimResult(miss, acc, parts, 1.0 - (sum(miss) / total if total else 0.0))
