"""Tile-granularity L2 model (paper §3.5, Eqs. 6–10).

Input : a sequence of tile keys (ints) in issue order, plus which stream each access
        belongs to; tile size in bytes.
Output: expected L2 misses per stream.

reuse distance D_T = number of *distinct* tiles touched between two accesses to the
same tile (computed exactly with a Fenwick tree, O(n log n)).
hit probability (set-associative, random set mapping):
    X ~ Binomial(D_T, A/B_T)         A = associativity, B_T = capacity in tiles
    P(hit | D_T) = P(X <= A-1)
exact binomial for D_T <= 256, Gaussian approximation with Zelen–Severo Φ otherwise
(Eqs. 8–10; we use a +0.5 continuity correction — documented deviation).
First touches are compulsory misses.
"""
from __future__ import annotations

import math


def _phi(x: float) -> float:
    """Standard normal CDF, Zelen & Severo (1964) 26.2.16 (paper Eq. 10)."""
    if x < 0:
        return 1.0 - _phi(-x)
    t = 1.0 / (1.0 + 0.33267 * x)
    a1, a2, a3 = 0.4361836, -0.1201676, 0.9372980
    return 1.0 - (a1 * t + a2 * t * t + a3 * t ** 3) * math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def hit_prob(d: int, assoc: int, cap_tiles: float) -> float:
    if d < assoc:
        return 1.0
    if cap_tiles <= assoc:
        return 0.0 if d >= cap_tiles else 1.0
    p = assoc / cap_tiles
    if d <= 256:
        # exact binomial CDF P(X <= A-1)
        q = 1.0 - p
        term = q ** d
        acc = term
        for a in range(1, assoc):
            term *= (d - a + 1) / a * p / q
            acc += term
        return min(1.0, acc)
    mu = d * p
    sigma = math.sqrt(d * p * (1 - p))
    return _phi((assoc - 1 + 0.5 - mu) / sigma)


class _BIT:
    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int, v: int) -> None:
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def prefix(self, i: int) -> int:        # sum of [0, i)
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def expected_misses(keys: list[int], streams: list[int], n_streams: int,
                    assoc: int, cap_tiles: float) -> list[float]:
    n = len(keys)
    bit = _BIT(n)
    last: dict[int, int] = {}
    miss = [0.0] * n_streams
    for t, (key, s) in enumerate(zip(keys, streams)):
        if key in last:
            lt = last[key]
            d = bit.prefix(t) - bit.prefix(lt + 1)   # distinct keys touched in (lt, t)
            miss[s] += 1.0 - hit_prob(d, assoc, cap_tiles)
            bit.add(lt, -1)
        else:
            miss[s] += 1.0
        bit.add(t, 1)
        last[key] = t
    return miss
