"""Correctness/regression tests for the C++ core.

There is only one implementation now (no Python reference engine to compare against), so this
no longer checks Python/C++ *parity* — it pins the engine's behavior against known-good
analytical bounds and a small hand-checkable cache trace, and guards against silent regressions
in the wave-decomposition/round-formula math and the deterministic tile cache.
"""
import random

import pytest

from tilesight import HardwareSpec, ModelSpec, RunConfig, _core, run_model

HW = HardwareSpec.load("b300")


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def test_cache_simulate_hand_checkable_trace():
    # one partition, capacity for exactly 2 tiles of 100 bytes, LRU: A B C A -> A B C are
    # compulsory misses, the second A evicted B already (capacity 2, so A is gone too) -> miss
    keys = [1, 2, 3, 1]
    addrs = [0, 100, 200, 0]
    sizes = [100.0] * 4
    streams = [0] * 4
    r = _core.cache_simulate(keys, addrs, sizes, streams, 1, capacity_bytes=200.0, n_partitions=1,
                             ports=1, granularity=1024, policy="lru")
    assert r["misses"] == [4.0]                     # A, B, C compulsory + A evicted before reuse
    assert r["accesses"] == [4]

    # now capacity for all 3 tiles: the second A must hit
    r2 = _core.cache_simulate(keys, addrs, sizes, streams, 1, capacity_bytes=300.0, n_partitions=1,
                              ports=1, granularity=1024, policy="lru")
    assert r2["misses"] == [3.0]


def test_cache_simulate_is_deterministic():
    rnd = random.Random(0)
    n = 500
    keys = [rnd.randint(0, 200) for _ in range(n)]
    addrs = [k * 128 for k in keys]
    sizes = [128.0] * n
    streams = [rnd.randint(0, 1) for _ in range(n)]
    a = _core.cache_simulate(keys, addrs, sizes, streams, 2, capacity_bytes=4096.0, n_partitions=2,
                             ports=2, granularity=128, policy="lru")
    b = _core.cache_simulate(keys, addrs, sizes, streams, 2, capacity_bytes=4096.0, n_partitions=2,
                             ports=2, granularity=128, policy="lru")
    assert a["misses"] == b["misses"] and a["hit_rate"] == b["hit_rate"]


def test_gemm_ddr_bound_matches_bandwidth_formula():
    N, K = 16384, 7168
    ks = _core.lower_gemm(HW, "g", 1, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                          tile=_core.GemmTile(bm=64, bn=64, bk=128))
    t = sum(_core.evaluate(HW, k).time_s for k in ks)
    ideal = N * K / (8.0e12 * 0.88)
    assert ideal <= t < 1.6 * ideal


def test_gemm_tc_bound_matches_flop_formula():
    M = N = K = 8192
    ks = _core.lower_gemm(HW, "g", M, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                          tile=_core.GemmTile(bm=128, bn=256, bk=64, cluster_m=2, cta_pair=True))
    r = _core.evaluate(HW, ks[0])
    ideal = 2 * M * N * K / (4500e12 * 0.92)
    assert r.bottleneck == "tc"
    assert ideal <= r.time_s < 1.25 * ideal


@pytest.mark.parametrize("phase,batch,expected_s,limiter", [
    ("decode", 64, 0.02663803906891158, "ddr"),
    ("prefill", 8, 0.16973208393601716, "tc"),
])
def test_kimi_step_time_regression(phase, batch, expected_s, limiter):
    """Pins the whole-model step time for a known config, so a change to the engine, the
    lowering, or the cache model shows up here even when no single-kernel test catches it."""
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(phase=phase, batch=batch, seq_len=4096, dp=8)
    rep = run_model(m, HW, rc)
    assert _rel(rep.step_time_s, expected_s) < 0.02
    assert next(iter(rep.limiter_breakdown())) == limiter


def test_run_model_is_deterministic():
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(phase="decode", batch=32, seq_len=2048, dp=8)
    a = run_model(m, HW, rc).step_time_s
    b = run_model(m, HW, rc).step_time_s
    assert a == b
