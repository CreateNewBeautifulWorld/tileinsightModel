"""Engine invariants (backend-agnostic)."""
import math

from tilesight import HardwareSpec
from tilesight.model.engine import reference
from tilesight.model.engine.cache import hit_prob, expected_misses
from tilesight.model.ir.kernel import Action, Kernel
from tilesight.model.kernels.gemm import lower_gemm
from tilesight.model.kernels.tiles import TileConfig

HW = HardwareSpec.load("b300")


def test_ddr_bound_decode_gemm_hits_bandwidth():
    # M=1 weight streaming: time must be ~ bytes / (DDR bw * eff)
    N, K = 16384, 7168
    ks = lower_gemm(HW, "g", 1, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                    tile=TileConfig(bm=64, bn=64, bk=128))
    t = sum(reference.evaluate(k, HW).time_s for k in ks)
    ideal = N * K / (8.0e12 * 0.88)
    assert ideal <= t < 1.6 * ideal


def test_large_gemm_is_tc_bound():
    M = N = K = 8192
    ks = lower_gemm(HW, "g", M, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                    tile=TileConfig(bm=128, bn=256, bk=64, cluster_m=2, cta_pair=True))
    r = reference.evaluate(ks[0], HW)
    ideal = 2 * M * N * K / (4500e12 * 0.92)
    assert r.bottleneck == "tc"
    assert ideal <= r.time_s < 1.25 * ideal


def test_more_ddr_bw_never_slower():
    prev = math.inf
    for bw in (4, 8, 16):
        h = HW.override({"memory.ddr.bandwidth_TBps": bw})
        ks = lower_gemm(h, "g", 16, 7168, 7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8")
        t = sum(reference.evaluate(k, h).time_s for k in ks)
        assert t <= prev + 1e-15
        prev = t


def test_recurrence_limits_serial_chain():
    body = [Action("a", {"tc": 1e-6}, recurrent=True), Action("b", {"sfu": 1e-6}, [0], recurrent=True)]
    k1 = Kernel("k", "attention", 160, 10, 2, 1, body, consumers=1)
    k2 = Kernel("k", "attention", 160, 10, 2, 1, body, consumers=2)
    t1, t2 = reference.evaluate(k1, HW), reference.evaluate(k2, HW)
    assert t1.bottleneck == "latency" and t2.time_s < t1.time_s


def test_tail_wave_counted():
    body = [Action("a", {"tc": 1e-6})]
    k = Kernel("k", "gemm", HW.sms + 1, 1, 2, 1, body)
    assert reference.evaluate(k, HW).waves == 2


def test_hit_prob_limits():
    assert hit_prob(0, 16, 1000) == 1.0
    assert hit_prob(100000, 16, 1000) < 1e-6
    assert hit_prob(500, 16, 1000) > 0.99
    # binomial/gaussian branches agree near the switch point
    assert abs(hit_prob(256, 16, 256) - hit_prob(257, 16, 256)) < 0.05


def test_reuse_within_capacity_hits():
    keys = list(range(50)) * 3
    (m,) = expected_misses(keys, [0] * len(keys), 1, 16, 10000)
    assert abs(m - 50) < 1e-6          # only compulsory misses
