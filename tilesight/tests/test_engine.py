"""Engine invariants (the C++ core is the only engine now — no python/cpp parity to check)."""
import math

from tilesight import HardwareSpec, _core

HW = HardwareSpec.load("b300")

TileConfig = _core.GemmTile


def lower_gemm(cur_gpu_config, name, M, N, K, **kw):
    kw.setdefault("tile", _core.GemmTile())
    return _core.lower_gemm(cur_gpu_config, name, M, N, K, **kw)


def evaluate(k, hw):
    return _core.evaluate(hw, k)


def test_ddr_bound_decode_gemm_hits_bandwidth():
    # M=1 weight streaming: time must be ~ bytes / (DDR bw * eff)
    N, K = 16384, 7168
    ks = lower_gemm(HW, "g", 1, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                    tile=TileConfig(bm=64, bn=64, bk=128))
    t = sum(evaluate(k, HW).time_s for k in ks)
    # weights are a matrix operand: DMA'd, so the ceiling is min(HBM, the DMA engines)
    lanes = {l.name: l for l in HW.lanes()}
    ideal = N * K / min(8.0e12 * 0.88, lanes["dma"].total_rate)
    assert ideal <= t < 1.6 * ideal


def test_large_gemm_is_tc_bound():
    M = N = K = 8192
    ks = lower_gemm(HW, "g", M, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                    tile=TileConfig(bm=128, bn=256, bk=64, cluster_m=2, cta_pair=True))
    r = evaluate(ks[0], HW)
    ideal = 2 * M * N * K / (4500e12 * 0.92)
    assert r.bottleneck == "tc"
    assert ideal <= r.time_s < 1.25 * ideal


def test_more_ddr_bw_never_slower():
    prev = math.inf
    for bw in (4, 8, 16):
        h = HW.override({"memory.ddr.bandwidth_TBps": bw})
        ks = lower_gemm(h, "g", 16, 7168, 7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8")
        t = sum(evaluate(k, h).time_s for k in ks)
        assert t <= prev + 1e-15
        prev = t


def test_recurrence_limits_serial_chain():
    body = [_core.TraceAction("a", {"tc": 1e-6}, recurrent=True),
            _core.TraceAction("b", {"sfu": 1e-6}, [0], recurrent=True)]
    k1 = _core.LoweredKernel(_core.TraceKernel("k", "attention", 160, 10, 2, 1, body, consumers=1))
    k2 = _core.LoweredKernel(_core.TraceKernel("k", "attention", 160, 10, 2, 1, body, consumers=2))
    t1, t2 = evaluate(k1, HW), evaluate(k2, HW)
    assert t1.bottleneck == "latency" and t2.time_s < t1.time_s


def test_tail_wave_counted():
    body = [_core.TraceAction("a", {"tc": 1e-6})]
    k = _core.LoweredKernel(_core.TraceKernel("k", "gemm", HW.sms + 1, 1, 2, 1, body))
    assert evaluate(k, HW).waves == 2
