"""C++ core must match the Python reference (engine + cache model)."""
import random

import pytest

from tilesight import HardwareSpec, ModelSpec, RunConfig
from tilesight.model.engine import cache as pycache
from tilesight.model.engine import reference
from tilesight.model.lower import lower_model
from tilesight.interfaceAndModelRun import runner

core = pytest.importorskip("tilesight._core")


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def test_cache_parity():
    rnd = random.Random(0)
    for _ in range(20):
        n = rnd.randint(10, 3000)
        keys = [rnd.randint(0, rnd.randint(5, 800)) for _ in range(n)]
        streams = [rnd.randint(0, 1) for _ in range(n)]
        a = pycache.expected_misses(keys, streams, 2, 16, rnd.choice([32.0, 100.0, 5000.0]))
        cap = 100.0
        a = pycache.expected_misses(keys, streams, 2, 16, cap)
        b = core.expected_misses(keys, streams, 2, 16, cap)
        for x, y in zip(a, b):
            assert _rel(x, y) < 1e-9


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_engine_parity_on_kimi(phase):
    from tilesight.model.engine.cpp_bridge import evaluate_cpp
    cur_gpu_config = HardwareSpec.load("b300")
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(phase=phase, batch=64 if phase == "decode" else 8, seq_len=4096, dp=8)
    n = 0
    for _, _, ops in lower_model(m, rc):
        for op in ops:
            ks, _ = runner.resolve_op(op, cur_gpu_config, rc)
            for kr in ks:
                pass
    # re-lower a sample of kernels and compare engines directly
    from tilesight.model.kernels.gemm import lower_gemm
    from tilesight.model.kernels.tiles import gemm_search_space
    from tilesight.model.kernels.attention import lower_attention_decode, lower_attention_prefill
    for k in (lower_attention_decode(cur_gpu_config, "a", B=16, H=64, kv_heads=1, S=8192, d_qk=576, d_v=512, v_in_k=True)
              + lower_attention_prefill(cur_gpu_config, "p", B=1, H=8, kv_heads=1, S=8192, d_qk=128, d_v=128)):
        p, c = reference.evaluate(k, cur_gpu_config), evaluate_cpp(core, k, cur_gpu_config)
        assert _rel(p.time_s, c.time_s) < 1e-9 and p.limiter_detail.keys() == c.limiter_detail.keys()
    for M, N, K in [(8, 7168, 2048), (4096, 4096, 7168), (1, 163840, 7168)]:
        for t in gemm_search_space(M, N, K)[::5]:
            ks = lower_gemm(cur_gpu_config, "g", M, N, K, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=t)
            if not ks:
                continue
            for k in ks:
                p, c = reference.evaluate(k, cur_gpu_config), evaluate_cpp(core, k, cur_gpu_config)
                assert _rel(p.time_s, c.time_s) < 1e-9
                assert p.bottleneck == c.bottleneck
                for lane, v in p.util.items():
                    assert _rel(v, c.util[lane]) < 1e-9
                assert set(p.limiter_detail) == set(c.limiter_detail)
                for key, v in p.limiter_detail.items():
                    assert _rel(v, c.limiter_detail[key]) < 1e-9
                n += 1
    assert n > 20
