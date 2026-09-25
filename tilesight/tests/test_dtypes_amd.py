"""Datatype coverage (int4 / nvfp4 / fp4 / fp8 / int8 / bf16 / fp16 / fp32) and AMD presets."""
import pytest

from tilesight import HardwareSpec, ModelSpec, RunConfig, run_model
from tilesight.hw.spec import DTYPE_BYTES
from tilesight.kernels.gemm import lower_gemm
from tilesight.kernels.tiles import TileConfig
from tilesight.server import _kernel_candidates

NV = ["b300", "b200", "h200"]
AMD = ["mi300x", "mi325x", "mi355x", "mi450"]


@pytest.mark.parametrize("name", NV + AMD)
def test_preset_loads_and_runs_a_gemm(name):
    cur_gpu_config = HardwareSpec.load(name)
    c = _kernel_candidates(cur_gpu_config, dict(kernel="gemm", M=4096, N=4096, K=7168,
                                    a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8"))[0]
    assert c["time_us"] >= c["ideal_us"] > 0
    assert 0 < c["peak_pct"] <= 1.0


@pytest.mark.parametrize("dt", ["int4", "nvfp4", "fp4", "fp8", "int8", "bf16", "fp16", "fp32"])
def test_every_dtype_is_usable(dt):
    cur_gpu_config = HardwareSpec.load("b300")
    ks = lower_gemm(cur_gpu_config, "g", 1024, 1024, 1024, a_dtype="bf16" if dt == "int4" else dt,
                    b_dtype=dt, compute_dtype="bf16" if dt == "int4" else dt,
                    tile=TileConfig(bm=64, bn=64, bk=64))
    assert ks and DTYPE_BYTES[dt] > 0


def test_dtype_ordering_on_blackwell():
    cur_gpu_config = HardwareSpec.load("b300")
    t = {}
    for dt in ("nvfp4", "fp8", "bf16", "fp32"):
        t[dt] = _kernel_candidates(cur_gpu_config, dict(kernel="gemm", M=4096, N=4096, K=7168,
                                            a_dtype=dt, b_dtype=dt, compute_dtype=dt))[0]["time_us"]
    assert t["nvfp4"] < t["fp8"] < t["bf16"] < t["fp32"]


def test_fp32_uses_cuda_cores_not_tensor_cores():
    cur_gpu_config = HardwareSpec.load("b300")
    lane, _ = cur_gpu_config.mma_cost(1e12, "fp32")
    assert lane == "cuda"
    assert cur_gpu_config.mma_cost(1e12, "fp8")[0] == "tc"


def test_nvfp4_aliases_to_the_fp4_datapath():
    cur_gpu_config = HardwareSpec.load("b300")
    assert cur_gpu_config.tc_datapath("nvfp4") == "fp4" and cur_gpu_config.tc_datapath("mxfp4") == "fp4"
    assert abs(cur_gpu_config.tc_time_per_sm(1e12, "nvfp4") - cur_gpu_config.tc_time_per_sm(1e12, "fp4")) < 1e-18


def test_missing_datapath_widens():
    h200 = HardwareSpec.load("h200")            # no FP4 datapath on Hopper
    assert h200.tc_datapath("nvfp4") == "fp8"
    mi300 = HardwareSpec.load("mi300x")         # no FP4 on CDNA3
    assert mi300.tc_datapath("fp4") == "fp8"


def test_int4_is_weight_only_and_halves_traffic():
    cur_gpu_config = HardwareSpec.load("b300")
    tile = TileConfig(bm=64, bn=128, bk=128)
    a = lower_gemm(cur_gpu_config, "g", 8, 7168, 7168, a_dtype="bf16", b_dtype="int4",
                   compute_dtype="bf16", tile=tile)[0]
    b = lower_gemm(cur_gpu_config, "g", 8, 7168, 7168, a_dtype="bf16", b_dtype="fp8",
                   compute_dtype="bf16", tile=tile)[0]
    assert a.meta["weight_bytes"] == b.meta["weight_bytes"] / 2


def test_amd_has_no_clusters_or_tmem():
    for name in AMD:
        cur_gpu_config = HardwareSpec.load(name)
        assert not cur_gpu_config.get("compute.cluster_multicast", False)
        assert not cur_gpu_config.has_tmem                       # accumulators live in AGPRs
        assert cur_gpu_config.resolve_path("tma") in (cur_gpu_config.get("load_paths") or {})   # falls back to a real path
        ks = lower_gemm(cur_gpu_config, "g", 4096, 4096, 4096, tile=TileConfig(bm=128, bn=128, bk=64, cluster_m=2))
        assert ks is None


def test_amd_model_run_and_kv():
    cur_gpu_config = HardwareSpec.load("mi355x")
    rep = run_model(ModelSpec.load("llama3_70b.hf"), cur_gpu_config,
                    RunConfig(phase="decode", batch=64, seq_len=4096, tp=8, dp=1))
    assert rep.step_time_s > 0 and rep.memory.weights_GB > 0
    assert "tmem" not in rep.limiter_breakdown()
