from tilesight import HardwareSpec, ModelSpec, RunConfig, run_model
from tilesight.gpuTilingPerfHWModel.model.memory import kv_bytes_per_token


def test_kimi_import_shapes():
    m = ModelSpec.load("kimi_k2.hf")
    assert [g.repeat for g in m.layers] == [1, 60]
    moe = m.layers[1].blocks[-1]
    assert moe["experts"] == 384 and moe["topk"] == 8 and moe["d_ff"] == 2048


def test_kimi_total_params_about_1T():
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(tp=1, dp=1, ep=1, weight_dtype="fp8", expert_dtype="fp8", batch=1, seq_len=128)
    rep = run_model(m, HardwareSpec.load("b300"), rc)
    # fp8 routed+dense weights ~1.0e12 bytes, embeddings/lm_head bf16 add ~4.7 GB
    assert 0.98e3 < rep.memory.weights_GB < 1.1e3


def test_kv_bytes_mla():
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(kv_dtype="bf16")
    assert kv_bytes_per_token(m, rc) == 61 * 576 * 2


def test_decode_report_runs_and_is_ddr_dominated():
    rep = run_model(ModelSpec.load("kimi_k2.hf"), HardwareSpec.load("b300"),
                    RunConfig(phase="decode", batch=128, seq_len=4096, dp=8))
    mix = rep.limiter_breakdown()
    assert next(iter(mix)) in ("ddr", "dma")      # HBM or the DMA engines feeding from it
    assert rep.memory.fits


def test_tile_override_is_used():
    rc = RunConfig(phase="decode", batch=64, seq_len=2048, dp=8)
    cur_gpu_config = HardwareSpec.load("b300").override(
        {"compute.tile_policy.overrides": {"*.experts_gate_up": {"bm": 64, "bn": 128, "bk": 128}}})
    rep = run_model(ModelSpec.load("kimi_k2.hf"), cur_gpu_config, rc)
    tiles = {o.op.name: o.tile for o in rep.ops}
    assert tiles["moe.3.moe.experts_gate_up"].startswith("64x128x128")
