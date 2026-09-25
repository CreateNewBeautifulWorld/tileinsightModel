"""MHA / GQA / MLA configs, flash vs naive, sliding window, per-block overrides."""
from tilesight import HardwareSpec, ModelSpec, RunConfig, run_model
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.lower import lower_model
from tilesight.gpuTilingPerfHWModel.model.memory import kv_bytes_per_token

HW = HardwareSpec.load("b300")


def _attn_time(rep):
    return sum(o.total_s for o in rep.ops if ".attn" in o.op.name and "allreduce" not in o.op.name)


def test_hf_import_picks_family():
    assert ModelSpec.load("llama2_7b.hf").layers[0].blocks[1]["type"] == "mha"
    assert ModelSpec.load("llama3_70b.hf").layers[0].blocks[1]["type"] == "gqa"
    assert ModelSpec.load("kimi_k2.hf").layers[0].blocks[1]["type"] == "mla"


def test_kv_bytes_per_family():
    rc = RunConfig(kv_dtype="bf16", tp=1)
    assert kv_bytes_per_token(ModelSpec.load("llama2_7b.hf"), rc) == 32 * 32 * 256 * 2
    assert kv_bytes_per_token(ModelSpec.load("llama3_70b.hf"), rc) == 80 * 8 * 256 * 2
    # tp shards kv heads (min 1 per rank)
    assert kv_bytes_per_token(ModelSpec.load("llama3_70b.hf"), RunConfig(tp=16)) == 80 * 1 * 256 * 2


def test_naive_gqa_folds_query_heads_into_M():
    rc = RunConfig(phase="decode", batch=8, seq_len=1024, tp=1, dp=1, attn_impl="naive")
    ops = {o.name: o for _, _, ops in lower_model(ModelSpec.load("llama3_70b.hf"), rc) for o in ops}
    sc = ops["dense.1.gqa.attn_scores"].p
    assert sc["M"] == 64 // 8 and sc["batch"] == 8 * 8 and sc["N"] == 1024 and sc["c_dtype"] == "fp32"


def test_naive_prefill_slower_ddr_bound_and_memory_hungry():
    m = ModelSpec.load("llama3_70b.hf")
    base = dict(phase="prefill", batch=1, seq_len=16384, tp=8, dp=1)
    f = run_model(m, HW, RunConfig(attn_impl="flash", **base))
    n = run_model(m, HW, RunConfig(attn_impl="naive", **base))
    assert _attn_time(n) > 2 * _attn_time(f)
    assert n.memory.activations_GB > 10 * f.memory.activations_GB
    lim = {o.op.name.split(".")[-1]: o.bottleneck for o in n.ops}
    assert lim["attn_softmax"] == "ddr"


def test_per_block_impl_override():
    spec = ModelSpec.load("llama2_7b.hf").to_dict()
    spec["layers"][0]["blocks"][1]["impl"] = "naive"
    m = ModelSpec.from_dict(spec)
    rep = run_model(m, HW, RunConfig(phase="decode", batch=8, seq_len=2048, tp=1, dp=1, attn_impl="flash"))
    names = [o.op.name for o in rep.ops]
    assert any(n.endswith("attn_scores") for n in names) and not any(n.endswith(".attn") for n in names)


def test_sliding_window_cuts_kv_and_time():
    spec = ModelSpec.load("llama3_70b.hf").to_dict()
    full = ModelSpec.from_dict(spec)
    spec["layers"][0]["blocks"][1]["sliding_window"] = 4096
    win = ModelSpec.from_dict(spec)
    for phase, b in (("decode", 32), ("prefill", 1)):
        rc = RunConfig(phase=phase, batch=b, seq_len=32768, tp=8, dp=1)
        rf, rw = run_model(full, HW, rc), run_model(win, HW, rc)
        assert _attn_time(rw) < 0.5 * _attn_time(rf)
        assert rw.memory.kv_cache_GB < 0.2 * rf.memory.kv_cache_GB


def test_mla_absorb_switch():
    spec = ModelSpec.load("kimi_k2.hf").to_dict()
    for g in spec["layers"]:
        g["blocks"][1]["absorb"] = False
    m = ModelSpec.from_dict(spec)
    rc = RunConfig(phase="decode", batch=64, seq_len=4096, dp=8)
    names = [o.name for _, _, ops in lower_model(m, rc) for o in ops]
    assert any(n.endswith("kv_b") for n in names) and not any("absorb" in n for n in names)
