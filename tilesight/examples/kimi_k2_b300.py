"""Kimi-K2 on 8x B300: decode + prefill reports, memory, bottlenecks, tile comparison."""
from tilesight import CurModelConfig, HardwareSpec, ModelSpec, RunConfig, run
from tilesight.generateResult.report.table import model_summary

# The model's only two inputs: cur_gpu_config (what the part is, incl. how it tiles) and
# cur_model_config (what runs on it: architecture + run settings).
cur_gpu_config = HardwareSpec.load("b300")
m = ModelSpec.load("kimi_k2.hf")

decode = CurModelConfig(spec=m, run=RunConfig(phase="decode", batch=256, seq_len=8192, dp=8))
prefill = CurModelConfig(spec=m, run=RunConfig(phase="prefill", batch=8, seq_len=8192, dp=8, comm_overlap=0.7))

print(model_summary(run(cur_gpu_config, decode), top=12))
print()
print(model_summary(run(cur_gpu_config, prefill), top=12))

# fixed tile vs auto-searched tile for the routed experts (tile policy is a GPU-side config)
print("\nexperts_gate_up tile study (decode b=256):")
for fields in ({"bm": 64, "bn": 64, "bk": 64}, {"bm": 128, "bn": 128, "bk": 64}, {"bm": 64, "bn": 256, "bk": 64}, None):
    cur_gpu_config_i = cur_gpu_config.override(
        {"compute.tile_policy.overrides": {"*.experts_gate_up": fields}} if fields else {})
    rep = run(cur_gpu_config_i, decode)
    o = next(o for o in rep.ops if o.op.name.endswith("experts_gate_up"))
    print(f"  {str(fields or 'auto'):42s} tile={o.tile:18s} {o.time_s*1e6:7.1f} us  limiter={o.bottleneck}")
