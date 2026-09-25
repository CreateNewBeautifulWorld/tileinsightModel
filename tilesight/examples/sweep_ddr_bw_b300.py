"""Experiment: how much DDR (HBM) bandwidth does Kimi-K2 decode need on a B300-class GPU?

1. sweep DDR BW with L2 fixed, and with L2 scaled alongside (L2 = 2.56x DDR, the B300 ratio)
2. repeat for several batch sizes (the DDR/compute balance moves with batch)
3. bisect the DDR BW required to hit a TPOT target

Run:  PYTHONPATH=python python examples/sweep_ddr_bw_b300.py
"""
from tilesight import HardwareSpec, ModelSpec, RunConfig
from tilesight.gpuTilingPerfHWModel.model.dse.sweep import required_value, rows_to_csv, sweep

cur_gpu_config = HardwareSpec.load("b300")
model = ModelSpec.load("kimi_k2.hf")
BW = [4, 6, 8, 10, 12, 16, 24, 32]
L2_RATIO = cur_gpu_config.get("memory.l2.bandwidth_TBps") / cur_gpu_config.get("memory.ddr.bandwidth_TBps")

for batch in (64, 256, 1024):
    rc = RunConfig(phase="decode", batch=batch, seq_len=8192, tp=1, dp=8)
    print(f"\n### batch={batch} seq=8192 (dp=8, ep=8), L2 fixed at {cur_gpu_config.get('memory.l2.bandwidth_TBps')} TB/s")
    print(rows_to_csv(sweep(model, cur_gpu_config, rc, "memory.ddr.bandwidth_TBps", BW)))
    print(f"### batch={batch}, L2 scaled with DDR (x{L2_RATIO:.2f})")
    print(rows_to_csv(sweep(model, cur_gpu_config, rc, "memory.ddr.bandwidth_TBps", BW,
                            linked=lambda v: {"memory.l2.bandwidth_TBps": v * L2_RATIO})))

rc = RunConfig(phase="decode", batch=256, seq_len=8192, tp=1, dp=8)
for target in (30.0, 25.0, 20.0, 15.0):
    v = required_value(model, cur_gpu_config, rc, "memory.ddr.bandwidth_TBps", target, lo=1, hi=64,
                       linked=lambda v: {"memory.l2.bandwidth_TBps": v * L2_RATIO})
    print(f"TPOT <= {target:5.1f} ms needs DDR BW >= {v if v is None else round(v, 2)} TB/s (L2 scaled)")
