"""How big should an extra on-chip buffer be, and what should live in it?

Run: PYTHONPATH=python python examples/onchip_buffer_study.py
"""
from tilesight import HardwareSpec, ModelSpec, RunConfig
from tilesight.dse.buffer import capacity_curve, optimize_buffer, with_buffer
from tilesight.model.runner import run_model

cur_gpu_config, model = HardwareSpec.load("b300"), ModelSpec.load("kimi_k2.hf")
rc = RunConfig(phase="decode", batch=256, seq_len=8192, dp=8)
L2 = cur_gpu_config.get("memory.l2.capacity_MB")

print(f"# capacity curve (L2 = {L2} MB), buffer bandwidth = L2 bandwidth, bypass+prefetch on")
for r in capacity_curve(model, cur_gpu_config, rc, (0, L2, 2 * L2, 4 * L2, 8 * L2, 16 * L2),
                        bypass_l2=True, prefetch=True, costream=True):
    print(f"  {r['capacity_MB']:7.0f} MB ({r['x_L2']:5.2f}x L2)  {r['step_ms']:7.2f} ms  "
          f"x{r['speedup']:.3f}  bound {r['top_bound']}")

cap = 4 * L2
print(f"\n# what to put in a {cap:.0f} MB buffer (4x L2)")
best, rows = optimize_buffer(model, cur_gpu_config, rc, cap, steps=2)
for r in rows[:8]:
    print(f"  {r['policy']:5s} w={r['pin_weight']} kv={r['pin_kv']} act={r['pin_act']} "
          f"byp={int(r['bypass_l2'])} pre={int(r['prefetch'])} cos={int(r['costream'])}  {r['step_ms']:6.2f} ms  "
          f"x{r['speedup_vs_no_buffer']}  {r['top_bound']}")

print(f"\n# buffer bandwidth matters once capacity is large ({8 * L2:.0f} MB)")
for bw in (5, 10, 20, 40):
    h = with_buffer(cur_gpu_config, 8 * L2, bandwidth_TBps=bw, bypass_l2=True, prefetch=True, costream=True)
    rep = run_model(model, h, rc)
    print(f"  {bw:4.0f} TB/s  {rep.step_time_s * 1e3:7.2f} ms  bound {next(iter(rep.detail_breakdown()))}")

print("\n# prefill is compute-bound: the same buffer buys much less")
rc_p = RunConfig(phase="prefill", batch=8, seq_len=8192, dp=8)
for cap_mb in (0, 4 * L2):
    h = with_buffer(cur_gpu_config, cap_mb, bypass_l2=True, prefetch=True, costream=True)
    rep = run_model(model, h, rc_p)
    print(f"  {cap_mb:7.0f} MB  {rep.step_time_s * 1e3:8.2f} ms  bound {next(iter(rep.detail_breakdown()))}")
