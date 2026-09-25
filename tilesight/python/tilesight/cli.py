"""CLI.

  python -m tilesight.cli run   --model kimi_k2.hf --hw b300 --phase decode --batch 256 --seq 8192 --dp 8
  python -m tilesight.cli request --model kimi_k2.hf --hw b300 --batch 256 --dp 8 --prompt 8192 --output 4096
  python -m tilesight.cli sweep --model kimi_k2.hf --hw b300 --param memory.ddr.bandwidth_TBps --values 4,6,8,12,16
  python -m tilesight.cli need  --model kimi_k2.hf --hw b300 --param memory.ddr.bandwidth_TBps --target-ms 20
  python -m tilesight.cli dump-model --model kimi_k2.hf      # editable block-level YAML
  python -m tilesight.cli serve --host 0.0.0.0 --port 8000   # web UI; computation stays on this machine
"""
from __future__ import annotations

import argparse
import json

import yaml

from . import HardwareSpec, ModelSpec, RunConfig, run_model
from .dse.sweep import required_value, rows_to_csv, sweep
from .report.table import model_summary, ops_csv


def _rc(a) -> RunConfig:
    kw = {}
    if a.run_config:
        with open(a.run_config) as f:
            kw.update(yaml.safe_load(f))
    for k in ("phase", "batch", "tp", "dp", "ep", "weight_dtype", "expert_dtype", "kv_dtype", "compute_dtype",
              "page_size", "attn_impl"):
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    if a.seq is not None:
        kw["seq_len"] = a.seq
    if a.prompt is not None:
        kw["prompt_len"] = a.prompt
    if a.output is not None:
        kw["output_len"] = a.output
    return RunConfig(**kw)


def _hw(a) -> HardwareSpec:
    hw = HardwareSpec.load(a.hw)
    if getattr(a, "ideal", False):
        hw = hw.lossless()
    for s in a.set or []:
        k, v = s.split("=")
        hw = hw.override({k: yaml.safe_load(v)})
    if getattr(a, "tile", None):
        hw = hw.override({"compute.tile_policy.gemm": a.tile})
    return hw


def main(argv=None):
    ap = argparse.ArgumentParser("tilesight")
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("buffer", help="size and configure an extra on-chip shared buffer")
    bp.add_argument("--model", required=True)
    bp.add_argument("--hw", default="b300")
    bp.add_argument("--run-config", dest="run_config")
    bp.add_argument("--phase", default="decode")
    bp.add_argument("--batch", type=int, default=256)
    bp.add_argument("--seq", type=int, default=8192)
    bp.add_argument("--tp", type=int, default=1)
    bp.add_argument("--dp", type=int, default=8)
    bp.add_argument("--capacity", type=float, help="MB; omit to sweep the capacity curve")
    bp.add_argument("--capacities", default="0,128,256,512,1024,2048")
    bp.add_argument("--bw", type=float, help="buffer bandwidth TB/s (default: L2 bandwidth)")
    bp.add_argument("--steps", type=int, default=2, help="pin-share grid resolution")
    bp.add_argument("--csv")

    kp = sub.add_parser("kernel", help="single-GPU kernel study (tile ranking + Figure-3e timeline)")
    kp.add_argument("--hw", default="b300")
    kp.add_argument("--kernel", default="gemm",
                    choices=["gemm", "grouped_gemm", "attn_decode", "attn_prefill", "elementwise"])
    kp.add_argument("--shape", required=True, help='JSON, e.g. \'{"M":4096,"N":4096,"K":7168}\'')
    kp.add_argument("--tile", default="auto")
    kp.add_argument("--dtype", default="fp8", help="a/b/compute dtype shortcut")
    kp.add_argument("--top", type=int, default=8)
    kp.add_argument("--unit", default="us", choices=["us", "cyc"], help="gantt time unit")
    kp.add_argument("--trace-out", dest="trace_out", help="write the full text trace to this file")
    kp.add_argument("--csv-out", dest="csv_out", help="write a per-cycle CSV (one row per cycle, one column per unit)")
    kp.add_argument("--base", default="0x0", help="base address of the first tensor")
    kp.add_argument("--addr", action="store_true",
                    help="tile-granularity address analysis: how the wave's tiles spread over L2 slices / HBM ports")
    kp.add_argument("--layout-a", dest="layout_a", default="row",
                    help="row | col | swizzle:G | xor:B | zorder")
    kp.add_argument("--layout-b", dest="layout_b", default="row")
    kp.add_argument("--fig3", dest="fig3", help="write Figure 3(d)(e)(f) as a self-contained HTML file")
    kp.add_argument("--fig3-json", dest="fig3_json", help="same three panels as JSON")
    kp.add_argument("--xlsx-out", dest="xlsx_out", help="write the colour-coded per-cycle Excel grid")
    kp.add_argument("--csv-full", dest="csv_full", action="store_true",
                    help="CSV covers prologue + every iteration + epilogue (large) instead of the drawn rounds")
    kp.add_argument("--set", action="append", help="hw override path=value")
    kp.add_argument("--ideal", action="store_true", help="switch every modelled loss off")

    gp = sub.add_parser("gpu", help="slice-based GPU config: describe, derive, translate")
    gp.add_argument("--file", help="slice config YAML (omit to list the schema)")
    gp.add_argument("--workload", help="workload YAML, to add the memory breakdown")
    gp.add_argument("--md", help="write the slice-config reference as markdown")
    gp.add_argument("--out", help="write the translated flat hardware config here")

    cp2 = sub.add_parser("config", help="the hardware config interface: list / validate / template")
    cp2.add_argument("--list", action="store_true", help="list every field the model can see")
    cp2.add_argument("--section", help="only this section")
    cp2.add_argument("--tag", choices=["spec", "calib", "policy", "loss"], help="only this tag")
    cp2.add_argument("--validate", help="check a hardware file or preset name against the schema")
    cp2.add_argument("--md", help="write the reference as markdown")
    cp2.add_argument("--csv", help="write the reference as CSV")
    cp2.add_argument("--workload", action="store_true", help="list the workload fields instead")

    ap2 = sub.add_parser("addrmap", help="dump the address -> L2 slice / HBM port mapping")
    ap2.add_argument("--hw", default="b300")
    ap2.add_argument("--set", action="append", help="override, e.g. memory.addressing.l2.mode=hash")
    ap2.add_argument("--rows", type=int, default=8)
    ap2.add_argument("--base", default="0x0")
    ap2.add_argument("--decode", help="comma-separated addresses to decode")
    ap2.add_argument("--csv", help="write the full stripe/range table")
    ap2.add_argument("--side", default="both", choices=["l2", "ddr", "both"])

    mp = sub.add_parser("memmap", help="per-GPU memory map: base address of every tensor")
    mp.add_argument("--model", required=True)
    mp.add_argument("--hw", default="b300")
    mp.add_argument("--phase", default="decode")
    mp.add_argument("--batch", type=int, default=256)
    mp.add_argument("--seq", type=int, default=8192)
    mp.add_argument("--tp", type=int, default=1)
    mp.add_argument("--dp", type=int, default=8)
    mp.add_argument("--base", default="0x0", help="base offset (hex or decimal)")
    mp.add_argument("--limit", type=int, default=40)
    mp.add_argument("--csv")

    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let other machines reach it")
    sp.add_argument("--port", type=int, default=8000)

    for name in ("run", "request", "sweep", "need", "dump-model"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--hw", default="b300")
        p.add_argument("--run-config")
        p.add_argument("--phase")
        p.add_argument("--batch", type=int)
        p.add_argument("--seq", type=int)
        p.add_argument("--prompt", type=int, help="prompt length (request mode)")
        p.add_argument("--output", type=int, help="generated tokens per request (request mode)")
        p.add_argument("--page-size", dest="page_size", type=int)
        p.add_argument("--attn-impl", dest="attn_impl")
        p.add_argument("--tp", type=int)
        p.add_argument("--dp", type=int)
        p.add_argument("--ep", type=int)
        p.add_argument("--weight-dtype", dest="weight_dtype")
        p.add_argument("--expert-dtype", dest="expert_dtype")
        p.add_argument("--kv-dtype", dest="kv_dtype")
        p.add_argument("--compute-dtype", dest="compute_dtype")
        p.add_argument("--tile", help='fixed GEMM tile, JSON e.g. \'{"bm":128,"bn":256,"bk":64}\'')
        p.add_argument("--set", action="append", help="hw override path=value (repeatable)")
        p.add_argument("--ideal", action="store_true",
                       help="switch every modelled loss off (efficiency, derates, queueing, per-SM caps)")
        p.add_argument("--csv")
        if name in ("sweep", "need"):
            p.add_argument("--param", required=True)
        if name == "sweep":
            p.add_argument("--values", required=True)
        if name == "need":
            p.add_argument("--target-ms", type=float, required=True)
            p.add_argument("--lo", type=float, default=1.0)
            p.add_argument("--hi", type=float, default=64.0)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .server import serve
        return serve(a.host, a.port)
    if a.cmd == "gpu":
        from .hw.slice_config import (SLICE_FIELDS, as_markdown as s_md, derive,
                                      to_hardware_spec, validate_slice_config)
        if a.md:
            open(a.md, "w").write(s_md())
            print(f"written to {a.md}")
            return
        if not a.file:
            cur = None
            for f in SLICE_FIELDS:
                if f.section != cur:
                    cur = f.section
                    print(f"\n[{cur}]")
                print(f"  {f.path:38s} {f.kind:5s} {f.unit:8s} {str(f.default):>12s}  {f.doc[:62]}")
            print(f"\n{len(SLICE_FIELDS)} fields")
            return
        cfg = yaml.safe_load(open(a.file))
        probs = validate_slice_config(cfg)
        if probs:
            print("problems:")
            for p_ in probs:
                print("  -", p_)
            return
        d = derive(cfg)
        print(f"{cfg.get('name', 'gpu')}")
        print(f"  compute      {d['shader_cores']} shader cores x {d['tensor_cores'] // d['shader_cores']}"
              f" tensor cores = {d['tensor_cores']} TCs")
        print(f"               MMA tile {d['mma_tile']}  ->  {d['tflops_per_tensor_core']:.2f} TFLOP/s per TC,"
              f" {d['tflops_per_shader_core']:.1f} per core, {d['pflops_total']:.2f} PFLOP/s total")
        print(f"               vector {d['vector_tflops']:.1f} TFLOP/s · attention tile {d['attention_tile']}")
        print(f"  memory       {d['memory_slices']} slices · HBM {d['hbm_capacity_GB']:.0f} GB @ "
              f"{d['hbm_TBps']:.2f} TB/s · L2 {d['l2_total_MB']:.0f} MB @ {d['l2_TBps']:.2f} TB/s")
        print(f"               L1 {d['l1_total_MB']:.2f} MB @ {d['l1_TBps']:.2f} TB/s · addressing {d['addressing']}")
        print(f"  gmem         down r/w {d['downstream_read_TBps']:.2f}/{d['downstream_write_TBps']:.2f} TB/s ·"
              f" up r/w {d['upstream_read_TBps']:.2f}/{d['upstream_write_TBps']:.2f} TB/s")
        print(f"  buffer       {d['buffer_MB']:.0f} MB @ {d['buffer_TBps']:.2f} TB/s "
              f"({d['buffer_MB_per_slice']:.1f} MB next to each slice) — {d['buffer_contents']}")
        print(f"  switch       {d['switch_TBps']:.2f} TB/s aggregate "
              f"({d['switch_TBps_per_slice']:.2f} TB/s per slice)")
        print(f"  per TC       SMEM {d['smem_per_tensor_core_KB']:.0f} KB · GPR {d['gpr_per_tensor_core_KB']:.0f} KB")
        if a.workload:
            from .model.workload import memory_breakdown
            wl = yaml.safe_load(open(a.workload))
            m = memory_breakdown(wl)
            print(f"\nworkload {wl.get('name', '')} — {m['layers']} layers, {m['sparsity']}")
            print(f"  weights      {m['weights_GB']:.2f} GB  (attention {m['attention_weights_GB']:.2f}"
                  f" + experts {m['expert_weights_GB']:.2f} + dense {m['dense_weights_GB']:.2f})")
            print(f"  KV cache     {m['kv_cache_GB']:.2f} GB  ({m['kv_bytes_per_token_per_layer']:.0f} B"
                  f" per token per layer x {m['layers']} x {m['kv_seq_len']} x {m['batch']})")
            print(f"  total        {m['total_GB']:.2f} GB of {d['hbm_capacity_GB']:.0f} GB HBM")
        if a.out:
            import json as _json
            open(a.out, "w").write(yaml.safe_dump(to_hardware_spec(cfg).raw, sort_keys=False))
            print(f"\ntranslated flat hardware config -> {a.out}")
            del _json
        return
    if a.cmd == "config":
        from .hw.schema import FIELDS, SECTIONS, as_csv, as_markdown, _Required
        if a.validate:
            hw = HardwareSpec.load(a.validate)
            probs = hw.validate()
            print(f"{hw.name}: {'ok' if not probs else str(len(probs)) + ' problem(s)'}")
            for p_ in probs:
                print("  -", p_)
        if a.md:
            open(a.md, "w").write(as_markdown())
            print(f"written to {a.md}")
        if a.csv:
            open(a.csv, "w").write(as_csv())
            print(f"written to {a.csv}")
        if a.workload:
            from .model.workload import WORKLOAD_FIELDS, W_SECTIONS, as_markdown as w_md
            if a.md:
                open(a.md, "w").write(w_md())
                print(f"written to {a.md}")
                return
            cur = None
            for f in WORKLOAD_FIELDS:
                if f.section != cur:
                    cur = f.section
                    print(f"\n[{cur}]")
                print(f"  {f.path:34s} {f.kind:6s} {f.unit:10s} {str(f.default):>8s}  {f.doc[:64]}")
            print(f"\n{len(WORKLOAD_FIELDS)} workload fields in {len(W_SECTIONS)} sections")
            return
        if a.list or not (a.validate or a.md or a.csv):
            sel = [f for f in FIELDS if (not a.section or f.section == a.section)
                   and (not a.tag or f.tag == a.tag)]
            cur = None
            for f in sel:
                if f.section != cur:
                    cur = f.section
                    print(f"\n[{cur}]")
                d = ("REQUIRED" if isinstance(f.default, _Required)
                     else "-" if f.default is None else str(f.default))
                print(f"  {f.path:42s} {f.kind:6s} {f.unit:8s} {d:>10s}  {f.tag:6s} {f.doc[:70]}")
            print(f"\n{len(sel)} of {len(FIELDS)} fields in {len(SECTIONS)} sections; "
                  f"tags: spec={sum(1 for f in FIELDS if f.tag == 'spec')} "
                  f"calib={sum(1 for f in FIELDS if f.tag == 'calib')} "
                  f"policy={sum(1 for f in FIELDS if f.tag == 'policy')} "
                  f"loss={sum(1 for f in FIELDS if f.tag == 'loss')}")
        return
    if a.cmd == "addrmap":
        from .report.addressing import AddressMap
        hw = HardwareSpec.load(a.hw)
        for s_ in a.set or []:
            k_, v_ = s_.split("=")
            hw = hw.override({k_: yaml.safe_load(v_)})
        sides = ["l2", "ddr"] if a.side == "both" else [a.side]
        for side in sides:
            m = AddressMap.from_hw(hw, side)
            print(m.dump(rows=a.rows, base=int(a.base, 0)))
            print()
        if a.decode:
            maps = {s2: AddressMap.from_hw(hw, s2) for s2 in ("l2", "ddr")}
            print(f"{'address':>16s} {'L2 slice':>9s} {'HBM port':>9s}")
            for tok in a.decode.split(","):
                addr = int(tok, 0)
                print(f"0x{addr:014x} {maps['l2'].port_of(addr):9d} {maps['ddr'].port_of(addr):9d}")
        if a.csv:
            with open(a.csv, "w") as f:
                f.write(AddressMap.from_hw(hw, sides[0]).dump_csv(base=int(a.base, 0)))
            print(f"written to {a.csv}")
        return
    if a.cmd == "memmap":
        from .model.memmap import build_memory_map
        mm = build_memory_map(ModelSpec.load(a.model),
                              RunConfig(phase=a.phase, batch=a.batch, seq_len=a.seq, tp=a.tp, dp=a.dp),
                              base=int(a.base, 0))
        print(mm.text(limit=a.limit))
        if a.csv:
            with open(a.csv, "w") as f:
                f.write("region,kind,base,size_bytes,rows,cols,elem_bytes\n")
                for r in mm.regions:
                    f.write(f"{r.name},{r.kind},0x{r.base:x},{r.size},{r.rows},{r.cols},{r.elem_bytes}\n")
            print(f"\nwritten to {a.csv} ({len(mm.regions)} regions)")
        return
    if a.cmd == "buffer":
        from .dse.buffer import best_alloc, capacity_curve, optimize_buffer, profile_classes, with_buffer
        from .dse.sweep import rows_to_csv
        model_ = ModelSpec.load(a.model)
        hw_ = HardwareSpec.load(a.hw)
        rc_ = RunConfig(phase=a.phase, batch=a.batch, seq_len=a.seq, tp=a.tp, dp=a.dp)
        kw = {"bandwidth_TBps": a.bw} if a.bw else {}
        l2 = hw_.get("memory.l2.capacity_MB", 1)
        if a.capacity:
            # closed form first: per class, saving per byte of capacity = HBM traffic / footprint
            prof = profile_classes(model_, hw_, rc_)
            print(f"baseline {prof['baseline_step_s'] * 1e3:.2f} ms · per class (HBM GB/step, "
                  f"footprint GB, value/byte):")
            for c in ("weight", "kv", "act"):
                print(f"  {c:7s} {prof['traffic_bytes'][c] / 1e9:8.2f} GB "
                      f"{prof['footprint_bytes'][c] / 1e9:9.2f} GB {prof['value_per_byte'][c]:8.3f}")
            cf = best_alloc(prof, a.capacity)
            cf_ms = run_model(model_, with_buffer(hw_, a.capacity, policy="pin", pin=cf["pin"], **kw),
                              rc_).step_time_s * 1e3
            print(f"closed form: fill {' > '.join(cf['order'])}, pin "
                  f"{{{', '.join(f'{k}:{v:.2f}' for k, v in cf['pin'].items())}}} -> {cf_ms:.2f} ms "
                  f"(minimises HBM bytes, not time)\n")
            best, rows = optimize_buffer(model_, hw_, rc_, a.capacity, steps=a.steps, **kw)
            print(f"{a.capacity:.0f} MB buffer ({a.capacity / l2:.1f}x L2) on {hw_.name}, "
                  f"{len(rows)} configurations")
            print(f"{'policy':6s} {'w':>4s} {'kv':>4s} {'act':>4s} {'byp':>4s} {'pre':>4s} "
                  f"{'step_ms':>8s} {'x':>6s}  top bound")
            for r in rows[:12]:
                print(f"{r['policy']:6s} {str(r['pin_weight']):>4s} {str(r['pin_kv']):>4s} "
                      f"{str(r['pin_act']):>4s} {int(r['bypass_l2']):>4d} {int(r['prefetch']):>4d} "
                      f"{r['step_ms']:8.2f} {r['speedup_vs_no_buffer']:6.3f}  {r['top_bound']}")
            out = rows_to_csv(rows)
        else:
            caps = [float(x) for x in a.capacities.split(",")]
            rows = capacity_curve(model_, hw_, rc_, caps, bypass_l2=True, prefetch=True, **kw)
            print(f"capacity curve on {hw_.name} (L2 = {l2} MB), bypass_l2 + prefetch on")
            print(f"{'MB':>8s} {'xL2':>6s} {'step_ms':>8s} {'speedup':>8s}  top bound")
            for r in rows:
                print(f"{r['capacity_MB']:8.0f} {r['x_L2']:6.2f} {r['step_ms']:8.2f} "
                      f"{r['speedup']:8.3f}  {r['top_bound']}")
            out = rows_to_csv(rows)
        if a.csv:
            open(a.csv, "w").write(out)
        return
    if a.cmd == "kernel":
        from .report.timeline import cycle_csv, machine_timeline, steady_timeline, timeline_text, trace_text
        from .server import _kernel_candidates
        hw = HardwareSpec.load(a.hw)
        if a.ideal:
            hw = hw.lossless()
        for s_ in a.set or []:
            k_, v_ = s_.split("=")
            hw = hw.override({k_: yaml.safe_load(v_)})
        spec = {"kernel": a.kernel, "tile": a.tile, "a_dtype": a.dtype, "b_dtype": a.dtype,
                "compute_dtype": a.dtype, **json.loads(a.shape)}
        cands = _kernel_candidates(hw, spec)
        b = cands[0]
        print(f"{hw.name} ({hw.sms} SMs) · {a.kernel} · {len(cands)} legal tiles")
        print(f"best {b['tile']}  {b['time_us']:.2f} us  {b['tflops']:.0f} TFLOP/s "
              f"({100 * b['peak_pct']:.1f}% of {a.dtype} peak)  HBM {b['ddr_GBps']:.0f} GB/s  "
              f"occ {b['occupancy']}  bound {b['bound']}")
        print(f"{'tile':24s} {'us':>9s} {'TF/s':>8s} {'%peak':>7s} {'occ':>14s}  bound")
        for c in cands[:a.top]:
            print(f"{c['tile']:24s} {c['time_us']:9.2f} {c['tflops']:8.0f} {100 * c['peak_pct']:6.1f}% "
                  f"{c['occupancy']:>14s}  {c['bound']}")
        print()
        addr_fn = None
        if a.kernel in ("gemm", "grouped_gemm"):
            from .hw.spec import DTYPE_BYTES as _DT
            from .kernels.tiles import TileConfig as _TC
            from .report.addressing import kernel_addr_fn
            _sh = json.loads(a.shape)
            _t = _TC(**json.loads(a.tile)) if a.tile != "auto" else \
                _TC(*[int(x) for x in b["tile"].split("/")[0].split("x")])
            addr_fn = kernel_addr_fn(hw, _sh.get("M", 1), _sh.get("N", 1), _sh.get("K", 1), _t,
                                     _DT[a.dtype], _DT[a.dtype], base=int(a.base, 0),
                                     layout_a=a.layout_a, layout_b=a.layout_b)
        sim = b["_ks"][0].meta.get("l2_sim")
        if sim is not None:
            print()
            print("L2 (deterministic tile-level simulation):")
            print("  " + sim.summary().replace("\n", "\n  "))
            res = sim.residency(top=6)
            if res:
                print("  still resident: " + ", ".join(f"p{p_}:tile{k_ & 0xffff}({sz // 1024}KB)"
                                                       for p_, k_, sz in res))
        l1m = b["_ks"][0].meta.get("l1_miss")
        if l1m:
            print(f"  L1 ({b['_ks'][0].meta['l1_capacity_KB']:.0f} KB usable, private per SM/cluster): "
                  f"hit {', '.join(f'{100 * (1 - m):.0f}%' for m in l1m)} per stream")
        tl = steady_timeline(b["_ks"][0], hw, addr_fn=addr_fn)
        print(timeline_text(tl, unit=a.unit))
        if a.trace_out:
            txt = trace_text(tl, f"{a.kernel} {a.shape} {a.dtype} tile={b['tile']}", hw.name,
                             {"time_us": f"{b['time_us']:.3f}", "tflops": f"{b['tflops']:.0f}",
                              "bound": b["bound"], "occupancy": b["occupancy"]})
            with open(a.trace_out, "w") as f:
                f.write(txt + "\n")
            print(f"\ntrace written to {a.trace_out} ({len(txt.splitlines())} lines)")
        if a.addr and a.kernel in ("gemm", "grouped_gemm"):
            from .hw.spec import DTYPE_BYTES
            from .kernels.tiles import TileConfig
            from .report.addressing import analyze_gemm
            sh = json.loads(a.shape)
            tcfg = TileConfig(**json.loads(a.tile)) if a.tile != "auto" else \
                TileConfig(*[int(x) for x in b["tile"].split("/")[0].split("x")])
            rep = analyze_gemm(hw, sh.get("M", 1), sh.get("N", 1), sh.get("K", 1), tcfg,
                               DTYPE_BYTES[a.dtype], DTYPE_BYTES[a.dtype],
                               layout_a=a.layout_a, layout_b=a.layout_b,
                               resident=int(b["occupancy"].split("/")[0]))
            print(f"\naddress spread (A={a.layout_a}, B={a.layout_b}, tile-granular):")
            print(rep["both"].text())
        m = machine_timeline(tl)
        print(f"\ngrid: {m['blocks']} blocks -> {len(m['waves'])} wave(s) of {m['sms']}x{m['resident']} "
              f"({m['wave_s'] * 1e6:.2f} us each); all SMs in a wave are identical in this model")
        if a.fig3 or a.fig3_json:
            from .report.figure3 import figure3_html, figure3_json
            title = f"{a.kernel} {a.shape} {a.dtype} · tile {b['tile']}"
            if a.fig3:
                with open(a.fig3, "w") as f:
                    f.write(figure3_html(b["_ks"][0], hw, title, addr_fn=addr_fn))
                print(f"Figure 3(d)(e)(f) written to {a.fig3}")
            if a.fig3_json:
                with open(a.fig3_json, "w") as f:
                    f.write(figure3_json(b["_ks"][0], hw))
                print(f"Figure 3 data written to {a.fig3_json}")
        if a.xlsx_out:
            from .report.excel import write_excel
            info = write_excel(tl, a.xlsx_out, f"{a.kernel} {a.shape} {a.dtype} tile={b['tile']}",
                               hw.name, full=a.csv_full,
                               extra={"time_us": f"{b['time_us']:.3f}", "bound": b["bound"],
                                      "occupancy": b["occupancy"], "tflops": f"{b['tflops']:.0f}"})
            print(f"Excel written to {a.xlsx_out} ({info['rows']} cycles"
                  f"{', truncated' if info['truncated'] else ''}, colour = iteration)")
        if a.csv_out:
            csv_txt = cycle_csv(tl, full=a.csv_full)
            with open(a.csv_out, "w") as f:
                f.write(csv_txt)
            print(f"per-cycle CSV written to {a.csv_out} ({len(csv_txt.splitlines()) - 1} cycles, "
                  f"{'whole kernel' if a.csv_full else 'drawn rounds'})")
        return
    model = ModelSpec.load(a.model)
    if a.cmd == "dump-model":
        print(model.dump_yaml())
        return
    hw, rc = _hw(a), _rc(a)
    if a.cmd == "request":
        from .model.request import request_summary, run_request
        print(request_summary(run_request(model, hw, rc)))
        return
    if a.cmd == "run":
        rep = run_model(model, hw, rc)
        print(model_summary(rep))
        if a.csv:
            open(a.csv, "w").write(ops_csv(rep))
    elif a.cmd == "sweep":
        vals = [float(v) for v in a.values.split(",")]
        rows = sweep(model, hw, rc, a.param, vals)
        out = rows_to_csv(rows)
        print(out)
        if a.csv:
            open(a.csv, "w").write(out)
    elif a.cmd == "need":
        v = required_value(model, hw, rc, a.param, a.target_ms, a.lo, a.hi)
        print(f"{a.param} needed for step <= {a.target_ms} ms: {v if v is None else round(v, 3)}")


if __name__ == "__main__":
    main()
