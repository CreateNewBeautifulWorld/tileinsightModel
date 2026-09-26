"""Build the JSON a results page needs, for one job — the part of `cli/server.py`'s job
handling that has nothing to do with HTTP or progress bookkeeping.

Pulled out of `server.py` so two callers can share it and stay byte-for-byte in agreement on
what a result looks like: `server.py`'s `_work()` (an ephemeral wizard run, tracked in memory
only) and `gpuTilingPerfHWModel/regr/worker.py` (a persisted regression job, tracked in
`gpuTilingPerfHWModel/regr/`). Both eventually just call `build_report_json(mode, cfg, prog)`.
"""
from __future__ import annotations

import json
from dataclasses import fields

from tilesight import _core
from tilesight.gpuTilingPerfHWModel.model.dse.sweep import required_value, sweep
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import DTYPE_BYTES
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import DB_DIR, HardwareSpec
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.request import request_summary, run_request
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.runner import CurModelConfig, run
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.model_spec import ModelSpec
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import run_workload, validate_workload
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.slice_config import derive as slice_derive, to_hardware_spec, validate_slice_config
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import compare_attention_impl, memory_breakdown
from tilesight.gpuTilingPerfHWModel.genResult.archdiagram import arch_svg
from tilesight.gpuTilingPerfHWModel.genResult.table import bound_report, model_summary, resource_class
from tilesight.gpuTilingPerfHWModel.genResult.timeline import machine_timeline, steady_timeline, trace_text


# ------------------------------------------------------------------ config -> the model's own inputs
def _load_model(cfg: dict) -> ModelSpec:
    text = (cfg.get("model_yaml") or "").strip()
    if text:
        import yaml
        raw = yaml.safe_load(text)
        return ModelSpec.from_dict(raw) if "layers" in raw else ModelSpec.from_hf_config(raw)
    return ModelSpec.load(cfg.get("model", "kimi_k2.hf"))


def _load_hw(cfg: dict) -> HardwareSpec:
    # the results page's "cache simulation window" slider (memory.l2.sim_max_accesses; 0 = no cap)
    sim = {"memory.l2.sim_max_accesses": float(cfg["sim_max_accesses"])} \
        if cfg.get("sim_max_accesses") is not None else {}
    sl = cfg.get("slice_cfg")
    if sl:
        probs = validate_slice_config(sl)
        if probs:
            raise ValueError("gpu config: " + "; ".join(probs[:5]))
        hw = to_hardware_spec(sl)
        return hw.override(sim) if sim else hw
    text = (cfg.get("gpuTilingPerfHWModelYaml") or "").strip()
    if text:
        import yaml as _yaml
        cur_gpu_config = HardwareSpec(_yaml.safe_load(text) or {})
        probs = cur_gpu_config.validate()
        if probs:
            raise ValueError("hardware config: " + "; ".join(probs[:5]))
    else:
        cur_gpu_config = HardwareSpec.load(cfg.get("gpuTilingPerfHWModel", "b300"))
    ov = cfg.get("gpuTilingPerfHWModelOverrides") or {}
    if isinstance(ov, str):
        import yaml
        ov = yaml.safe_load(ov) or {}
    ov = dict(ov)
    # tile policy is GPU-side (compute.tile_policy.*), but the UI still collects it next to the
    # rest of the run config, so pick it up from cfg["run"] here rather than in _run_config().
    run_cfg = cfg.get("run") or {}
    for k, path in (("gemm_tile", "compute.tile_policy.gemm"), ("attn_tile", "compute.tile_policy.attn"),
                    ("tile_overrides", "compute.tile_policy.overrides")):
        v = run_cfg.get(k)
        if isinstance(v, str) and v.strip() and v != "auto":
            v = json.loads(v)
        if v not in (None, "", {}):
            ov[path] = v
    ov.update(sim)
    return cur_gpu_config.override(ov) if ov else cur_gpu_config


def _run_config(cfg: dict) -> RunConfig:
    names = {f.name for f in fields(RunConfig)}
    kw = {k: v for k, v in (cfg.get("run") or {}).items() if k in names and v not in ("", None)}
    for k in ("batch", "seq_len", "tp", "dp", "ep", "prompt_len", "output_len", "page_size",
              "decode_samples", "prefill_batch"):
        if k in kw:
            kw[k] = int(kw[k])
    for k in ("comm_overlap",):
        if k in kw:
            kw[k] = float(kw[k])
    return RunConfig(**kw)


# ------------------------------------------------------------------ report -> JSON
def _top_kernels(rep, cur_gpu_config, n: int = 3):
    """Re-lower the heaviest ops so the PDF / trace views can show their timelines."""
    out = []
    for o in sorted(rep.ops, key=lambda o: -o.total_s):
        if o.op.kind not in ("gemm", "attn_decode", "attn_prefill"):
            continue
        try:
            p = o.op.p
            tile_s = o.tile.split("/")[0]
            if o.op.kind == "gemm":
                bm, bn, bk = (int(x) for x in tile_s.split("x"))
                ks = _core.lower_gemm(cur_gpu_config, o.op.name, p["M"], p["N"], p["K"], batch=p["batch"],
                                      a_dtype=p["a_dtype"], b_dtype=p["b_dtype"], c_dtype=p["c_dtype"],
                                      compute_dtype=p["compute_dtype"], tile=_core.GemmTile(bm, bn, bk))
            else:
                fn = _core.lower_attention_decode if o.op.kind == "attn_decode" else _core.lower_attention_prefill
                ks = fn(cur_gpu_config, o.op.name, B=p["B"], H=p["H"], kv_heads=p["kv_heads"], S=p["S"],
                        d_qk=p["d_qk"], d_v=p["d_v"], tile=_core.AttnTile(),
                        **({"v_in_k": p["v_in_k"]} if o.op.kind == "attn_decode" else {}))
            if ks:
                out.append(ks[0])
        except Exception:                                # noqa: BLE001, S112
            continue
        if len(out) >= n:
            break
    return out


def _ops_json(rep) -> list[dict]:
    tot = rep.step_time_s
    return [dict(name=o.op.name, kind=o.op.kind, repeat=o.repeat, tile=o.tile,
                 time_us=o.time_s * 1e6, total_us=o.total_s * 1e6, share=o.total_s / tot,
                 occupancy=o.occupancy, bound=o.bottleneck_detail,
                 util={k: round(v, 4) for k, v in sorted(o.kernels[0].util.items(),
                                                         key=lambda kv: -kv[1])[:4]} if o.kernels else {})
            for o in sorted(rep.ops, key=lambda o: -o.total_s)]


def _bounds_json(rep) -> dict:
    tot = rep.step_time_s
    det = rep.detail_breakdown()
    cls: dict[str, float] = {}
    for k, v in det.items():
        cls[resource_class(k)] = cls.get(resource_class(k), 0.0) + v
    return {"by_resource": [{"name": k, "share": v / tot} for k, v in sorted(cls.items(), key=lambda kv: -kv[1])],
            "by_tensor": [{"name": k, "share": v / tot} for k, v in list(det.items())[:15]]}


def _model_json(rep) -> dict:
    m = rep.memory
    return {"summary": model_summary(rep, top=40), "bound_text": bound_report(rep),
            "step_ms": rep.step_time_s * 1e3, "tok_s_gpu": rep.tokens_per_s_per_gpu,
            "backend": rep.backend,
            "memory": {"weights_GB": m.weights_GB, "kv_GB": m.kv_cache_GB, "act_GB": m.activations_GB,
                       "reserve_GB": m.reserve_GB, "total_GB": m.total_GB, "capacity_GB": m.capacity_GB,
                       "fits": m.fits, "max_seqs": m.max_seqs_per_rank(rep.rc.seq_len)},
            "bounds": _bounds_json(rep), "ops": _ops_json(rep)}


def _sim_window_json(rep, cur_gpu_config) -> dict:
    """How much of each kernel the cache simulation replayed (meta.sim_coverage), time-weighted."""
    cov = [(k.meta.get("sim_coverage"), o.total_s, o.op.name) for o in rep.ops for k in o.kernels
           if k.meta.get("sim_coverage") is not None]
    if not cov:
        return {}
    tot = sum(t for _, t, _ in cov) or 1.0
    worst = min(cov, key=lambda c: c[0])
    return {"budget": cur_gpu_config.get("memory.l2.sim_max_accesses"),
            "weighted": sum(c * t for c, t, _ in cov) / tot,
            "min": worst[0], "min_op": worst[2],
            "full": sum(1 for c, _, _ in cov if c >= 0.999), "kernels": len(cov)}


def _memmap_json(model_spec, rc, cur_gpu_config, prog) -> dict:
    """Where every tensor sits in HBM (contiguous, 2 MB-aligned regions from
    memory.addressing.base) and how it spreads over the HBM ports; also written to out/memmap
    as CSV + Excel, which /api/memmap serves."""
    from tilesight.gpuTilingPerfHWModel.genResult.memmap_report import memmap_json, write_all
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.memmap import build_memory_map
    prog(1, 1, "allocating the HBM memory map")
    mm = build_memory_map(model_spec, rc, base=int(cur_gpu_config.get("memory.addressing.base")))
    out = memmap_json(mm, cur_gpu_config)
    stem = f"memmap_{getattr(model_spec, 'name', 'model')}_{cur_gpu_config.name}_{rc.phase}".replace(" ", "_")
    out["files"] = {k: str(v) for k, v in write_all(mm, cur_gpu_config, None, stem,
                                                     f"{getattr(model_spec, 'name', '')} on {cur_gpu_config.name}, {rc.phase}").items()}
    return out


def _workload_report_json(rep, wl: dict, cur_gpu_config, cfg: dict, prog) -> dict:
    """Everything the results page shows for one phase's report — shared by the single-phase
    "workload" mode and the prefill+decode "workload_both" mode (one call per phase)."""
    out = _model_json(rep)
    out["workload"] = wl
    out["workload_memory"] = memory_breakdown(wl, cur_gpu_config)
    out["by_block"] = {k: v / rep.step_time_s for k, v in rep.by_domain().items()}
    out["activity_by_block"] = rep.activity_by_domain()
    if cfg.get("compare_attention"):
        prog(1, 1, "comparing flash vs naive attention")
        out["attn_compare"] = compare_attention_impl(wl, cur_gpu_config)
    if cfg.get("slice_cfg"):
        out["derived"] = slice_derive(cfg["slice_cfg"])
    out["arch_svg"] = arch_svg(cur_gpu_config)
    out["gpu_name"] = cur_gpu_config.name
    out["sim_window"] = _sim_window_json(rep, cur_gpu_config)
    try:
        from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import to_model_spec
        out["memmap"] = _memmap_json(to_model_spec(wl), rep.rc, cur_gpu_config, prog)
    except Exception as e:                              # never lose the run over the map
        out["memmap"] = {"error": str(e)}
    # attach the heaviest kernel's timeline so the Excel / CSV / PDF downloads AND the inline
    # Gantt view on the results page work
    prog(1, 1, "building the trace of the heaviest kernel")
    ks = _top_kernels(rep, cur_gpu_config, n=1)
    if ks:
        tl = steady_timeline(ks[0], cur_gpu_config)
        out["timeline"] = tl
        out["trace_text"] = trace_text(tl, ks[0].name, cur_gpu_config.name)
        out["trace_kernel"] = ks[0].name
    return out


# ------------------------------------------------------------------ single-GPU kernel mode
def _cand(ks, cur_gpu_config, tile, flops, bytes_min, comp="bf16"):
    res = [_core.evaluate(cur_gpu_config, x) for x in ks]
    t = sum(r.time_s for r in res)
    m = ks[0].meta
    det: dict[str, float] = {}
    for r in res:
        for n, v in r.limiter_detail.items():
            det[n] = det.get(n, 0.0) + v
    ddr_peak = cur_gpu_config.get("memory.ddr.bandwidth_TBps") * 1e12
    tc_table = cur_gpu_config.get("compute.tc_dense_tflops")
    tc_peak = tc_table.get(comp, tc_table.get("bf16")) * 1e12
    return {"_ks": ks, "tile": tile, "time_us": t * 1e6,
            "tflops": flops / t / 1e12 if t else 0.0,
            "ddr_GBps": res[0].util.get("ddr", 0.0) * ddr_peak / 1e9,
            "ideal_us": max(flops / tc_peak, bytes_min / ddr_peak) * 1e6,
            "peak_pct": (flops / tc_peak) / (sum(r.time_s for r in res) or 1),
            "bound": max(det.items(), key=lambda kv: kv[1])[0] if det else "-",
            "occupancy": f"{m.get('resident', '?')}/{m.get('occ_limiter', '?')}",
            "stages": m.get("stages", "-"), "regs": m.get("regs_per_thread", "-"),
            "l2_miss": round(m.get("l2_miss_B", m.get("l2_miss_KV", 0.0)), 3),
            "kernels": len(ks),
            "util": {k2: round(v, 3) for k2, v in sorted(res[0].util.items(), key=lambda kv: -kv[1])[:4]}}


def _kernel_candidates(cur_gpu_config: HardwareSpec, k: dict, progress=None):
    """Evaluate one kernel on ONE GPU over a tile search space; return ranked candidates."""
    kind = k.get("kernel", "gemm")
    fixed = (k.get("tile") or "auto").strip()
    gi = lambda n, d=0: int(float(k.get(n, d) or d))          # noqa: E731
    out = []

    if kind in ("gemm", "grouped_gemm"):
        M, N, K = gi("M", 1), gi("N", 1), gi("K", 1)
        batch = gi("experts", 1) if kind == "grouped_gemm" else gi("batch", 1)
        a, b = k.get("a_dtype", "fp8"), k.get("b_dtype", "fp8")
        comp = k.get("compute_dtype", "fp8")
        tiles = [_core.GemmTile(**json.loads(fixed))] if fixed != "auto" else _core.gemm_search_space(M, N, K)
        names = ("act", "expert_weight" if kind == "grouped_gemm" else "weight", "out")
        flops = 2.0 * M * N * K * batch
        bytes_min = (M * K * DTYPE_BYTES[a] + K * N * DTYPE_BYTES[b] + M * N * 2) * batch
        for i, t in enumerate(tiles):
            if progress:
                progress(i, len(tiles), f"tile {t.bm}x{t.bn}x{t.bk}")
            ks = _core.lower_gemm(cur_gpu_config, "k", M, N, K, batch=batch, a_dtype=a, b_dtype=b,
                                  compute_dtype=comp, tile=t, a_name=names[0], b_name=names[1], c_name=names[2])
            if not ks:
                continue
            out.append(_cand(ks, cur_gpu_config, ks[0].meta.get("tile", "-"), flops, bytes_min, comp))

    elif kind in ("attn_decode", "attn_prefill"):
        B, H, kvh = gi("B", 1), gi("H", 1), gi("kv_heads", 1)
        S, dq, dv = gi("S", 1), gi("d_qk", 128), gi("d_v", 128)
        kw = dict(B=B, H=H, kv_heads=max(1, kvh), S=S, d_qk=dq, d_v=dv,
                  kv_dtype=k.get("kv_dtype", "bf16"), compute_dtype=k.get("compute_dtype", "bf16"))
        tiles = [_core.AttnTile(**json.loads(fixed))] if fixed != "auto" else _core.attn_search_space()
        kvb = DTYPE_BYTES[kw["kv_dtype"]]
        for i, t in enumerate(tiles):
            if progress:
                progress(i, len(tiles), f"tile {t.block_m}x{t.block_n}")
            if kind == "attn_decode":
                ks = _core.lower_attention_decode(cur_gpu_config, "k", v_in_k=bool(k.get("v_in_k", True)), tile=t, **kw)
                flops = 2.0 * B * H * S * (dq + dv)
                bytes_min = B * max(1, kvh) * S * (dq + (0 if k.get("v_in_k", True) else dv)) * kvb
            else:
                ks = _core.lower_attention_prefill(cur_gpu_config, "k", causal=bool(k.get("causal", True)), tile=t, **kw)
                frac = 0.5 if k.get("causal", True) else 1.0
                flops = 2.0 * B * H * S * S * frac * (dq + dv)
                bytes_min = B * max(1, kvh) * S * (dq + dv) * kvb
            if not ks:
                continue
            out.append(_cand(ks, cur_gpu_config, ks[0].meta.get("tile", "-"), flops, bytes_min,
                             kw["compute_dtype"]))

    elif kind == "elementwise":
        bi, bo = float(k.get("bytes_in", 0)), float(k.get("bytes_out", 0))
        ks = _core.lower_elementwise(cur_gpu_config, "k", bytes_in=bi, bytes_out=bo,
                                     flops=float(k.get("flops", 0)), sfu_ops=float(k.get("sfu_ops", 0)))
        out.append(_cand(ks, cur_gpu_config, "-", float(k.get("flops", 0)), bi + bo, "bf16"))
    else:
        raise ValueError(f"unknown kernel {kind}")

    out.sort(key=lambda c: c["time_us"])
    seen, uniq = set(), []                      # different configs can collapse to the same tile
    for c in out:
        if c["tile"] not in seen:
            seen.add(c["tile"])
            uniq.append(c)
    return uniq


# ------------------------------------------------------------------ the one entry point
def build_report_json(mode: str, cfg: dict, prog) -> dict:
    """Runs the model for one job and returns exactly what a results page needs. `prog(done,
    total, label)` is the only side channel — the caller decides what to do with progress (an
    in-memory job dict for the ephemeral wizard, a persisted job record for a regression job)."""
    cur_gpu_config = _load_hw(cfg)
    cur_model_config = CurModelConfig(spec=_load_model(cfg), run=_run_config(cfg))
    model, rc = cur_model_config.spec, cur_model_config.run
    if mode == "run":
        rep = run(cur_gpu_config, cur_model_config, progress=prog)
        out = _model_json(rep)
        out["memmap"] = _memmap_json(model, rc, cur_gpu_config, prog)
        return out
    if mode == "request":
        r = run_request(model, cur_gpu_config, rc, progress=prog)
        return {"summary": request_summary(r), "ttft_ms": r.ttft_s * 1e3,
               "tpot_first_ms": r.points[0].tpot_s * 1e3, "tpot_last_ms": r.points[-1].tpot_s * 1e3,
               "tpot_avg_ms": r.tpot_avg_s * 1e3, "e2e_s": r.e2e_s,
               "peak_total_GB": r.peak_total_GB, "peak_kv_GB": r.peak_kv_GB,
               "capacity_GB": r.capacity_GB, "fits": r.fits,
               "max_requests": r.max_requests_per_rank,
               "points": [{"step": p.step, "kv_len": p.kv_len, "tpot_ms": p.tpot_s * 1e3,
                           "kv_GB": p.kv_GB, "total_GB": p.total_GB, "bound": p.top_bound}
                          for p in r.points],
               "bounds_last": _bounds_json(r.points[-1].report),
               "bounds_prefill": _bounds_json(r.prefill), "ops": _ops_json(r.points[-1].report)}
    if mode == "workload":
        wl = cfg.get("workload") or {}
        problems = validate_workload(wl)
        if problems:
            raise ValueError("; ".join(problems))
        rep = run_workload(wl, cur_gpu_config, progress=prog)
        return _workload_report_json(rep, wl, cur_gpu_config, cfg, prog)
    if mode == "workload_both":
        # prefill + decode together, one job, so the results page can show both without a
        # second run — the "single token generation step" decode needs to sit right next to
        # the prompt-processing prefill step to compare.
        wl = cfg.get("workload") or {}
        problems = validate_workload(wl)
        if problems:
            raise ValueError("; ".join(problems))
        from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import run_workload_both_phases
        reps = run_workload_both_phases(wl, cur_gpu_config, progress=prog)
        return {phase: _workload_report_json(rep, wl, cur_gpu_config, cfg, prog)
               for phase, rep in reps.items()}
    if mode == "kernel":
        cands = _kernel_candidates(cur_gpu_config, cfg.get("kernel") or {}, progress=prog)
        if not cands:
            raise ValueError("no legal tile for this shape (try smaller tiles / fewer stages)")
        tl = steady_timeline(cands[0]["_ks"][0], cur_gpu_config)      # Figure 3(e) view of the best tile
        k = cfg.get("kernel") or {}
        trace = trace_text(tl, f"{k.get('kernel', 'gemm')} {json.dumps({x: y for x, y in k.items() if x != 'tile'})}"
                           f" tile={cands[0]['tile']}", cur_gpu_config.name,
                           {"time_us": f"{cands[0]['time_us']:.3f}", "tflops": f"{cands[0]['tflops']:.0f}",
                            "bound": cands[0]["bound"], "occupancy": cands[0]["occupancy"]})
        cands = [{k2: v for k2, v in c.items() if k2 != "_ks"} for c in cands]
        return {"gpu_name": cur_gpu_config.name, "sms": cur_gpu_config.sms, "backend": "cpp", "timeline": tl,
               "machine": machine_timeline(tl), "trace_text": trace,
               "best": cands[0], "candidates": cands[:25], "tried": len(cands)}
    if mode == "sweep":
        param = cfg["sweep"]["param"]
        values = [float(v) for v in str(cfg["sweep"]["values"]).replace(" ", "").split(",") if v]
        linked = None
        if cfg["sweep"].get("link_l2"):
            ratio = cur_gpu_config.get("memory.l2.bandwidth_TBps") / cur_gpu_config.get("memory.ddr.bandwidth_TBps")
            linked = (lambda v: {"memory.l2.bandwidth_TBps": v * ratio}) if "ddr" in param else None
        rows = sweep(model, cur_gpu_config, rc, param, values, linked=linked, progress=prog)
        return {"param": param, "rows": rows}
    if mode == "need":
        s = cfg["sweep"]
        v = required_value(model, cur_gpu_config, rc, s["param"], float(s["target_ms"]),
                           float(s.get("lo", 1)), float(s.get("hi", 64)))
        return {"param": s["param"], "target_ms": float(s["target_ms"]), "value": v}
    raise ValueError(f"unknown mode {mode}")
