"""Local web server: your machine computes, others just open the page.

    python -m tilesight.cli serve --host 0.0.0.0 --port 8000

Stdlib only (http.server + json + threading), no external dependencies.
The page is served from tilesight/web/index.html and is fully self-contained
(no CDN, no fonts, no network access), so it also works opened from file:// if you
point it at a server URL.

API
  GET  /api/options            -> models, hardware presets, defaults
  POST /api/jobs               -> {mode, config}  =>  {job_id}
  GET  /api/jobs/<id>          -> {state, done, total, label, result|error}
  DELETE /api/jobs/<id>        -> cancel bookkeeping (a running job finishes on its own)

Every job runs in a worker thread and reports progress, so the UI can show a
"processing ..." state with the current op / sweep point instead of freezing.
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dse.sweep import required_value, sweep
from .engine import backend
from .hw.spec import DTYPE_BYTES
from .kernels.attention import lower_attention_decode, lower_attention_prefill
from .kernels.gemm import lower_elementwise, lower_gemm
from .kernels.tiles import AttnTileConfig, TileConfig, attn_search_space, gemm_search_space
from .hw.spec import DB_DIR, HardwareSpec
from .model.request import run_request
from .model.run_config import RunConfig
from .model.runner import CurModelConfig, run
from .model.spec import PRESET_DIR, ModelSpec
from .model.workload import WORKLOAD_FIELDS, W_SECTIONS, run_workload, validate_workload
from .hw.slice_config import SLICE_FIELDS, derive as slice_derive, to_hardware_spec, validate_slice_config
from .model.workload import compare_attention_impl, memory_breakdown
from .report.archdiagram import arch_svg
from .report.table import bound_report, model_summary, resource_class
from .report.timeline import cycle_csv, machine_timeline, steady_timeline, trace_text
from .model.request import request_summary

WEB_DIR = Path(__file__).parent / "web"
_KERNELS: dict = {}


def _top_kernels(rep, hw, n: int = 3):
    """Re-lower the heaviest ops so the PDF can show their timelines."""
    from .model.runner import resolve_op
    out = []
    for o in sorted(rep.ops, key=lambda o: -o.total_s):
        if o.op.kind not in ("gemm", "attn_decode", "attn_prefill"):
            continue
        try:
            from .kernels.attention import lower_attention_decode, lower_attention_prefill
            from .kernels.gemm import lower_gemm
            from .kernels.tiles import AttnTileConfig, TileConfig
            p = o.op.p
            tile_s = o.tile.split("/")[0]
            if o.op.kind == "gemm":
                bm, bn, bk = (int(x) for x in tile_s.split("x"))
                ks = lower_gemm(hw, o.op.name, p["M"], p["N"], p["K"], batch=p["batch"],
                                a_dtype=p["a_dtype"], b_dtype=p["b_dtype"], c_dtype=p["c_dtype"],
                                compute_dtype=p["compute_dtype"], tile=TileConfig(bm, bn, bk))
            else:
                fn = lower_attention_decode if o.op.kind == "attn_decode" else lower_attention_prefill
                ks = fn(hw, o.op.name, B=p["B"], H=p["H"], kv_heads=p["kv_heads"], S=p["S"],
                        d_qk=p["d_qk"], d_v=p["d_v"], tile=AttnTileConfig(),
                        **({"v_in_k": p["v_in_k"]} if o.op.kind == "attn_decode" else {}))
            if ks:
                out.append(ks[0])
        except Exception:                                # noqa: BLE001, S112
            continue
        if len(out) >= n:
            break
    return out
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


# ------------------------------------------------------------------ helpers
def _options() -> dict:
    models = sorted(p.stem.replace(".hf", "") + (".hf" if ".hf" in p.name else "")
                    for p in PRESET_DIR.glob("*") if p.suffix in (".json", ".yaml"))
    hws = sorted(p.stem for p in DB_DIR.glob("*.yaml"))
    rc = RunConfig()
    defaults = {f.name: getattr(rc, f.name) for f in fields(rc)
                if isinstance(getattr(rc, f.name), (int, float, str, bool))}
    return {"models": models, "hardware": hws, "defaults": defaults,
            "kernels": {
                "gemm": ["M", "N", "K", "batch", "a_dtype", "b_dtype", "compute_dtype"],
                "grouped_gemm": ["M", "N", "K", "experts", "a_dtype", "b_dtype", "compute_dtype"],
                "attn_decode": ["B", "H", "kv_heads", "S", "d_qk", "d_v", "kv_dtype", "compute_dtype", "v_in_k"],
                "attn_prefill": ["B", "H", "kv_heads", "S", "d_qk", "d_v", "kv_dtype", "compute_dtype", "causal"],
                "elementwise": ["bytes_in", "bytes_out", "flops", "sfu_ops"]},
            "sweep_params": ["memory.ddr.bandwidth_TBps", "memory.l2.bandwidth_TBps",
                             "memory.l2.per_sm_max_GBps", "memory.l2.effective_capacity_MB",
                             "memory.ddr.capacity_GB", "sms", "compute.tc_dense_tflops.fp8",
                             "compute.tc_dense_tflops.bf16", "compute.sfu_tops",
                             "network.nvlink.bandwidth_GBps", "runtime.launch_overhead_us"]}


def _load_model(cfg: dict) -> ModelSpec:
    text = (cfg.get("model_yaml") or "").strip()
    if text:
        import yaml
        raw = yaml.safe_load(text)
        return ModelSpec.from_dict(raw) if "layers" in raw else ModelSpec.from_hf_config(raw)
    return ModelSpec.load(cfg.get("model", "kimi_k2.hf"))


def _load_hw(cfg: dict) -> HardwareSpec:
    sl = cfg.get("slice_cfg")
    if sl:
        probs = validate_slice_config(sl)
        if probs:
            raise ValueError("gpu config: " + "; ".join(probs[:5]))
        return to_hardware_spec(sl)
    text = (cfg.get("hw_yaml") or "").strip()
    if text:
        import yaml as _yaml
        hw = HardwareSpec(_yaml.safe_load(text) or {})
        probs = hw.validate()
        if probs:
            raise ValueError("hardware config: " + "; ".join(probs[:5]))
    else:
        hw = HardwareSpec.load(cfg.get("hw", "b300"))
    ov = cfg.get("hw_overrides") or {}
    if isinstance(ov, str):
        import yaml
        ov = yaml.safe_load(ov) or {}
    ov = dict(ov)
    # tile policy is GPU-side (compute.tile_policy.*), but the UI still collects it next to the
    # rest of the run config, so pick it up from cfg["run"] here rather than in _run_config().
    run = cfg.get("run") or {}
    for k, path in (("gemm_tile", "compute.tile_policy.gemm"), ("attn_tile", "compute.tile_policy.attn"),
                    ("tile_overrides", "compute.tile_policy.overrides")):
        v = run.get(k)
        if isinstance(v, str) and v.strip() and v != "auto":
            v = json.loads(v)
        if v not in (None, "", {}):
            ov[path] = v
    return hw.override(ov) if ov else hw


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


# ------------------------------------------------------------------ single-GPU kernel mode
def _kernel_candidates(hw: HardwareSpec, k: dict, progress=None):
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
        tiles = [TileConfig(**json.loads(fixed))] if fixed != "auto" else gemm_search_space(M, N, K)
        names = ("act", "expert_weight" if kind == "grouped_gemm" else "weight", "out")
        flops = 2.0 * M * N * K * batch
        bytes_min = (M * K * DTYPE_BYTES[a] + K * N * DTYPE_BYTES[b] + M * N * 2) * batch
        for i, t in enumerate(tiles):
            if progress:
                progress(i, len(tiles), f"tile {t.short()}")
            ks = lower_gemm(hw, "k", M, N, K, batch=batch, a_dtype=a, b_dtype=b,
                            compute_dtype=comp, tile=t, names=names)
            if not ks:
                continue
            out.append(_cand(ks, hw, ks[0].meta.get("tile", t.short()), flops, bytes_min, comp))

    elif kind in ("attn_decode", "attn_prefill"):
        B, H, kvh = gi("B", 1), gi("H", 1), gi("kv_heads", 1)
        S, dq, dv = gi("S", 1), gi("d_qk", 128), gi("d_v", 128)
        kw = dict(B=B, H=H, kv_heads=max(1, kvh), S=S, d_qk=dq, d_v=dv,
                  kv_dtype=k.get("kv_dtype", "bf16"), compute_dtype=k.get("compute_dtype", "bf16"))
        tiles = [AttnTileConfig(**json.loads(fixed))] if fixed != "auto" else attn_search_space()
        kvb = DTYPE_BYTES[kw["kv_dtype"]]
        for i, t in enumerate(tiles):
            if progress:
                progress(i, len(tiles), f"tile {t.short()}")
            if kind == "attn_decode":
                ks = lower_attention_decode(hw, "k", v_in_k=bool(k.get("v_in_k", True)), tile=t, **kw)
                flops = 2.0 * B * H * S * (dq + dv)
                bytes_min = B * max(1, kvh) * S * (dq + (0 if k.get("v_in_k", True) else dv)) * kvb
            else:
                ks = lower_attention_prefill(hw, "k", causal=bool(k.get("causal", True)), tile=t, **kw)
                frac = 0.5 if k.get("causal", True) else 1.0
                flops = 2.0 * B * H * S * S * frac * (dq + dv)
                bytes_min = B * max(1, kvh) * S * (dq + dv) * kvb
            if not ks:
                continue
            out.append(_cand(ks, hw, ks[0].meta.get("tile", t.short()), flops, bytes_min,
                             kw["compute_dtype"]))

    elif kind == "elementwise":
        bi, bo = float(k.get("bytes_in", 0)), float(k.get("bytes_out", 0))
        ks = lower_elementwise(hw, "k", bytes_in=bi, bytes_out=bo,
                               flops=float(k.get("flops", 0)), sfu_ops=float(k.get("sfu_ops", 0)))
        out.append(_cand(ks, hw, "-", float(k.get("flops", 0)), bi + bo, "bf16"))
    else:
        raise ValueError(f"unknown kernel {kind}")

    out.sort(key=lambda c: c["time_us"])
    seen, uniq = set(), []                      # different configs can collapse to the same tile
    for c in out:
        if c["tile"] not in seen:
            seen.add(c["tile"])
            uniq.append(c)
    return uniq


def _cand(ks, hw, tile, flops, bytes_min, comp="bf16"):
    res = [backend.evaluate(x, hw) for x in ks]
    t = sum(r.time_s for r in res)
    m = ks[0].meta
    det: dict[str, float] = {}
    for r in res:
        for n, v in r.limiter_detail.items():
            det[n] = det.get(n, 0.0) + v
    ddr_peak = hw.get("memory.ddr.bandwidth_TBps") * 1e12
    tc_table = hw.get("compute.tc_dense_tflops")
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


# ------------------------------------------------------------------ job worker
def _work(job_id: str, mode: str, cfg: dict) -> None:
    import inspect

    def prog(done, total, label):
        # where we are, so a stuck run is visible: caller file:function + what it is chewing on
        try:
            fr = inspect.stack()[1]
            where = f"{fr.filename.rsplit('/', 1)[-1]}:{fr.function}"
        except Exception:                      # noqa: BLE001
            where = "?"
        line = f"{where} → {label}"
        with _LOCK:
            j = _JOBS[job_id]
            j.update(done=done, total=total, label=label, where=where)
            log = j.setdefault("log", [])
            if not log or log[-1] != line:
                log.append(line)
                del log[:-8]
    try:
        # The two configs the model actually takes. Whatever the request came from (a preset
        # pick, a custom hw/model YAML, or the "workload" one-layer shortcut below), it's built
        # here and only here — the model layer downstream never sees "workload" or raw request
        # JSON, only cur_gpu_config + cur_model_config.
        cur_gpu_config = _load_hw(cfg)
        cur_model_config = CurModelConfig(spec=_load_model(cfg), run=_run_config(cfg))
        model, rc = cur_model_config.spec, cur_model_config.run
        hw = cur_gpu_config  # alias: report/timeline helpers below take `hw` by convention
        if mode == "run":
            rep = run(cur_gpu_config, cur_model_config, progress=prog)
            out = _model_json(rep)
        elif mode == "request":
            r = run_request(model, cur_gpu_config, rc, progress=prog)
            out = {"summary": request_summary(r), "ttft_ms": r.ttft_s * 1e3,
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
        elif mode == "workload":
            wl = cfg.get("workload") or {}
            problems = validate_workload(wl)
            if problems:
                raise ValueError("; ".join(problems))
            rep = run_workload(wl, hw, progress=prog)
            out = _model_json(rep)
            out["workload"] = wl
            out["workload_memory"] = memory_breakdown(wl)
            out["by_block"] = {k: v / rep.step_time_s for k, v in rep.by_domain().items()}
            out["activity_by_block"] = rep.activity_by_domain()
            if cfg.get("compare_attention"):
                prog(1, 1, "comparing flash vs naive attention")
                out["attn_compare"] = compare_attention_impl(wl, hw)
            if cfg.get("slice_cfg"):
                out["derived"] = slice_derive(cfg["slice_cfg"])
            out["arch_svg"] = arch_svg(hw)
            out["hw_name"] = hw.name
            # attach the heaviest kernel's timeline so the Excel / CSV / PDF downloads work
            prog(1, 1, "building the trace of the heaviest kernel")
            ks = _top_kernels(rep, hw, n=1)
            if ks:
                tl = steady_timeline(ks[0], hw)
                out["timeline"] = tl
                out["trace_text"] = trace_text(tl, ks[0].name, hw.name)
                out["trace_kernel"] = ks[0].name
        elif mode == "kernel":
            cands = _kernel_candidates(hw, cfg.get("kernel") or {}, progress=prog)
            if not cands:
                raise ValueError("no legal tile for this shape (try smaller tiles / fewer stages)")
            tl = steady_timeline(cands[0]["_ks"][0], hw)      # Figure 3(e) view of the best tile
            k = cfg.get("kernel") or {}
            trace = trace_text(tl, f"{k.get('kernel', 'gemm')} {json.dumps({x: y for x, y in k.items() if x != 'tile'})}"
                               f" tile={cands[0]['tile']}", hw.name,
                               {"time_us": f"{cands[0]['time_us']:.3f}", "tflops": f"{cands[0]['tflops']:.0f}",
                                "bound": cands[0]["bound"], "occupancy": cands[0]["occupancy"]})
            cands = [{k2: v for k2, v in c.items() if k2 != "_ks"} for c in cands]
            out = {"hw": hw.name, "sms": hw.sms, "backend": backend.BACKEND, "timeline": tl,
                   "machine": machine_timeline(tl), "trace_text": trace,
                   "best": cands[0], "candidates": cands[:25], "tried": len(cands)}
        elif mode == "sweep":
            param = cfg["sweep"]["param"]
            values = [float(v) for v in str(cfg["sweep"]["values"]).replace(" ", "").split(",") if v]
            linked = None
            if cfg["sweep"].get("link_l2"):
                ratio = hw.get("memory.l2.bandwidth_TBps") / hw.get("memory.ddr.bandwidth_TBps")
                linked = (lambda v: {"memory.l2.bandwidth_TBps": v * ratio}) if "ddr" in param else None
            rows = sweep(model, hw, rc, param, values, linked=linked, progress=prog)
            out = {"param": param, "rows": rows}
        elif mode == "need":
            s = cfg["sweep"]
            v = required_value(model, hw, rc, s["param"], float(s["target_ms"]),
                               float(s.get("lo", 1)), float(s.get("hi", 64)))
            out = {"param": s["param"], "target_ms": float(s["target_ms"]), "value": v}
        else:
            raise ValueError(f"unknown mode {mode}")
        with _LOCK:
            _JOBS[job_id].update(state="done", result=out, done=_JOBS[job_id].get("total", 1))
    except Exception as e:                                  # noqa: BLE001 - surface to the UI
        with _LOCK:
            _JOBS[job_id].update(state="error", error=f"{type(e).__name__}: {e}",
                                 trace=traceback.format_exc()[-2000:])


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "tilesight"

    def log_message(self, fmt, *args):                      # quieter console
        if "/api/jobs/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode())

    def do_OPTIONS(self):                                   # noqa: N802
        self._send(204, b"")

    def do_GET(self):                                       # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/app", "/app.html"):
            return self._send(200, (WEB_DIR / "app.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/expert", "/index.html"):
            return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/xlsx":                     # colour-coded per-cycle Excel grid
            import tempfile
            from .report.excel import write_excel
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            res = (job or {}).get("result") or {}
            tl = res.get("timeline")
            if tl is None:
                return self._json(404, {"error": "no timeline for this job"})
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
                write_excel(tl, f.name, res.get("best", {}).get("tile", ""), res.get("hw", ""),
                            full=q.get("full") == "1")
                body = open(f.name, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", "attachment; filename=tilesight_cycles.xlsx")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return self.wfile.write(body)
        if path == "/api/csv":                      # per-cycle CSV of a finished kernel job
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            tl = ((job or {}).get("result") or {}).get("timeline")
            if tl is None:
                return self._json(404, {"error": "no timeline for this job"})
            body = cycle_csv(tl, full=q.get("full") == "1").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=tilesight_cycles.csv")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return self.wfile.write(body)
        if path == "/api/hw_yaml":
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            f = DB_DIR / f"{q.get('hw', 'b300').lower()}.yaml"
            if not f.exists():
                return self._json(404, {"error": "no such preset"})
            return self._send(200, f.read_bytes(), "text/plain; charset=utf-8")
        if path == "/api/slice_schema":
            return self._json(200, {"fields": [
                {"path": f.path, "kind": f.kind, "unit": f.unit, "section": f.section,
                 "doc": f.doc, "default": f.default} for f in SLICE_FIELDS]})
        if path == "/api/schema":
            from .hw.schema import FIELDS as HW_FIELDS
            return self._json(200, {
                "hardware": [{"path": f.path, "kind": f.kind, "unit": f.unit, "tag": f.tag,
                              "section": f.section, "doc": f.doc,
                              "default": None if f.default is None or not isinstance(
                                  f.default, (int, float, str, bool)) else f.default}
                             for f in HW_FIELDS],
                "workload": [{"path": f.path, "kind": f.kind, "unit": f.unit, "section": f.section,
                              "doc": f.doc, "default": f.default} for f in WORKLOAD_FIELDS],
                "workload_sections": list(W_SECTIONS)})
        if path == "/api/arch.svg":
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            try:
                hw = HardwareSpec.load(q.get("hw", "b300"))
            except Exception as e:                      # noqa: BLE001
                return self._json(400, {"error": str(e)})
            return self._send(200, arch_svg(hw).encode(), "image/svg+xml")
        if path == "/api/pdf":
            import tempfile
            from .report.pdfreport import write_pdf
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            res = (job or {}).get("result") or {}
            cfg = (job or {}).get("config") or {}
            try:
                hw = _load_hw(cfg)
                wl = cfg.get("workload") or {}
                rep = run_workload(wl, hw) if wl else None
                kernels = []
                if rep is not None:
                    for o in sorted(rep.ops, key=lambda o: -o.total_s)[:3]:
                        kernels += [k for k in _KERNELS.get(id(o), [])] or []
                from .model.runner import resolve_op  # noqa: F401
                if rep is not None and not kernels:
                    kernels = _top_kernels(rep, hw)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    write_pdf(f.name, hw, kernels, res.get("title", "TileSight report"), wl, rep)
                    body = open(f.name, "rb").read()
            except Exception as e:                      # noqa: BLE001
                return self._json(400, {"error": f"{type(e).__name__}: {e}"})
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", "attachment; filename=tilesight_report.pdf")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return self.wfile.write(body)
        if path == "/api/options":
            return self._json(200, _options())
        if path.startswith("/api/jobs/"):
            jid = path.rsplit("/", 1)[-1]
            with _LOCK:
                job = _JOBS.get(jid)
            if job is None:
                return self._json(404, {"error": "no such job"})
            return self._json(200, job)
        self._json(404, {"error": "not found"})

    def do_POST(self):                                      # noqa: N802
        if self.path.split("?")[0] == "/api/derive":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            sl = body.get("slice_cfg") or {}
            probs = validate_slice_config(sl)
            out = {"problems": probs}
            if not probs:
                out["derived"] = slice_derive(sl)
                out["arch_svg"] = arch_svg(to_hardware_spec(sl))
            wl = body.get("workload")
            if wl:
                out["memory"] = memory_breakdown(wl)
            return self._json(200, out)
        if self.path.split("?")[0] == "/api/validate_hw":
            import yaml as _yaml
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                raw = _yaml.safe_load(body.get("yaml") or "") or {}
            except Exception as e:                          # noqa: BLE001
                return self._json(200, {"problems": [f"YAML error: {e}"], "fields": 0})
            from .hw.schema import validate as _v
            n_set = sum(1 for _ in json.dumps(raw))
            return self._json(200, {"problems": _v(raw), "fields": len(json.dumps(raw).split(","))})
        if self.path.split("?")[0] != "/api/jobs":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            return self._json(400, {"error": str(e)})
        jid = uuid.uuid4().hex[:12]
        with _LOCK:
            _JOBS[jid] = {"state": "running", "done": 0, "total": 1, "label": "starting"}
            if len(_JOBS) > 200:                            # keep the table bounded
                for k in list(_JOBS)[:50]:
                    if _JOBS[k]["state"] != "running":
                        _JOBS.pop(k, None)
        with _LOCK:
            _JOBS[jid]["config"] = body.get("config", {})
        threading.Thread(target=_work, args=(jid, body.get("mode", "run"), body.get("config", {})),
                         daemon=True).start()
        self._json(200, {"job_id": jid})

    def do_DELETE(self):                                    # noqa: N802
        jid = self.path.rsplit("/", 1)[-1]
        with _LOCK:
            _JOBS.pop(jid, None)
        self._json(200, {"ok": True})


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    from .engine.backend import BACKEND
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"tilesight web UI on http://{host}:{port}  (engine backend: {BACKEND})")
    print("computation runs here; clients only need a browser")
    httpd.serve_forever()
