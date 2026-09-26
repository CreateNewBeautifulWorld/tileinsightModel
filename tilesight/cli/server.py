"""Local web server: your machine computes, others just open the page.

    python -m tilesight.cli.cli serve --host 0.0.0.0 --port 8000

Stdlib only (http.server + json + threading), no external dependencies.
The page is served from tilesight/web/index.html and is fully self-contained
(no CDN, no fonts, no network access), so it also works opened from file:// if you
point it at a server URL.

API
  GET  /api/options            -> models, hardware presets, defaults
  POST /api/jobs               -> {mode, config}  =>  {job_id}
  GET  /api/jobs/<id>          -> {state, done, total, label, result|error}
  DELETE /api/jobs/<id>        -> cancel bookkeeping (a running job finishes on its own)

  GET  /api/regr/jobs          -> the persisted regression-job dashboard's list (id/name/tag/
                                  status/submitted_by/timestamps), capped at 100, oldest pruned
  POST /api/regr/jobs          -> {name, tag, mode, config}  =>  {job_id}; queued, run one at a
                                  time by gpuTilingPerfHWModel/regr/worker.py
  GET  /api/regr/jobs/<id>     -> that job's metadata + config + result (once done)
  DELETE /api/regr/jobs/<id>   -> remove a job and its output directory
  GET  /api/regr/file          -> ?job=<id>&phase=<p>&name=<file> — an artifact already written
                                  into that job's own out directory

Every job runs in a worker thread and reports progress, so the UI can show a
"processing ..." state with the current op / sweep point instead of freezing. A regression job
instead runs one at a time in its own background worker (see gpuTilingPerfHWModel/regr/), so a
long, wide-search-window regression run never competes with someone using the interactive wizard.
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tilesight import _core
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import DB_DIR, HardwareSpec
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.model_spec import PRESET_DIR, ModelSpec
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import WORKLOAD_FIELDS, W_SECTIONS, run_workload, validate_workload
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.slice_config import SLICE_FIELDS, derive as slice_derive, to_hardware_spec, validate_slice_config
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import memory_breakdown
from tilesight.gpuTilingPerfHWModel.genResult.archdiagram import arch_svg
from tilesight.gpuTilingPerfHWModel.genResult.timeline import cycle_csv, steady_timeline, trace_text
from tilesight.gpuTilingPerfHWModel.regr.store import store as regr_store
from tilesight.cli.report_json import _kernel_candidates, _load_hw, _memmap_json, _top_kernels, build_report_json

WEB_DIR = Path(__file__).parent.parent / "html"
_KERNELS: dict = {}
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


# ------------------------------------------------------------------ job worker
def _work(job_id: str, mode: str, cfg: dict) -> None:
    """Run one ephemeral wizard job and report progress into the in-memory `_JOBS` table (gone
    on restart). `gpuTilingPerfHWModel/regr/worker.py` runs a persisted regression job the same
    way, through the same `build_report_json()` — this is only the in-memory bookkeeping."""
    import inspect

    def prog(done, total, label):
        # where we are, so a stuck run is visible: caller file:function + what it is chewing on
        try:
            fr = inspect.stack()[1]
            where = f"{fr.filename.rsplit('/', 1)[-1]}:{fr.function}"
        except Exception:                      # noqa: BLE001
            where = "?"
        line = f"{where} \u2192 {label}"
        with _LOCK:
            j = _JOBS[job_id]
            j.update(done=done, total=total, label=label, where=where)
            log = j.setdefault("log", [])
            if not log or log[-1] != line:
                log.append(line)
                del log[:-8]
    try:
        out = build_report_json(mode, cfg, prog)
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
        if path == "/api/build_status":          # cli/server_boot.py swaps this handler in once
            return self._json(200, {"stage": "ready", "message": ""})   # ready by construction
        if path in ("/", "/app", "/app.html"):
            return self._send(200, (WEB_DIR / "app.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/expert", "/index.html"):
            return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/regr", "/regr.html"):                 # page 1: the regression job dashboard
            return self._send(200, (WEB_DIR / "regr.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/regr/new", "/regr_new.html"):         # page 2: create a regression job (its own page)
            return self._send(200, (WEB_DIR / "regr_new.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/regr/result", "/regr_result.html"):   # page 3: one regression job's results
            return self._send(200, (WEB_DIR / "regr_result.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/results.js":                # results-rendering JS shared by app.html + regr_result.html
            return self._send(200, (WEB_DIR / "results.js").read_bytes(), "application/javascript; charset=utf-8")
        if path == "/api/xlsx":                     # colour-coded per-cycle Excel grid
            import tempfile
            from tilesight.gpuTilingPerfHWModel.genResult.excel import write_excel
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            res = (job or {}).get("result") or {}
            if (job or {}).get("mode") == "workload_both":
                res = res.get(q.get("phase", "decode"), {})
            tl = res.get("timeline")
            if tl is None:
                return self._json(404, {"error": "no timeline for this job"})
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
                write_excel(tl, f.name, res.get("best", {}).get("tile", ""), res.get("gpu_name", ""),
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
        if path == "/api/memmap":                   # the job's HBM address map, as xlsx or csv
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            res = (job or {}).get("result") or {}
            if (job or {}).get("mode") == "workload_both":
                res = res.get(q.get("phase", "decode"), {})
            fmt = "csv" if q.get("fmt") == "csv" else "xlsx"
            f = ((res.get("memmap") or {}).get("files") or {}).get(fmt)
            if not f or not Path(f).exists():
                return self._json(404, {"error": "no memory map for this job"})
            body = Path(f).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv" if fmt == "csv" else
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", f"attachment; filename={Path(f).name}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return self.wfile.write(body)
        if path == "/api/csv":                      # per-cycle CSV of a finished kernel job
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            res = (job or {}).get("result") or {}
            if (job or {}).get("mode") == "workload_both":
                res = res.get(q.get("phase", "decode"), {})
            tl = res.get("timeline")
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
        if path == "/api/gpu_tiling_perf_hw_model_yaml":
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            f = DB_DIR / f"{q.get('gpuTilingPerfHWModel', 'b300').lower()}.yaml"
            if not f.exists():
                return self._json(404, {"error": "no such preset"})
            return self._send(200, f.read_bytes(), "text/plain; charset=utf-8")
        if path == "/api/slice_schema":
            return self._json(200, {"fields": [
                {"path": f.path, "kind": f.kind, "unit": f.unit, "section": f.section,
                 "doc": f.doc, "default": f.default} for f in SLICE_FIELDS]})
        if path == "/api/schema":
            from tilesight.gpuTilingPerfHWModel.interfaceAndRun.schema import FIELDS as HW_FIELDS
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
                cur_gpu_config = HardwareSpec.load(q.get("gpuTilingPerfHWModel", "b300"))
            except Exception as e:                      # noqa: BLE001
                return self._json(400, {"error": str(e)})
            return self._send(200, arch_svg(cur_gpu_config).encode(), "image/svg+xml")
        if path == "/api/pdf":
            import tempfile
            from tilesight.gpuTilingPerfHWModel.genResult.pdfreport import write_pdf
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            with _LOCK:
                job = _JOBS.get(q.get("job", ""))
            res = (job or {}).get("result") or {}
            cfg = (job or {}).get("config") or {}
            phase = q.get("phase", "decode")
            if (job or {}).get("mode") == "workload_both":
                res = res.get(phase, {})
            try:
                cur_gpu_config = _load_hw(cfg)
                wl = cfg.get("workload") or {}
                if wl and (job or {}).get("mode") == "workload_both":
                    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.workload import run_workload_both_phases
                    rep = run_workload_both_phases(wl, cur_gpu_config).get(phase)
                else:
                    rep = run_workload(wl, cur_gpu_config) if wl else None
                kernels = []
                if rep is not None:
                    for o in sorted(rep.ops, key=lambda o: -o.total_s)[:3]:
                        kernels += [k for k in _KERNELS.get(id(o), [])] or []
                from tilesight.gpuTilingPerfHWModel.interfaceAndRun.runner import resolve_op  # noqa: F401
                if rep is not None and not kernels:
                    kernels = _top_kernels(rep, cur_gpu_config)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    write_pdf(f.name, cur_gpu_config, kernels, res.get("title", "TileSight report"), wl, rep)
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
        if path == "/api/regr/jobs":                # the dashboard's job list (page 1)
            from tilesight.gpuTilingPerfHWModel.regr.store import MAX_JOBS
            return self._json(200, {"jobs": regr_store.list(), "max_jobs": MAX_JOBS})
        if path.startswith("/api/regr/jobs/"):       # one job's metadata + its result, once done
            jid = path.rsplit("/", 1)[-1]
            job = regr_store.get(int(jid)) if jid.isdigit() else None
            if job is None:
                return self._json(404, {"error": "no such regression job"})
            d = job.summary()
            d["config"] = job.config
            if job.result_file.exists():
                try:
                    d["result"] = json.loads(job.result_file.read_text())
                except Exception:                    # noqa: BLE001 - a job mid-write, or a corrupt file
                    d["result"] = None
            return self._json(200, d)
        if path == "/api/regr/file":                 # an artifact this job's out dir already has
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            jid = q.get("job", "")
            job = regr_store.get(int(jid)) if jid.isdigit() else None
            if job is None:
                return self._json(404, {"error": "no such regression job"})
            try:
                fp = (job.out_dir / q.get("phase", "decode") / q.get("name", "")).resolve()
                fp.relative_to(job.out_dir.resolve())        # refuse to leave the job's own directory
            except (ValueError, RuntimeError):
                return self._json(400, {"error": "bad path"})
            if not fp.is_file():
                return self._json(404, {"error": "no such file"})
            ctype = {".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".csv": "text/csv", ".txt": "text/plain; charset=utf-8"}.get(fp.suffix, "application/octet-stream")
            body = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f"attachment; filename={fp.name}")
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
        if self.path.split("?")[0] == "/api/validate_gpu_tiling_perf_hw_model":
            import yaml as _yaml
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                raw = _yaml.safe_load(body.get("yaml") or "") or {}
            except Exception as e:                          # noqa: BLE001
                return self._json(200, {"problems": [f"YAML error: {e}"], "fields": 0})
            from tilesight.gpuTilingPerfHWModel.interfaceAndRun.schema import validate as _v
            n_set = sum(1 for _ in json.dumps(raw))
            return self._json(200, {"problems": _v(raw), "fields": len(json.dumps(raw).split(","))})
        if self.path.split("?")[0] == "/api/regr/jobs":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": str(e)})
            cfg = body.get("config") or {}
            wl = cfg.get("workload") or {}
            mode = body.get("mode") or (
                "workload_both" if (wl.get("run") or {}).get("phase") == "both" else "workload")
            if mode not in ("workload", "workload_both"):
                return self._json(400, {"error": "regression jobs only support mode workload / workload_both"})
            problems = validate_workload(wl)
            if problems:
                return self._json(400, {"error": "; ".join(problems)})
            try:
                _load_hw(cfg)                        # fail fast on a bad GPU config, before queueing
            except Exception as e:                      # noqa: BLE001
                return self._json(400, {"error": f"{type(e).__name__}: {e}"})
            from tilesight.gpuTilingPerfHWModel.regr.worker import ensure_started
            job = regr_store.add(body.get("name", ""), body.get("tag", ""), mode, cfg)
            ensure_started()
            return self._json(200, {"job_id": job.id})
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
            _JOBS[jid]["mode"] = body.get("mode", "run")
        threading.Thread(target=_work, args=(jid, body.get("mode", "run"), body.get("config", {})),
                         daemon=True).start()
        self._json(200, {"job_id": jid})

    def do_DELETE(self):                                    # noqa: N802
        if self.path.split("?")[0].startswith("/api/regr/jobs/"):
            jid = self.path.rsplit("/", 1)[-1]
            return self._json(200, {"ok": regr_store.delete(int(jid)) if jid.isdigit() else False})
        jid = self.path.rsplit("/", 1)[-1]
        with _LOCK:
            _JOBS.pop(jid, None)
        self._json(200, {"ok": True})


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    BACKEND = "cpp"
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"tilesight web UI on http://{host}:{port}  (engine backend: {BACKEND})")
    print("computation runs here; clients only need a browser")
    httpd.serve_forever()
