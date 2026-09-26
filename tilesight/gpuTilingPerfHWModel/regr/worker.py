"""Serial background processor for regression jobs (see `store.py`): pulls the oldest pending
job, runs it through the exact same `build_report_json()` the interactive wizard uses
(`cli/report_json.py`, so a regression job's result has the identical shape a live run would
have shown), then writes the result and whatever the model produced for it — the HBM address
map, the steady-state cycle trace as Excel/CSV, a text summary — into the job's own directory,
one job at a time so it never competes with an interactive run for the machine.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from pathlib import Path

from tilesight.gpuTilingPerfHWModel.regr.store import RegrJob, now_iso, store

_started = False
_start_lock = threading.Lock()


def _write_phase_artifacts(res: dict, out_dir: Path) -> None:
    """Whatever the model produced for one phase: copy the HBM memory map it already wrote
    under `out/memmap` (see `report_json._memmap_json`) into this job's own directory instead of
    leaving it in the shared, overwritten-by-the-next-run one, and write the heaviest kernel's
    steady-state trace (Excel + CSV) and a text summary alongside it — the same artifacts the
    results page can regenerate on demand for an ephemeral wizard job, kept here so they survive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for _kind, path in ((res.get("memmap") or {}).get("files") or {}).items():
        try:
            shutil.copy2(path, out_dir / Path(path).name)
        except OSError:
            pass
    summary = res.get("summary")
    if summary:
        (out_dir / "summary.txt").write_text(summary)
    tl = res.get("timeline")
    if tl:
        try:
            from tilesight.gpuTilingPerfHWModel.genResult.excel import write_excel
            write_excel(tl, str(out_dir / "cycles.xlsx"), "", res.get("gpu_name", ""), full=False)
        except Exception:                                  # noqa: BLE001 - the result itself still stands
            traceback.print_exc()
        try:
            from tilesight.gpuTilingPerfHWModel.genResult.timeline import cycle_csv
            (out_dir / "cycles.csv").write_text(cycle_csv(tl, full=False))
        except Exception:                                  # noqa: BLE001
            traceback.print_exc()


def _run_job(job: RegrJob) -> None:
    from tilesight.cli.report_json import build_report_json

    store.update(job.id, status="running", started_at=now_iso(), done=0, total=1, label="starting")

    def prog(done, total, label):
        store.update(job.id, done=done, total=total, label=label)

    try:
        out = build_report_json(job.mode, job.config, prog)
        phases = out if job.mode == "workload_both" else {job.phase or "decode": out}
        for phase, res in phases.items():
            _write_phase_artifacts(res, job.out_dir / phase)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        (job.out_dir / "result.json").write_text(json.dumps(out))
        store.update(job.id, status="done", completed_at=now_iso(), done=1, total=1, label="done")
    except Exception as e:                                  # noqa: BLE001 - surface to the dashboard
        store.update(job.id, status="error", completed_at=now_iso(),
                     error=f"{type(e).__name__}: {e}", label="failed")


def _loop() -> None:
    while True:
        job = store.next_pending()
        if job is None:
            time.sleep(1.0)
            continue
        try:
            _run_job(job)
        except Exception:                                   # noqa: BLE001 - the loop must not die
            traceback.print_exc()


def ensure_started() -> None:
    """Start the one background worker thread, at most once, however the server got here (the
    real `serve()` entry point, a test harness binding `Handler` directly, ...)."""
    global _started
    with _start_lock:
        if _started:
            return
        threading.Thread(target=_loop, daemon=True, name="regr-worker").start()
        _started = True
