"""Persistent registry of regression jobs: pending / running / done / error, capped at
`MAX_JOBS` (oldest auto-deleted), name + a free-text tag, who submitted it, when.

Unlike `cli/server.py`'s `_JOBS` (in-memory, one ad hoc wizard run, gone on restart), a
regression job is meant to be looked back at later: submitted from `html/regr_new.html` with a
name and a tag, queued (`html/regr.html` lists it as "pending"), and processed one at a time by
the single background worker in `worker.py` — so it never competes with an interactive wizard
run for the machine — which writes its result and a copy of whatever the model produced under
`out/` into this job's own directory (`RegrJob.out_dir`), so both survive past the next run and
past a server restart.
"""
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGR_DIR = Path(__file__).resolve().parent
REGISTRY_FILE = REGR_DIR / "registry.json"
JOBS_DIR = REGR_DIR / "jobs"
MAX_JOBS = 100


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RegrJob:
    id: int
    name: str
    tag: str
    mode: str                       # "workload" | "workload_both"
    config: dict[str, Any]
    submitted_by: str = "local"
    status: str = "pending"         # pending | running | done | error
    gpu: str = ""
    phase: str = ""
    submitted_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    done: int = 0
    total: int = 1
    label: str = ""

    @property
    def out_dir(self) -> Path:
        return JOBS_DIR / str(self.id)

    @property
    def result_file(self) -> Path:
        return self.out_dir / "result.json"

    def summary(self) -> dict:
        """Everything the dashboard table needs — not `config` (small, but pointless there) and
        not the (possibly large) result, which the detail endpoint attaches separately."""
        d = asdict(self)
        d.pop("config")
        d["has_result"] = self.result_file.exists()
        return d


class RegrStore:
    def __init__(self, registry_file: Path | str = REGISTRY_FILE):
        self._file = Path(registry_file)
        self._lock = threading.RLock()
        self._jobs: dict[int, RegrJob] = {}
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text())
        except Exception:                                  # noqa: BLE001 - a corrupt registry starts fresh
            return
        for r in raw.get("jobs", []):
            if r.get("status") == "running":                 # a server restart lost its progress
                r["status"], r["error"] = "pending", None
            try:
                j = RegrJob(**r)
            except TypeError:                                # a field the schema no longer has
                continue
            self._jobs[j.id] = j
        self._next_id = raw.get("next_id", max(self._jobs, default=0) + 1)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        raw = {"next_id": self._next_id, "jobs": [asdict(j) for j in self._jobs.values()]}
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(self._file)

    def add(self, name: str, tag: str, mode: str, config: dict, submitted_by: str = "local") -> RegrJob:
        with self._lock:
            wl = config.get("workload") or {}
            jid = self._next_id
            job = RegrJob(id=jid, name=name.strip() or f"job {jid}", tag=(tag or "").strip(),
                          mode=mode, config=config, submitted_by=(submitted_by or "local").strip() or "local",
                          gpu=config.get("gpuTilingPerfHWModel") or ("custom" if config.get("slice_cfg") or
                                                                     config.get("gpuTilingPerfHWModelYaml") else "b300"),
                          phase=(wl.get("run") or {}).get("phase") or "decode")
            self._next_id += 1
            self._jobs[job.id] = job
            job.out_dir.mkdir(parents=True, exist_ok=True)
            self._prune()
            self._save()
            return job

    def _prune(self) -> None:
        """Keep at most MAX_JOBS, oldest first, never touching one that is currently running."""
        if len(self._jobs) <= MAX_JOBS:
            return
        for jid in sorted(self._jobs):
            if len(self._jobs) <= MAX_JOBS:
                break
            j = self._jobs[jid]
            if j.status == "running":
                continue
            shutil.rmtree(j.out_dir, ignore_errors=True)
            del self._jobs[jid]

    def list(self) -> list[dict]:
        with self._lock:
            return [j.summary() for j in sorted(self._jobs.values(), key=lambda j: -j.id)]

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def get(self, jid: int) -> RegrJob | None:
        with self._lock:
            return self._jobs.get(jid)

    def next_pending(self) -> RegrJob | None:
        with self._lock:
            pend = [j for j in self._jobs.values() if j.status == "pending"]
            return min(pend, key=lambda j: j.id) if pend else None

    def update(self, jid: int, **fields: Any) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j is None:
                return
            for k, v in fields.items():
                setattr(j, k, v)
            self._save()

    def delete(self, jid: int) -> bool:
        with self._lock:
            j = self._jobs.pop(jid, None)
            if j is None:
                return False
            shutil.rmtree(j.out_dir, ignore_errors=True)
            self._save()
            return True


store = RegrStore()
