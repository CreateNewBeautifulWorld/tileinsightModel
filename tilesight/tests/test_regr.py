"""The regression-job dashboard: gpuTilingPerfHWModel/regr/ (persistent job store + serial
background worker) and its HTTP surface on cli/server.py (/api/regr/..., html/regr*.html)."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tilesight.cli.server import Handler
from tilesight.gpuTilingPerfHWModel.regr.store import RegrStore

CFG = {"gpuTilingPerfHWModel": "b300", "workload": {
    "name": "x", "hidden": 4096, "layers": 1,
    "attention": {"type": "mla", "impl": "flash", "heads": 64, "head_dim": 576, "v_head_dim": 512},
    "ffn": {"type": "moe", "d_ff": 2048, "experts": 32},
    "dtypes": {"weight": "fp8", "kv": "bf16"},
    "run": {"phase": "decode", "batch": 8, "prefill_seq_len": 1024, "cur_decoding_seq_len": 1024,
           "max_seq_len": 8192}},
    "sim_max_accesses": 2e6}


# ------------------------------------------------------------------ store.py, no server needed
def test_store_prunes_the_oldest_completed_job_past_the_cap(tmp_path):
    # one store for the whole test, so job ids stay unique and out_dir (always under the real,
    # shared JOBS_DIR — every job's artifacts live in one place regardless of which registry
    # file tracks it) never collides between two independently-numbered stores.
    from tilesight.gpuTilingPerfHWModel.regr.store import MAX_JOBS
    s = RegrStore(tmp_path / "registry.json")
    try:
        for i in range(1, MAX_JOBS + 5):
            j = s.add(f"job{i}", "t", "workload", {"gpuTilingPerfHWModel": "b300", "workload": {}})
            s.update(j.id, status="done")           # only non-running jobs are ever pruned
        assert s.count() == MAX_JOBS
        ids = sorted(row["id"] for row in s.list())
        assert ids[0] == 5 and ids[-1] == MAX_JOBS + 4   # the oldest 4 were dropped to stay at MAX_JOBS

        # a currently-running job is never pruned even if it is the oldest
        old = s.add("old", "", "workload", {})
        s.update(old.id, status="running")
        for i in range(10):
            j = s.add(f"new{i}", "", "workload", {})
            s.update(j.id, status="done")
        assert s.count() == MAX_JOBS               # still capped
        assert old.id in {row["id"] for row in s.list()}
        assert s.get(old.id).status == "running"
    finally:
        for row in s.list():
            s.delete(row["id"])


def test_store_persists_across_reload_and_recovers_interrupted_jobs(tmp_path):
    reg = tmp_path / "registry.json"
    s = RegrStore(reg)
    j = s.add("a job", "my-tag", "workload", CFG, submitted_by="alice")
    s.update(j.id, status="running", started_at="2020-01-01T00:00:00+00:00")
    try:
        # a fresh store re-reads what's on disk — a job stuck "running" (a server restart lost
        # its progress) comes back as "pending" so the worker picks it up again
        s2 = RegrStore(reg)
        got = s2.get(j.id)
        assert got is not None and got.status == "pending" and got.error is None
        assert got.tag == "my-tag" and got.submitted_by == "alice" and got.config == CFG
    finally:
        s.delete(j.id)


def test_job_out_dir_is_the_only_thing_deleted(tmp_path):
    # a fresh registry file, but out_dir/jobs/<id> is always under the real package (by design:
    # every job's artifacts live in one place regardless of which registry tracks it) — that's
    # exactly what this test is checking, so it uses the real JOBS_DIR on purpose.
    s = RegrStore(tmp_path / "registry.json")
    j = s.add("throwaway", "", "workload", {"gpuTilingPerfHWModel": "b300", "workload": {}})
    d = j.out_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "marker.txt").write_text("x")
    assert d.exists()
    assert s.delete(j.id)
    assert not d.exists()
    assert s.get(j.id) is None


# ------------------------------------------------------------------ the HTTP surface end to end
@pytest.fixture(scope="module")
def url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _post(url, path, body):
    req = urllib.request.Request(url + path, method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def _wait_done(url, jid, timeout=120):
    for _ in range(int(timeout / 0.25)):
        j = json.load(urllib.request.urlopen(f"{url}/api/regr/jobs/{jid}"))
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.25)
    raise AssertionError("regression job timed out")


def test_regr_pages_are_served_and_self_contained(url):
    for path in ("/regr", "/regr.html", "/regr/new", "/regr_new.html", "/regr/result", "/regr_result.html"):
        page = urllib.request.urlopen(url + path).read().decode()
        assert "<html" in page
        # same self-containment discipline as app.html: no third-party link before the first
        # inline <script> (the API default's "http://127.0.0.1:8000" lives inside a script, fine)
        assert "http://" not in page.split("<script>")[0]
    js = urllib.request.urlopen(url + "/results.js").read().decode()
    assert "function renderPhaseBody" in js and "function memmapHtml" in js


def test_creating_a_regression_job_runs_it_and_writes_its_own_artifacts(url):
    jid = json.load(_post(url, "/api/regr/jobs", {"name": "kimi decode smoke", "tag": "ci-run", "config": CFG}))["job_id"]
    try:
        # shows up immediately as pending/running on the dashboard's list
        listing = json.load(urllib.request.urlopen(url + "/api/regr/jobs"))
        row = next(r for r in listing["jobs"] if r["id"] == jid)
        assert row["name"] == "kimi decode smoke" and row["tag"] == "ci-run" and row["gpu"] == "b300"
        assert row["status"] in ("pending", "running", "done")
        assert "config" not in row                    # the list is metadata only, not the full config

        j = _wait_done(url, jid)
        assert j["status"] == "done", j.get("error")
        assert j["mode"] == "workload" and j["result"]["step_ms"] > 0
        assert j["config"] == CFG                      # the detail endpoint does include it

        # the worker copied/wrote artifacts into this job's own directory (not the shared out/)
        files = (j["result"]["memmap"]["files"])
        import os
        name = os.path.basename(files["xlsx"])
        r = urllib.request.urlopen(f"{url}/api/regr/file?job={jid}&phase=decode&name={name}")
        assert "spreadsheet" in r.headers["Content-Type"] and len(r.read()) > 100
        r = urllib.request.urlopen(f"{url}/api/regr/file?job={jid}&phase=decode&name=cycles.xlsx")
        assert len(r.read()) > 100
        r = urllib.request.urlopen(f"{url}/api/regr/file?job={jid}&phase=decode&name=summary.txt")
        assert len(r.read()) > 100

        # a path that tries to leave the job's directory is refused, not served (the server's
        # query parsing doesn't URL-decode, so the traversal has to use literal dots/slashes)
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{url}/api/regr/file?job={jid}&phase=decode&name=../../../../../../etc/passwd")
        assert exc.value.code == 400

        # results.js's DOM-free path also renders without throwing, given this exact JSON shape
        # (a lightweight parity check with the browser-side renderer, run in a JS-less way: just
        # confirm the shape renderPhaseBody expects is present)
        res = j["result"]
        assert "memmap" in res and "ops" in res and "bounds" in res and "step_ms" in res
    finally:
        req = urllib.request.Request(f"{url}/api/regr/jobs/{jid}", method="DELETE")
        urllib.request.urlopen(req)


def test_regr_job_mode_follows_the_workload_phase(url):
    both_cfg = json.loads(json.dumps(CFG))
    both_cfg["workload"]["run"]["phase"] = "both"
    jid = json.load(_post(url, "/api/regr/jobs", {"name": "both", "tag": "", "config": both_cfg}))["job_id"]
    try:
        j = _wait_done(url, jid)
        assert j["status"] == "done", j.get("error")
        assert j["mode"] == "workload_both"
        assert set(j["result"].keys()) >= {"decode", "prefill"}
    finally:
        urllib.request.urlopen(urllib.request.Request(f"{url}/api/regr/jobs/{jid}", method="DELETE"))


def test_a_bad_workload_is_rejected_before_it_is_ever_queued(url):
    bad = {"gpuTilingPerfHWModel": "b300", "workload": {"attention": {"type": "nope"}}}
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(url, "/api/regr/jobs", {"name": "bad", "tag": "", "config": bad})
    assert exc.value.code == 400
    # never made it into the job list
    listing = json.load(urllib.request.urlopen(url + "/api/regr/jobs"))
    assert all(r["name"] != "bad" for r in listing["jobs"])


def test_deleting_and_fetching_unknown_jobs_are_well_behaved(url):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(url + "/api/regr/jobs/999999999")
    assert exc.value.code == 404
    req = urllib.request.Request(f"{url}/api/regr/jobs/999999999", method="DELETE")
    assert json.load(urllib.request.urlopen(req)) == {"ok": False}
