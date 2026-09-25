"""Web server: job API, progress reporting, page is self-contained."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tilesight.interfaceAndModelRun.server import WEB_DIR, Handler


@pytest.fixture(scope="module")
def url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _post(url, mode, cfg):
    req = urllib.request.Request(url + "/api/jobs", method="POST",
                                 data=json.dumps({"mode": mode, "config": cfg}).encode(),
                                 headers={"Content-Type": "application/json"})
    jid = json.load(urllib.request.urlopen(req))["job_id"]
    states = []
    for _ in range(600):
        j = json.load(urllib.request.urlopen(f"{url}/api/jobs/{jid}"))
        states.append(j)
        if j["state"] != "running":
            return j, states
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_page_is_self_contained():
    html = (WEB_DIR / "index.html").read_text()
    for bad in ("cdn.", "googleapis", "unpkg", "jsdelivr", "<link rel=\"stylesheet\" href=\"http"):
        assert bad not in html
    assert "<script>" in html and "fetch(" in html


def test_options(url):
    o = json.load(urllib.request.urlopen(url + "/api/options"))
    assert "kimi_k2.hf" in o["models"] and "b300" in o["hardware"]
    assert "memory.ddr.bandwidth_TBps" in o["sweep_params"]


def test_run_job_reports_progress_and_result(url):
    cfg = {"model": "kimi_k2.hf", "gpuTilingPerfHWModel": "b300",
           "run": {"phase": "decode", "batch": 64, "seq_len": 4096, "dp": 8}}
    j, states = _post(url, "run", cfg)
    assert j["state"] == "done"
    r = j["result"]
    assert r["step_ms"] > 0 and r["ops"] and r["bounds"]["by_resource"]
    assert r["memory"]["total_GB"] > 0
    assert any(s["total"] > 1 for s in states)          # progress was reported
    assert all(s["done"] <= s["total"] for s in states)


def test_hw_override_and_request_mode(url):
    cfg = {"model": "kimi_k2.hf", "gpuTilingPerfHWModel": "b300", "gpuTilingPerfHWModelOverrides": '{"memory.ddr.bandwidth_TBps": 16}',
           "run": {"batch": 64, "dp": 8, "prompt_len": 2048, "output_len": 512,
                   "prefill_batch": 8, "decode_samples": 3}}
    j, _ = _post(url, "request", cfg)
    assert j["state"] == "done"
    r = j["result"]
    assert r["ttft_ms"] > 0 and len(r["points"]) == 3
    assert r["points"][-1]["kv_len"] > r["points"][0]["kv_len"]
    assert r["peak_total_GB"] >= r["points"][-1]["total_GB"] - 1e-6


def test_custom_model_yaml_and_sweep(url):
    yaml_text = """
name: tiny
hidden: 2048
vocab: 32000
layers:
  - name: block
    repeat: 4
    blocks:
      - {type: norm}
      - {type: gqa, heads: 16, kv_heads: 4, head_dim: 128}
      - {type: norm}
      - {type: mlp, d_ff: 8192}
"""
    cfg = {"model_yaml": yaml_text, "gpuTilingPerfHWModel": "h200",
           "run": {"phase": "decode", "batch": 8, "seq_len": 2048, "tp": 1, "dp": 1},
           "sweep": {"param": "memory.ddr.bandwidth_TBps", "values": "2,4,8", "link_l2": False}}
    j, _ = _post(url, "sweep", cfg)
    assert j["state"] == "done"
    rows = j["result"]["rows"]
    assert len(rows) == 3 and rows[0]["step_ms"] > rows[-1]["step_ms"]


def test_bad_config_reports_error(url):
    j, _ = _post(url, "run", {"model": "nope.hf", "gpuTilingPerfHWModel": "b300", "run": {}})
    assert j["state"] == "error" and j["error"]


def test_kernel_mode_single_gpu(url):
    cfg = {"gpuTilingPerfHWModel": "b300", "kernel": {"kernel": "gemm", "M": 4096, "N": 4096, "K": 7168,
                                     "a_dtype": "fp8", "b_dtype": "fp8", "compute_dtype": "fp8",
                                     "tile": "auto"}}
    j, _ = _post(url, "kernel", cfg)
    assert j["state"] == "done"
    r = j["result"]
    assert r["gpu_name"] == "B300" and r["tried"] > 10
    b = r["best"]
    assert b["time_us"] >= b["ideal_us"] > 0            # never beats the roofline lower bound
    assert b["bound"].startswith("tc:") and 0 < b["peak_pct"] <= 1.0
    times = [c["time_us"] for c in r["candidates"]]
    assert times == sorted(times)                       # ranked best first


def test_kernel_mode_fixed_tile_is_slower_than_search(url):
    base = {"kernel": "gemm", "M": 4096, "N": 4096, "K": 7168, "compute_dtype": "fp8"}
    auto, _ = _post(url, "kernel", {"gpuTilingPerfHWModel": "b300", "kernel": {**base, "tile": "auto"}})
    fixed, _ = _post(url, "kernel", {"gpuTilingPerfHWModel": "b300",
                                     "kernel": {**base, "tile": '{"bm":64,"bn":64,"bk":64}'}})
    assert fixed["result"]["tried"] == 1
    assert fixed["result"]["best"]["time_us"] > auto["result"]["best"]["time_us"]


def test_kernel_mode_attention_and_hw_override(url):
    k = {"kernel": "attn_decode", "B": 32, "H": 64, "kv_heads": 1, "S": 8192,
         "d_qk": 576, "d_v": 512, "v_in_k": True, "tile": "auto"}
    slow, _ = _post(url, "kernel", {"gpuTilingPerfHWModel": "b300", "kernel": k,
                                    "gpuTilingPerfHWModelOverrides": '{"memory.ddr.bandwidth_TBps": 4}'})
    fast, _ = _post(url, "kernel", {"gpuTilingPerfHWModel": "b300", "kernel": k})
    assert slow["result"]["best"]["bound"].startswith("ddr:")
    assert fast["result"]["best"]["time_us"] < slow["result"]["best"]["time_us"]


def test_cycle_csv_endpoint(url):
    cfg = {"gpuTilingPerfHWModel": "b300", "kernel": {"kernel": "gemm", "M": 2048, "N": 2048, "K": 4096,
                                     "compute_dtype": "fp8", "tile": "auto"}}
    req = urllib.request.Request(url + "/api/jobs", method="POST",
                                 data=json.dumps({"mode": "kernel", "config": cfg}).encode(),
                                 headers={"Content-Type": "application/json"})
    jid = json.load(urllib.request.urlopen(req))["job_id"]
    for _ in range(600):
        j = json.load(urllib.request.urlopen(f"{url}/api/jobs/{jid}"))
        if j["state"] != "running":
            break
        time.sleep(0.05)
    assert j["state"] == "done" and j["result"]["machine"]["waves"]
    r = urllib.request.urlopen(f"{url}/api/csv?job={jid}")
    assert r.headers["Content-Type"] == "text/csv"
    body = r.read().decode()
    head = body.splitlines()[0].split(",")
    assert head[:4] == ["cycle", "time_ns", "phase", "round"] and "tc" in head and "tc_busy" in head
    assert len(body.splitlines()) > 50
