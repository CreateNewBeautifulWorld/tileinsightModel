# CLAUDE.md — working guide for Claude Code in this repo

## What this project is
A re-implementation of **TileSight** (arXiv 2607.22432, "A First-Principles Tile-Centric
Analytical GPU Performance Model from Cores to Clusters") extended with:
1. a **model layer**: give layer sizes (or a HF `config.json`, e.g. Kimi-K2) → per-GPU
   kernels → step latency, throughput, **memory footprint**, and **bottleneck** per op;
2. **configurable tiles** per op (fixed, per-op override, or auto-search);
3. **extended hardware resources**: multiple load paths (TMA/LSU, split), extra on-chip
   memories (TMEM, …), cluster multicast / 2-CTA MMA, per-SM load caps;
4. **design-space exploration**: sweep any hardware field (e.g. DDR bandwidth on a
   B300-class GPU) and bisect the value needed to hit a latency target.

Python = configuration, lowering, reports. C++ (nanobind) = engine hot path.
The Python engine in `engine/reference.py` is the **executable spec**; C++ must match it.

## Layout
```
python/tilesight/
  gpuTilingPerfHWModel/spec.py, gpuTilingPerfHWModel/db/*.yaml   hardware description -> resource lanes (units matter, see DESIGN §2)
                             NVIDIA b300/b200/h200 + AMD mi300x/mi325x/mi355x/mi450;
                             dtype aliases + widening + weight-only live in gpuTilingPerfHWModel/spec.py
  ir/kernel.py               Action / Kernel / KernelResult  (engine contract)
  engine/reference.py        pipeline-envelope engine (DESIGN §3)       <- spec
  engine/cache.py            tile reuse-distance + SDCM L2 model (DESIGN §4) <- spec
  engine/backend.py          picks C++ _core if built (TILESIGHT_BACKEND=python|cpp)
  engine/cpp_bridge.py       packs Python IR into lane-indexed C++ structs
  kernels/gemm.py            GEMM / batched / grouped(MoE) lowering + elementwise
  kernels/attention.py       decode (split-KV, MLA) and causal prefill lowering
  kernels/comm.py            alpha-beta collectives
  kernels/tiles.py           TileConfig / AttnTileConfig + search spaces
  model/spec.py              block-level model spec + HF importer
  model/lower.py             blocks -> Ops for ONE GPU (parallelism semantics, DESIGN §6)
  model/attention_blocks.py  mha / gqa(+mqa) / mla blocks, flash | naive cores, KV-cache sizing (DESIGN §5.2b)
  model/runner.py            tile resolution, evaluation, ModelReport
  model/memory.py            weights / KV / activations per GPU
  model/request.py           prompt_len + output_len -> TTFT, TPOT curve, peak memory, max concurrency
  dse/sweep.py               sweep + required_value (bisection)
  report/table.py, report/timeline.py (Fig-3e schedule reconstruction),
  report/figure3.py (Fig 3 d/e/f artifact), report/addressing.py, report/excel.py, cli.py
  server.py                  stdlib HTTP server: /api/options, /api/jobs (async + progress);
                             modes: kernel (single GPU) | run | request | sweep | need
  web/index.html             self-contained UI (no CDN/fonts/external requests) — keep it that way
cpp/include/tilesight/engine.hpp, cpp/src/{engine,cache}.cpp, cpp/bindings/bind.cpp
tests/   examples/   docs/DESIGN.md   docs/TASKS.md   docs/research/*.md (background + specs)
```

## Commands
```bash
pip install -e ".[dev]"                     # builds C++ via scikit-build-core + nanobind
# or, for fast iteration:
cmake -S . -B build && cmake --build build -j   # drops _core*.so into python/tilesight/
PYTHONPATH=python pytest -q                     # all tests (parity tests skip if no _core)
TILESIGHT_BACKEND=python PYTHONPATH=python pytest -q   # force reference engine
PYTHONPATH=python python -m tilesight.cli run --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode --batch 256 --seq 8192 --dp 8
PYTHONPATH=python python -m tilesight.cli request --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --run-config examples/request_kimi_b300.yaml
PYTHONPATH=python python -m tilesight.cli sweep --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode --batch 256 --seq 8192 --dp 8 \
    --param memory.ddr.bandwidth_TBps --values 4,8,12,16
PYTHONPATH=python python examples/sweep_ddr_bw_b300.py
PYTHONPATH=python python -m tilesight.cli serve --host 0.0.0.0 --port 8000   # web UI on your machine
```

## Invariants (do not break)
1. **Parity**: any change to `engine/reference.py` or `engine/cache.py` must be mirrored
   in `cpp/src/*.cpp` in the same change; `tests/test_cpp_parity.py` requires ≤1e-9 rel
   error and identical bottleneck labels (tie-break = insertion order, like Python `max`).
2. **Units**: per-SM lanes carry *seconds on one SM*; shared lanes (`l2`, `ddr`) carry
   *bytes*. Lowerings convert with `HardwareSpec.*_time_per_sm`. Never mix.
3. **Equations live in docs/DESIGN.md.** If you change a formula, update the DESIGN
   section and add/adjust a test that pins the behavior. Mark deviations from the paper.
4. **Hardware numbers are tagged** `[spec]` or `[calib]` in YAML. Never silently change
   a `[spec]` value; `[calib]` values are replaced only by the calibration suite output.
5. **Hardware is data.** New HW features = new YAML fields + lanes; the engine core stays
   lane-generic. Do not hard-code GPU names in engine/kernels code (use capability flags
   like `compute.cta_pair`, presence of `memory.onchip.tmem`).
6. **One GPU's view.** `model/lower.py` emits the ops executed by one GPU; parallelism
   only changes shapes and adds collectives there. Keep this property.
7. **Attribution**: every second of kernel time is charged to exactly one coarse limiter
   and one fine `limiter_detail` key; the sums must match (`test_detail_attributes_tensor`).
   New actions must be named `<verb>:<tensor>` (e.g. `load:kv_cache(K)`) so reports can
   say which tensor/cache/register resource binds.
8. **Web UI**: `web/index.html` must stay self-contained (no CDN, no external fonts/scripts) and
   every long-running call must go through the job API so the UI can show progress. Long work in
   `server.py` runs in a worker thread and reports `progress(done, total, label)`.
9. **Monotonicity**: more bandwidth/compute must never make a kernel slower
   (`test_more_ddr_bw_never_slower`). Add similar tests when adding knobs.

## How to extend
- **New GPU**: copy the closest YAML, tag every value `[spec]`/`[calib]`, set capability flags
  (`compute.cta_pair`, `compute.cluster_multicast`, presence of `memory.onchip.tmem`,
  `load_paths_default`), and add it to `tests/test_dtypes_amd.py`'s preset list.
- **New datatype**: add bytes to `DTYPE_BYTES`, an alias in `DTYPE_ALIAS` if it runs on an existing
  datapath, `WEIGHT_ONLY` if the MMA runs wider, and a peak entry in each hardware table.
- **New hardware knob**: add to YAML (tagged) → read via `cur_gpu_config.get("a.b.c")` in the lowering
  → if it is a new resource, add it in `HardwareSpec.lanes()` (C++ needs nothing: lanes
  are data). Add a DSE example if it is a design parameter.
- **New on-chip memory** (e.g. DSMEM, larger SRAM): add under `memory.onchip.<name>` with
  capacity + bandwidth; it automatically becomes a lane `<name>`; charge it in lowerings.
- **New load path**: add under `load_paths.<name>`; select per tile via
  `TileConfig(load_path="name")` or split `"split:tma=0.7,lsu=0.3"`.
- **New attention variant** (e.g. DSA/NSA sparse, linear attention): add a lowering in
  `model/attention_blocks.py`, reuse `attn_core` if it is softmax attention, extend
  `kv_bytes_per_seq`, and add a flash-vs-naive test in `tests/test_attention_variants.py`.
- **New block type** (e.g. `mamba`): add a branch in
  `model/lower.py::lower_block`, KV/state accounting in `model/memory.py`, and a test.
- **New kernel family**: write `lower_*` returning `list[Kernel]` (or None if the tile is
  illegal), register its op kind in `model/runner.py::resolve_op`, add a search space.

## Working style for tasks
- Take tasks from `docs/TASKS.md` in order unless told otherwise. Each task lists
  acceptance criteria; write the test first, then the code.
- Keep the reference engine readable; optimize only in C++.
- After each task: run the full test suite with both backends and
  `examples/kimi_k2_b300.py`; report step-time deltas vs. before in the PR/commit message.
- When a paper detail is ambiguous, implement the simplest defensible version, document
  it in DESIGN.md under "Deviations/assumptions", and make it switchable if cheap.
