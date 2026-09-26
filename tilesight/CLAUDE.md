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

Python = configuration, shape-only lowering, reports. C++ (nanobind, `tilesight._core`) = the
whole computation (`gpuTilingPerfHWModel/model/`) — tiling search, wave decomposition, the
round-time engine, the deterministic tile cache. There is no second (Python) implementation to
keep in sync; `gpuTilingPerfHWModel/model/` is pure C++.

## Layout
This project's own root doubles as the "tilesight" package root — flat layout, no `src/` or
`python/` wrapper (`pyproject.toml`'s `wheel.packages`/`pythonpath` point back at this same
directory; see the comment there for why). It has one big self-contained folder
(`gpuTilingPerfHWModel/`, the model itself — no b200/Kimi specifics inside it, could be lifted
out and run standalone) plus the things that *use* it, sitting next to it, next to `docs/`,
`tests/`, `examples/` and this file:

```
tilesight/                   (this directory - both the project root and the package root)
  gpuTilingPerfHWModel/       the model, standalone. No preset data lives in here — it reads
                              gpuPresets/ and modelPresets/ (both outside it) by relative path.
    interfaceAndRun/           the ONLY entry point into model/ — nothing outside this folder
                                (not even genResult/) calls into model/ directly. Both config
                                schemas are defined here.
      hardware_spec.py           HardwareSpec: hardware description -> resource lanes (units
                                 matter, see DESIGN §2); this is cur_gpu_config. NVIDIA
                                 b300/b200/h200 + AMD mi300x/mi325x/mi355x/mi450 (data in
                                 ../../gpuPresets/); dtype aliases + widening + weight-only live here.
      schema.py                  the 99-field flat schema hardware_spec.py validates against
      slice_config.py             the 42-field "architect's view" alternate input form
      model_spec.py                ModelSpec: block-level model spec + HF importer (data in
                                   ../../modelPresets/)
      run_config.py                 RunConfig: phase/batch/seq/parallelism/dtypes
      runner.py                     CurModelConfig (model_spec+run_config bundle), run() the
                                     public entry point, run_model() the lower-level 3-arg one,
                                     tile resolution, ModelReport — the only Python caller of
                                     tilesight._core (gpuTilingPerfHWModel/model/)
      lower.py                      blocks -> Ops for ONE GPU (parallelism semantics, DESIGN
                                     §6) — shape/dtype arithmetic over ModelSpec/RunConfig only,
                                     never HardwareSpec, so it stays Python alongside the schemas
      attention_blocks.py           mha / gqa(+mqa) / mla blocks, flash | naive cores, KV-cache
                                     sizing (DESIGN §5.2b) — same reason as lower.py
      memmap.py                     per-GPU memory map (weights/KV/activations base+size), used
                                     by genResult/addressing.py and the `memmap` CLI command
      workload.py                    one-layer "workload" shortcut -> CurModelConfig (model/
                                     never sees a workload dict, only the CurModelConfig it becomes)
      request.py                      prompt_len + output_len -> TTFT, TPOT curve, peak memory,
                                      max concurrency
    model/                      the actual computation: a pure C++ nanobind extension
                                (tilesight._core). Reached only through interfaceAndRun/runner.py,
                                never a public entry point of its own. No Python implementation
                                lives here — see the folder-by-folder map right below.
      gpu_top/                    orchestrator: lowering (shape+tile -> a tile execution plan,
                                  per op kind: gemm/attention/comm) and the wave-decomposition
                                  scheduler that calls the other four folders (DESIGN §3, §5);
                                  the nanobind bindings (bindings.cpp) live here too
      shader_core/                 one core's per-lane time; the steady-state K-loop round
                                   formula (DESIGN §3)
      shader_slice/                 grid -> waves; the slice-shared L1
      on_chip_buffer/                the optional staging SRAM between the shader slices and the
                                     memory slices
      memory_slice/                   the L2 port + DMA port + HBM behind it
      common/cache/                    the deterministic tile-level cache simulation (DESIGN §4),
                                       shared by on_chip_buffer and memory_slice
      memory.py                    weights / KV / activations per GPU (stays Python — standalone
                                   capacity math over many ops, not part of one op's engine)
      dse/sweep.py                  sweep + required_value (bisection); repeatedly runs the model
    genResult/                  turns a ModelReport into something to look at or download; pulls
                                its data from whatever interfaceAndRun/runner.py already computed.
      report/table.py, report/timeline.py (Fig-3e schedule reconstruction), report/figure3.py
      (Fig 3 d/e/f artifact), report/addressing.py, report/excel.py, report/pdfreport.py
      memmap_report.py: HBM address map (every tensor's region from 0x8000_0000, bytes per HBM
      port) -> out/memmap/*.csv|xlsx|txt and the results page's address-map section
    out/                        the real output: where a run's generated trace/xlsx/pdf/svg
                                files land (git-ignored except a .gitkeep; created on demand)
  gpuPresets/                 GPU YAML presets (b200.yaml, b300.yaml, h200.yaml, mi300x.yaml, ...)
                              — data only, read by gpuTilingPerfHWModel/interfaceAndRun/hardware_spec.py
  modelPresets/               model architecture presets (kimi_k2.hf.json, llama2_7b.hf.json, ...)
                              — data only, read by gpuTilingPerfHWModel/interfaceAndRun/model_spec.py
  cli/cli.py, cli/server.py   the two ways to drive the tool: `python -m tilesight.cli.cli`
                              (also the `tilesight` console script) and the stdlib HTTP server
                              (/api/options, /api/jobs, async + progress; modes: kernel (single
                              GPU) | run | request | sweep | need). Both import
                              gpuTilingPerfHWModel/interfaceAndRun and call run() — neither
                              reaches into model/ directly.
  html/index.html, html/app.html   self-contained UI (no CDN/fonts/external requests) served by
                              cli/server.py — keep it that way. Each has a #buildOverlay banner
                              that polls /api/build_status on load and hides once the C++ core
                              is ready (or shows the build error) — see cli/server_boot.py.
  cli/server_boot.py         `serve`'s real entry point (cli.py delegates to it): binds the
                              socket and serves html/ + /api/build_status immediately, builds
                              tilesight._core in a background thread, then swaps in
                              cli/server.py's Handler once it's ready. Stdlib-only at module
                              scope on purpose — see _core_builder.py below for why.
  _core_builder.py (project root)   builds tilesight._core (cmake configure+build) if it isn't
                              there yet. tilesight/__init__.py calls it synchronously on import
                              for every command except `serve` (checked via sys.argv, since
                              __init__.py runs before any of cli.py's own code can) — `serve`
                              defers it to cli/server_boot.py's background thread instead, so the
                              page can show build progress rather than the terminal hanging with
                              nothing listening yet.
The compiled extension (CMakeLists.txt builds every gpuTilingPerfHWModel/model/**/*.cpp into it)
lands right at this directory's root as _core*.so, since `from tilesight import _core` expects
it right next to gpuTilingPerfHWModel/.
tests/   examples/   docs/DESIGN.md   docs/TASKS.md   docs/research/*.md (background + specs)
```

## Commands
`import tilesight` auto-builds `tilesight._core` on first use if it isn't there yet (runs
`cmake -S . -B build && cmake --build build -j` for you, once; needs a C++17 compiler + cmake
on PATH). So plain `PYTHONPATH=.. python -m tilesight.cli.cli ...` or `pytest -q` from a fresh
checkout just works, no manual build step — the first call takes ~1-2 min, every one after is
instant. Manual build is still there for explicit control:
```bash
pip install -e ".[dev]"                     # builds C++ via scikit-build-core + nanobind
# or, for fast iteration:
cmake -S . -B build && cmake --build build -j   # drops _core*.so right here, next to gpuTilingPerfHWModel/
pytest -q                                   # all tests (pyproject.toml sets pythonpath); the C++
                                             # build is mandatory, there is no Python fallback
PYTHONPATH=.. python -m tilesight.cli.cli run --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode --batch 256 --seq 8192 --dp 8
PYTHONPATH=.. python -m tilesight.cli.cli request --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --run-config examples/request_kimi_b300.yaml
PYTHONPATH=.. python -m tilesight.cli.cli sweep --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode --batch 256 --seq 8192 --dp 8 \
    --param memory.ddr.bandwidth_TBps --values 4,8,12,16
PYTHONPATH=.. python examples/sweep_ddr_bw_b300.py
PYTHONPATH=.. python -m tilesight.cli.cli serve --host 0.0.0.0 --port 8000   # web UI on your machine
```

## Invariants (do not break)
1. **One implementation.** `gpuTilingPerfHWModel/model/` is pure C++ — there is no Python
   reference engine to keep in sync any more. `tests/test_cpp_parity.py` now pins the C++
   engine's behavior against known-good analytical bounds and hand-checkable cache traces
   (regression coverage), not Python/C++ parity.
2. **Units**: per-SM lanes carry *seconds on one SM*; shared lanes (`l2`, `ddr`) carry
   *bytes*. Lowerings convert with `HardwareSpec.*_time_per_sm`. Never mix.
3. **Equations live in docs/DESIGN.md.** If you change a formula, update the DESIGN
   section and add/adjust a test that pins the behavior. Mark deviations from the paper.
4. **Hardware numbers are tagged** `[spec]` or `[calib]` in YAML. Never silently change
   a `[spec]` value; `[calib]` values are replaced only by the calibration suite output.
5. **Hardware is data.** New HW features = new YAML fields + lanes; the engine core stays
   lane-generic. Do not hard-code GPU names in gpuTilingPerfHWModel/model/ C++ or in
   interfaceAndRun/lower.py / attention_blocks.py (use capability flags like
   `compute.cta_pair`, presence of `memory.onchip.tmem`).
6. **One GPU's view.** `gpuTilingPerfHWModel/interfaceAndRun/lower.py` emits the ops executed by
   one GPU; parallelism only changes shapes and adds collectives there. Keep this property.
7. **Attribution**: every second of kernel time is charged to exactly one coarse limiter
   and one fine `limiter_detail` key; the sums must match (`test_detail_attributes_tensor`).
   New actions must be named `<verb>:<tensor>` (e.g. `load:kv_cache(K)`) so reports can
   say which tensor/cache/register resource binds.
8. **Web UI**: `html/index.html` must stay self-contained (no CDN, no external fonts/scripts) and
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
  `gpuTilingPerfHWModel/interfaceAndRun/attention_blocks.py`, reuse `attn_core` if it is softmax
  attention, extend `kv_bytes_per_seq`, and add a flash-vs-naive test in
  `tests/test_attention_variants.py`.
- **New block type** (e.g. `mamba`): add a branch in
  `gpuTilingPerfHWModel/interfaceAndRun/lower.py::lower_block`, KV/state accounting in
  `gpuTilingPerfHWModel/model/memory.py`, and a test.
- **New kernel family**: add a `lower_*` function to `gpuTilingPerfHWModel/model/gpu_top/gpu_top.cpp`
  (typed args in, `optional<vector<LoweredKernel>>` out — `nullopt` for an illegal tile), bind it
  in `bindings.cpp`, register its op kind in
  `gpuTilingPerfHWModel/interfaceAndRun/runner.py::resolve_op`, add a search space.

## Working style for tasks
- Take tasks from `docs/TASKS.md` in order unless told otherwise. Each task lists
  acceptance criteria; write the test first, then the code.
- Keep the engine readable; it's all C++ now — no separate "optimize only in C++" step.
- After each task: run the full test suite and `examples/kimi_k2_b300.py`; report step-time
  deltas vs. before in the PR/commit message.
- When a paper detail is ambiguous, implement the simplest defensible version, document
  it in DESIGN.md under "Deviations/assumptions", and make it switchable if cheap.
