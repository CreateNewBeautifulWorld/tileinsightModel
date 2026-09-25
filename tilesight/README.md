# tilesight — tile-centric analytical GPU performance model + LLM simulator

Re-implementation of **TileSight** (arXiv 2607.22432) with a model layer on top:
give a model's layer sizes (or a HF `config.json`), a hardware YAML and a run config →
per-GPU step latency, tok/s/GPU, per-op tiles and limiters, HBM footprint, and
design-space sweeps over any hardware field.

Python front-end (config, lowering, reports) · C++17 core via nanobind (engine + L2 model),
bit-identical to the Python reference engine.

```bash
pip install -e ".[dev]"            # or: cmake -S . -B build && cmake --build build -j
PYTHONPATH=python pytest -q
PYTHONPATH=python python -m tilesight.cli run --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 \
    --phase decode --batch 256 --seq 8192 --dp 8
PYTHONPATH=python python -m tilesight.cli request --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --batch 256 --dp 8 \
    --prompt 8192 --output 4096            # TTFT, TPOT curve, peak memory, max concurrency
PYTHONPATH=python python -m tilesight.cli sweep --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode \
    --batch 256 --seq 8192 --dp 8 --param memory.ddr.bandwidth_TBps --values 4,8,12,16
PYTHONPATH=python python -m tilesight.cli need --model kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode \
    --batch 256 --seq 8192 --dp 8 --param memory.ddr.bandwidth_TBps --target-ms 25
PYTHONPATH=python python -m tilesight.cli dump-model --model kimi_k2.hf > my_model.yaml   # edit sizes
PYTHONPATH=python python examples/sweep_ddr_bw_b300.py
```

## Web UI (compute stays on your machine)
```bash
PYTHONPATH=python python -m tilesight.cli serve --host 0.0.0.0 --port 8000
```
Open `http://<your-machine>:8000`. The page is a **single self-contained HTML file**
(no CDN, no fonts, no external requests) served by a stdlib HTTP server — clients only need
a browser; every evaluation runs on the serving machine with the C++ engine.

- Modes: **single GPU (one kernel + tile study)** / single step / request (prompt+output) /
  hardware sweep / required-value bisection.
- Single-GPU mode: pick the card (b300 / b200 / h200, plus dotted overrides), pick a kernel
  (GEMM, grouped/MoE GEMM, attention decode, attention prefill, elementwise), enter the shapes,
  and get latency, TFLOP/s, % of the dtype peak, HBM traffic, occupancy (`1/smem+regs`),
  L2 miss ratio and the bound (`tc:mma`, `ddr:load:expert_weight`, `latency:softmax(sfu)`),
  with the ranked tile candidates behind it, plus a **steady-state timeline** (the paper's
  Figure 3e): one row per resource lane, one dashed segment per K-loop round, showing how
  multi-buffered loads overlap with compute and which lane (or dependency chain) sets the round.
  No TP/DP/EP, no collectives.
  The timeline reads in **microseconds or cycles** (toggle); above it a wave-level bar shows how
  the grid maps onto the SMs (full waves + tail), since every SM in a wave is identical here.
  Exports: a **coloured Excel grid** (one row per cycle, one column per unit, cell = the tile,
  colour = iteration so a tile keeps its colour from HBM to the tensor core), a **text trace** (header + per-action events with ns and cycles + both gantts + lane
  occupancy) and a **per-cycle CSV** (one row per cycle, one column per unit + busy fraction),
  for the drawn rounds or the whole kernel.
  Same thing in the terminal:
  `python -m tilesight.cli kernel --gpu-tiling-perf-hw-model b300 --kernel gemm --dtype fp8 --shape '{"M":4096,"N":4096,"K":7168}' --unit cyc --trace-out trace.txt --csv-out cycles.csv --xlsx-out cycles.xlsx`
- Config: model preset **or pasted layer-size YAML / HF config**, hardware preset + dotted
  overrides (`{"memory.ddr.bandwidth_TBps": 12}`), phase, batch, seq, prompt/output, TP/DP/EP,
  dtypes, attention impl, tile policy and per-op tile overrides.
- Datatypes: **int4 · nvfp4 · fp4 · fp8 · int8 · bf16 · fp16 · fp32** (+ mxfp4/mxfp6/mxfp8/tf32 aliases).
  `int4` is weight-only (MMA at the activation width, half the weight traffic); a dtype with no
  tensor-core datapath on the chosen card widens to the next one it has (fp4 → fp8 on Hopper/CDNA3);
  fp32 falls back to the CUDA/vector cores.
- Jobs are asynchronous: the UI polls `/api/jobs/<id>` and shows **"Processing… 12/38 · op name · 1.4s"**
  with a progress bar, so a long sweep never looks frozen.
- API (JSON): `GET /api/options`, `POST /api/jobs {mode, config}`, `GET /api/jobs/<id>`.
  CORS is open, so you can also open the HTML from `file://` and point it at the server.

Cost per request on one CPU core (Kimi-K2, 8×B300): single step ≈ 0.35 s, request mode ≈ 0.6 s,
6-point sweep ≈ 1.4 s, bisection (18 evals) ≈ 5 s. Repeat runs hit the lowering cache and are
faster. For many concurrent users, run several server processes behind a reverse proxy — Python
lowering holds the GIL, so one process serializes CPU-bound jobs.

**Interface: see `INTERFACE.md`** — the three configs, their schemas, and the rule that the
model only ever sees config.

## Three inputs, three schemas
1. **GPU config** — `gpuTilingPerfHWModel/schema.py`, 99 fields. `tilesight config --list`.
1b. **GPU config, slice form** — `gpuTilingPerfHWModel/slice_config.py`, 42 fields: shader slices (cores per slice,
   tensor cores per core, wave32 units, MMA tile M/N/K and its latency in cycles, shared memory
   and GPR shared-or-split), memory slices (HBM + L2 port + DMA port each, L2 may be 0),
   addressing (linear or N-KB interleave), the on-chip buffer and what is pinned on it, and gmem
   up/down read/write bandwidth. `derive()` gives TFLOP/s per tensor core and per shader core,
   whole-GPU PFLOP/s and every aggregate bandwidth; `to_hardware_spec()` translates it into the
   flat form. `tilesight gpu --file my_gpu.yaml --workload wl.yaml`.
2. **Workload config** — `model/workload.py`, 34 fields: one layer on one GPU (attention type and
   dims, FFN/MoE, datatypes, phase/batch/seq). `tilesight config --workload`.
3. **Tile policy** — part of the GPU config, not the workload (`gpuTilingPerfHWModel/schema.py`'s
   `compute.tile_policy.gemm`/`.attn`/`.overrides`): `auto` or a fixed tile. It's a modelling
   choice about how *this part* executes a GEMM/attention op, independent of which model runs on it.

## Guided web app
`tilesight serve` now opens a five-step wizard at `/` (the expert single-page UI moved to
`/expert`): pick a supported GPU or a custom one → edit and validate the hardware config →
fill in the workload → watch it run (progress bar plus a live log showing `file:function → op`,
so a stuck run is obvious) → results. The results page has the GPU architecture drawn **from the
config**, where the time goes by resource and by tensor, the operator table, and downloads:
per-cycle **Excel** (columns grouped into shader / L1+scratchpad / on-chip buffer / DMA / L2 /
HBM sections, scroll horizontally), CSV, and a **PDF report** with the grouped timeline.

## The hardware interface
A GPU is input, not model. `gpuTilingPerfHWModel/schema.py` declares all 99 config fields (15 sections; tags:
39 spec, 33 calib, 15 policy, 12 loss) and is the only place defaults live.
`tilesight config --list|--validate <file>|--md docs/CONFIG.md|--csv ref.csv`.
A test enforces that no engine/kernel/model file names a device or vendor and that every
preset validates.

## Inputs
- **Model** (`model/spec.py`): block-level YAML (`mha`, `gqa`/MQA, `mla`, `mlp`, `moe`,
  `norm`, raw `gemm`/`elementwise`) with `repeat` per layer group, or a HF config (presets:
  `kimi_k2.hf` MLA, `llama3_70b.hf` GQA, `llama2_7b.hf` MHA). Every attention block takes
  `impl: flash|naive`, `causal`, `sliding_window` (+ `absorb` for MLA). See
  `examples/custom_model.yaml`, `examples/attention_variants.yaml`.
- **Losses are opt-in**: every `capacity_derate` ships at 1.0, `efficiency.*` defaults to 1.0 when
  absent and `memory.queueing.coef` to 0. `--ideal` (or `HardwareSpec.lossless()`) turns every
  modelled loss off at once for a theoretical-peak baseline.
- **Per-unit latency** is configurable in cycles: `compute.mma_latency_cycles: 64` (one tile
  MMA), `sfu_latency_cycles`, `cuda_latency_cycles`, SMEM/TMEM `latency_cycles`, L2/HBM
  `latency_ns`, load-path issue overhead. Independent latency only costs pipeline fill;
  loop-carried latency (online softmax, accumulators) bounds the steady state.
- **Optional extra on-chip buffer**: add `memory.sram: {capacity_MB: 64, bandwidth_TBps: 15, …}`
  to any hardware file (or `--set`). Bytes that miss L2 are served there, and the share of a
  tensor that stays resident across calls skips HBM entirely. Policies: `policy: cache|pin`
  (+ per-class shares), `keep_intermediates`, `bypass_l2`, `prefetch`, `costream`.
  `memory.sram.stage: {share_blocks: N}` models panel-granularity async staging shared between
  shaders. `tilesight kernel … --addr --layout-a zorder` reports how a layout/swizzle spreads a
  wave's tiles over L2 slices and HBM ports (tile-granular, no cache lines).
  `tilesight buffer --model … --capacity 32768` prints the closed-form allocation
  (value/byte = HBM traffic ÷ footprint per class) and then searches policies × splits;
  `--capacities 0,128,…` sweeps the capacity curve.
- **Hardware** (`gpuTilingPerfHWModel/db/*.yaml`): NVIDIA **B300** (default), **B200**, **H200**; AMD **MI300X**,
  **MI325X**, **MI355X**, **MI450** (CDNA3/4/5 — `sms` = CUs, on-chip `smem` = LDS, `l2` = Infinity
  Cache/MALL, no TMA/TMEM/clusters, load paths `tdm`/`buffer_lds`/`lsu`). Every field is overridable by
  dotted path (`--set memory.ddr.bandwidth_TBps=12`). New on-chip memories / load paths
  become resource lanes automatically.
- **Run config** (`examples/run_decode.yaml`): phase, batch, seq, tp/dp/ep, dtypes, tile
  policy (`auto`, fixed, or per-op fnmatch overrides like `"*.experts_*": {bm: 64, bn: 256}`),
  `attn_impl: flash|naive` default for all attention blocks.

## Outputs
- Step time / TPOT, tok/s/GPU; request mode: TTFT, TPOT first/last/avg over the KV growth
  from prompt to prompt+output, end-to-end latency.
- **Where it is bound**: by resource class (tensor core, CUDA core, SFU, SMEM, TMEM,
  registers/spill, L2, DDR/HBM, load path, interconnect, latency, launch) and by
  resource:tensor (e.g. `ddr:load:expert_weight`, `ddr:load:kv_cache(latent)`,
  `latency:softmax(sfu)`), plus the occupancy limiter per kernel (`1/smem+regs`, `2/tmem`).
- Memory: weights / KV / activations / reserve, **peak** over the generation, paged KV,
  max concurrent requests per rank. CSV export.

## Snapshot: Kimi-K2 decode, 8× B300 (dp=8, ep=8, FP8), batch 256, 8K context, L2 BW scaled 2.56× DDR
| DDR TB/s | TPOT ms | tok/s/GPU | top limiter | DDR share |
|---|---|---|---|---|
| 4 | 54.5 | 588 | ddr | 86% |
| 6 | 38.8 | 824 | ddr | 80% |
| 8 (B300) | 31.1 | 1,030 | ddr | 73% |
| 12 | 24.0 | 1,331 | ddr | 54% |
| 16 | 20.6 | 1,557 | ddr | 46% |
| 24 | 17.9 | 1,792 | l2 (per-SM load cap) | 1% |

Above ~20 TB/s the wall moves to the per-SM global-load cap (`memory.l2.per_sm_max_GBps`,
a calibration placeholder), then latency/launch — raising HBM alone stops paying off.
**All `[calib]` numbers are placeholders until the calibration suite (docs/TASKS.md
Phase C) runs on real hardware; treat absolute values as indicative, trends as the result.**

Docs:
- a Chinese-language learning guide at the repo root, one level above this folder — how
  the model computes, what each config field does, how to read the output, hands-on exercises
- `CLAUDE.md` — how to work in this repo (for Claude Code)
- `docs/DESIGN.md` — equations, semantics, deviations from the paper
- `docs/TASKS.md` — ordered backlog with acceptance criteria (Phase H = items from research report 02)
- `docs/research/01_reimplementation_plan.md` — paper summary + re-implementation plan
- `docs/research/02_accuracy_tuning_presets_kimi_k3.md` — paper accuracy & caveats, "what to tune
  if inaccurate" guide, GB300 / H200 / MI450 specs, Kimi-K3 config, single-GPU vs system split
