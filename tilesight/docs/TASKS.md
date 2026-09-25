# TASKS — ordered backlog for Claude Code

Status legend: ✅ done in the scaffold · ⬜ todo. Each task: goal → files → acceptance.
Always: tests first, both backends green, update DESIGN.md if a formula changes.

## Phase 0 — scaffold ✅
✅ Hardware YAML DB (B300, B200, H200) with `[spec]/[calib]` tags, dotted overrides, lanes
✅ Engine (reference + C++ parity), L2 reuse-distance/SDCM model (Python + C++)
✅ GEMM / batched / grouped GEMM, attention decode (split-KV, MLA) + prefill, elementwise, collectives
✅ Tile configs, per-op overrides, auto search incl. split-K, cluster multicast, 2-CTA MMA
✅ Model spec (block level) + HF importer (Kimi-K2 / DeepSeek-V3 / Llama-style GQA)
✅ Memory report, bottleneck mix, CLI (`run`, `sweep`, `need`, `dump-model`)
✅ DSE sweep + bisection, B300 DDR-bandwidth experiment
✅ Request mode: prompt_len/output_len, paged KV, TPOT curve, TTFT, E2E, peak memory, max concurrency
✅ Fine-grained bound attribution (resource:tensor, critical-path latency node), occupancy
   limiter incl. registers (GPR) and TMEM, register-spill traffic
✅ Attention families mha / gqa (mqa) / mla with `impl: flash|naive`, sliding window,
   qk_norm, partial rope, MLA absorb switch; presets llama2_7b (MHA), llama3_70b (GQA), kimi_k2 (MLA)

## Phase A — engine fidelity (paper-faithful)
⬜ **A1 Topological-order search (Eq. 5).** Enumerate legal orders of the body DAG (Kahn
  with pruning, cap 10k), evaluate an issue-order-aware steady time per order, take the
  min. Files: model/engine/reference.py, gpuTilingPerfHWModel/model/cpp/src/engine.cpp. Accept: golden test — an 11-action
  MLA-decode DAG yields exactly 132 orders; parity test extended; no regression on GEMMs.
⬜ **A2 Paper envelope mode.** `Kernel.envelope = "fill" | "paper"`; paper: `T = T_pro +
  max(N−d,0)·R + T_epi`, `d = stages·resident−1`. Accept: both modes tested; deep-K sweep
  (K=1k..64k) shows the paper mode's bias vs fill mode in a notebook/report.
⬜ **A3 Per-block iteration counts.** Causal attention and ragged MoE groups have
  different `iters` per block. Represent as a histogram `[(iters, count)]` per kernel;
  waves pick blocks in issue order. Accept: causal prefill time within 2% of an exact
  per-block simulation in a unit test.
⬜ **A4 Stream-K / persistent GEMM.** Plan template with work units split over SMs + fixup.
  Accept: removes the tail-wave cliff on shapes with blocks ≈ 1.1×SMs.
⬜ **A5 Tile scheduler / persistent kernels** (CLC on Blackwell): blocks = SMs·resident,
  work queue; epilogue/mainloop overlap. Accept: toggle in TileConfig, test on 8K GEMM.
⬜ **A6 D_T perturbation** (paper §3.5.3): seeded jitter + cross-tensor aging, off by default.

## Phase B — cache model
⬜ **B1 Cross-wave reuse.** Sample the first two waves; weights re-read by later waves can
  hit when N·K fits L2. Accept: prefill GEMM with N·K·b ≤ 0.5·L2 shows DDR miss < 20%.
⬜ **B2 Die/partition cascade.** `dies: 2` → per-die L2 halves, far-L2 hits pay extra
  latency and consume NV-HBI bandwidth (new lane `d2d`). Accept: B300 L2-BW-vs-working-set
  curve reproduces a smeared cliff (paper Fig. 2 style) in a plot script.
⬜ **B3 Sequence generation in C++.** Move key-sequence construction for GEMM/attention to
  C++ (`gemm_misses(M,N,K,tile,...)`). Accept: `run` on Kimi decode < 0.15 s with C++.
⬜ **B4 NCU validation hooks.** `validate/ncu.py` parses `lts__t_sector_hit_rate` etc.

## Phase C — calibration suite (needs a GPU)  → writes `[calib]` fields
⬜ C1 DDR stream BW (read/write/copy, vs #SMs → also yields `per_sm_max_GBps`)
⬜ C2 L2 BW vs working set (effective capacity, assoc fit) ⬜ C3 L2 / DDR latency (p-chase)
⬜ C4 TC peak per dtype via CUTLASS/DeepGEMM microkernels (incl. 2-CTA) ⬜ C5 SFU exp rate
⬜ C6 TMA vs LSU per-SM issue rates ⬜ C7 launch overhead with/without CUDA graphs
⬜ C8 NCCL/NVSHMEM alpha-beta per message size (allreduce, all-to-all)
Accept: `python -m tilesight.calibrate --gpu 0 --out gpuPresets/<gpu>.calib.yaml`; loader
merges `*.calib.yaml` over the base file; every `[calib]` field covered.

## Phase D — validation harness
⬜ D1 GEMM suite (≥500 shapes incl. decode M≤256, **no filtering** of stream-K/SIMT
  paths) vs cuBLASLt/CUTLASS/DeepGEMM; report MAPE with median and IQR over ≥30 runs.
⬜ D2 FlashMLA / FlashInfer decode + FA prefill vs measurement.
⬜ D3 End-to-end: SGLang/vLLM Kimi-K2 decode TPOT at several batch/seq; wMAPE.
Targets (paper): GEMM MAPE ≈12%, distributed ≈16% wMAPE, serving ≈14% wMAPE.

## Phase E — model layer
⬜ E1 Fused MoE kernels (DeepEP-style dispatch+GEMM, fused gate_up+act) as alternatives
⬜ E2 Overlap scheduler (two-batch overlap / comm streams) replacing `comm_overlap`
✅ E3 Hierarchical collectives (reduce-scatter intra / all-reduce inter / all-gather intra)
✅ linked DSE knobs (`sms` scales the compute peaks with it)
⬜ E4 MTP / speculative decoding (k draft tokens → M×k in verify step)
⬜ E5 Quantization kernels (per-token FP8/NVFP4 act quant, block scales) + scale traffic
⬜ E6 Chunked prefill / mixed prefill+decode batches
⬜ E7 Expert-load imbalance (Zipf or measured routing histogram) → max over EP ranks
⬜ E8 Paged KV (page size effects), FP8/FP4 KV cache, sparse attention (NSA/DSA) block
⬜ E10 More attention impls: `sdpa_mem_efficient` (xformers-style, fused but no TMA/TMEM),
   FlashAttention-2 vs -3 vs -4 presets (different consumers/stages/SFU tricks), naive with
   causal-skip, FlashDecoding vs FlashInfer split heuristics. Accept: each is a named `impl`
   with a test comparing it to flash/naive on a 16K prefill.
⬜ E11 Hybrid models: alternating full / sliding-window layers (gpt-oss, Gemma-3) from HF
   `layer_types`; linear-attention / Mamba blocks with state instead of KV
⬜ E9 Training step (fwd+bwd+optimizer, activation recompute) — optional

⬜ E12 Request distributions: P/O histograms (from traces) -> expected TTFT/TPOT/E2E +
   p50/p90/p99, continuous batching steady state (mixed KV lengths in one decode batch)
⬜ E13 Memory timeline with KV eviction/offload (CPU/NVMe tiers as extra memory levels)

## Phase F — performance
⬜ F1 Batch autotune: lower all candidates, one `evaluate_batch` call (GIL released)
⬜ F2 Parallel DSE sweeps (process pool over values) ⬜ F3 Lowering cache keyed on fingerprint (✅ basic)

## Phase G — extended hardware resources
⬜ G1 **Instruction tables**: `compute.mma: [{dtype, m, n, k, cta_group, cycles, operand_src}]`
  replacing scalar TFLOPS; TC time from issue count × cycles. Accept: H200 vs B300 small-N
  efficiency difference reproduced (Hopper SMEM-bound below N=128).
⬜ G2 **DSMEM lane**: cluster-shared SMEM (Hopper/Blackwell) with bandwidth/latency; use in
  A-tile sharing across a cluster (cluster_n) and in split-KV combine.
⬜ G3 Operand sourcing: SS vs RS (A from registers/TMEM) changes SMEM reads.
⬜ G4 AMD preset (MI355X: MFMA table, 160 KB LDS, buffer_load→LDS path, Infinity Fabric).
⬜ G5 Large-SRAM accelerator preset (weights resident on chip; DDR lane only for KV).
⬜ G11 Buffer extras worth modelling once it is large: layer-ahead prefetch (stage layer L+1
   weights while computing L — only helps if HBM has spare bandwidth, so it needs a real
   overlap rule), cross-wave B-tile reuse (today only within-wave reuse is simulated), MoE
   hot-expert caching under skewed routing, collective payload staging, and an eviction
   policy per class instead of a static share
✅ DMA vs LSU engine attributes (issue slots, staging registers, smem_direct, multicast gating)
✅ tile-granularity address/swizzle analysis over a wave window (L2 slices, HBM ports, imbalance)
✅ mega-tile staging through the buffer; L2-vs-buffer fixed-budget trade (`l2_tradeoff`)
✅ L2 blocks/ports/sectors, outstanding-request (MSHR) ceiling, DMA destination & engine layout,
   L1 ownership config; Figure 3(d)(e)(f) as an HTML/JSON artifact; Chinese learning guide
⬜ G12 Memory-system depth, next: queueing delay as a lane approaches saturation (M/D/1-style
   inflation instead of a hard cap), sector-granularity waste for strided tiles, L2 write-allocate
   and read-modify-write for split-K, atomics, per-die near/far L2 on multi-die parts, HBM
   row-buffer/bank-group effects, DVFS under sustained tensor-core load. Calibrate blocks,
   ports and per_sm_lines with a stride + concurrency microbenchmark (Phase C).
✅ configurable address→unit map (interleave / range / hash, per-side ports & granularity, 48-bit)
   with a dump (`tilesight addrmap`, CSV export)
✅ per-GPU memory map (`tilesight memmap`) + tile addresses in trace/CSV/Excel (slice, port)
⬜ G8b Address model, next steps: allocator realism (reuse, fragmentation, paged KV pages), imbalance fed
   back per lane instead of one global efficiency factor, bank conflicts inside SMEM/LDS
⬜ G8 Address model: tensor layout + swizzle + interleave granularity -> L2 set index, SMEM/LDS
   bank, HBM channel; enables set/bank/channel conflict terms and per-port HBM imbalance
⬜ G9 DMA/copy-engine lane: H2D/P2P transfers overlapping compute, with an overlap rule
⬜ G10 Tile-to-SM assignment policies (raster, swizzle, persistent/CLC work queue) as an explicit
   scheduler, so the Excel/timeline can show which block ran where instead of one representative SM
✅ hardware config schema + validation + generated reference (`tilesight config`), enforced
   GPU/model boundary (no device names in the model, defaults only in the schema)
✅ every loss knob is explicit config with a no-loss default (derates 1.0, efficiency 1.0 when
   absent, queueing 0) + `--ideal` / `lossless()` theoretical-peak baseline
✅ L1 as a modeled level (private per SM / shared per cluster, derated capacity, only for paths
   that go through it) + unified `capacity_derate` on every cache level
⬜ G7 Register-file bandwidth lane (`rf`) for CUDA-core-heavy epilogues/softmax, and
   L1 / texture as a separate cache level (per-SM, with its own reuse model)
⬜ G6 Power/energy lane (pJ/byte, pJ/flop) → tokens/J in reports and DSE.

## Phase W (updated)
✅ slice-based GPU config (`gpuTilingPerfHWModel/interfaceAndRun/slice_config.py`) with derived TFLOPS/PFLOPS/bandwidths and a
   translation into the flat config; Kimi-K3 10-layer workload preset; INTERFACE.md
✅ model organised in three blocks (shader slice / gmem / memory), L1 as a per-slice resource,
   `by_domain()` roll-up everywhere; flash-vs-naive comparison
⬜ W7 Show the best schedule found by the tile search next to the current one in the trace
   (the paper's cost-model-driven selection, as a diff on the timeline)
✅ workload config schema (single-GPU layer) + validation + `tilesight config --workload`
✅ five-step wizard UI (GPU -> gpuTilingPerfHWModel config -> workload -> run with live log -> results)
✅ GPU architecture diagram drawn from the config; PDF report; Excel columns grouped by unit
⬜ W6 Wizard extras: save/load a whole run (gpuTilingPerfHWModel + workload) as one file, compare two runs
   side by side, and show the Figure 3(d)(e)(f) panels inline on the results page

## Phase W — web UI / serving (partly done)
✅ stdlib server with async jobs + progress, self-contained HTML UI, gpuTilingPerfHWModel overrides, custom model YAML
✅ buffer policies (cache/pin shares, keep_intermediates, bypass_l2, prefetch, costream),
   closed-form allocation + policy search + capacity curve (`tilesight buffer`)
✅ configurable extra on-chip shared buffer (`memory.sram`, e.g. 64 MB A/B staging) with
   cross-call residency; coloured per-cycle Excel export (`--xlsx-out`, `/api/xlsx`)
✅ single-GPU kernel mode (pick card + kernel + shapes -> latency, % of peak, occupancy, tile ranking)
⬜ W1 Result caching keyed by (model, gpuTilingPerfHWModel fingerprint, run config) so repeated configs return instantly
⬜ W2 Multi-process workers (`--workers N`, socket reuse) for concurrent users; queue depth in the UI
✅ Steady-state timeline (Fig. 3e) in the web UI (SVG, µs/cycle toggle) and CLI (ASCII);
   full text trace export (`--trace-out`, download button) with cycles and wave decomposition;
   per-cycle CSV (`--csv-out`, `/api/csv`) and a wave-level machine view instead of N SM rows
⬜ W3b Timeline extras: draw prologue/epilogue and the tail wave, overlay the `blocks_per_sm`
   resident blocks as separate rows, emit Chrome-trace (`chrome://tracing`/Perfetto) JSON,
   per-action tooltips with the bound reason, and a timeline for any op picked in model mode
⬜ W3a Kernel mode extras: roofline chart (achieved vs peak/HBM), shape sweep (e.g. M = 1..4096),
   side-by-side cards (b300 vs h200 vs mi450) for the same kernel
⬜ W3 Compare mode: run N configs (e.g. 3 GPUs x 2 dtypes) and show a diff table + chart
⬜ W4 Save/share: permalink encoding the config in the URL; CSV/JSON download buttons
⬜ W5 Auth/limits if exposed beyond the LAN (token header, per-IP job cap, max sweep points)

## Phase H — from research report 02 (docs/research/02_accuracy_tuning_presets_kimi_k3.md)
⬜ **H1 GB300 preset** `gpuPresets/gb300.yaml` (NVL72 domain 72, NVLink5 1.8 TB/s bidir); resolve the
  BF16/FP8 peak conflict (2250/4500 vs 2500/5000) with a probe; keep `[calib]` tags.
🟡 **H2 AMD presets added** (mi300x/mi325x/mi355x/mi450: LDS as on-chip, MALL modeled as `l2`,
  no TMA/TMEM/clusters, buffer_lds/tdm load paths, MFMA `tc_min_m` 16). **Still to do:**
  wavefront-64 occupancy math, VGPR/AGPR split, per-XCD L2 as a level separate from MALL,
  MFMA instruction table, rocprof validation, `default_cache_mode` flag. Original task text: `memory.onchip.lds` instead of smem/tmem,
  `load_paths.buffer_lds` and `load_paths.tdm` (Tensor Data Mover), wave64 + shared VGPR/AGPR
  occupancy, MFMA instruction table, extra cache level MALL between L2 and HBM (new shared lane
  `mall`), per-XCD L2 partitions. Accept: runs Kimi-K2/K3 decode; `default_cache_mode` flag.
⬜ **H3 Kimi-K3 preset**: `kimi_k3.hf.json` + importer support for `kimi_linear` configs:
  new block `linear_attention` (KDA: constant-size state, no KV growth, short conv k=4),
  per-layer attention pattern (full MLA on listed layers), latent MoE (routed hidden 3584,
  down/up projections), 2 shared experts, MXFP4 experts (`expert_dtype: fp4` + scale bytes),
  dense FFN 33792. Accept: KV bytes/token counts only full-attention layers; request mode
  shows flat KDA state memory.
⬜ **H4 Split into subsystems**: `single_gpu/` (lowerings, OpLatencyModel, OpLatencyTable,
  validate_ncu) and `system/` (parallelism, collectives, overlap, moe_balance, serving,
  disagg, validate_vllm) with the OpSpec/OpResult contract (bound + confidence fields);
  keep `model/` feeding both. Accept: system layer never imports kernels/*; table cache keyed
  by (gpuTilingPerfHWModel fingerprint, calib hash, version).
⬜ **H5 Accuracy guardrails from the tuning guide**: deep-K L2 hit clamp option, measured-SFU
  enforcement (warn when using spec SFU), per-symptom diagnostic CLI
  (`tilesight diagnose --ncu report.csv`) that maps residuals to the parameters in report 02 §2.
⬜ **H6 Cross-GPU example**: `examples/kimi_k3_gb300_h200_mi450.py` comparing decode/prefill/request
  on the three presets with bound breakdowns.
