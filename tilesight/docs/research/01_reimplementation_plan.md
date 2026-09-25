# Engineering Plan: Re-Implementing the TileSight GPU Performance Model From Scratch (Python front-end + C++ core), With Extended Configurable-Hardware Modeling

> Research report (deep-research output, Sep 2026). Source paper: arXiv:2607.22432v1,
> "TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters".
> The repo in this archive implements the core of this plan; see docs/DESIGN.md for what is built
> and where it deviates, and docs/TASKS.md for the remaining backlog.

## TL;DR
- **TileSight is a fully analytical, white-box tile-centric performance model with a small, closed set of equations (Eqs. 1–11) and two algorithms; it is re-implementable from the arXiv paper alone.** Build it as a thin Python front-end (configs, Triton/TileLang ingestion, sweeps, reporting) over a C++17 core (resource-vector cost engine, topological-order overlap scheduler, tile-reuse-distance SDCM cache model, α–β communication model, threaded batch evaluator) bound with **nanobind + scikit-build-core**.
- **The paper fully specifies the math and abstractions but leaves the rate-conversion formulas, prologue/epilogue closed forms, occupancy solver, D_T perturbation, routing granularity, and DSL parsers for you to design** — mark these as inferred and calibrate them from one-shot microbenchmarks (~minutes/architecture), targeting ~12% GEMM MAPE, ~1 pp L2-hit error, ~16% distributed wMAPE, ~14% vLLM wMAPE.
- **Extend beyond the paper by making hardware a data-driven graph of memory levels + data paths + compute units with instruction tables**, so multiple load paths (TMA/cp.async/LSU, AMD buffer_load→LDS), extra on-chip memories (Blackwell TMEM, distributed shared memory, large accelerator SRAM), detailed MMA instruction tables (wgmma/tcgen05/MFMA, per-datatype throughput, operand sourcing, 2-CTA), and partitioned-cache bandwidths are added by YAML alone.

## 1. Paper summary — modeling details and equations

TileSight (Mo, Cheng, Wang, … Fan; Imperial/PKU/SJTU/Tile-AI/MSR/Edinburgh; 24 Jul 2026) is a first-principles, white-box, tile-centric analytical GPU model. Reference implementation ≈6K lines of Python; supports NVIDIA and AMD. Code "will be open-sourced upon publication".

**Core thesis.** The tile is both a programming primitive (Triton, TileLang, CUDA Tile, CuteDSL) and the analysis primitive: deterministic (fixed per-tile resource use), composable (intra → inter → cross-device), portable (A100, H100/H200, B200, RTX PRO 6000 Blackwell "B6000", MI210).

**Input → tile execution plan (§3.1).** A workload (tiled GEMM, fused attention, all-gather+GEMM, MoE routing) with tensors/placements fixed but schedule unspecified is lifted to a plan exposing tile shape, loop/reduction order, block swizzle, software-pipeline depth, resident blocks per SM, distributed partitioning, collective implementation.

**Intra-tile resource vector (§3.2), Eq. (1):**
u(o) = ⟨t_TC, t_CUDA, t_SFU, t_TMEM, t_SMEM, t_L1.5, t_L2, t_DDR, t_Net⟩ from operation + footprint + placement + calibrated rates. Tiles on different pipelines overlap; same-pipeline tiles serialize. Algorithm 1 ("Hierarchical Tile-Pipeline Evaluation"): build grid/order/swizzle, tensor accesses, action DAG, pipeline params; for distributed plans partition → infer remote accesses → decompose into stages → logical exchanges → route → α–β stage time → Net lane; then CacheTraffic → ResidentTilesPerSM → d = stages×resident−1 → PipelineEnvelope → WaveAggregate.

**Pipeline envelope (§3.4).**
- Eq. (2): T = T_pro + max(N−d, 0)·T_steady + T_epi (recursive: waves → K-loop → inner actions)
- Eq. (3): d = stages × resident_tiles_per_SM − 1
- Eq. (4): T_steady(σ) = max_r Σ_{o∈σ} u_r(o)
- Eq. (5): T_steady = min over topological orders σ of the action DAG (MLA decode: 11 actions → 132 legal orders)
Tail waves use fewer active SMs, giving each a larger share of L2/DDR bandwidth.

**Cache traffic via tile reuse distance (§3.5).**
- Eq. (6): key(x,R) = Linearize(x_d | d ∉ R)
- D_T = distinct tile-blocks between two accesses to the same block (~64× fewer entries than 128 B lines)
- Eq. (7) binomial SDCM: P(h|D_T) = Σ_{a=0}^{A−1} C(D_T,a)(A/B_T)^a((B_T−A)/B_T)^{D_T−a}
- Eqs. (8–9) Gaussian: μ = D_T·A/B_T, σ² = D_T·(A/B_T)(1−A/B_T)
- Eq. (10) Zelen–Severo Φ approximation
- K-axis sampling; two-level cascade (L1.5 per SM group, L2 global, DDR residual); D_T perturbed for nondeterminism.

**Cross-device (§3.6).** Remote tile accesses gain a Net entry; collectives decompose into stages.
- Eq. (11): T_k = max_{(s,d,b)∈E_k} Σ_{l∈P_sd} α_l + max_{l∈L} β_l·B_{l,k}

**Calibration (§3.8, Table 3)** — spec / measured:
A100 (108 SMs; FP32 19.5/19.0; TC 312/299; SFU 2.4/2.4; L2 3.2; DDR 1.9/1.7);
H200 (132; 61.8/49.5; 989/928; 3.9/4.1; L2 9.2; DDR 4.8/4.2);
B6000 (188; 117/88.6; 468/433; 7.3/6.7; L2 7.6; DDR 1.8/1.4);
B200 (148; 74.5/57.7; 2382/2185; 4.7/4.5; L2 20.5; DDR 8.0/7.0);
MI210 (104; 45.3/34.4; 181/167; 2.8/1.1; L2 4.8; DDR 1.6/1.4).
Effective L2 cliffs: B200 dual-die smeared at ~83 MB (L1.5 ~22.5 TB/s); B6000 sharp at ~130 MB.

**Evaluation (§5).** 703 GEMM shapes (after filtering stream-K and SIMT fallback) → 12.35% pooled MAPE vs PipeWeave 21.97%, NeuSight 32.95%, Roofline 33.85%, GenZ 34.89%. FA-3 vs NCU (Table 4). L2 ~1 pp MAE over 4,680 persistent-kernel cases. 304 fused distributed kernels ≤32 GPUs → 16.18% wMAPE. 166 vLLM decode configs → 13.52% wMAPE.

**Limitations.** Validated as a TileLang cost model only on selected GEMM workloads. Review (Pith) flags: no code/data; stream-K/SIMT filtered; deep-K bias; cache claim possibly overstated; no run-to-run variance.

## 2. Gaps / ambiguities (must be designed and flagged)
Stated: Eqs. 1–11, Algorithm structure and subroutine names, lanes, placements, Table 3.
Inferred: rate formulas (time = work / effective rate); T_pro/T_epi closed forms; occupancy solver; SDCM cascade grouping; D_T perturbation; routing; SFU/CUDA accounting for softmax; Triton/TileLang ingestion. The reviewer gaps (stream-K, SIMT fallback, deep-K bias, variance) are design opportunities.

## 3. System architecture — Python config + C++ core
- Python: pydantic/YAML configs (hardware, workload, plan), frontends (Triton, TileLang, manual), builders (GEMM, FA, MLA, AG-GEMM, MoE), sweeps/autotune, reports, calibration and validation harness.
- C++: resource-vector cost engine; overlap scheduler + envelope; reuse-distance cache model; α–β comm model; plan enumerator with a thread pool (GIL released).
- Binding: nanobind (smaller binaries, lower call overhead than pybind11, zero-copy ndarray/DLPack) built with scikit-build-core + CMake; one `evaluate_batch` call per sweep; deterministic seeded jitter.

## 4. Extended hardware-resource modeling
Generalize the fixed 9-lane vector to named lanes auto-derived from a hardware graph:
- **Multiple load paths** (TMA / cp.async / LSU; AMD buffer_load→LDS direct, which saves ~100 VGPR/wave on CDNA4): choose or split bytes between paths, each with bandwidth, latency, issue limit.
- **Extra on-chip memories**: Blackwell TMEM (256 KB/SM, tcgen05-only, accumulators resident), distributed shared memory across clusters (Hopper DSM SM-to-SM latency ~180 cycles, ~3.3 TB/s at cluster size 2), large-SRAM accelerators, AMD LDS — all as `MemoryLevel` nodes.
- **Tensor-core instruction tables**: wgmma (>95% of Hopper peak vs ~63% for legacy mma), tcgen05.mma (~11-cycle latency, SMEM/TMEM operands, cta_group::2), AMD MFMA shapes and block-scaled FP8/FP6/FP4; per-datatype throughput, operand sourcing, issue latency.
- **Cache bandwidth parameters**: L1/SMEM ~128 B/clk/SM on Hopper, partitioned L2 (H100 2×25 MB; B200 4 partitions), HBM per product; SDCM uses effective calibrated capacity.

## 5. Calibration & validation plan
Microbenchmarks (compute rates per dtype incl. SFU; bandwidth-vs-working-set sweeps; per-path issue limits; per-link α/β via NCCL). Validation vs Nsight Compute, CUPTI, cutlass_profiler / Triton do_bench; metrics MAPE / pp-MAE / wMAPE. Add stream-K and SIMT-fallback templates, a deep-K check, and median+IQR reporting over ≥30 repeats.

## 6. Roadmap (original plan)
Phase 0 scaffolding → 1 single-GPU GEMM core → 2 cache model → 3 fused kernels + diagnosis → 4 distributed → 5 frontends/autotune/reporting → 6 vLLM + gap coverage + extended HW DB (~24 weeks for one engineer). The shipped scaffold already covers large parts of phases 0–3 and 5; see docs/TASKS.md.

## Caveats
The paper is v1 without released code; constants here will differ from the authors'. Headline accuracy excludes stream-K/SIMT paths; the ~1 pp cache figure is over persistent sweeps. Some hardware constants have source-dependent spread (B200 HBM 8 vs 7.7 TB/s by SKU; L2 capacity vs effective cliff). Prefer your own measured values over datasheets, as the paper does.
