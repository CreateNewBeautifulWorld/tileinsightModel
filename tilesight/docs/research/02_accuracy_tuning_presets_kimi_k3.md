# Re-implementing TileSight: Accuracy, Tuning, Hardware Presets (GB300/H200/MI450), Kimi-K3, and a Single-GPU/System Split

> Research report (deep-research output, Sep 2026). Numbers marked [spec] are vendor figures,
> [calib] must be measured, [rumor]/[conflict] need verification before use.
> Status in this repo: the tuning guide and specs below are NOT yet wired into code —
> see docs/TASKS.md Phase H for the implementation tasks derived from this report.

## TL;DR
- TileSight reports **12.35% pooled GEMM MAPE** (A100/H200/B200/B6000), **~1 pp L2 hit-rate error**, **~2.7% FA-3 latency error vs NCU**, **16.18% wMAPE on fused distributed kernels**, **13.52% wMAPE on vLLM decode**. Main documented weakness: optimistic L2 hit rate in the deep-K / high-occupancy regime (82% predicted vs 43% measured on one H200 case).
- Presets are largely spec-derivable but need calibrated efficiency factors: GB300/B300, H200 SXM, AMD MI450/MI455X (CDNA 5). Kimi-K3's config: hidden 7168, 93 layers, 96 heads, 896 experts top-16, MLA + KDA linear attention, MXFP4 weights.
- Split the model layer into a **single-GPU op-latency subsystem** and a **system subsystem**, joined by a versioned per-op latency contract.

## 1. TileSight accuracy claims & caveats

### 1.1 Headline numbers
Calibrated once per architecture (bandwidth/throughput/latency sweeps in minutes, short GEMM probes in seconds), then analytical.
- **GEMM:** 12.35% pooled MAPE over 703 BF16/FP16 shapes (cutlass_profiler ground truth, stream-K and SIMT fallback filtered). Baselines: PipeWeave 21.97%, NeuSight (retrained) 32.95%, Roofline 33.85%, GenZ 34.89%.
- **Per GPU (Fig. 5):** B6000 5.2%, H200 10.4%, B200 14.9%, A100 18.7%, MI210 23.4%. NeuSight only wins on A100 (in its training set).
- **MI210 (CDNA2):** 23.4%, still best (PipeWeave 25.5%, NeuSight 26.4%, Roofline 38.8%, GenZ 40.4%); ran in default cache mode because Composable Kernel exposes no rasterization/swizzle control.
- **L2 hit rate (4,680 persistent-kernel cases):** A100 1.46 pp MAE; H200 0.88; B200 1.05; B6000 0.78.
- **FA-3 vs NCU (Table 4, H100, batch 1, 64 heads, d=128):** NCU 5.58 ms / L2 hit 96.50% / L2 util 38.66% / SMEM 51.14% / TC 74.78% / SFU 38.58%; TileSight 5.73 ms / 95.26 / 35.72 / 43.13 / 70.30 / 35.42. Whole-kernel only — no per-op (QK / softmax / PV) latency rows.
- **Distributed (≤32 GPUs, 304 kernels):** 16.18% wMAPE; pure collectives 12.22%; fused compute-comm 14.83%.
- **vLLM decode (166 configs):** 13.52% wMAPE; MoE 10.35%; per machine 7.5–18.0%. PipeWeave 31.84% on dense only.
- **Cost-model use:** keeping the predicted top-5% TileLang schedules prunes 95% of candidates and reaches 99.66% of exhaustive-search best.

### 1.2 Calibration (Table 3)
| GPU | SMs | FP32 spec/meas | TC FP16 spec/meas | SFU spec/meas | L2 TB/s meas | DDR spec/meas |
|---|---|---|---|---|---|---|
| A100 | 108 | 19.5 / 19.0 | 312 / 299 | 2.4 / 2.4 | 3.2 | 1.9 / 1.7 |
| H200 | 132 | 61.8 / 49.5 | 989 / 928 | 3.9 / 4.1 | 9.2 | 4.8 / 4.2 |
| B6000 | 188 | 117 / 88.6 | 468 / 433 | 7.3 / 6.7 | 7.6 | 1.8 / 1.4 |
| B200 | 148 | 74.5 / 57.7 | 2382 / 2185 | 4.7 / 4.5 | 20.5 | 8.0 / 7.0 |
| MI210 | 104 | 45.3 / 34.4 | 181 / 167 | 2.8 / 1.1 | 4.8 | 1.6 / 1.4 |

TC measured ≈ 90–94% of spec; DDR ≈ 85–90%. **SFU diverges most** (MI210 1.1 vs 2.4) → always use measured SFU. L2 effective capacity from bandwidth-vs-working-set sweeps: B200 L1.5 22,465 GB/s, L2 20,549 GB/s, DDR 7,407 GB/s, smeared cliff ~83 MB; B6000 L2 7,434 GB/s, DDR 1,563 GB/s, sharp cliff ~130 MB.

### 1.3 Criticisms
- **Deep-K optimistic cache bias:** M=N=8192, K=28672 on H200 → 82% predicted vs 43% measured L2 hit (39 pp); flows into the envelope when memory-bound. The paper's own optimization workloads sit in this regime.
- Cache headline is a mean; stream-K/SIMT filtered; no run-to-run variance; calibrated (not vendor) inputs. An independent paper (PASCAL, arXiv:2609.10515) reports TileSight's cache mode at 44.79% MAPE / 2.80 pp on a GB10 test set — the SDCM cache model is the weakest link.

## 2. If predictions are inaccurate — what to tune

| Symptom | Likely root cause | What to tune | Measure with |
|---|---|---|---|
| Small-M decode GEMM off | tail wave / occupancy; launch overhead; tc_min_m padding | fill term, resident tiles, `compute.tc_min_m`, `runtime.launch_overhead_us` (CUDA graphs) | NCU `launch__waves_per_multiprocessor`, `sm__warps_active`; graph-replay timing |
| Large GEMM off by a constant | TC rate at spec instead of measured | `efficiency.tc` | cutlass_profiler peak probe; `sm__pipe_tensor_op_*` |
| Error grows with K | optimistic L2 hit rate | `memory.l2.effective_capacity_MB`, `assoc`, cascade, hit-rate clamp | `lts__t_sector_hit_rate.pct`; BW-vs-working-set sweep |
| Attention prefill off | overlap/stages/consumers assumptions, SMEM rate | `AttnTileConfig.stages/consumers`, SMEM lane | NCU tensor pipe, SMEM throughput |
| MLA decode off | SFU rate, TMEM traffic, recurrence | measured `compute.sfu_tops`, TMEM rates, consumers | SFU pipe counters (`xu`), TMEM counters |
| L2 hit rate off | traversal order / swizzle / capacity | swizzle, effective capacity, cascade | `lts__t_sector_hit_rate`, reuse-distance histogram |
| Tail-wave shapes off | active-SM bandwidth share | `memory.l2.per_sm_max_GBps`, wave decomposition | per-wave timeline, `sm__cycles_active` |
| MoE grouped GEMM off | expert imbalance, grouped lowering, kernel choice | imbalance factor (TASK E7), tile plan, kernel impl | per-expert token counts; grouped-GEMM probe |
| Collectives off, small msgs | α | `network.*.alpha_us` | nccl-tests / rccl-tests latency sweep |
| Collectives off, large msgs | β, topology | `network.*.bandwidth_GBps`, domain size | nccl-tests bus bandwidth 1–16 GB |
| End-to-end serving off | overlap, KV, batching | `comm_overlap`, KV/page settings, batching model | vLLM/SGLang TTFT/TPOT traces |
| AMD off | no TMA/TMEM; LDS; wave64; MALL | LDS lane, buffer_load→LDS path, VGPR/AGPR occupancy, XCD L2 + Infinity Cache level | rocprofv3 `MfmaUtil`, `VALUBusy`, `OccupancyPercent`, LDS bank conflicts |

Priorities: (a) calibrate measured SFU and TC rates; (b) validate/clamp L2 effective capacity against a real sweep before trusting decode/attention or deep-K GEMMs.

## 3. Hardware presets (YAML-ready)

### 3.1 NVIDIA GB300 / B300 (Blackwell Ultra)
```yaml
gb300_b300:
  arch: blackwell_ultra          # [spec] dual-die, NV-HBI, 208B transistors
  sms: 160                       # [spec] full config; varies by SKU
  boost_clock_ghz: ~2.6          # [spec]
  tdp_w: 1400                    # [spec]
  dense_tflops:
    nvfp4: 15000                 # [spec]
    fp8_fp6: 5000                # [spec]  (other sources list 4500 — verify)
    fp16_bf16: 2500              # [spec]  (other sources list 2250 — verify)
    fp32_cuda: 80                # [spec]
    int8: 330                    # [spec] de-emphasized
  sfu: needs_calib               # ~2x Blackwell exp throughput
  hbm: {type: HBM3e_12hi, capacity_gb: 288, bandwidth_tbs: 8.0}   # measured likely ~7.0 [calib]
  l2_mb: needs_calib
  smem_per_sm_kb: 228            # [calib]
  tmem_per_sm_kb: 256            # [calib]
  nvlink5_gbs: 1800              # [spec] bidirectional per GPU
  domain: nvl72                  # [spec] 72 GPUs, 130 TB/s NVLink domain
```

### 3.2 NVIDIA H200 SXM
```yaml
h200_sxm:
  sms: 132
  clock_mhz: {default: 1830, max: 1980}
  dense_tflops: {bf16_fp16: 989, fp8: 1979, fp32_cuda: 67}   # measured TC 928 [calib]
  sfu: {spec: 3.9, meas: 4.1}
  hbm: {type: HBM3e, capacity_gb: 141, bandwidth_tbs: 4.8}   # measured 4.2 [calib]
  l2_mb: 50
  smem_per_sm_kb: 228
  nvlink_gbs: 900
  l2_bw_tbs_meas: 9.2
```
wgmma reaches >96% of BF16 peak for N≥64 (vs ~63% for legacy mma.sync).

### 3.3 AMD Instinct MI450 / MI455X (CDNA 5, Helios)
```yaml
mi455x:
  arch: cdna5                    # 8x XCD (N2) + fabric/cache + IO dies
  transistors_b: 320
  wgps: 256
  xcds: 8
  engine_clock_ghz: 2.4
  tdp_w: ~900                    # [leak]
  dense_pflops:
    fp4_mxfp4: 40                # [spec]
    fp8_mxfp8: 20                # [spec headline]; one deep-dive lists ~3.2 [conflict]
    fp32_fp16_vector_tflops: 315 # [spec]
  hbm: {type: HBM4, capacity_gb: 432, bandwidth_tbs: 23.3}    # 19.6 was the 2025 preview
  l2_mb: 192                     # per-XCD partitioned
  infinity_cache_mb: needs_confirm   # MALL (256 MB on MI350)
  lds_per_cu_kb: 160             # CDNA4 value; confirm CDNA5
  vgpr_file: 512                 # shared VGPR+AccVGPR
  wavefront: 64
  tensor_data_mover: true        # async global<->LDS (TMA analogue)
  scale_up: ualink_over_ethernet # 3.6 TB/s per GPU
  helios_rack: {gpus: 72, hbm4_tb: 31, fp4_exaflops: 2.9, scale_up_tbs: 260}
```
SKUs: MI450 (volume, same 432 GB), MI455X (Helios flagship), MI430X (HPC/FP64), MI440X (8-GPU server).

**Model changes needed for AMD:** drop TMEM lane; LDS replaces SMEM (bank-conflict-aware); Tensor Data Mover as async LDS load path; buffer_load→LDS direct path; wave64 occupancy; shared 512-entry VGPR/AGPR file bounds occupancy; MFMA instruction table; hierarchy becomes reg → LDS → L1 → per-XCD L2 → MALL → HBM; expect default-cache-mode accuracy (~MI210-like) until CK/TileLang expose swizzle control.

## 4. Kimi-K3 (from moonshotai/Kimi-K3 config.json)
```yaml
kimi_k3:
  total_params: 2.8e12
  active_params: ~50e9
  hidden_size: 7168
  num_hidden_layers: 93
  num_attention_heads: 96
  intermediate_size: 33792        # dense FFN
  vocab_size: 163840
  max_position_embeddings: 1048576
  hidden_act: situ
  # MLA
  kv_lora_rank: 512
  q_lora_rank: 1536
  qk_nope_head_dim: 128
  qk_rope_head_dim: 64
  v_head_dim: 128
  mla_use_output_gate: true
  # MoE
  n_routed_experts: 896
  num_experts_per_tok: 16
  moe_intermediate_size: 3072
  routed_expert_hidden_size: 3584 # latent MoE projection
  n_shared_experts: 2
  first_k_dense_replace: 1
  num_nextn_predict_layers: 0
  # hybrid attention: full MLA every 4th layer (+ last), KDA linear attention elsewhere
  full_attn_layers: [4,8,12,...,88,92,93]
  kda: {num_heads: 96, head_dim: 128, short_conv_kernel_size: 4}
  attn_res_block_size: 12
  vision: {layers: 27, hidden: 1024, heads: 12, patch: 14}
  quantization: {format: mxfp4-pack-quantized, bits: 4, group_size: 32,
                 ignore: [self_attn, shared_experts, dense mlp, lm_head, vision]}
```

| Field | Kimi-K2 | Kimi-K3 |
|---|---|---|
| layers | 61 | 93 |
| heads | 64 | 96 |
| attention | MLA everywhere | MLA (≈1/4 layers) + KDA linear |
| routed experts / top-k | 384 / 8 | 896 / 16 |
| shared experts | 1 | 2 |
| moe_intermediate | 2048 | 3072 |
| context | 128K | 1M |
| total / active | 1T / 32B | 2.8T / ~50B |
| released weights | bf16/fp8 | MXFP4 (experts) |

Simulator implications: KDA layers keep a constant-size recurrent state (no KV growth) → needs a new `linear_attention` block type; only full-attention layers grow the KV cache; latent MoE adds down/up projections around the experts (routed hidden 3584).

## 5. Single-GPU vs multi-GPU split

**Why:** the single-GPU part is deterministic, analytical, validated against NCU/cutlass_profiler/rocprof and used for kernel/tile tuning and hardware DSE. The system part (TP/DP/EP/PP/SP, collectives and topology, compute-comm overlap, expert imbalance, disaggregated prefill/decode, continuous batching, KV capacity) owns all non-determinism and is validated against vLLM/SGLang.

**Related tools:** LLMCompass (hardware-centric, static), GenZ (single iteration), Vidur (profiled op-latency predictor → discrete-event serving simulator, <9% error), LLMServingSim / Frontier (on ASTRA-sim), ASTRA-sim (compute/network layer API), Calculon, DistServe/Splitwise simulators. Pattern: an op-latency layer exposes latency(op, shape, dtype, hw); a system layer consumes it.

**Proposed layout**
```
tilesight/
  gpuTilingPerfHWModel/         # YAMLs + calib/
  single_gpu/               # SUBSYSTEM A: lowerings, op_model, op_table, validate_ncu
  system/                   # SUBSYSTEM B: parallelism, collectives, overlap, moe_balance,
                            #              serving, disagg, validate_vllm
  model/                    # model spec / HF import, feeds both
```

**Contract A → B**
```python
@dataclass(frozen=True)
class OpSpec:
    op: str            # gemm | flash_attn | mla_decode | grouped_gemm | elementwise
    shape: dict
    dtype: str
    impl: str | None
    hw_id: str

@dataclass(frozen=True)
class OpResult:
    latency_s: float
    bound: str         # resource:tensor attribution
    util: dict
    bytes_hbm: int
    cache_hit: dict
    confidence: str    # calibrated | default_cache_mode | extrapolated

class OpLatencyModel(Protocol):
    def predict(self, spec: OpSpec) -> OpResult: ...
```
Plus a versioned `OpLatencyTable` keyed by (hw_id, calib_hash, version); a collective contract `stage_time(op, bytes, group, topology)`; and the model layer emitting a layer graph consumed by both. Rules: A is pure and hardware-parameterized; B owns batching/arrival/routing; calibration hashes invalidate cached tables; `bound`/`confidence` propagate upward; AMD/NVIDIA differences stay inside A.

## 6. References
- TileSight, arXiv:2607.22432v1; Pith review (pith.science/paper/2607.22432); PASCAL, arXiv:2609.10515.
- moonshotai/Kimi-K3 config.json and Kimi K3 blog; Kimi K2 tech report arXiv:2507.20534.
- NVIDIA "Inside NVIDIA Blackwell Ultra"; H200 datasheet; Hopper/Blackwell microbenchmark papers (arXiv:2501.12084, 2402.13499, 2512.02189).
- AMD MI400/MI455X product page; ROCm blogs on CDNA5/Helios and CDNA4 occupancy; ROCm device glossary.
- Vidur (arXiv:2405.05465), GenZ, LLMCompass, LLMServingSim 2.0 (arXiv:2602.23036), ASTRA-sim, Calculon; FlashAttention-3 (arXiv:2407.08608), TileLang (arXiv:2504.17577).
