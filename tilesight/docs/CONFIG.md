# Hardware configuration reference

The model only ever sees these fields. `spec` = copy it from the vendor, `calib` = measure it, `policy` = a modelling choice, `loss` = a derating knob whose default is always no loss.

## identity

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `name` | str | — | **required** | spec | Display name of the part |
| `arch` | str | — | `` | spec | Architecture tag (informational) |
| `sms` | int | count | **required** | spec | Streaming multiprocessors / compute units. Compute peaks are whole-GPU, so the model derives the per-SM rate as peak / sms |
| `clock_ghz` | float | GHz | `1.8` | spec | Clock used for every cycle <-> second conversion |
| `dies` | int | count | `1` | spec | Dies/XCDs; default number of L2 slices is 8 per die |
| `wavefront` | int | threads | `32` | spec | Warp / wavefront width (informational today) |

## compute

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `compute.tc_dense_tflops` | map | TFLOP/s | **required** | spec | Dense tensor-core peak per datapath: fp4 / fp6 / fp8 / int8 / bf16 / fp16 / tf32. A dtype the part lacks widens to the next one present |
| `compute.cuda_fp32_tflops` | float | TFLOP/s | **required** | spec | Vector FP32 peak |
| `compute.sfu_tops` | float | TOP/s | **required** | spec | Transcendental (exp) throughput; drives softmax. Measured is often far below spec |
| `compute.tc_min_m` | int | rows | `64` | spec | Smallest M one MMA instruction covers; smaller tiles are padded to it |
| `compute.attention_tile_m` | int | rows | `0` | spec | Attention tile M fixed by the part (0 = let the tile search choose) |
| `compute.attention_tile_n` | int | cols | `0` | spec | Attention tile N fixed by the part (0 = let the tile search choose) |
| `compute.cta_pair` | bool | — | `False` | spec | 2-CTA MMA (Blackwell tcgen05 cta_group::2) |
| `compute.cluster_multicast` | bool | — | `False` | spec | Thread-block clusters with TMA multicast; required for cluster_m > 1 tiles |
| `compute.mma_latency_cycles` | float | cycles | `64` | calib | Issue-to-result of one tile MMA. Independent MMAs overlap, so this costs pipeline fill, not throughput |
| `compute.cuda_latency_cycles` | float | cycles | `4` | calib | Vector op latency |
| `compute.sfu_latency_cycles` | float | cycles | `16` | calib | Transcendental latency; loop-carried in online softmax, so it does bound attention |

## losses

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `efficiency.tc` | float | ratio | `1.0` | loss | Achieved / peak tensor core |
| `efficiency.cuda` | float | ratio | `1.0` | loss | Achieved / peak vector |
| `efficiency.sfu` | float | ratio | `1.0` | loss | Achieved / peak SFU |
| `efficiency.smem` | float | ratio | `1.0` | loss | Achieved / peak SMEM or LDS |
| `efficiency.l2` | float | ratio | `1.0` | loss | Achieved / peak L2 |
| `efficiency.ddr` | float | ratio | `1.0` | loss | Achieved / peak HBM |
| `efficiency.sram` | float | ratio | `1.0` | loss | Achieved / peak on-chip buffer |

## memory.ddr

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.ddr.capacity_GB` | float | GB | **required** | spec | HBM capacity per GPU |
| `memory.ddr.bandwidth_TBps` | float | TB/s | **required** | spec | HBM peak bandwidth |
| `memory.ddr.latency_ns` | float | ns | `800` | calib | Idle load-to-use latency from HBM |
| `memory.ddr.ports` | int | count | `8` | spec | HBM ports/stacks; used by the address map (aggregated for bandwidth) |
| `memory.ddr.blocks` | int | count | `0` | calib | Independent controllers (0 = do not port-cap) |
| `memory.ddr.ports_per_block` | int | count | `1` | calib | Ports per controller |
| `memory.ddr.bytes_per_clk_per_port` | float | B/clk | `0` | calib | Port width; blocks x ports x width x clock caps the level |

## memory.l2

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.l2.capacity_MB` | float | MB | **required** | spec | Physical L2 (or MALL) capacity |
| `memory.l2.capacity_derate` | float | ratio | `1.0` | loss | Fraction of the capacity the simulation may use. DEFAULT 1.0 = no loss |
| `memory.l2.effective_capacity_MB` | float | MB | — | calib | Direct override of the usable capacity (a measured bandwidth-vs-working-set cliff) |
| `memory.l2.bandwidth_TBps` | float | TB/s | **required** | calib | L2 peak bandwidth |
| `memory.l2.latency_ns` | float | ns | `300` | calib | L2 hit latency |
| `memory.l2.per_sm_max_GBps` | float | GB/s | `1000000000.0` | calib | Most one SM can pull from the shared levels; binds when the grid is small |
| `memory.l2.partitions` | int | count | `1` | spec | Independent partitions, each a tile-level cache; a tile goes to the one its address maps to |
| `memory.l2.policy` | str | — | `lru` | policy | Replacement: lru | fifo | mru |
| `memory.l2.model` | str | — | `lru` | policy | lru = deterministic tile simulation (default) | sdcm = the paper's probabilistic model |
| `memory.l2.waves_simulated` | int | count | `2` | policy | Waves replayed in the simulation; >1 makes cross-wave reuse visible |
| `memory.l2.assoc` | int | ways | `16` | calib | Associativity (only used by the sdcm model) |
| `memory.l2.slices` | int | count | `0` | calib | Slices for the address map (0 = 8 per die) |
| `memory.l2.blocks` | int | count | `0` | calib | Independent L2 blocks, each with its own load/store port (0 = do not port-cap) |
| `memory.l2.ports_per_block` | int | count | `1` | calib | Ports per block |
| `memory.l2.bytes_per_clk_per_port` | float | B/clk | `0` | calib | Port width; blocks x ports x width x clock caps the level |
| `memory.l2.line_bytes` | int | B | `128` | spec | Cache line (used by the Little's-law cap) |
| `memory.l2.sector_bytes` | int | B | `32` | spec | Sector a miss actually fetches |
| `memory.l2.apply_conflict_penalty` | bool | — | `False` | policy | Fold the measured address-spread imbalance into efficiency.l2 |

## memory.l1

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.l1.owner` | str | — | `sm` | spec | sm | cluster (a cluster shares its L1s) |
| `memory.l1.capacity_KB` | float | KB | `0` | spec | Unified L1 + scratchpad per SM |
| `memory.l1.smem_carveout_KB` | float | KB | `0` | spec | What the tile plan takes as scratchpad; only the remainder caches global loads |
| `memory.l1.cluster_size` | int | SMs | `1` | spec | SMs per cluster when owner = cluster |
| `memory.l1.cache_global_loads` | bool | — | `True` | policy | False = global loads bypass L1 |
| `memory.l1.capacity_derate` | float | ratio | `1.0` | loss | DEFAULT 1.0 = no loss |
| `memory.l1.bytes_per_clk` | float | B/clk | `128` | calib | L1 bandwidth per SM |
| `memory.l1.latency_cycles` | float | cycles | `20` | calib | L1 hit latency |
| `memory.l1.policy` | str | — | `lru` | policy | Replacement policy |

## memory.onchip

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.onchip.smem.capacity_KB` | float | KB | **required** | spec | Shared memory / LDS per SM available to a tile plan |
| `memory.onchip.smem.bytes_per_clk` | float | B/clk | `128` | spec | SMEM bandwidth per SM |
| `memory.onchip.smem.latency_cycles` | float | cycles | `30` | calib | SMEM latency |
| `memory.onchip.tmem.capacity_KB` | float | KB | `0` | spec | Tensor memory per SM (Blackwell); its presence also moves accumulators out of registers |
| `memory.onchip.tmem.read_TBps` | float | TB/s | `0` | calib | TMEM read bandwidth per SM |
| `memory.onchip.tmem.write_TBps` | float | TB/s | `0` | calib | TMEM write bandwidth per SM |
| `memory.onchip.tmem.latency_cycles` | float | cycles | `16` | calib | TMEM latency |

## memory.sram

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.sram.capacity_MB` | float | MB | `0` | spec | Extra on-chip shared buffer between L2 and HBM (absent = no such buffer) |
| `memory.sram.capacity_derate` | float | ratio | `1.0` | loss | DEFAULT 1.0 = no loss |
| `memory.sram.bandwidth_TBps` | float | TB/s | `0` | calib | Buffer bandwidth |
| `memory.sram.switch_TBps` | float | TB/s | `0` | calib | Aggregate bandwidth of the switch between the shader slices and the buffer pieces (0 = same as the buffer bandwidth). A slice draws its 1/slices share of it |
| `memory.sram.latency_ns` | float | ns | `400` | calib | Buffer latency |
| `memory.sram.per_sm_max_GBps` | float | GB/s | `1000000000.0` | calib | Per-SM ceiling |
| `memory.sram.policy` | str | — | `cache` | policy | cache | pin |
| `memory.sram.pin` | map | share | — | policy | Capacity shares per tensor class when policy = pin: {weight, kv, act} |
| `memory.sram.alloc` | map | share | — | policy | Alias of pin |
| `memory.sram.keep_intermediates` | bool | — | `False` | policy | Elementwise intermediates stay on chip instead of round-tripping HBM |
| `memory.sram.bypass_l2` | bool | — | `False` | policy | Resident bytes skip the L2 datapath (otherwise L2 becomes the next wall) |
| `memory.sram.prefetch` | bool | — | `False` | policy | Resident bytes cost no exposed latency |
| `memory.sram.costream` | bool | — | `False` | policy | Buffer and HBM serve in parallel |
| `memory.sram.stage` | map | — | — | policy | Mega-tile staging: {share_blocks: N} — one DMA fills a panel that N blocks consume |
| `memory.sram.placement` | str | — | `none` | policy | What is pre-allocated in the buffer, at tile granularity: none | ab | abc | abc_kv | abc_kv_moe (1: A and B, 2: A B C, 3: + KV cache, 4: + every expert) |
| `memory.sram.unified` | bool | — | `True` | policy | Treat the per-slice buffers as one pool (physically each slice holds a piece) |

## tensor_core

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `compute.tensor_core.tile_m` | int | rows | `64` | spec | M of one tensor-core tile instruction (all tensor datatypes use the KQV dtype here) |
| `compute.tensor_core.tile_n` | int | cols | `64` | spec | N of one tensor-core tile |
| `compute.tensor_core.tile_k` | int | depth | `32` | spec | K of one tensor-core tile |
| `compute.tensor_core.tile_latency_cycles` | float | cycles | `16` | calib | Issue-to-result of ONE tile instruction |
| `compute.tensor_core.tile_interval_cycles` | float | cycles | `4` | calib | Issue interval between back-to-back tile instructions on one tensor core (throughput) |
| `compute.tensor_core.per_shader_core` | int | count | `4` | spec | Tensor cores inside one shader core (2, 4, 16 ...) |

## tile_policy

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `compute.tile_policy.gemm` | str | — | `auto` | policy | GEMM block tile: "auto" searches the space per op shape, or a JSON TileConfig override, e.g. {"bm":128,"bn":256,"bk":64} |
| `compute.tile_policy.attn` | str | — | `auto` | policy | "auto" or a JSON AttnTileConfig override. Ignored on parts where compute.attention_tile_m/n force a fixed attention tile. |
| `compute.tile_policy.overrides` | map | — | `{}` | policy | fnmatch pattern on op name -> partial tile fields, applied before gemm/attn above |

## shader

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `shader.cores_per_slice` | int | count | `1` | spec | Shader cores in one shader slice; a slice shares one L1 |
| `shader.slices` | int | count | `0` | spec | Shader slices (0 = sms / cores_per_slice). Slices are copies that run staggered in time |
| `shader.wave32_per_core` | int | count | `4` | spec | wave32 (SIMD-32) issue slots per shader core; with the clock this gives the vector rate |
| `shader.clock_ghz` | float | GHz | `0` | spec | Core clock (0 = use the top-level clock_ghz) |
| `shader.smem_per_core_KB` | float | KB | `0` | spec | Scratchpad per shader core (0 = use memory.onchip.smem.capacity_KB) |
| `shader.smem_sharing` | str | — | `shared` | spec | How the tensor cores inside a shader core see the scratchpad: shared | split |
| `shader.gpr_total_KB` | float | KB | `0` | spec | Register file per shader core (0 = from occupancy.regs_per_sm) |
| `shader.gpr_sharing` | str | — | `split` | spec | Register file: shared | split across the tensor cores |
| `shader.l1_per_slice_KB` | float | KB | `0` | spec | L1 per shader slice (0 = memory.l1.capacity_KB). Tile-granular, deterministic residency |

## memory.slices

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.slices.count` | int | count | `0` | spec | Memory slices; each owns its HBM stack, its L2 port and its DMA port (0 = ddr.ports) |
| `memory.slices.l2_per_slice_MB` | float | MB | `0` | spec | L2 inside one memory slice. 0 = no L2 at all: the port and the DMA talk to the on-chip buffer |
| `memory.slices.hbm_per_slice_GB` | float | GB | `0` | spec | HBM capacity per slice (0 = ddr.capacity_GB / slices) |
| `memory.slices.address_mode` | str | — | `interleave` | spec | How a tensor is spread over the slices: linear | interleave |
| `memory.slices.interleave_KB` | float | KB | `1` | spec | Interleave granularity when address_mode = interleave (1, 4, 16 ...) |

## memory.gmem

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.gmem.downstream_read_GBps_per_slice` | float | GB/s | `0` | spec | Read bandwidth from one memory slice towards the fabric |
| `memory.gmem.downstream_write_GBps_per_slice` | float | GB/s | `0` | spec | Write bandwidth into one memory slice |
| `memory.gmem.upstream_read_GBps_per_slice` | float | GB/s | `0` | spec | Read bandwidth delivered to one shader slice |
| `memory.gmem.upstream_write_GBps_per_slice` | float | GB/s | `0` | spec | Write bandwidth out of one shader slice |

## memory.addressing

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.interleave_KB` | float | KB | `2` | calib | Default interleave granularity when a side does not set its own |
| `memory.addressing.addr_bits` | int | bits | `48` | spec | Physical address width |
| `memory.addressing.l2` | map | — | — | calib | {ports, mode: interleave|range|hash, granularity_KB} for L2 slices / load ports |
| `memory.addressing.ddr` | map | — | — | calib | {ports, mode, granularity_KB} for the HBM ports the DMA engines target |

## memory.parallelism

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.outstanding.per_sm_lines` | int | lines | `0` | calib | Cache lines in flight per SM (MSHR-style). Little's law: BW_per_SM <= lines x line / latency. 0 = unlimited |
| `memory.outstanding.dma_per_engine_lines` | int | lines | `0` | calib | In-flight lines per DMA engine (recorded; not yet a cap) |
| `memory.queueing.coef` | float | — | `0.0` | loss | M/D/1-style latency inflation 1 + coef*u/(1-u) as a lane saturates. DEFAULT 0 = no loss |
| `memory.queueing.max_factor` | float | — | `3.0` | loss | Cap on that inflation |

## memory.dma

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `memory.dma.engines` | int | count | `1` | spec | Copy engines |
| `memory.dma.per_l2_block` | bool | — | `False` | spec | One engine per L2 block (false = a single shared engine) |
| `memory.dma.destination` | str | — | `smem` | policy | Where a DMA drops data: smem (through the L2 datapath) | l2 (fills L2) | bypass |
| `memory.dma.hugepage_KB` | float | KB | `0` | calib | Fixed size of one DMA operation: a DMA moves exactly one hugepage, never less (0 = unset: fall back to an occupancy proxy for L2/DDR slice spread; NVIDIA DMA/TMA moves 2048). Splits evenly across every memory slice by construction; if it doesn't divide evenly by memory.addressing.l2's granularity x port count (e.g. 2048 over 12 ports), it is padded up to what the fullest slice would get, never rejected or underestimated. Also changes the L2 simulation's cache atom (for GEMM A/B and attention K/V) to hugepage granularity, so real reuse across loop iterations shows up as a hit instead of a fresh miss every call |

## load_paths

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `load_paths` | map | — | **required** | spec | One entry per path (tma, lsu, buffer_lds, tdm ...), each with: engine (dma|lsu), per_sm_bytes_per_clk (this path's own clock domain, same as every other on-chip lane — scales with core.freq_ghz), latency_ns (issue overhead), smem_direct, multicast, regs_per_thread, issue_bytes_per_clk |
| `load_paths_default` | str | — | `` | spec | Path a request resolves to when the named one does not exist on this part |

## occupancy

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `occupancy.max_blocks_per_sm` | int | count | `32` | spec | Hardware limit on resident blocks |
| `occupancy.max_threads_per_sm` | int | threads | `2048` | spec | Thread limit per SM |
| `occupancy.regs_per_sm` | int | registers | `65536` | spec | Register file per SM, in 4-byte registers |
| `occupancy.max_regs_per_thread` | int | registers | `255` | spec | Above this the accumulator spills to local memory |

## network

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `network.nvlink.domain_size` | int | GPUs | `8` | spec | Size of the fast scale-up domain; collectives go hierarchical beyond it |
| `network.nvlink.bandwidth_GBps` | float | GB/s | **required** | spec | Per-GPU scale-up bandwidth (one direction) |
| `network.nvlink.alpha_us` | float | us | `3.0` | calib | Scale-up startup latency |
| `network.scaleout.bandwidth_GBps` | float | GB/s | **required** | spec | Per-GPU scale-out bandwidth |
| `network.scaleout.alpha_us` | float | us | `8.0` | calib | Scale-out startup latency |

## runtime

| field | type | unit | default | tag | meaning |
|---|---|---|---|---|---|
| `runtime.launch_overhead_us` | float | us | `2.0` | calib | Per-kernel launch cost; ~0.5-1 with CUDA graphs |

Total: 132 fields (63 spec, 37 calib, 20 policy, 12 loss).