# DESIGN — equations, semantics, deviations

## 0. The interface: GPU config in, model behind it
A GPU is **input**, not part of the model. `gpuTilingPerfHWModel/interfaceAndRun/schema.py` declares every field
the model may see — 99 fields in 15 sections, each with a type, unit, default, and a tag:
`spec` (copy it from the vendor), `calib` (measure it), `policy` (a modelling choice) or `loss`
(a derating knob, default always no loss). The reference is generated from it:
`docs/CONFIG.md`, or `tilesight config --list [--section … --tag …] [--md f.md] [--csv f.csv]`.

Three rules keep the boundary clean, and a test enforces all three:
1. the model asks `HardwareSpec.get(path)` and never carries a literal default — absent fields
   resolve to the schema default, so hardware knowledge lives in exactly one file;
2. no file under `model/` (or `interfaceAndRun/lower.py`, `attention_blocks.py`, `memmap.py`) may name a device, vendor or architecture
   (`b300`, `hopper`, `cdna` …) outside comments;
3. every shipped preset must pass `HardwareSpec.validate()` (unknown field, wrong type, missing
   required field), which is also what `tilesight config --validate <file>` runs for a new part.

Adding a GPU therefore means writing a YAML and nothing else; adding a hardware *concept* means
adding a field to the schema and using it in the model.

## 1. Layers
```
ModelSpec (blocks: mla/gqa/mlp/moe/norm/raw)      interfaceAndRun/model_spec.py  <- user gives layer sizes / HF config
   │  RunConfig (phase, batch, seq, tp/dp/ep, dtypes)
   ▼
Ops for one GPU (gemm, attn_decode, attn_prefill, elementwise, allreduce, a2a)   interfaceAndRun/lower.py
   │  tile policy (GPU config, gpuTilingPerfHWModel/interfaceAndRun/schema.py's compute.tile_policy.*):
   │  fixed | per-op override (fnmatch) | auto search (gpu_top's tile search spaces)
   ▼
Kernels = tile execution plans lowered to numbers, in C++          model/gpu_top/*.cpp
   │  HardwareSpec -> lanes
   ▼
Engine (model/shader_core, model/shader_slice) -> KernelResult (time, limiter, util)
   ▼
ModelReport: step time, tok/s/GPU, per-op table, limiter mix, memory   interfaceAndRun/runner.py
DSE: sweep / required_value over any hardware field                      dse/sweep.py
```

## 2. Hardware → resource lanes (paper Eq. 1, generalized)
The paper's fixed vector ⟨TC, CUDA, SFU, TMEM, SMEM, L1.5, L2, DDR, Net⟩ becomes a
**named, data-driven lane list** built from the YAML:

| lane | source | work unit in Action.work | rate |
|---|---|---|---|
| `tc` | compute.tc_dense_tflops[dtype] | seconds on one SM | peak·eff/SMs |
| `cuda` | compute.cuda_fp32_tflops | seconds on one SM | |
| `sfu` | compute.sfu_tops | seconds on one SM | |
| `<onchip>` (smem, tmem, …) | memory.onchip.* | seconds on one SM | bytes/clk or TB/s per SM |
| `path:<p>` (tma, lsu, …) | load_paths.* | seconds on one SM | per_sm_bytes_per_clk x clock |
| `l2` | memory.l2 | **bytes** | min(BW·eff / active_SMs, per_sm_max) |
| `ddr` | memory.ddr | **bytes** | min(BW·eff / active_SMs, per_sm_max) |

Shared lanes model the paper's observation that a partially filled wave gives each
active SM a larger share of L2/DDR bandwidth, bounded by what one SM can pull
(`per_sm_max_GBps`, a calibration target). Net is handled by comm kernels (§5.4).

### 2b. Datatypes
`DTYPE_BYTES` gives storage width; `DTYPE_ALIAS` maps a name to the tensor-core datapath it runs on
(nvfp4/mxfp4 → fp4, mxfp8 → fp8); `WEIGHT_ONLY` (int4) means the weights are narrow but the MMA runs
at the activation width. `HardwareSpec.tc_datapath` widens to the next datapath a part actually has
(fp4 → fp8 on Hopper/CDNA3); `mma_cost` returns the `cuda` lane when there is no tensor-core path at
all (fp32). Capability flags gate tile features: `compute.cta_pair` (2-CTA MMA),
`compute.cluster_multicast` (thread-block clusters + TMA multicast; false on AMD),
`load_paths_default` (a request for `tma` resolves to the part's real path, e.g. CDNA5 `tdm`).

## 3. Engine (paper §3.3–3.4, Eqs. 2–5; Algorithms 1–2)
For a kernel with `num_blocks`, `iters`, `stages`, `resident`, `consumers` and action
lists `prologue`, `body` (one K-loop iteration), `epilogue`:

**Waves.** `conc = SMs·resident`; `full, tail = divmod(num_blocks, conc)`. For a wave of
`b` blocks: `active = min(SMs, ⌈b/resident⌉)`, `bps = ⌈b/active⌉`.

**Per-action lane time.** `u_r(o) = work_r(o)` (per-SM lane) or `work_r(o)/share_r(active)`
(shared lane). Node weight `w(o) = max_r u_r(o) + latency(o)`.

**Steady round (Eq. 4').**
```
R = max( bps · max_r Σ_o u_r(o)            # contention: same-lane work serializes (paper Eq. 4)
         CP / stages,                       # SMEM-buffer recycling recurrence
         CPrec / consumers )                # loop-carried compute recurrence
CP    = longest path over the body DAG with weights w(o)
CPrec = longest path using only actions flagged `recurrent`
```
Limiter label = the lane attaining the first term, else `latency`.

**Wave time (Eq. 2').** `T = T_pro + T_fill + iters·R + T_epi`, with
`T_fill = max(0, CP − R)` (first iteration's exposed latency) and
`T_pro/T_epi = max(bps·max_r Σ u_r, longest path)` over the prologue/epilogue lists.
Kernel time = Σ_waves count·T + launch overhead.

**Utilization.** busy fraction per lane = total work of all blocks / (machine rate × time).

**Bound attribution (coarse + fine).** Every phase's time is charged to a limiter:
- coarse (`limiter_time`): lane name | `latency` | `launch` | `net`
- fine (`limiter_detail`):
  - resource-bound → `"<lane>:<action>"`, action = the one putting most work on that lane,
    e.g. `ddr:load:expert_weight`, `l2:load:act`, `smem:mma`, `tc:gemm_qk`, `sfu:softmax`
  - latency-bound → `"latency:<action>(<lane>|mem-lat)"`, action = heaviest node on the
    critical path of whichever recurrence set the bound (CP/stages or CPrec/consumers),
    lane = that node's dominant lane or `mem-lat` if its memory latency dominates
  - `latency:fill`, `launch`, `net:<algo>`
Action names carry the **tensor** (`load:weight`, `load:kv_cache(K)`, `load:scores(S)`,
`store:partials`, `spill:regs` …), set by the lowering from `names=(A,B,C)`.
Reports roll details up into resource classes (compute:tensor-core / cuda-core / sfu,
on-chip:smem / tmem / regs(spill), cache:L2, memory:DDR/HBM, load-path:*, interconnect,
latency<-x, launch-overhead) and list the top resource:tensor pairs.

**Deviations from the paper (documented, switchable later — TASKS A1/A2):**
- Paper Eq. 5 minimizes over topological orders of the action DAG. We use a
  recurrence-based bound (CP/stages, CPrec/consumers) that is order-free; exact
  order enumeration is task A1 (needed for fused kernels with several independent
  chains, e.g. the paper's MLA-decode example with 132 legal orders).
- Paper Eq. 2 uses `N−d` steady iterations with `d = stages·resident−1`; we use all
  `iters` and an explicit fill term. This avoids the double-count the paper's reviewers
  suspected as the source of its deep-K bias. Task A2 adds the paper form behind a flag.

## 4. L2 model — deterministic tile-level simulation (default) or SDCM
The paper's SDCM answers "how likely is a hit at this reuse distance". For a hardware-managed
L2 that is the wrong question: what is resident is decidable, so `model/common/cache/cache_sim.cpp` decides
it. The tile is the atom (no cache lines). `memory.l2.partitions` independent partitions each
hold whole tiles and run an explicit policy (`memory.l2.policy`: `lru` | `fifo` | `mru`); a tile
goes to the partition **its address maps to** (`report.addressing.AddressMap`), so the layout /
swizzle decides the balance and a power-of-two pitch really does pile everything into one
partition. Capacity is in bytes, so different tile sizes compete honestly.

**Simulation window.** By default the simulation replays the **whole kernel**: every wave, and in
each wave every K step, all blocks of the wave stepping through their K loop in lockstep
(`sim_window()` in `gpu_top.cpp`). It used to replay 2 waves x the first 4 K steps — 0.5% of an
8192³ GEMM's accesses, 2% of a 4096x4096x7168 one — which never touches enough data to see a
capacity eviction in a 132 MB L2 (or in a buffer shared by every shader core), only compulsory
misses, and then extrapolated those. `memory.l2.waves_simulated` / `.ksteps_simulated` (0 = all)
can shorten it deliberately; `memory.l2.sim_max_accesses` (default 2e6 tile accesses per
kernel, ~0.2-0.4 s) caps the cost, and when a kernel is over budget waves are dropped first (the
waves of one kernel are close to statistically alike) and K steps only last (cutting K is what
biases a hugepage trace — see below). Every kernel reports `meta.sim_coverage` (simulated /
total accesses). Attention replays the same way over the **real KV-cache layout**
(`[B][kv_heads][kv tiles]`, K then V per tile): the query heads of a GQA group and the query
tiles of a head walk the same addresses, so they share tiles the way the hardware does — the
old attention trace gave every access a fresh synthetic address in replay order. The whole
test suite takes ~10 min instead of ~40 s; the model is meant to be accurate first.

Output is not just a miss fraction: `SimResult` gives per-partition residency (exactly which
tiles are still in L2), occupancy and eviction counts — printed by `tilesight kernel` and shown
in the Figure 3(f) panel. Example (B300, 4096³ FP8 GEMM): 93.8% hit; with the default
interleave every resident tile lands in partition 0, with `mode: hash` they spread 40/40/40/40.
Shrinking the effective L2 from 83 MB to 1 MB produces 208 evictions and drops the hit rate.

**Losses are configured, and off by default.** Every cache level takes `capacity_derate`, and
every preset ships it at **1.0**: out of the box the simulation may use the full physical
capacity, so nothing is silently taken away. Lower it only with a measurement behind it — it is
the one knob that absorbs what the tile-level simulation does not model (cache-line
granularity, conflicts inside a partition, other kernels' traffic, streaming/evict-first
hints). `effective_capacity_MB` is an outright override for when a bandwidth-vs-working-set
sweep gives the cliff directly (TileSight measured ~83 MB on B200-class parts, i.e. ~0.66 of
the 126 MB physical; that value now lives in a comment, not in the defaults).
The same rule holds for the other loss knobs: `efficiency.*` defaults to 1.0 when absent,
`memory.queueing.coef` to 0, and `HardwareSpec.lossless()` (CLI `--ideal`) switches all of them
off at once — efficiency factors, derates, queueing, per-SM caps and outstanding-request
ceilings — which gives the theoretical-peak baseline. On B300 the modelled losses account for
1.14x on a big FP8 GEMM, 1.24x on a decode GEMM and 1.26x on Kimi-K2 decode (35.5 vs 28.2 ms).

**L1 is modeled the same way.** `memory.l1`: `owner` (`sm` | `cluster` — a Hopper/Blackwell
cluster shares its L1s through DSMEM, so the usable capacity is multiplied by `cluster_size`),
`capacity_KB`, `smem_carveout_KB` (what the tile plan already takes as scratchpad; only the
rest caches global loads), `cache_global_loads`, `capacity_derate`, `bytes_per_clk`,
`latency_cycles`, `policy`. B300: (256 - 228) x 0.75 x 2 = 42 KB usable. L1 is simulated over
**one SM's slice of the wave** (it is private, unlike L2) and only for paths that actually go
through it — a DMA engine writing SMEM/LDS directly never touches it, which the `smem_direct`
path attribute decides. An L1 hit does not reach the L2 datapath: a 1024x1024x512 BF16 GEMM on
the LSU path gets 37%/12% hit and its L2 traffic drops from 16 to 12 KB, while the same GEMM on
TMA bypasses L1 entirely. There is an `l1` lane for its bandwidth.

**The on-chip buffer's residency is decided one of two ways**, chosen by how its capacity is
allocated. A *shared* buffer (`memory.sram.policy: cache`, no `alloc`/`pin` split) is a real cache
that every tensor class competes for via actual access order, so `lower_gemm`/`lower_attention_*`
give it the same tile-keyed LRU trace as L1/L2 — reusing the exact A/B (or K/V) access stream
already built for the L2 simulation, run through `cache::simulate` a second time against
`sram_capacity_for(class)`. This is the same reuse-distance idea the paper's SDCM approximates
with a probability (see "why a simulation instead of the paper's probability formula" below),
computed exactly instead, because we do have the real trace. A *pinned* buffer (`memory.sram.alloc` or `.pin`, e.g. `{weight: 1.0}`) models
a dedicated, pre-staged carve-out instead — assumed already resident before this kernel's own
accesses run, not something its access order fills from cold — so it keeps the closed-form
`resident_frac` (capacity/footprint ratio): charging the trace's compulsory first-touch misses
against a pinned share would wrongly treat pre-loaded weights as starting from an empty cache.
See `tests/test_paths_addr_buffer.py::
test_shared_buffer_uses_a_real_lru_trace_pinned_buffer_stays_closed_form`.

The paper's probabilistic SDCM model (§3.5, Eqs. 6–10: reuse-distance -> hit probability via a
binomial/Gaussian approximation) answered a different, weaker question ("how likely is a hit")
and was dropped from the port — it was dead code even before this rewrite (nothing in the real
simulation path called it, only its own tests did); the deterministic tile-level simulation
above is what every lowering actually uses.

**Why a simulation instead of the paper's probability formula, precisely.** The paper's Eq. 6
computes a tile's reuse distance `D_T` (distinct tile-blocks touched since its last access) —
that step is exact, a plain count over a trace. Eqs. 7–10 then turn `D_T` into a hit *probability*
via a binomial/Gaussian approximation, because the paper doesn't necessarily know which cache
*set* each of those `D_T` tiles actually mapped to, so it assumes uniform-random placement across
`B_T` sets and asks "what's the chance ≥`A` of them collided with mine". We are not in that
position: every access's real address is generated from the tile geometry, so `AddrCfg`/`port_of`
already gives the exact set/partition for every access. With that, there's nothing left to
estimate — `cache_sim.cpp`'s per-partition LRU directly computes whether enough same-set tiles
really did land within the reuse window, which is the deterministic answer Eqs. 7–10 exist to
approximate when the real mapping isn't known. The one approximation that *can* remain here is
unrelated to probability: a kernel over `memory.l2.sim_max_accesses` is replayed over fewer waves
(or, last, fewer K steps) than it really has — a coverage limit, reported per kernel as
`meta.sim_coverage`, not a statistical model.

**L2/DDR bandwidth is not one GPU-wide pool.** §2's `l2`/`ddr` lanes model the *configured
aggregate* bandwidth, fair-shared across active cores (`shader_core::per_core_rate`) — but
physically each memory slice owns its own L2 port and its own HBM channel, and a tile's address
routes it to exactly one slice (`memory.addressing.l2`, the same `AddrCfg`/`port_of()` used
above for the tile-cache simulation). A kernel can only draw the full aggregate bandwidth if it
keeps at least as many blocks concurrently in flight as there are slices; with fewer, only that
many slices are ever touched and the reachable bandwidth is capped proportionally. `gpu_top.cpp`
applies this as `slice_bw_frac(concurrent_blocks, n_slices) = min(1, concurrent_blocks /
n_slices)`, scaling up the `l2`/`ddr` bytes charged in `gload`/`gstore` by `1/slice_bw_frac`
wherever it is below 1 (GEMM A/B/C, attention KV). `concurrent_blocks` is the same
occupancy-derived quantity already used for on-chip-buffer residency (`sms() × resident`, capped
by the kernel's total block count) — deliberately *not* derived from the synthetic per-tile
address stream the L2 hit/miss simulation builds, since that stream is tuned for cache behavior
and aliases badly (power-of-two tile strides against the interleave granularity) if reused to
judge bandwidth spread. This mainly shows up for small-grid kernels (e.g. low-batch decode
attention, whose block count is `batch·kv_heads·groups·splits`): a big compute-bound GEMM has
far more concurrent blocks than slices and sees no penalty; a decode step with too few blocks in
flight to reach every slice does. See `tests/test_paths_addr_buffer.py::
test_l2_ddr_bandwidth_is_capped_by_memory_slices`.

**`memory.dma.hugepage_KB`** models a DMA that moves exactly one **hugepage** per operation, never
less (matching how NVIDIA's DMA/TMA engines actually burst — the name is deliberate: 2MB, the huge
page size, is the expected real value). Unset (default 0), `gload`/`gstore` fall back to the
occupancy proxy above unchanged — no behavior change for any preset that hasn't opted in. A
hugepage spans every memory slice evenly **by construction, never by simulated address**, unlike
two earlier, failed attempts at slice-aware bandwidth that derived slice spread from the L2 hit/
miss simulation's synthetic per-tile address stream (aliases on power-of-two tile strides).
Real interleaving distributes it across slices one granule (`memory.addressing.l2`'s granularity)
at a time, round-robin, so `hugepage_bytes()` in `gpu_top.cpp` doesn't require the hugepage to
divide evenly by the port count — when it doesn't (e.g. a 2048 KB hugepage over H200's 12 L2 ports
at 1 KB granules: 2048/12 = 170.67 granules/slice), some slices simply end up with one fewer
granule than others in reality; rather than track which slice that is, it pads up to what the
fullest slice gets (`ceil(granules / n_slices) × n_slices × granularity` — 171 granules ×
12 × 1KB = 2052 KB here), never an underestimate of the real transfer.

Configuring it also changes what the L2 simulation's cache atom *is*, for the tensor operand loads
that already go through `simulate_l2` (GEMM A/B, attention K/V — "the matrix part"; Q, output
stores, register spill are untouched, they never went through `simulate_l2` to begin with — "the
rest goes through L2 ports" at their own raw byte size, never hugepage-priced). Each of those
loads' access stream is keyed by **hugepage index** (`address / hugepage_bytes`) instead of loop
index, so a second access landing inside an already-resident hugepage is a real hit in the
simulation — `miss_fraction()` then means "how often a genuinely new hugepage is needed", and
`gload` bills that directly as `miss × hugepage_bytes` (replacing the plain `to_memory × miss` it
uses when hugepages are off). A's and B's (K's and V's) miss fractions are independent per-stream
results from one shared simulation call, so no proportional splitting is needed for GEMM; decode
and prefill attention model K+V as one *combined* access unit with one shared miss fraction (a
pre-existing simplification, unrelated to hugepages), so each operand's `gload` call is billed its
`k_tile/(k_tile+v_tile)` or `v_tile/(k_tile+v_tile)` share of the one combined hugepage fetch, so
the two calls still sum to exactly one hugepage per real miss rather than double-charging it.

Getting this right took a real wrong turn worth recording: an earlier version left `simulate_l2`
alone and instead rounded each call's *already piece-wise-amortized* raw-tile miss bytes up to a
whole hugepage independently, every call — since `gload` represents one steady-state loop
iteration that the round-time formula then multiplies by `iters`, that turned "occasionally fetch
a whole page, which then serves ~20 iterations for free" into "possibly re-fetch a whole page on
every one of 128 iterations", inflating a genuinely `tc`-bound 8192³ FP8 GEMM (0.315ms) to a
falsely `ddr`-bound 63.7ms — about 200x. Keying the cache itself by hugepage index fixes this at
the source, because reuse across iterations is now a real, simulated hit rather than something the
billing formula has to guess at after the fact. See
`tests/test_paths_addr_buffer.py::test_dma_hugepage_keys_the_l2_simulation_instead_of_rounding_after_the_fact`.

Three more things the hugepage trace needs to be right, each found by turning it on for every
preset and chasing the failures:
- **The window must cover the K loop.** A page holds many K steps' worth of one operand (2 rows of
  A x 128 K steps for an 8192³ FP8 GEMM). The first step into it pays the whole page, the next
  127 are free; the old 4-step window saw the expensive steps and extrapolated them — 3.4 GB of
  DDR for a 67 MB operand. Over the full K loop it is 107 MB, identical to the tile-granular
  trace, and the GEMM is `tc`-bound again. Hence the whole-kernel window above.
- **A page lives in every L2 partition, one shard each.** Routing the whole page to the partition
  of whichever tile happened to touch it put one page in several partitions, each charged as its
  own 2 MB miss (a streamed decode weight: 235 MB of DDR for 117 MB of weight). `cache::simulate`'s
  `page_fill` mode stores `hugepage/n_partitions` per partition, looks the shard up in the
  partition the tile's address maps to, and on a miss refills every partition's shard (the DMA
  moved the whole page); each partition still evicts on its own.
- **L1 stays tile-granular.** L1 caches LSU loads, not DMA pages; a 2 MB entry can never fit a
  ~42 KB L1, so reusing the page-keyed trace for it silently switched L1 off. `lower_gemm` keeps a
  tile-keyed copy of the access stream (`tkeys`/`tsizes`) for L1.

**The decode thrash, and why pages are now on by default.** In pure page mode, long-context
decode attention (Kimi-K2, batch 256, seq 8192, dp 8) thrashes: every split streams its own KV
range, 148–320 concurrent streams x 2 MB pages > a 132 MB L2, so LRU evicts a page before its
stream has finished it and nearly every tile step refetches 2 MB. That is not a bug of the trace —
it is what "every DMA is a whole 2 MB page into L2" really costs. It is resolved by making *how* a
matrix operand comes in a GPU config choice (below) and by giving page mode a destination that can
hold every stream's page: the on-chip buffer.

### 4c. HBM fetch paths: DMA engines vs. L2 ports (`memory.dma.fetch_mode`)

HBM (`ddr` lane) is one bandwidth pool shared by two engines, each with its own ceiling lane:

| lane | who uses it | rate |
|---|---|---|
| `dma` | matrix operands (GEMM A/B, attention K/V), per `fetch_mode` | `memory.dma.engines x memory.dma.port_bytes x clock` (default 16 x 128 B) — a DMA moves a contiguous range into space reserved up front, so it needs no outstanding entries |
| `l2port` | everything else (Q, activations, elementwise) + all write-back | `min(ports x ddr_port_bytes_per_clk x clock, ports x floor(ddr_outstanding / ceil(op_bytes / width)) x op_bytes / (ddr_entry_cycles / clock))` |
| `ddr` | both (sum) | HBM bandwidth |

`l2port` is Little's law on the L2 ports' outstanding entries, shared by reads and writes: one
entry covers one port width (64 B), an `op_bytes` request (256 B) takes `ceil(256/64) = 4`
entries and holds them `ddr_entry_cycles` (300); 128 entries per port = 32 requests in flight.
`ports` = `memory.l2.ddr_ports`, 0 = one per `memory.addressing.l2` port. With B300 defaults: DMA
16 x 128 B x 1.9 GHz = 3.9 TB/s, L2 ports 16 x 32 x 256 B / 158 ns = 0.83 TB/s, HBM 8 TB/s x eff.
So with 16 DMA engines the DMA path alone cannot fill HBM; more engines can. A miss also pays the
path's fill latency (`memory.dma.fill_latency_cycles` 8 per 128 B beat, `memory.l2.
ddr_fill_latency_cycles` 16 per 64 B beat) on top of the DDR latency.

`memory.l2.op_bytes` is the L2 operation = eviction granularity (default 256 B). It has to divide
the L2 interleave stripe (`memory.addressing.l2.granularity_KB`) or one op would straddle two
slices — asserted in both `HardwareSpec.l2_op_bytes()` and `gpu_top.cpp::l2_op_bytes()`. An
L2-port fetch of a tile is rounded up to it.

`plan_fetch()` (`gpu_top.cpp`) turns the kernel's real access trace into fetch atoms and runs
the L2 simulation on them. Every access carries its **stream's needed range** — the contiguous
bytes one block walks over its K loop (a GEMM block's A row strip and B column strip, with A laid
out `[batch][M-tile][K-tile]` and B `[batch][N-tile][K-tile]`, K contiguous like an nn.Linear
weight; a decode split's KV range; a prefill query tile's KV window). Modes:

| mode | atom of a miss | engine |
|---|---|---|
| `dma_page` | the whole padded page (`hugepage_KB`, default 2 MB) | DMA |
| `dma_page_tail_l2` | pages fully inside the stream's range: page; the partial page at either end: the tile, rounded up to `op_bytes` | DMA / L2 port |
| `l2_port` | the tile rounded up to `op_bytes` | L2 port |
| `dma_smart` | a chunk `c` = largest multiple of `dma.port_bytes` with `c <= page`, `c <= the stream's range`, and `c <= destination capacity / concurrent streams` (so one chunk per stream fits); never below one tile | DMA |
| `auto` (default) | `dma_page` when the DMA lands in a shared on-chip buffer that holds one page per concurrent stream, `dma_smart` otherwise (no buffer, too small a buffer, or a `pin`/`alloc`/`stage` buffer, which keep their closed-form model and DMA via L2) | |

Concurrent streams = distinct needed ranges in the first wave. The destination is the buffer
when there is one, else L2. DMA atoms are sharded over every L2 partition (`page_fill`, a
per-access `fill_mask`); L2-port tiles stay whole in the partition their address maps to. The
simulation's per-(operand, engine) misses x atom bytes / loads gives each `gload` its HBM bytes
per load on each engine (`HbmFetch`), which the on-chip buffer's hit fraction then reduces.

**The buffer.** DMA'd pages land in the buffer when one exists; unlike L2 it holds a page until
its streams are done with it. Its in-kernel trace (L1-style LRU over the same atoms) uses the
**whole** buffer capacity for a shared buffer: the trace replays from cold and the running kernel's
streams are the only ones being touched, so the footprint split (`sram_capacity_for`, which is
about what stays resident *between* kernels) does not apply inside the trace; a static
`alloc`/`pin` share still does (`trace_buffer_cap`).

**Measured (Kimi-K2 decode, batch 256, seq 8192, dp 8, whole step; attention in brackets):**

| config | step |
|---|---|
| B300, before this change (tile-granular, DDR 8 TB/s only) | 36.8 ms |
| B300, `dma_smart` (default) | 57.1 ms (attn 9.4) — `dma`-bound: 16 DMA engines = 3.9 TB/s < HBM |
| B300, `dma_page` | 196.7 ms (attn 126.6) — KV page thrash in a 132 MB L2 |
| B300, `l2_port` / `dma_page_tail_l2` | 209.5 ms (attn 26.9) — `l2port`-bound (0.83 TB/s); a split's KV range and an expert's weight strip are < 2 MB, so tail mode sends them all through the L2 ports |
| B200, `dma_smart` | 60.9 ms (attn 11.5) |
| B200, `dma_page` | 219.4 ms (attn 148.4) |
| B200 + buffer (buffer = 4 x L2 = 504 MB, L2 = L2/4 = 31.5 MB), `auto` -> `dma_page` into the buffer | 61.0 ms (attn 11.5) — every page fetched exactly once |

Whole 2 MB pages only work when the destination holds one per concurrent stream: a small,
hardware-managed L2 cannot (it evicts a page before its stream is done), a large buffer can — with
it, page mode reaches the same compulsory traffic as the smart DMA. At the single-kernel level
(`test_b200_with_buffer_holds_whole_pages_that_l2_thrashes_on`): 2 MB pages into B200's L2 move
7.3 GB for 302 MB of KV; into the buffer, 307 MB. Smart chunks are aligned to the stream's own
start and trimmed to its end, so they too move exactly the needed bytes. Prefill on B300 goes
345 ms -> 552 ms, mostly elementwise activation traffic limited by the L2 ports' outstanding
entries (`l2port`). Matrix outputs (GEMM C, attention O) are written back by DMA; elementwise
outputs and split partials through the L2 ports.

All trace addresses start at `memory.addressing.base` (default 0x8000_0000, 2 MB-aligned), and so
do the memory map and the CLI address dumps.

**HBM address map output** (`genResult/memmap_report.py`). The per-GPU memory map
(`interfaceAndRun/memmap.py`: every tensor one contiguous, 2 MB-aligned region, allocated in
execution order — weights layer by layer, then the KV cache per attention layer, then
double-buffered activations and a workspace) is written to `out/memmap/` as CSV, text and Excel
(`regions`; `address_map`, a Gantt-style chart of the address space by tensor kind; `hbm_ports`,
bytes per HBM port by kind, stacked; `summary`). Per-port bytes come from the interleave map
exactly (whole stripes counted, partial first/last stripe trimmed), not sampled. Every web run
shows the same map on the results page (address strip, per-port bars, a filterable region table,
Excel/CSV download via `/api/memmap`). Standalone: `python -m
tilesight.gpuTilingPerfHWModel.genResult.memmap_report ...` or `tilesight memmap ... [--out DIR]`.

See
`tests/test_paths_addr_buffer.py` (`test_fetch_modes_*`, `test_l2_op_bytes_*`,
`test_b200_with_buffer_*`).

## 5. Kernel lowerings
### 5.1 GEMM (`model/gpu_top/gpu_top.cpp`, `lower_gemm`)
`C[b] = A[b]·B[b]`, grid `batch·⌈M/bm⌉·⌈N/bn⌉·split_k`, `iters = ⌈⌈K/bk⌉/split_k⌉`.
- stages: explicit or max fitting SMEM (≤8); resident = min over max_blocks, SMEM, threads,
  registers (GPR), TMEM; the report shows `resident/limiter`, all binding resources joined
  with `+` (e.g. `1/smem+regs`).
- **Registers (GPR)**: regs/thread = 40 base + fp32 accumulator spread over consumer
  threads (no-TMEM parts) or +16 epilogue regs (TMEM parts). Above
  `occupancy.max_regs_per_thread` the excess spills: `spill:regs` action charges
  2×spilled bytes per block to the L2 lane (spread over iterations).
- body: `load_A`, `load_B` (TMA/LSU/split paths), `mma` (tc time with M padded to
  `tc_min_m`; smem operand reads). Cluster multicast `cluster_m` divides B's L2 traffic;
  `cta_pair` (2-CTA MMA, capability `compute.cta_pair`) also halves B's SMEM footprint/reads.
- epilogue: TMEM accumulator read + convert, store C (fp32 partials if split-K, followed
  by a reduce kernel).
- low-precision datapath ⇒ activations are quantized to it (quant kernel cost: TASK E5).
### 5.2 Attention (`model/gpu_top/gpu_top.cpp`, `lower_attention_decode`/`lower_attention_prefill`)
- decode: block = (batch, kv_head, head-group of ≤block_m heads, kv-split); per KV tile:
  load K(/V) → gemm_qk → softmax (SFU exp + CUDA + TMEM S/P traffic) → gemm_pv;
  qk/softmax/pv are `recurrent`; `consumers` ping-pong warpgroups hide the chain.
  MLA absorbed: `d_qk = kv_lora+rope`, `d_v = kv_lora`, one kv head, `v_in_k`.
  auto split-KV fills the machine; a combine kernel follows.
- prefill: causal FA-style; iterations averaged over q-tiles (per-block iters: TASK A3).
### 5.2b Attention families and implementations (`interfaceAndRun/attention_blocks.py`)
Three block types, all with `impl: flash | naive` (default `RunConfig.attn_impl`),
`causal`, `sliding_window`:

| block | shape fields | KV cache / token / layer (one GPU) | core |
|---|---|---|---|
| `mha` | heads, head_dim [, v_head_dim] | `heads/tp · (hd+vd) · kv_bytes` | H=KVH |
| `gqa` | heads, kv_heads, head_dim (kv_heads=1 → MQA) | `max(1,kv_heads/tp) · (hd+vd) · kv_bytes` | H/KVH query heads per KV head |
| `mla` | q_lora_rank, kv_lora_rank, qk_nope, qk_rope, v_head | `(kv_lora+rope) · kv_bytes` (replicated over tp) | absorbed: 1 latent KV head, d_qk=kv_lora+rope, d_v=kv_lora; expanded: MHA with d_qk=nope+rope, d_v=v_head |

Extra mha/gqa options: `fused_qkv`, `qk_norm`, `rope_dim`; MLA: `absorb: auto|true|false`.

**flash** → one fused kernel (§5.2): S/P never leave the SM; causal and sliding-window
tiles are skipped (per-q-tile iteration counts; average used until TASK A3).

**naive** → three ops, each a normal kernel lowering:
```
attn_scores : GEMM  M=(H/KVH)·Sq, N=Skv, K=d_qk, batch=B·KVH, out=attn_scores_dtype (fp32)
attn_softmax: elementwise  read S (fp32), write P (act dtype), 1 exp + ~5 flops / element
attn_pv     : GEMM  M=(H/KVH)·Sq, N=d_v, K=Skv, batch=B·KVH
```
Query heads sharing a KV head are folded into M so K/V are read once per KV head (the
GEMM's L2 model keys operands by batch index). Causal masking does not skip work.
S and P are counted as live activations → long-context naive prefill shows up as
activation-memory OVERFLOW and a `ddr`-bound softmax.

### 5.3 Elementwise
bytes in/out + CUDA flops + SFU ops, 32 KB chunks per block, LSU path.
### 5.4 Collectives (`model/gpu_top/gpu_top.cpp`, `allreduce`/`all_to_all`, Eq. 11)
Flat inside the fast domain, **hierarchical** beyond it: a group larger than
`network.nvlink.domain_size` reduce-scatters inside each node, all-reduces the 1/d shards
between nodes and all-gathers inside the node again; all-to-all only sends the
`(group-d)/group` fraction over the slow fabric. Modelling it flat overestimated a 16-GPU
all-reduce by ~5x (5040 µs vs 868 µs on B300 with 256 MB).
allreduce: min(ring `2(p−1)α + 2(p−1)/p·S/β`, recursive doubling `⌈log2 p⌉(α+S/β)`).
all-to-all: `⌈log2 p⌉α + S·(p−1)/p/β`. NVLink inside `domain_size`, scale-out NIC beyond.
`RunConfig.comm_overlap` hides a fraction (placeholder for a real overlap scheduler, E2).

## 6. Model layer semantics (`interfaceAndRun/lower.py`)
- Attention: `dp` groups of `tp` GPUs; heads split by tp; tokens per attention rank
  `T = batch/dp` (decode) or `batch/dp·seq` (prefill). MLA latent KV is replicated across
  tp ranks (not head-sharded).
- MLA: decode uses weight absorption (W_UK/W_UV as per-head batched GEMMs, attention
  over the 576-dim latent); prefill uses the non-absorbed path (kv_b up-projection, MHA
  with d_qk=192, d_v=128). Override with `RunConfig.mla_absorb`.
- Dense MLP / shared expert: column+row split by tp, all-reduce over tp.
- MoE: EP over `ep` GPUs (default all). Unique tokens per GPU `Tu = T/tp`. Pairs landing
  in the EP group `Tu·ep·k`; P(expert active) = `1−(1−k/E)^(Tu·ep)`; active local experts
  `E/ep·P`; tokens per active expert `Tu·ep·k/E/P`. Grouped GEMM batch = active experts —
  this is what makes small-batch decode stream (almost) all expert weights. Uniform
  routing is assumed (skew: TASK E7).
- lm_head: vocab/tp, only the last token per sequence.

## 7. Memory model (`model/memory.py`)
weights (per-op resident bytes × repeat, with sharding) + KV cache (per family, see §5.2b;
sliding-window layers store only `min(seq, window)` tokens) × seqs/rank + 2× peak live
activation + runtime reserve (4 GB, calib). Reports fit/overflow and max seqs per rank.

## 7b. Request-level runs (`gpuTilingPerfHWModel/interfaceAndRun/request.py`)
Config: `prompt_len` (P), `output_len` (O), `page_size`, `kv_reserve: peak|current`,
`decode_samples`, `prefill_batch`.
- TTFT = prefill step over P tokens (`prefill_batch` sequences).
- Decode step t has KV length P+t. The full model is evaluated at `decode_samples`
  values of t in [1, O]; TPOT(t) is integrated with the trapezoid rule
  (avg TPOT = Σ/O; attention cost is linear in KV length, weights constant).
- E2E = TTFT + O·avg TPOT.
- KV per request at step t = kv_bytes_per_seq(ceil((P+t)/page)·page) (sliding-window
  layers capped at the window). Peak KV = at t=O. Peak total = weights + peak KV +
  max(prefill act, decode act) + reserve (conservative).
- max concurrent requests/rank = free HBM / per-request KV (peak: P+O; current: P+O/2).

## 7c. Timeline view (`genResult/timeline.py`) — the paper's Figure 3(e)
The engine says how long a round is and what bounds it; `steady_timeline` reconstructs a
concrete schedule consistent with that, to render (SVG in the web UI, ASCII in the CLI):
actions are walked in topological order and start when their dependencies are done and
every lane they need is free; an action holds lane r for u_r and ends at max_r u_r + latency.
With `stages > 1` a load edge does not constrain the round (the consumer reads a buffer
filled stages-1 rounds earlier), which is exactly the load/compute overlap Figure 3(e)
draws; loads are therefore labelled with iteration i+stages-1. Reported alongside:
`round_s` (what the model uses), `makespan_s` (this schedule's length), the resource and
dependency bounds, and prologue/epilogue lengths. The picture is a rendering of the model,
not a second model: `tests/test_request_and_bounds.py` pins it to the engine's numbers.

Units: every event carries seconds and **cycles** (`clock_hz` from the hardware YAML), so
`timeline_text(tl, unit="cyc")` and the UI toggle show the same schedule in either unit.
`trace_text` writes the full text trace: header (hardware, clock, grid → waves, schedule,
round with both bounds, kernel time), one row per action event (phase, round, iteration,
start/duration in ns and cycles, per-lane occupancy), both gantts, and per-lane occupancy
of one round. CLI: `tilesight kernel ... --unit cyc --trace-out trace.txt`; the web UI has
a "download trace (.txt)" button.

**Per-cycle CSV.** `cycle_csv(tl, full=)` emits one row per GPU cycle and one column per
hardware unit: `cycle, time_ns, phase, round, <lane>…, <lane>_busy…` where `<lane>` is the
action occupying that unit in that cycle (empty when idle) and `<lane>_busy` is the fraction
of the cycle it is busy. A unit is modeled as occupied contiguously from the action's start,
so a 37 ns DDR share of a 162 ns round shows as ~70 busy cycles followed by idle ones.
`full=False` covers the drawn rounds, `full=True` prologue + every iteration + epilogue.
CLI `--csv-out FILE [--csv-full]`; web UI: two download buttons (served by `GET /api/csv?job=…`).

**Excel grid.** `report/excel.py` writes the same cycle grid as a workbook: one row per
cycle, one column per unit, the cell holding the tile that occupies it (`A#7`, `B#7`,
`MMA#5`) and **coloured by iteration**, so one tile keeps its colour as it moves from
HBM/L2 to SMEM and, `stages-1` rounds later, into the tensor core. The legend sheet states
the stagger in rounds, cycles and ns, plus every assumption; the summary sheet has the round
composition and lane occupancy. CLI `--xlsx-out`, web UI "cycles Excel (coloured)".

**Everything is a cycle count.** Each lane time comes from a closed form and the clock:
`cycles = round(seconds x clock_hz)`, where seconds is `bytes / bandwidth` for memory lanes
(bandwidth per SM = min(total/active_SMs, per_sm_cap)), `bytes / (bytes_per_clk x clock)`
for SMEM/LDS, `flops / (peak x efficiency / SMs)` for the tensor core, `ops / rate` for SFU,
and a fixed latency term (path issue + hit-weighted L2/HBM latency) added to the node. A tile
that needs 200 or 500 cycles on a lane simply occupies that many rows in the Excel grid.

**Extra on-chip shared buffer.** `memory.sram` (capacity, bandwidth, latency, assoc) adds a
level between L2 and HBM and a `sram` lane. Bytes that miss L2 are charged to it, and it
serves the share of a tensor that stays resident across calls
(`resident_frac = min(1, capacity / tensor_bytes)`) — which is where a 64 MB A/B staging
buffer actually pays off: not inside one streaming kernel, but because the next decode step
re-reads the same weights/KV. See `tests/test_request_and_bounds.py`.

**Using a large on-chip buffer (`dse/buffer.py`).** Once the buffer is a multiple of L2, what
you put in it matters more than its size. Policies in `memory.sram`:
`policy: cache|pin` (+ `pin: {weight, kv, act}` shares; a shared cache is split in proportion
to the classes' footprints so no class double-counts the capacity), `keep_intermediates`
(elementwise intermediates never reach HBM), `bypass_l2` (resident bytes skip the L2
datapath, otherwise L2 becomes the next wall), `prefetch` (resident bytes cost no exposed
memory latency) and `costream` (buffer and HBM serve in parallel, bandwidths add).
Residency is capacity-share based (`resident_frac`) against the GLOBAL per-class footprint
that `run_model` installs, because one buffer serves every op.

Two ways to choose the split:
- `best_alloc` (closed form): pinning x_c bytes of class c removes `T_c·min(1, x_c/F_c)` HBM
  bytes, so value per byte is `T_c/F_c` (traffic / footprint) and greedy fill by that ratio is
  exactly optimal *for traffic*. Free — no model re-runs.
- `optimize_buffer` (search): re-runs the model per split × policy combination, so it captures
  what the closed form cannot — traffic removed off the critical path is worth nothing, and
  `bypass_l2`/`prefetch` change which lane binds.
On Kimi-K2 decode with a 32 GB buffer on B300 the closed form picks act > kv > weight and
lands at 27.65 ms, while the search finds weight-pinned + bypass_l2 + prefetch at 25.72 ms
(1.21x over no buffer): KV traffic is real but its attention kernels are latency-bound, so
removing those bytes buys less than removing expert-weight bytes.

**Per-unit latency.** Every unit has a latency, configurable in cycles (memory levels may use
ns instead): `compute.mma_latency_cycles` (default 64 — one tile MMA issue-to-result),
`compute.cuda_latency_cycles`, `compute.sfu_latency_cycles`,
`memory.onchip.smem.latency_cycles`, `memory.onchip.tmem.latency_cycles`,
`memory.l2.latency_ns`, `memory.ddr.latency_ns`, `memory.sram.latency_ns`, plus the load
path's issue overhead. Latency never changes throughput; it lengthens dependency chains.
Where it lands depends on whether the dependency is loop-carried:
- **independent work** (a tile MMA, a multi-buffered load): several are in flight, so the
  latency shows up only in the pipeline **fill**. Raising the MMA latency from 11 to 256
  cycles moves a B300 FP8 GEMM from 82.5 to 83.1 µs (fill 2.4 → 2.9 µs) and it stays tc-bound.
- **loop-carried work** (`recurrent`: the online-softmax state, the attention accumulator):
  the result must land before the next iteration, so the latency divided by `consumers` is a
  hard floor on the round. At 1024-cycle MMA latency MLA decode flips to `latency`-bound.
This split is enforced by tests; the engine computes the fill from the full critical path and
the steady bound from a path where non-recurrent latency is removed.

**L2 blocks, ports and outstanding requests.** `memory.l2` now describes the physical
structure: `blocks` (independent L2 blocks, each with its own load/store port(s)),
`ports_per_block`, `bytes_per_clk_per_port`, `line_bytes`, `sector_bytes`. A level can never
exceed blocks x ports x width x clock, whatever aggregate bandwidth is quoted (on B300 this
caps the modelled L2 at 17.4 TB/s instead of 20.5). `memory.outstanding.per_sm_lines` is the
MSHR-style limit on cache lines in flight per SM and gives a Little's-law ceiling
`BW_per_SM <= lines x line_bytes / latency`, which is why a latency-bound kernel does not
speed up when HBM gets wider: 64 lines -> 10 GB/s/SM and a 345 µs GEMM, 512 lines ->
77 GB/s/SM and 83 µs, 2048 lines -> the 180 GB/s per-SM cap and 82.7 µs.

**Where a DMA drops the data.** `memory.dma`: `engines` and `destination`:
`smem` (global -> SMEM, the bytes still cross the L2 datapath), `l2` (the engine fills L2 and
the consumer reads it back — extra L2 write traffic) or `bypass` (engine -> consumer, the L2
datapath is skipped). Same B300 GEMM: 82.7 / 131.4 / 82.5 µs with L2 occupancy 68% / 84% / 2%.

**L1 ownership.** `memory.l1`: `owner: sm | cluster`, `capacity_KB`, `cluster_size` — Hopper
and Blackwell share through DSMEM inside a thread-block cluster, AMD keeps a per-CU vector
cache. Recorded and used for cluster gating today; a separate `l1` lane is TASKS G7.

**Figure 3(d)(e)(f) as an artifact.** `report/figure3.py` renders the paper's three panels as a
self-contained HTML page (or JSON): (d) the per-action resource vectors, the DAG and the
envelope with both bounds written out, (e) the timeline over the lanes with the round
boundaries and the software-pipeline offset, (f) the per-tile report — latency in ns and
cycles, waves, occupancy and its limiter, lane utilisation, cache hit, overlap rate and the
time charged to each resource:tensor. CLI: `tilesight kernel … --fig3 fig.html --fig3-json fig.json`.

**DMA vs ordinary loads.** Load paths carry an engine description, not just a bandwidth:
`engine: dma|lsu`, `smem_direct`, `multicast`, `regs_per_thread`, `issue_bytes_per_clk`.
A DMA engine (TMA, CDNA5 TDM, `buffer_load→LDS`) issues a descriptor, writes SMEM/LDS
directly, costs no registers and no SM issue slots, and can multicast to a cluster. An
ordinary vector load charges the `cuda` lane for address generation, adds a second SMEM
crossing (register staging), costs `regs_per_thread` that reduce occupancy, and cannot
multicast — so a `cluster_m > 1` tile is rejected on such a path. On a B300 FP8 GEMM the two
differ by ~7% in time and far more in which lane binds (LSU: cuda 66%, smem 73%).

**Mega-tile staging.** `memory.sram.stage: {share_blocks: N}` models a panel staged into the
on-chip buffer by one DMA transfer and consumed by up to N blocks (async copy at panel
granularity, shared between shaders). B panels are shared by the blocks along M, A panels by
those along N, so the HBM fraction of that operand is divided by the achievable share.

**Where addresses come from (`model/memmap.py`).** The model layer knows how many layers this
GPU holds and every matrix shape in them, so addresses are an allocation: walk the ops in
execution order and hand out 2 MB-aligned regions from a base offset — one per *layer
instance* weight tensor (61 layers x ~9 tensors = 548 regions, 141.7 GB on Kimi-K2), one per
attention layer's KV cache, plus double-buffered activations and a workspace. `Region.tile_addr(i, j, bm, bn, layout)`
then gives the byte address of any tile. `tilesight memmap --model … --base 0x0 [--csv map.csv]`
prints or exports the map. A standalone kernel study uses `kernel_addr_fn`, which lays out
A | B | C from a base offset instead.

Addresses flow into every export: the text trace gains `addr / sl / pt` columns, the cycle CSV
gains `<lane>_addr` columns, and the Excel workbook gains an **addresses** sheet (phase, round,
iteration, action, tile label with its colour, address, size, L2 slice, HBM port, start cycle).
Example finding this makes visible: with row-major B tiles of 256 KB and a 2 KB interleave over
16 slices, every B tile aliases onto slice 0 — a power-of-two stride collapsing onto one slice,
which is invisible without addresses.

**Address → unit mapping is configuration (`memory.addressing`).** Two independent maps,
because the L2 side and the memory side are different hardware: `l2` (slices / load ports) and
`ddr` (HBM ports, what the DMA engines target). Each takes `ports`, `granularity_KB`,
`addr_bits` (48) and a `mode`:
`interleave` (port = (addr >> log2 gran) % ports, the usual case), `range` (the address space
split into equal contiguous ranges) and `hash` (interleave after XOR-folding the higher address
bits, which breaks power-of-two stride aliasing). `tilesight addrmap --gpu-tiling-perf-hw-model b300 [--side l2|ddr]
[--decode 0x…] [--csv table.csv]` dumps the mapping — which stripes or which range each port
owns — and decodes individual addresses. Measured effect on one wave of a 4096³ FP8 GEMM
(B300, 16 slices): 1 KB interleave 1.33x imbalance (75% of peak), 2 KB 2.00x (50%), `hash`
1.04x (96%), `range` 16x (6%, everything in one range). The mapping feeds the spread analysis
and the per-event `slice`/`port` columns in the trace, CSV and Excel exports.

**Tile-granularity addressing (`report/addressing.py`).** Tiles get real addresses from a
layout — `row`, `col`, `swizzle:G`, `xor:B`, `zorder` — and the *set touched together by one
wave* is spread over L2 slices and HBM ports at `memory.interleave_KB` granularity. A tile is
not a cache line: it covers several interleave chunks and its bytes are spread over them. The
report gives per-slice/per-port bytes, an imbalance factor (max/mean) and the implied fraction
of peak; `with_conflict_penalty` folds that into `efficiency.l2/ddr`. Example (B300, 16 slices,
8 ports, one wave of a 4096³ FP8 GEMM): row-major A gives 2.00x slice imbalance (50% of peak),
z-order 1.33x (75%), and 64x64 tiles 3.14x (32%) — smaller tiles cover fewer chunks and spread
worse. The full set of tile addresses is layout-independent; what a layout changes is which
tiles are live at the same time, which is why the analysis is over a wave window.

**Trading L2 for buffer (`l2_tradeoff`).** Spend one SRAM budget on a fast L2 or a big slow
buffer, bandwidth scaling with capacity. At an L2-sized budget (126 MB on B300) shifting
silicon to the buffer loses — every load still crosses the L2 datapath, so the narrower L2
becomes the limiter; at 8 GB, moving 90–98% into the buffer wins ~4% on Kimi-K2 decode. The
buffer has to be large relative to the working set, not just relative to L2.

**Many SMs.** Like the paper (§3.4), one representative SM is modeled and waves are
aggregated rather than simulating each SM: `WaveDecompose` splits the grid into full waves
of `SMs x resident` blocks plus a tail wave; resident blocks on an SM are interleaved
instances of the same pipeline (their lane work is multiplied by `blocks_per_sm`); shared
lanes (L2/DDR) are divided by the *active* SM count, so a tail wave gives each SM a larger
share; and cross-SM locality enters through the tile reuse-distance sequence, which is
generated over the whole wave's block order. The trace header reports the wave split.
Because every SM in a wave runs the identical pipeline here, the UI never draws N SM rows:
`machine_timeline` gives a wave-level bar (how many full waves, the tail wave and its active
SMs, per-wave duration) above the one-SM detail view. Per-SM differences would need a load
imbalance model (TASKS E7) — until then, N identical rows would carry no extra information.

## 8. DSE (`dse/sweep.py`)
**Queueing (G12).** A shared lane close to saturation does not only run out of bandwidth, it
makes every access wait. `memory.queueing.coef` applies an M/D/1-style latency multiplier
`1 + coef·u/(1-u)`, capped by `max_factor`, where `u` is the busiest shared lane's utilisation
in that round. Throughput is still bounded by the resource term, so this only moves
latency-sensitive parts (fill, loop-carried chains): a decode GEMM goes 20.2 → 22.2 µs at
coef 0.5. Implemented in both engines (parity-tested).

**Overlap scheduler (E2).** `RunConfig.overlap_mode`: `none` (default, every collective
exposed), `stream` (a collective hides behind the compute that follows it in the same layer),
`two_batch` (two micro-batches in flight, so it can hide behind the whole layer's compute —
the DeepEP trick) or `manual` (the old scalar `comm_overlap`). `overlap_efficiency` (0.8)
says how much of the hideable compute really overlaps. Kimi-K2 decode tp=2: 38.3 ms with
nothing hidden, 35.9 ms with either overlap mode.

**Linked knobs.** Compute peaks in the YAML are whole-GPU numbers and the engine derives the
per-SM rate as `peak / sms`, so sweeping `sms` alone keeps total FLOPS constant and makes every
SM weaker — an audit caught this as a monotonicity violation. `default_links` supplies the
companion overrides (scale `tc_dense_tflops.*`, `cuda_fp32_tflops`, `sfu_tops` with the SM
count) and `sweep`/`required_value` apply them automatically; pass `auto_link=False` to opt
out. With the link, doubling the SMs of a B300 speeds a compute-bound FP8 GEMM by 1.34x — not
2x, because L2 then becomes the limiter.
`sweep(path, values, linked=f)` re-runs the model per value (kernels are re-lowered, so
tile auto-search re-adapts to the new hardware — important for fair comparisons).
`required_value` bisects the smallest value meeting a step-time target (assumes
monotonicity, which tests enforce for bandwidth knobs). Typical linked knobs: L2 BW ∝ DDR
BW; `memory.l2.per_sm_max_GBps` (otherwise it becomes the wall above ~25 TB/s on B300).

## 8b. What is *not* addressed (and what it would take)
- **Addresses.** Tiles are identified symbolically (tensor + tile coordinate), never by
  physical address. That is enough for capacity/reuse effects but not for L2 *set* conflicts,
  SMEM/LDS *bank* conflicts, or HBM *channel/port* imbalance. `memory.ddr.ports` is recorded
  and reported but the ports are aggregated into one bandwidth; per-port modelling needs an
  address map (tensor layout, swizzle, interleave granularity) — TASKS G8.
- **Contention that *is* visible**: L2 and HBM bandwidth contention between SMs (shared lanes
  divided by active SMs, `per_sm_max_GBps` cap), SMEM bandwidth per SM, capacity contention
  through occupancy (SMEM/TMEM/registers) and through the reuse-distance cache model
  (concurrent blocks in a wave evict each other). Bank/set conflicts are approximated by the
  associativity term and the SMEM efficiency factor.
- **DMA.** The async copy engines that matter inside a kernel (TMA, CDNA5 TDM,
  `buffer_load→LDS`) are modelled as load paths with their own issue-rate lanes. A separate
  copy engine doing H2D/P2P transfers concurrently with compute is not — it would be a lane
  plus an overlap rule (TASKS G9).

## 9. Known limitations
No warp-level issue model, no register allocation, no instruction-level scheduling
(same as the paper). Expert-load imbalance, MTP/speculative decoding, paged-KV page
effects, quantization kernels, chunked prefill and real compute–comm overlap are
backlog items (docs/TASKS.md). All `[calib]` constants are placeholders until the
calibration suite runs on real hardware.
