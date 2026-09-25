# Interface

Three inputs, three schemas, one rule: **the model only ever sees config.** No device name, no
vendor assumption and no default value lives inside `engine/`, `kernels/` or `model/` — a test
enforces it.

```
   GPU config                 workload config              tile policy
   (what the part is)         (what runs on it)            (how it is tiled)
        │                            │                          │
        ├── slice form  ─────┐       │                          │
        │   hw/slice_config  │       │                          │
        │        derive()    ▼       ▼                          ▼
        └── flat form ──►  HardwareSpec ──►  the model  ◄────────┘
            hw/schema.py                     engine + kernels + model
```

## 1. GPU config

Two equivalent forms.

**Flat form** (`hw/schema.py`, 99 fields) — a datasheet view: peaks, capacities, latencies,
ports. This is what the model consumes. `tilesight config --list`, `tilesight config --validate
my_gpu.yaml`. Every field carries a tag:

| tag | meaning |
|---|---|
| `spec` | copy it from the vendor |
| `calib` | measure it on the part; the shipped value is a placeholder |
| `policy` | a modelling choice, not a property of the silicon |
| `loss` | a derating knob — **every one defaults to no loss** |

**Slice form** (`hw/slice_config.py`, 42 fields) — an architect's view, and what the web app
asks for:

- **shader core**: clock, wave32 units per core, tensor cores per core, resident waves per SIMD,
  shared memory and GPR file (each either shared by the tensor cores or split evenly)
- **tensor core**: MMA tile M/N/K, cycles per tile, datatype (simplification: all tensors use
  the KQV datatype), attention tile M/N
- **shader slice**: cores per slice (they share the slice's L1), slice count, L1 size / latency /
  width. Slices are copies of each other; the model staggers them in time
- **memory slice**: count, and per slice its HBM channel, L2 (may be **0** — then the port and
  the DMA talk straight to the on-chip buffer), L2 port width and latency, DMA ports
- **addressing**: `linear` or `interleave` with a configurable stripe (1 KB, 4 KB, …)
- **on-chip buffer**: capacity, bandwidth, latency, and **what is pre-allocated on it** at tile
  granularity: `none` | `ab` | `abc` | `abc_kv` | `abc_kv_moe` (the last one because which
  experts are active is not known ahead of time). It is logically one buffer, not part of any
  memory slice
- **gmem**: upstream (towards the shader slices) and downstream (towards the memory slices)
  bandwidth, read and write separately

`derive()` turns that into the numbers you want to see before running anything — TFLOP/s per
tensor core and per shader core, whole-GPU PFLOP/s, vector TFLOP/s, aggregate L1 / L2 / HBM /
buffer bandwidth and capacity, per-tensor-core SMEM and GPR — and `to_hardware_spec()`
translates it into the flat form. `tilesight gpu --file my_gpu.yaml [--out flat.yaml]`.

## 2. Workload config

`model/workload.py`, 36 fields, **one layer on one GPU**. The model tells the simulator:

- attention: type (`mha` / `gqa` / `mla`), `impl` (`flash` / `naive`), head counts, Q/K/V dims
  (MLA: `q_lora_rank`, `kv_lora_rank`, `qk_nope`, `qk_rope`, `v_head`), causal, sliding window
- FFN: `mlp` or sparse `moe` with `experts` (all of them resident — which are active is not
  known), `topk` (the sparsity), `d_ff`, `shared_experts`
- datatypes: weight, expert, activation, **KV cache**, compute, attention compute
- run: phase, batch, `seq_len`, `max_seq_len` (the capacity question), tile policy
- `layers`: how many layers of this kind this GPU holds

From the dims, dtypes, layer count and max sequence length alone,
`memory_breakdown()` gives the footprint before any simulation:

```
kimi-k3-10L — 10 layers, 16/112 experts per token
  weights   39.77 GB  (attention 1.44 + experts 36.99 + dense 1.34)
  KV cache   3.02 GB  (1152 B per token per layer x 10 x 8192 x 32)
  total     42.79 GB
```

Example preset: `model/presets/kimi_k3_10L.yaml` — 10 Kimi-K3-shaped layers, flash attention,
sparse MoE. `tilesight config --workload` lists every field.

## 3. Tile policy

Part of the workload (`run.gemm_tile`, `run.attn_tile`): `auto` searches the space, or give a
fixed tile as JSON. When the GPU config fixes the attention tile
(`compute.attention_tile_m/n`, which the slice form always sets), the model uses that tile and
only searches the pipeline knobs around it.

## How the model is organised

The model mirrors the config: three blocks, and a shader slice built from one core.

```
shader slice ──┐
 tc cuda sfu   │                on-chip buffer            memory slice
 smem tmem     ├── switch ──►  one piece per slice,  ──►  L2 port + DMA port + HBM
 L1 (slice)    │               interconnected, so
 load paths ───┘               logically one buffer
               └──────────────────────────────────────►  (bypass the buffer)
```

The buffer sits **between** the shader slices and L2, not underneath it. A request either goes
over the switch to the buffer — and then touches neither L2 nor HBM — or bypasses it straight
to a memory slice. Bypassing is not automatically faster: with `l2_MB: 0` the ports go straight
to HBM and pay its latency (hundreds of cycles), which the model weighs against the buffer route.

The switch has one **aggregate** bandwidth (`memory.sram.switch_TBps`, or
`onchip_buffer.switch_bytes_per_clk` in the slice form); one slice may draw its `1/slices`
share, and it shows up as its own `switch` lane that can be the bottleneck on its own.

- **one core is modelled, a slice is composed from it, and slices are copies.** Per-core lanes
  (`tc`, `cuda`, `sfu`, `smem`, `tmem`) carry seconds on one core. `L1` is a *slice* resource:
  its modelled total is `slices x per-slice bandwidth`, so the cores in a slice share it
  exactly. Slices are identical, so the wave decomposition (paper §3.4) covers them: full waves
  of `cores x resident blocks`, then a tail wave with fewer active cores and therefore a larger
  share of the shared lanes.
- **what the paper's machinery computes stays the same**: the action DAG and its resource
  vectors, the pipeline envelope, and the tile reuse distance — now with the reuse distance run
  through a deterministic per-partition tile cache, so "which tile is in L2 / L1" is an answer,
  not a probability.
- **every report rolls up into the three blocks.** `ModelReport.by_domain()`, the `bound by
  block` line in the text report, the web results page, and the Excel / PDF column groups
  (`shader slice · cores`, `shader slice · L1/scratchpad`, `shader slice · load paths`,
  `on-chip buffer`, `memory · L2 ports`, `memory · HBM`). Two views: `by_domain()` splits the
  critical time, `activity_by_domain()` shows how busy each block is whether or not it limits —
  a fast on-chip buffer is 0% of the critical time but 6% busy.

Example (Kimi-K3 10 layers on B300): decode is `memory 79%, shader_slice 16%`, prefill is
`shader_slice 81%, memory 18%`.

## Flash attention is a choice

`attention.impl: flash | naive` is part of the workload, because the two are different
computations: flash keeps S and P on chip, naive writes the score matrix to HBM and reads it
back. On an 8K prefill of the K3 layer: flash 92.8 ms vs naive 226.4 ms (2.44x), attention
14.6 vs 148.2 ms, and naive needs 70 GB more activation memory — and the block mix flips from
shader-bound to memory-bound. `compare_attention_impl()` runs both; the web app has a checkbox.

## Putting it together

```bash
tilesight gpu --file examples/slice_gpu.yaml \
              --workload python/tilesight/model/presets/kimi_k3_10L.yaml   # derived numbers
tilesight serve --host 0.0.0.0 --port 8000                                  # the 5-step web app
```

JSON API: `GET /api/schema` (flat + workload), `GET /api/slice_schema`,
`POST /api/derive {slice_cfg, workload}` → derived numbers + memory breakdown + architecture SVG,
`POST /api/jobs {mode: "workload", config: {slice_cfg | hw_yaml | hw, workload}}`,
`GET /api/jobs/<id>` (progress + live log), then `/api/xlsx|csv|pdf?job=<id>`.
