"""GEMM / batched GEMM / grouped (MoE) GEMM -> Kernel lowering.

C[b] (M x N) = A[b] (M x K) @ B[b] (K x N)    for b in range(batch)
  * dense linear      : batch=1
  * batched (per-head): batch=H           (MLA absorbed W_UK / W_UV)
  * grouped (MoE)     : batch=#active experts, M = tokens per active expert
Tile config -> grid, K-loop, occupancy, per-iteration action vector, L2 misses.
"""
from __future__ import annotations

import math

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import DTYPE_BYTES, HardwareSpec
from tilesight.model.ir.kernel import Action, Kernel
from tilesight.model.engine import backend
from tilesight.model.kernels.tiles import TileConfig, parse_path_split


# ----------------------------------------------------------------------------- helpers
def gload(cur_gpu_config: HardwareSpec, name: str, nbytes: float, miss: float, load_path: str,
          deps: list[int] | None = None, sram_miss: float | None = None,
          l1_miss: float | None = None) -> Action:
    """One tile load, routed through the topology.

        shader slice ──┐
                       ├── switch ── on-chip buffer (one piece per slice, all interconnected,
        shader slice ──┘                             so logically one buffer)
                       └───────────────────────────► memory slice (L2 port, DMA port, HBM)

    The buffer sits BETWEEN the shader slices and L2, not underneath it, and a request may
    bypass it and go straight to the memory slice. So a tile that is resident in the buffer is
    answered over the switch and never touches L2 or HBM; everything else goes to the memory
    slice, hits L2 if L2 exists, and otherwise pays the HBM latency — which is why bypassing
    the buffer is not automatically the faster route (HBM latency is hundreds of cycles).

    Fractions, all of `nbytes`:
      `l1_miss`    what the slice's L1 could not answer (DMA paths into SMEM skip L1 entirely)
      `sram_miss`  what the buffer could not answer, i.e. what goes to the memory slice
      `miss`       of the bytes reaching the memory slice, what L2 could not answer either
    """
    # --- L1 (private to the slice) ------------------------------------------------------
    l1_m = 1.0
    if l1_miss is not None and cur_gpu_config.l1_capacity_bytes > 0:
        l1_m = max(0.0, min(1.0, l1_miss))
    work: dict[str, float] = {}
    if l1_m < 1.0:
        work["l1"] = cur_gpu_config.l1_time_per_sm(nbytes)
    after_l1 = nbytes * l1_m

    # --- the buffer, reached over the switch --------------------------------------------
    buf = cur_gpu_config.sram
    from_buffer = 0.0
    if buf is not None:
        to_memory_frac = miss if sram_miss is None else min(miss, sram_miss)
        from_buffer = max(0.0, min(1.0, 1.0 - (to_memory_frac / miss if miss > 0 else 0.0)
                                   if miss > 0 else (0.0 if sram_miss is None else 1.0 - sram_miss)))
        if buf.get("costream", False) and from_buffer > 0 and miss > 0:
            # the buffer and the memory slices are independent datapaths: splitting the resident
            # bytes so both finish together beats serving everything from one of them
            bws = buf["bandwidth_TBps"] * cur_gpu_config.eff("sram")
            bwd = cur_gpu_config.get("memory.ddr.bandwidth_TBps") * cur_gpu_config.eff("ddr")
            S, D = from_buffer, 1.0 - from_buffer
            x = min(max((S * bwd - D * bws) / (bws + bwd), 0.0), S)
            from_buffer -= x
        work["sram"] = after_l1 * from_buffer                   # the buffer itself
        work["switch"] = after_l1 * from_buffer                 # slice <-> buffer interconnect

    # --- the memory slice ----------------------------------------------------------------
    to_memory = after_l1 * (1.0 - from_buffer)
    work["l2"] = to_memory
    work["ddr"] = to_memory * miss
    lat = 0.0
    for path, frac in parse_path_split(load_path).items():
        real = cur_gpu_config.resolve_path(path)                 # AMD parts have no TMA: fall back to theirs
        work[f"path:{real}"] = work.get(f"path:{real}", 0.0) + cur_gpu_config.path_time_per_sm(real, nbytes * frac)
        lat = max(lat, cur_gpu_config.path_latency_s(real))
        if cur_gpu_config.path_is_dma(real):
            dest = cur_gpu_config.dma_destination                # smem | l2 (engine fills L2) | bypass
            if dest == "bypass":
                work["l2"] -= to_memory * frac
            elif dest == "l2":
                work["l2"] += to_memory * frac
        ipc = float(cur_gpu_config.path_attr(real, "issue_bytes_per_clk", 0) or 0)
        if ipc > 0:                                  # vector loads occupy SM issue slots
            work["cuda"] = work.get("cuda", 0.0) + nbytes * frac / (ipc * cur_gpu_config.clock_hz)
        if not cur_gpu_config.path_attr(real, "smem_direct", True):
            work["smem"] = work.get("smem", 0.0) + cur_gpu_config.smem_time_per_sm(nbytes * frac)
    work["l2"] = max(0.0, work["l2"])

    # --- latency: whichever route the bytes took ----------------------------------------
    buf_lat = float((buf or {}).get("latency_ns", 0)) * 1e-9
    mem_lat = ((1 - miss) * cur_gpu_config.get("memory.l2.latency_ns") + miss * cur_gpu_config.get("memory.ddr.latency_ns")) * 1e-9
    if cur_gpu_config.get("memory.l2.capacity_MB", 0) <= 0:
        mem_lat = cur_gpu_config.get("memory.ddr.latency_ns") * 1e-9   # no L2: every miss pays HBM latency
    lat += from_buffer * buf_lat + (1.0 - from_buffer) * mem_lat
    if buf is not None and buf.get("prefetch", False):
        lat *= max(0.0, 1.0 - from_buffer)           # resident bytes can be staged ahead of time
    return Action(name, work, deps or [], latency_s=lat)


def gstore(name: str, nbytes: float, deps: list[int] | None = None, ddr_frac: float = 1.0,
           sram_frac: float = 0.0) -> Action:
    work = {"l2": nbytes, "ddr": nbytes * ddr_frac}
    if sram_frac:
        work["sram"] = nbytes * sram_frac
    return Action(name, work, deps or [])


BASE_REGS = 40            # per-thread registers for addresses, loop state, softmax scalars  [calib]


def reg_estimate(cur_gpu_config: HardwareSpec, acc_bytes: int, threads: int) -> tuple[int, int]:
    """(registers per thread, spilled bytes per block) for register-resident accumulators.

    On TMEM parts (Blackwell) the accumulator lives in TMEM, so only BASE_REGS plus an
    epilogue fragment is needed. On Hopper-class parts the fp32 accumulator is spread over
    the consumer threads' registers."""
    max_rpt = int(cur_gpu_config.get("occupancy.max_regs_per_thread"))
    acc_regs = 0 if cur_gpu_config.has_tmem else -(-acc_bytes // (4 * threads))
    rpt = BASE_REGS + acc_regs + (16 if cur_gpu_config.has_tmem else 0)
    spill = max(0, rpt - max_rpt) * 4 * threads
    return min(rpt, max_rpt), spill


def tensor_class(name: str) -> str:
    """weight | kv | act — which pool of the on-chip buffer a tensor draws from."""
    if "weight" in name:
        return "weight"
    if "kv_cache" in name:
        return "kv"
    return "act"


def l1_miss_fractions(cur_gpu_config: HardwareSpec, keys, addrs, sizes, streams, n_streams: int,
                      blocks_per_sm: int, blocks_in_wave: int) -> list[float] | None:
    """Miss fractions in L1, simulated the same deterministic way as L2.

    L1 is private to an SM (or to a cluster, if `memory.l1.owner: cluster`), so it only sees the
    accesses of the blocks resident on that SM — not the whole wave. Requests that take a DMA
    path straight into SMEM/LDS never touch it; that is decided by the caller."""
    cap = cur_gpu_config.l1_capacity_bytes
    if cap <= 0 or not keys:
        return None
    from tilesight.model.engine.cache_sim import simulate as _sim
    share = max(1, blocks_per_sm)
    if str((cur_gpu_config.get("memory.l1") or {}).get("owner", "sm")) == "cluster":
        share *= int((cur_gpu_config.get("memory.l1") or {}).get("cluster_size", 1))
    step = max(1, blocks_in_wave // share)          # keep this SM's slice of the wave
    sel = [i for i in range(len(keys)) if (i // 2) % step == 0]
    if not sel:
        return None
    sim = _sim([keys[i] for i in sel], [addrs[i] for i in sel], [sizes[i] for i in sel],
               [streams[i] for i in sel], n_streams, cap, 1, lambda a: 0,
               str((cur_gpu_config.get("memory.l1") or {}).get("policy", "lru")))
    return sim.miss_fraction


def resident_frac(cur_gpu_config: HardwareSpec, footprint_bytes: float, klass: str = "weight") -> float:
    """Fraction of a tensor class that stays resident in the optional on-chip buffer.

    A staging SRAM does not help a tensor streamed once inside one kernel — its value is that
    it is still there on the next call (the next decode step re-reads the same weights, the
    same KV). Modelled as a capacity share:  resident = min(1, capacity(klass) / footprint).
    A model-level run installs the GLOBAL footprint of each class on this GPU
    (`memory.sram.footprint`), because one buffer is shared by every op; a standalone kernel
    study falls back to that kernel's own tensor size.
    """
    if cur_gpu_config.sram is None:
        return 0.0
    fp = float(cur_gpu_config.get(f"memory.sram.footprint.{klass}") or footprint_bytes)
    cap = cur_gpu_config.sram_capacity_for(klass)
    return min(1.0, cap / fp) if fp > 0 and cap > 0 else 0.0


def tensor_class(name: str) -> str:
    """Map a tensor label (the `names` of a lowering) to a buffer class."""
    n = name.lower()
    if "weight" in n:
        return "weight"
    if "kv" in n or n in ("k", "v"):
        return "kv"
    return "act"


def occupancy(cur_gpu_config: HardwareSpec, smem_bytes: int, acc_bytes: int, threads: int = 256,
              extra_regs: int = 0) -> tuple[int, str]:
    """(resident blocks per SM, limiting resource).  limiter ∈ max_blocks|smem|threads|tmem|regs."""
    rpt, _ = reg_estimate(cur_gpu_config, acc_bytes, threads)
    rpt += extra_regs                    # LSU address/staging registers (DMA needs none)
    lim = [("max_blocks", int(cur_gpu_config.get("occupancy.max_blocks_per_sm"))),
           ("smem", cur_gpu_config.smem_per_sm // max(1, smem_bytes)),
           ("threads", int(cur_gpu_config.get("occupancy.max_threads_per_sm")) // threads),
           ("regs", int(cur_gpu_config.get("occupancy.regs_per_sm")) // max(1, rpt * threads))]
    if cur_gpu_config.has_tmem:
        lim.append(("tmem", cur_gpu_config.tmem_per_sm // max(1, acc_bytes)))
    val = min(v for _, v in lim)
    name = "+".join(n for n, v in lim if v == val)       # every resource binding at the minimum
    return max(0, val), name


def grouped_raster(mt: int, nt: int, group_m: int):
    """Triton-style GROUP_M swizzle: yields (m, n) in issue order."""
    group_m = max(1, min(group_m, mt))
    for g0 in range(0, mt, group_m):
        rows = min(group_m, mt - g0)
        for n in range(nt):
            for r in range(rows):
                yield g0 + r, n


# ----------------------------------------------------------------------------- GEMM
_CACHE: dict = {}


def lower_gemm(cur_gpu_config: HardwareSpec, name: str, M: int, N: int, K: int, *, batch: int = 1,
               a_dtype="bf16", b_dtype="bf16", c_dtype="bf16", compute_dtype="bf16",
               tile: TileConfig = TileConfig(), names: tuple[str, str, str] = ("act", "weight", "out")
               ) -> list[Kernel] | None:
    """Returns the kernel list (GEMM [+ split-K reduce]) or None if the tile is illegal.
    `names` = tensor labels of (A, B, C) used in bottleneck attribution."""
    key = (cur_gpu_config.fingerprint, name, M, N, K, batch, a_dtype, b_dtype, c_dtype, compute_dtype, tile, names)
    if key not in _CACHE:
        _CACHE[key] = _lower_gemm(cur_gpu_config, name, M, N, K, batch, a_dtype, b_dtype, c_dtype, compute_dtype, tile, names)
    return _CACHE[key]


def _lower_gemm(cur_gpu_config, name, M, N, K, batch, a_dt, b_dt, c_dt, comp_dt, t: TileConfig, names=("act", "weight", "out")):
    an, bn_, cn = names
    if M <= 0 or N <= 0 or K <= 0 or batch <= 0:
        return []
    ab, bb, cb = DTYPE_BYTES[a_dt], DTYPE_BYTES[b_dt], DTYPE_BYTES[c_dt]
    mt, nt, kt = math.ceil(M / t.bm), math.ceil(N / t.bn), math.ceil(K / t.bk)
    sk = max(1, min(t.split_k, kt))
    iters = math.ceil(kt / sk)
    a_tile, b_tile = t.bm * t.bk * ab, t.bn * t.bk * bb
    cl = max(1, t.cluster_m)
    if cl > 1 and (mt < cl or not cur_gpu_config.get("compute.cluster_multicast", False)
                   or (t.cta_pair and (cl != 2 or not cur_gpu_config.get("compute.cta_pair", False)))):
        return None      # AMD parts have no thread-block clusters / TMA multicast
    stage = int(a_tile + (b_tile / cl if t.cta_pair else b_tile))
    epi_smem = int(t.bm * t.bn * cb)
    stages = t.stages or min(8, (cur_gpu_config.smem_per_sm - epi_smem) // stage)
    if stages < 2 or stages * stage + epi_smem > cur_gpu_config.smem_per_sm:
        return None
    acc = t.bm * t.bn * 4
    threads = 128 * (1 + (2 if t.bm >= 128 else 1))     # producer WG + consumer WG(s)
    paths = list(parse_path_split(t.load_path))
    if cl > 1 and not all(cur_gpu_config.path_attr(pp, "multicast", False) for pp in paths):
        return None            # cluster multicast needs a DMA engine that can multicast
    extra_regs = max(int(cur_gpu_config.path_attr(pp, "regs_per_thread", 0) or 0) for pp in paths)
    resident, occ_lim = occupancy(cur_gpu_config, stages * stage + epi_smem, acc, threads, extra_regs=extra_regs)
    if resident < 1:
        return None
    rpt, spill = reg_estimate(cur_gpu_config, acc, threads - 128)
    blocks = batch * mt * nt * sk

    stage_cfg = (cur_gpu_config.sram or {}).get("stage") or {}
    share_a = share_b = 1.0
    if stage_cfg:
        share = float(stage_cfg.get("share_blocks", 1))
        share_b = max(1.0, min(share, mt))
        share_a = max(1.0, min(share, nt))
    # ---- L2 model: deterministic tile-level simulation over several waves ------------
    # `waves_simulated` waves are replayed so a panel re-read by a later wave can hit (B1).
    from tilesight.model.engine.cache_sim import simulate as _sim
    from tilesight.generateResult.report.addressing import AddressMap
    conc = cur_gpu_config.sms * resident
    order_all = [(b_, s_, m, n) for b_ in range(batch) for s_ in range(sk)
                 for m, n in grouped_raster(mt, nt, t.swizzle)]
    n_waves = min(int(cur_gpu_config.get("memory.l2.waves_simulated") or 1),
                  max(1, -(-len(order_all) // conc)))
    order = order_all[:conc * n_waves]
    a_base, b_base = 0, int(M * K * ab * batch)
    keys, addrs, sizes, streams = [], [], [], []
    ksteps = list(range(min(iters, 4)))
    for w in range(n_waves):                      # wave by wave, K-step by K-step
        wave = order[w * conc:(w + 1) * conc]
        for kk in ksteps:
            for (b_, s_, m, n) in wave:
                k = s_ * iters + kk
                keys.append(((b_ * 4096 + m) * 65536 + k) * 2)          # A(b,m,k)
                addrs.append(a_base + int(((b_ * mt + m) * kt + k) * a_tile))
                sizes.append(a_tile)
                streams.append(0)
                keys.append(((b_ * 4096 + n) * 65536 + k) * 2 + 1)      # B(b,n,k)
                addrs.append(b_base + int(((b_ * kt + k) * nt + n) * b_tile))
                sizes.append(b_tile)
                streams.append(1)
    lmap = AddressMap.from_hw(cur_gpu_config, "l2")
    n_part = int(cur_gpu_config.get("memory.l2.partitions") or 1)
    policy = str(cur_gpu_config.get("memory.l2.policy"))
    # L1 only sees traffic that goes through it: a DMA straight into SMEM does not
    paths_now = list(parse_path_split(t.load_path))
    through_l1 = any(not cur_gpu_config.path_attr(pp, "smem_direct", True) for pp in paths_now)
    l1_miss = l1_miss_fractions(cur_gpu_config, keys, addrs, sizes, streams, 2, resident, len(order) // max(1, n_waves)) \
        if through_l1 else None
    sim = _sim(keys, addrs, sizes, streams, 2, cur_gpu_config.l2_capacity_bytes, n_part,
               lmap.port_of, policy)
    fa, fb = sim.miss_fraction
    l2_sim = sim
    sa = sb = None
    if cur_gpu_config.sram is not None:
        # the on-chip buffer is explicitly managed, not a cache: a class gets the capacity its
        # pin share gives it, deterministically (see resident_frac / dse.buffer)
        ra = resident_frac(cur_gpu_config, M * K * ab * batch, tensor_class(an))
        rb_ = resident_frac(cur_gpu_config, K * N * bb * batch, tensor_class(bn_))
        sa = fa * (1 - ra) / share_a
        sb = fb * (1 - rb_) / share_b

    # ---- actions --------------------------------------------------------------------
    bm_c = max(t.bm, cur_gpu_config.tc_min_m)                 # tensor-core M padding
    mma_lane, mma_t = cur_gpu_config.mma_cost(2 * bm_c * t.bn * t.bk, comp_dt)
    body = [
        gload(cur_gpu_config, f"load:{an}", a_tile, fa, t.load_path, sram_miss=sa,
              l1_miss=l1_miss[0] if l1_miss else None),
        # cluster_m CTAs along M share one B tile via TMA multicast (L2 traffic / cl);
        # with 2-CTA MMA (tcgen05 cta_group::2) each CTA also stores/reads only half of B.
        gload(cur_gpu_config, f"load:{bn_}", b_tile / cl, fb, t.load_path, sram_miss=sb,
              l1_miss=l1_miss[1] if l1_miss else None),
        Action("mma", {mma_lane: mma_t,
                       "smem": cur_gpu_config.smem_time_per_sm(a_tile + (b_tile / cl if t.cta_pair else b_tile))},
               [0, 1], latency_s=cur_gpu_config.unit_latency_s(mma_lane) + cur_gpu_config.unit_latency_s("smem")),
    ]
    if spill:   # register spill: accumulator fragments round-trip through local memory (L1/L2)
        body.append(Action("spill:regs", {"l2": 2.0 * spill / max(1, iters)}, [2]))
    out_bytes = t.bm * t.bn * (4 if sk > 1 else cb)
    epilogue = [
        Action("acc_read" if cur_gpu_config.has_tmem else "acc_regs",
               {"tmem": cur_gpu_config.tmem_time_per_sm(acc, 0), "cuda": cur_gpu_config.cuda_time_per_sm(2 * t.bm * t.bn)},
               latency_s=cur_gpu_config.unit_latency_s("tmem" if cur_gpu_config.has_tmem else "cuda")),
        gstore(f"store:{cn if sk == 1 else 'partials'}", out_bytes, [0]),
    ]
    useful = 2.0 * M * N * K * batch
    meta = dict(M=M, N=N, K=K, batch=batch, tile=t.with_(stages=stages).short(), stages=stages, resident=resident,
                flops=useful, pad_eff=useful / (2.0 * batch * mt * t.bm * nt * t.bn * kt * t.bk),
                weight_bytes=K * N * bb * batch, l2_miss_A=fa, l2_miss_B=fb,
                sram_miss_A=sa, sram_miss_B=sb, stage_share=(share_a, share_b),
                l1_miss=l1_miss, l1_capacity_KB=cur_gpu_config.l1_capacity_bytes / 1024,
                l2_hit_rate=l2_sim.hit_rate, l2_partitions=n_part, l2_policy=policy,
                l2_waves_simulated=n_waves, l2_sim=l2_sim,
                dtype=f"{a_dt}x{b_dt}->{comp_dt}", occ_limiter=occ_lim, regs_per_thread=rpt,
                reg_spill_bytes=spill, smem_per_block=int(stages * stage + epi_smem),
                stages_limiter="smem" if not t.stages else "config")
    ks = [Kernel(name, "gemm", blocks, iters, stages, resident, body, [], epilogue, meta)]
    if sk > 1:
        ks += lower_elementwise(cur_gpu_config, f"{name}.splitk_reduce", bytes_in=sk * M * N * 4 * batch,
                                bytes_out=M * N * cb * batch, flops=sk * M * N * batch,
                                names=("partials", cn))
    return ks


# ----------------------------------------------------------------------------- elementwise
def lower_elementwise(cur_gpu_config: HardwareSpec, name: str, *, bytes_in: float, bytes_out: float,
                      flops: float = 0.0, sfu_ops: float = 0.0, kind: str = "elementwise",
                      chunk: int = 32 * 1024, names: tuple[str, str] = ("act", "act"),
                      on_chip: bool | None = None) -> list[Kernel]:
    if bytes_in + bytes_out <= 0:
        return []
    blocks = max(1, math.ceil(max(bytes_in, bytes_out) / chunk))
    per = 1.0 / blocks
    # with a large on-chip buffer the intermediate never reaches HBM (policy keep_intermediates)
    oc = cur_gpu_config.sram_keeps_intermediates if on_chip is None else on_chip
    res = resident_frac(cur_gpu_config, max(bytes_in, bytes_out), "act") if oc else 0.0
    body = [
        gload(cur_gpu_config, f"load:{names[0]}", bytes_in * per, 1.0, "lsu", sram_miss=1.0 - res),
        Action("compute", {"cuda": cur_gpu_config.cuda_time_per_sm(flops * per), "sfu": cur_gpu_config.sfu_time_per_sm(sfu_ops * per)},
               [0], latency_s=cur_gpu_config.unit_latency_s("sfu" if sfu_ops else "cuda")),
        gstore(f"store:{names[1]}", bytes_out * per, [1], ddr_frac=1.0 - res,
               sram_frac=res if cur_gpu_config.sram else 0.0),
    ]
    meta = dict(bytes=bytes_in + bytes_out, flops=flops, occ_limiter="-", on_chip_frac=res)
    return [Kernel(name, kind, blocks, 1, 1, 8, body, [], [], meta)]
