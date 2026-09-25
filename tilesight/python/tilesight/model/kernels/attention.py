"""Fused attention -> Kernel lowering.

decode  : FlashMLA / FlashDecoding style. One block = (batch, kv_head, head-group, kv-split).
          Per KV tile: load K(/V) -> S = QK^T (TC) -> softmax (SFU+CUDA, S/P in TMEM)
          -> O += PV (TC).  gemm_qk/softmax/gemm_pv are `recurrent` (online-softmax
          state is loop-carried): their chain is only hidden by `consumers` ping-pong
          warpgroups, so a slow SFU shows up as a `latency` limiter — the effect
          Blackwell Ultra's 2x SFU targets.
prefill : FlashAttention-style causal, one block = (batch, head, q-tile).
MLA     : set v_in_k=True (V is the first d_v columns of the cached latent, one kv head).
"""
from __future__ import annotations

import math

from tilesight.interfaceAndModelRun.gpuTilingPerfHWModel.spec import DTYPE_BYTES, HardwareSpec
from tilesight.model.ir.kernel import Action, Kernel
from tilesight.model.engine import backend
from tilesight.model.kernels.gemm import gload, gstore, lower_elementwise, occupancy, reg_estimate, resident_frac, tensor_class
from tilesight.model.kernels.tiles import AttnTileConfig


def _softmax(cur_gpu_config, rows, cols, deps):
    s_bytes = rows * cols * 4
    return Action("softmax", {"sfu": cur_gpu_config.sfu_time_per_sm(rows * cols),
                              "cuda": cur_gpu_config.cuda_time_per_sm(6 * rows * cols),
                              "tmem": cur_gpu_config.tmem_time_per_sm(s_bytes, s_bytes * 0.5)}, deps, recurrent=True,
                  latency_s=cur_gpu_config.unit_latency_s("sfu") + cur_gpu_config.unit_latency_s("tmem"))


def lower_attention_decode(cur_gpu_config: HardwareSpec, name: str, *, B: int, H: int, kv_heads: int, S: int,
                           d_qk: int, d_v: int, kv_dtype="bf16", compute_dtype="bf16",
                           v_in_k: bool = False, tile: AttnTileConfig = AttnTileConfig()) -> list[Kernel] | None:
    kvb = DTYPE_BYTES[kv_dtype]
    qpk = H // kv_heads                                   # query heads per kv head
    hb = min(qpk, tile.block_m)
    groups = math.ceil(qpk / hb)
    units = B * kv_heads * groups
    ntiles = math.ceil(S / tile.block_n)
    k_tile = tile.block_n * d_qk * kvb
    v_tile = 0 if v_in_k else tile.block_n * d_v * kvb
    smem = tile.stages * (k_tile + v_tile) + hb * d_qk * 2
    if smem > cur_gpu_config.smem_per_sm:
        return None
    threads = 128 * (1 + tile.consumers)
    resident, occ_lim = occupancy(cur_gpu_config, int(smem), hb * (d_v + tile.block_n) * 4, threads)
    if resident < 1:
        return None
    rpt, spill = reg_estimate(cur_gpu_config, hb * (d_v + tile.block_n) * 4, threads - 128)
    splits = tile.num_splits or max(1, min(ntiles, math.ceil(cur_gpu_config.sms * resident / units)))
    iters = math.ceil(ntiles / splits)
    blocks = units * splits

    # L2: KV tiles are shared by the `groups` head-groups of the same kv head
    order = [(b, h, g, s) for b in range(B) for h in range(kv_heads) for s in range(splits) for g in range(groups)]
    order = order[: cur_gpu_config.sms * resident]
    keys, streams = [], []
    for it in range(min(iters, 4)):
        for b, h, g, s in order:
            keys.append(((b * 256 + h) * 4096 + s) * 4096 + it)
            streams.append(0)
    from tilesight.model.engine.cache_sim import simulate as _sim
    from tilesight.generateResult.report.addressing import AddressMap
    lmap = AddressMap.from_hw(cur_gpu_config, "l2")
    n_part = int(cur_gpu_config.get("memory.l2.partitions") or 1)
    tile_b = k_tile + v_tile
    addrs = [i * int(tile_b) for i in range(len(keys))]        # KV pages laid out contiguously
    sim = _sim(keys, addrs, [tile_b] * len(keys), streams, 1, cur_gpu_config.l2_capacity_bytes, n_part,
               lmap.port_of, str(cur_gpu_config.get("memory.l2.policy")))
    f = sim.miss_fraction[0]
    fs = None
    if cur_gpu_config.sram is not None:
        kv_total = B * kv_heads * S * (d_qk + (0 if v_in_k else d_v)) * kvb
        fs = f * (1 - resident_frac(cur_gpu_config, kv_total, "kv"))

    hb_c = max(hb, cur_gpu_config.tc_min_m)
    qk_lane, qk_t = cur_gpu_config.mma_cost(2 * hb_c * d_qk * tile.block_n, compute_dtype)
    pv_lane, pv_t = cur_gpu_config.mma_cost(2 * hb_c * tile.block_n * d_v, compute_dtype)
    body = [gload(cur_gpu_config, "load:kv_cache(latent)" if v_in_k else "load:kv_cache(K)", k_tile, f, "tma",
                  sram_miss=fs)]
    iv = 0
    if not v_in_k:
        body.append(gload(cur_gpu_config, "load:kv_cache(V)", v_tile, f, "tma", sram_miss=fs))
        iv = 1
    body += [
        Action("gemm_qk", {qk_lane: qk_t,
                           "smem": cur_gpu_config.smem_time_per_sm(k_tile + hb * d_qk * 2)}, [0], recurrent=True,
               latency_s=cur_gpu_config.unit_latency_s(qk_lane) + cur_gpu_config.unit_latency_s("smem")),
    ]
    body.append(_softmax(cur_gpu_config, hb, tile.block_n, [len(body) - 1]))
    body.append(Action("gemm_pv", {pv_lane: pv_t,
                                   "smem": cur_gpu_config.smem_time_per_sm(tile.block_n * d_v * kvb)},
                       [len(body) - 1, iv], recurrent=True,
                       latency_s=cur_gpu_config.unit_latency_s(pv_lane) + cur_gpu_config.unit_latency_s("smem")))
    if spill:
        body.append(Action("spill:regs", {"l2": 2.0 * spill / max(1, iters)}, [len(body) - 1]))
    prologue = [gload(cur_gpu_config, "load:q", hb * d_qk * 2, 1.0, "tma")]
    out_b = 4 if splits > 1 else 2
    epilogue = [Action("o_read", {"tmem": cur_gpu_config.tmem_time_per_sm(hb * d_v * 4, 0)}),
                gstore("store:out" if splits == 1 else "store:partials", hb * d_v * out_b, [0])]
    kv_bytes = B * kv_heads * S * (d_qk + (0 if v_in_k else d_v)) * kvb
    meta = dict(B=B, H=H, S=S, tile=f"hb{hb}x{tile.block_n}/s{tile.stages}/sp{splits}", resident=resident,
                occ_limiter=occ_lim, regs_per_thread=rpt, reg_spill_bytes=spill, smem_per_block=int(smem),
                l2_hit_rate=sim.hit_rate, l2_partitions=n_part, l2_sim=sim,
                flops=2.0 * B * H * S * (d_qk + d_v), kv_bytes=kv_bytes, l2_miss_KV=f, splits=splits)
    ks = [Kernel(name, "attention", blocks, iters, tile.stages, resident, body, prologue, epilogue, meta,
                consumers=tile.consumers)]
    if splits > 1:
        ks += lower_elementwise(cur_gpu_config, f"{name}.combine", bytes_in=B * H * splits * (d_v + 1) * 4,
                                bytes_out=B * H * d_v * 2, flops=3 * B * H * splits * d_v,
                                sfu_ops=B * H * splits, names=("partials", "out"))
    return ks


def lower_attention_prefill(cur_gpu_config: HardwareSpec, name: str, *, B: int, H: int, kv_heads: int, S: int,
                            d_qk: int, d_v: int, kv_dtype="bf16", compute_dtype="bf16",
                            causal: bool = True, window: int = 0,
                            tile: AttnTileConfig = AttnTileConfig(block_m=128, block_n=128)
                            ) -> list[Kernel] | None:
    kvb = DTYPE_BYTES[kv_dtype]
    bm, bn = tile.block_m, tile.block_n
    qt = math.ceil(S / bm)
    iters_list = []
    for i in range(qt):
        hi = min(S, (i + 1) * bm) if causal else S           # last key visible to this q-tile
        lo = max(0, i * bm - window + 1) if window > 0 else 0  # first key inside the window
        iters_list.append(max(1, math.ceil(hi / bn) - lo // bn))
    iters = max(1, round(sum(iters_list) / qt))
    k_tile, v_tile = bn * d_qk * kvb, bn * d_v * kvb
    q_tile = bm * d_qk * 2
    smem = tile.stages * (k_tile + v_tile) + q_tile
    if smem > cur_gpu_config.smem_per_sm:
        return None
    threads = 128 * (1 + tile.consumers)
    resident, occ_lim = occupancy(cur_gpu_config, int(smem), bm * (d_v + bn) * 4, threads)
    if resident < 1:
        return None
    rpt, spill = reg_estimate(cur_gpu_config, bm * (d_v + bn) * 4, threads - 128)
    blocks = B * H * qt

    order = [(b, h, q) for b in range(B) for h in range(H) for q in range(qt)][: cur_gpu_config.sms * resident]
    qpk = H // kv_heads
    keys, streams = [], []
    for it in range(min(iters, 4)):
        for b, h, q in order:
            if it < iters_list[q]:
                keys.append(((b * 256 + h // qpk) * 65536 + it) * 2)
                streams.append(0)
                keys.append(((b * 256 + h // qpk) * 65536 + it) * 2 + 1)
                streams.append(0)
    from tilesight.model.engine.cache_sim import simulate as _sim
    from tilesight.generateResult.report.addressing import AddressMap
    lmap = AddressMap.from_hw(cur_gpu_config, "l2")
    n_part = int(cur_gpu_config.get("memory.l2.partitions") or 1)
    tb = (k_tile + v_tile) / 2
    addrs = [i * int(tb) for i in range(len(keys))]
    sim = _sim(keys, addrs, [tb] * len(keys), streams, 1, cur_gpu_config.l2_capacity_bytes, n_part,
               lmap.port_of, str(cur_gpu_config.get("memory.l2.policy")))
    f = sim.miss_fraction[0]
    fs = None
    if cur_gpu_config.sram is not None:
        kv_total = B * kv_heads * S * (d_qk + d_v) * kvb
        fs = f * (1 - resident_frac(cur_gpu_config, kv_total, "kv"))

    bm_c = max(bm, cur_gpu_config.tc_min_m)
    qk_lane, qk_t = cur_gpu_config.mma_cost(2 * bm_c * bn * d_qk, compute_dtype)
    pv_lane, pv_t = cur_gpu_config.mma_cost(2 * bm_c * bn * d_v, compute_dtype)
    body = [gload(cur_gpu_config, "load:kv_cache(K)", k_tile, f, "tma", sram_miss=fs),
            gload(cur_gpu_config, "load:kv_cache(V)", v_tile, f, "tma", sram_miss=fs),
            Action("gemm_qk", {qk_lane: qk_t,
                               "smem": cur_gpu_config.smem_time_per_sm(k_tile + q_tile)}, [0], recurrent=True,
                   latency_s=cur_gpu_config.unit_latency_s(qk_lane) + cur_gpu_config.unit_latency_s("smem"))]
    body.append(_softmax(cur_gpu_config, bm, bn, [2]))
    body.append(Action("gemm_pv", {pv_lane: pv_t,
                                   "smem": cur_gpu_config.smem_time_per_sm(v_tile)}, [3, 1], recurrent=True,
                       latency_s=cur_gpu_config.unit_latency_s(pv_lane) + cur_gpu_config.unit_latency_s("smem")))
    if spill:
        body.append(Action("spill:regs", {"l2": 2.0 * spill / max(1, iters)}, [4]))
    prologue = [gload(cur_gpu_config, "load:q", q_tile, 1.0, "tma")]
    epilogue = [Action("o_read", {"tmem": cur_gpu_config.tmem_time_per_sm(bm * d_v * 4, 0)}),
                gstore("store:out", bm * d_v * 2, [0])]
    pairs = 0                                                # (query, key) pairs actually scored
    for i in range(qt):
        rows = min(bm, S - i * bm)
        hi = min(S, (i + 1) * bm) if causal else S
        lo = max(0, i * bm - window + 1) if window > 0 else 0
        pairs += rows * (hi - lo) - (rows * (rows - 1) // 2 if causal else 0)
    meta = dict(B=B, H=H, S=S, tile=tile.short(), resident=resident, occ_limiter=occ_lim,
                l2_hit_rate=sim.hit_rate, l2_partitions=n_part, l2_sim=sim,
                regs_per_thread=rpt, reg_spill_bytes=spill, smem_per_block=int(smem),
                flops=2.0 * B * H * pairs * (d_qk + d_v), l2_miss_KV=f)
    return [Kernel(name, "attention", blocks, iters, tile.stages, resident, body, prologue, epilogue, meta,
                   consumers=tile.consumers)]
