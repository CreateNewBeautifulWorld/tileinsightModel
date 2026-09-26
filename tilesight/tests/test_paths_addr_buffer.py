"""DMA vs LSU engines, mega-tile staging, tile-granularity addressing, L2/buffer trade."""
import pytest

from tilesight import HardwareSpec, ModelSpec, RunConfig, _core, run_model
from tilesight.gpuTilingPerfHWModel.genResult.addressing import analyze_gemm, spread, tile_index
from tilesight.gpuTilingPerfHWModel.model.dse.buffer import l2_tradeoff, with_buffer

HW = HardwareSpec.load("b300")

TileConfig = _core.GemmTile


def evaluate(k, hw):
    return _core.evaluate(hw, k)


def _lower_gemm(cur_gpu_config, M=4096, N=4096, K=7168, **kw):
    kw.setdefault("tile", TileConfig())
    return _core.lower_gemm(cur_gpu_config, "g", M, N, K, **kw)


def _k(cur_gpu_config, path="tma", **kw):
    t = TileConfig(bm=128, bn=256, bk=64, load_path=path, **kw)
    ks = _lower_gemm(cur_gpu_config, 4096, 4096, 7168, a_dtype="fp8", b_dtype="fp8",
                     compute_dtype="fp8", tile=t)
    return ks[0] if ks else None


def test_dma_and_lsu_are_different_engines():
    dma, lsu = _k(HW, "tma"), _k(HW, "lsu")
    # the vector path burns SM issue slots and stages through SMEM; the DMA engine does neither
    assert sum(a.work.get("cuda", 0) for a in lsu.trace.body) > 0
    assert sum(a.work.get("cuda", 0) for a in dma.trace.body) == 0
    assert sum(a.work.get("smem", 0) for a in lsu.trace.body) > sum(a.work.get("smem", 0) for a in dma.trace.body)
    assert evaluate(lsu, HW).time_s > evaluate(dma, HW).time_s
    assert HW.path_is_dma("tma") and not HW.path_is_dma("lsu")


def test_lsu_costs_registers_and_blocks_multicast():
    # staging registers reduce occupancy
    r_dma, _ = _core.occupancy(HW, 64 * 1024, 128 * 256 * 4, 384, extra_regs=0)
    r_lsu, _ = _core.occupancy(HW, 64 * 1024, 128 * 256 * 4, 384, extra_regs=16)
    assert r_lsu <= r_dma
    # a cluster-multicast tile is illegal on a path that cannot multicast
    assert _k(HW, "lsu", cluster_m=2) is None
    assert _k(HW, "tma", cluster_m=2) is not None


def test_mega_tile_staging_cuts_hbm_traffic():
    buf = with_buffer(HW, 4096, policy="pin", pin={"weight": 1.0})
    staged = buf.override({"memory.sram.stage": {"share_blocks": 8}})
    a, b = _k(buf), _k(staged)
    assert sum(x.work.get("ddr", 0) for x in b.trace.body) < sum(x.work.get("ddr", 0) for x in a.trace.body)
    assert b.meta["stage_share"][1] > 1        # B panels shared along M


@pytest.mark.parametrize("layout", ["row", "col", "swizzle:8", "xor:3", "zorder"])
def test_tile_index_is_a_bijection(layout):
    ni, nj = 8, 12
    idx = sorted(tile_index(layout, i, j, ni, nj) for i in range(ni) for j in range(nj))
    assert len(set(idx)) == ni * nj          # every tile gets a distinct slot


def test_address_spread_is_tile_granular_and_layout_sensitive():
    # a tile covers several interleave chunks (not one cache line)
    rep = spread(HW, [(0, 8192), (8192, 8192)])
    assert rep.tile_bytes == 8192 and rep.interleave_bytes == 1024   # from memory.addressing.l2
    assert sum(rep.slices) == 16384
    row = analyze_gemm(HW, 4096, 4096, 7168, TileConfig(128, 256, 64), 1.0, 1.0)
    z = analyze_gemm(HW, 4096, 4096, 7168, TileConfig(128, 256, 64), 1.0, 1.0, layout_a="zorder")
    small = analyze_gemm(HW, 4096, 4096, 7168, TileConfig(64, 64, 64), 1.0, 1.0)
    assert z["slice_imbalance"] < row["slice_imbalance"]      # z-order spreads better
    assert small["slice_imbalance"] > row["slice_imbalance"]  # small tiles spread worse
    assert 0 < z["effective_l2_fraction"] <= 1.0
    assert "slice" in row["both"].text()


def test_l2_vs_buffer_trade_needs_a_big_buffer():
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(phase="decode", batch=64, seq_len=4096, dp=8)
    rows = l2_tradeoff(m, HW, rc, 126, buffer_shares=(0.0, 0.9), policy="pin",
                       pin={"weight": 1.0}, bypass_l2=True, prefetch=True)
    # at an L2-sized budget, trading L2 away for a buffer loses: a pinned buffer is not the DMA
    # destination, so the matrix DMAs still go through the (now small) L2 and bind there
    assert rows[-1]["step_ms"] > rows[0]["step_ms"]
    assert rows[-1]["top_bound"].startswith(("l2", "dma"))


def test_shared_buffer_uses_a_real_lru_trace_pinned_buffer_stays_closed_form():
    """A *shared* buffer (policy=cache, no alloc/pin split) is a real cache tiles compete for via
    the actual access order — so its residency comes from the same tile-keyed LRU trace as L1/L2
    (`cache::simulate` over the exact A/B access stream already built for the L2 simulation),
    not a closed-form capacity/footprint ratio: a tile's presence is decided by whether it was
    really touched recently enough to still be resident, exactly like the paper's reuse-distance
    D_T but computed exactly (real addresses, real order) instead of approximated by a binomial.

    A *pinned* buffer (an explicit memory.sram.alloc/pin split) models a dedicated, pre-staged
    carve-out instead — weights assumed already resident before this kernel's own accesses, not
    something filled from cold by them — so it deliberately keeps the closed-form resident_frac:
    charging this simulation's compulsory first-touch misses against a pinned share would be
    wrong. This is what test_l2_vs_buffer_trade_needs_a_big_buffer and
    test_buffer_sits_between_the_slices_and_l2_behind_a_switch already rely on."""
    tile = TileConfig(bm=64, bn=256, bk=64)

    def ddr_of(cur_gpu_config):
        k = _lower_gemm(cur_gpu_config, 4096, 4096, 7168, a_dtype="fp8", b_dtype="fp8",
                        compute_dtype="fp8", tile=tile)[0]
        return sum(a.work.get("ddr", 0) for a in k.trace.body)

    # Shared cache: real trace, so more capacity can only ever help (never worse), same
    # monotonicity CLAUDE.md already requires of every other bandwidth/capacity knob.
    small = ddr_of(with_buffer(HW, 1, policy="cache"))
    big = ddr_of(with_buffer(HW, 256, policy="cache"))
    plain = ddr_of(HW.override({"memory.sram": None}) if False else HW)
    # (a 1 MB buffer as the DMA destination is worse than no buffer at all -- it holds far less
    # than the 132 MB L2 the DMA would otherwise land in -- so only capacity monotonicity holds)
    assert big <= small and big <= plain

    # Pinned 100% to weight with ample capacity: still the old closed-form full-residency case
    # (zero leftover DDR for that operand), unaffected by the shared-buffer trace above.
    pinned = with_buffer(HW, 64 * 1024, policy="pin", pin={"weight": 1.0})
    k = _lower_gemm(pinned, 4096, 4096, 7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=tile)[0]
    assert k.trace.body[1].work["ddr"] == 0  # body[1] = load:weight (b_name defaults to "weight")


def test_every_unit_has_a_configurable_latency():
    # defaults are in cycles and convert with the clock
    assert abs(HW.unit_latency_s("tc") * HW.clock_hz - 64) < 1e-6      # one tile MMA
    assert abs(HW.unit_latency_s("sfu") * HW.clock_hz - 16) < 1e-6
    h = HW.override({"compute.mma_latency_cycles": 200})
    assert abs(h.unit_latency_s("tc") * h.clock_hz - 200) < 1e-6
    # memory levels may be given in ns instead
    assert abs(HW.unit_latency_s("ddr") - HW.get("memory.ddr.latency_ns") * 1e-9) < 1e-15


def test_independent_latency_only_costs_fill_but_loop_carried_costs_throughput():
    slow_mma = HW.override({"compute.mma_latency_cycles": 512})
    a, b = _k(HW), _k(slow_mma)
    ra, rb = evaluate(a, HW), evaluate(b, slow_mma)
    # a tile MMA is independent per iteration: more latency only lengthens the fill
    assert rb.breakdown["fill"] > ra.breakdown["fill"]
    assert abs(rb.breakdown["steady"] - ra.breakdown["steady"]) < 1e-9

    # the online-softmax chain in attention IS loop-carried: its latency hits the steady state
    def attn(cur_gpu_config):
        k = _core.lower_attention_decode(cur_gpu_config, "a", B=32, H=64, kv_heads=1, S=8192, d_qk=576,
                                         d_v=512, v_in_k=True,
                                         tile=_core.AttnTile(block_m=64, block_n=64, stages=2, consumers=1))[0]
        return evaluate(k, cur_gpu_config)
    slow_sfu = HW.override({"compute.sfu_latency_cycles": 4096})
    assert attn(slow_sfu).time_s > attn(HW).time_s


def test_memory_map_allocates_every_layer_instance():
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.memmap import build_memory_map
    m = ModelSpec.load("kimi_k2.hf")
    rc = RunConfig(phase="decode", batch=64, seq_len=4096, dp=8)
    mm = build_memory_map(m, rc, base=0x1000000)
    w = mm.by_kind("weight")
    assert len(w) > 500                                  # 61 layers x ~9 weight tensors
    assert all(r.base >= 0x1000000 for r in mm.regions)
    # regions never overlap and are 2 MB aligned
    ordered = sorted(mm.regions, key=lambda r: r.base)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.base and b.base % (2 * 1024 * 1024) == 0
    # totals match the memory report
    rep = run_model(m, HW, rc)
    assert abs(sum(r.size for r in w) / 1e9 - rep.memory.weights_GB) / rep.memory.weights_GB < 0.05
    assert mm.by_kind("kv") and mm.find("experts_gate_up") is not None


def test_tile_addresses_follow_the_layout():
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.memmap import build_memory_map
    mm = build_memory_map(ModelSpec.load("kimi_k2.hf"),
                          RunConfig(phase="decode", batch=64, seq_len=4096, dp=8), base=0)
    r = mm.find("experts_gate_up")
    a00 = r.tile_addr(0, 0, 64, 256)
    assert a00 == r.base
    assert r.tile_addr(0, 1, 64, 256) - a00 == 64 * 256 * r.elem_bytes        # row-major: next column
    assert r.tile_addr(0, 1, 64, 256, "zorder") != r.tile_addr(0, 1, 64, 256)


def test_trace_and_csv_carry_addresses():
    from tilesight.gpuTilingPerfHWModel.genResult.addressing import kernel_addr_fn
    from tilesight.gpuTilingPerfHWModel.genResult.timeline import cycle_csv, steady_timeline, trace_text
    t = TileConfig(bm=128, bn=256, bk=64, stages=4, cluster_m=2)
    k = _lower_gemm(HW, 4096, 4096, 7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=t)[0]
    fn = kernel_addr_fn(HW, 4096, 4096, 7168, t, 1.0, 1.0, base=0x10000000)
    tl = steady_timeline(k, HW, addr_fn=fn)
    loads = [it for it in tl["steady"] if it["action"].startswith("load")]
    assert loads and all(it["addr"] >= 0x10000000 for it in loads)
    assert all(0 <= it["l2_slice"] < 16 and 0 <= it["hbm_port"] < 8 for it in loads)
    # A tiles of consecutive iterations are one tile apart
    a = sorted({it["addr"] for it in loads if "act" in it["action"]})
    assert a[1] - a[0] == 128 * 64                       # bm*bk*1 byte (fp8)
    assert "addr" in trace_text(tl, "g", "B300")
    head = cycle_csv(tl).splitlines()[0]
    assert "ddr_addr" in head and "l2_addr" in head


def test_address_map_modes_and_dump():
    from tilesight.gpuTilingPerfHWModel.genResult.addressing import AddressMap, analyze_gemm
    m = AddressMap.from_hw(HW, "l2")
    assert m.ports == 16 and m.mode == "interleave" and m.granularity == 1024 and m.addr_bits == 48
    # interleave: consecutive stripes walk the ports
    assert [m.port_of(i * 1024) for i in range(4)] == [0, 1, 2, 3]
    # a tile spreads over the stripes it covers
    assert sum(b for _, b in m.spans(0, 8192)) == 8192
    assert len({p for p, _ in m.spans(0, 8192)}) == 8
    # range mode: equal contiguous slices of the 48-bit space
    rng = AddressMap.from_hw(HW.override({"memory.addressing.l2.mode": "range"}), "l2")
    span = (1 << 48) // 16
    assert rng.port_of(0) == 0 and rng.port_of(span) == 1 and rng.port_of((1 << 48) - 1) == 15
    # hash mode breaks power-of-two stride aliasing
    t = TileConfig(bm=128, bn=256, bk=64)
    plain = analyze_gemm(HW, 4096, 4096, 7168, t, 1.0, 1.0)
    hashed = analyze_gemm(HW.override({"memory.addressing.l2.mode": "hash"}), 4096, 4096, 7168,
                          t, 1.0, 1.0)
    assert hashed["slice_imbalance"] < plain["slice_imbalance"]
    assert "ports" in m.describe() and "stripes" in m.dump(rows=2)
    assert len(m.dump_csv(limit=32).splitlines()) == 33


def test_granularity_changes_the_spread():
    from tilesight.gpuTilingPerfHWModel.genResult.addressing import analyze_gemm
    t = TileConfig(bm=128, bn=256, bk=64)
    fine = analyze_gemm(HW.override({"memory.addressing.l2.granularity_KB": 1}),
                        4096, 4096, 7168, t, 1.0, 1.0)
    coarse = analyze_gemm(HW.override({"memory.addressing.l2.granularity_KB": 8}),
                          4096, 4096, 7168, t, 1.0, 1.0)
    assert fine["slice_imbalance"] <= coarse["slice_imbalance"]


def test_l2_blocks_and_outstanding_limits_are_enforced():
    lanes = {l.name: l for l in HW.lanes()}
    cfg = HW.get("memory.l2")
    port_cap = cfg["blocks"] * cfg["ports_per_block"] * cfg["bytes_per_clk_per_port"] * HW.clock_hz
    assert lanes["l2"].total_rate <= port_cap + 1        # blocks x ports cap the level
    # Little's law: fewer lines in flight -> lower per-SM ceiling -> slower memory-bound kernel
    slow = HW.override({"memory.outstanding.per_sm_lines": 64})
    fast = HW.override({"memory.outstanding.per_sm_lines": 2048})
    assert slow.outstanding_cap("ddr") < fast.outstanding_cap("ddr")
    assert abs(slow.outstanding_cap("ddr") - 64 * 128 / slow.unit_latency_s("ddr")) < 1e3
    a = evaluate(_k(slow), slow).time_s
    b = evaluate(_k(fast), fast).time_s
    assert a > b


def test_dma_and_l2_port_lanes_share_hbm():
    """HBM (`ddr`) is one pool that two engines draw from, each with its own ceiling lane:
    the DMA engines (`dma` = engines x port_bytes x clock, no outstanding entries: a DMA moves a
    contiguous range into space reserved up front) and the L2 ports (`l2port`, Little's law over
    the outstanding entries shared by reads and writes: an op_bytes request takes
    ceil(op_bytes / width) entries -- rounded up -- for ddr_entry_cycles)."""
    lanes = {l.name: l for l in HW.lanes()}
    clk = HW.clock_hz
    assert lanes["dma"].total_rate == pytest.approx(16 * 128 * clk)                  # 16 DMA ports x 128 B
    # 16 L2 ports, 128 entries of 64 B, 256 B op -> 4 entries -> 32 requests in flight each
    assert lanes["l2port"].total_rate == pytest.approx(min(16 * 64 * clk, 16 * 32 * 256 * clk / 300))
    # a request that isn't a whole number of port beats rounds its entries up: 256 B over a
    # 96 B port = 3 entries -> floor(128 / 3) = 42 requests
    odd = HW.override({"memory.l2.ddr_port_bytes_per_clk": 96})
    assert odd.l2port_rate() == pytest.approx(min(16 * 96 * clk, 16 * 42 * 256 * clk / 300))
    # more DMA engines -> a faster DMA-bound kernel, up to the shared HBM ceiling
    t = TileConfig(bm=64, bn=64, bk=128)
    few, many = HW.override({"memory.dma.engines": 4}), HW.override({"memory.dma.engines": 64})
    kf = _lower_gemm(few, 64, 16384, 7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=t)[0]
    km = _lower_gemm(many, 64, 16384, 7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=t)[0]
    rf, rm = evaluate(kf, few), evaluate(km, many)
    assert rf.bottleneck == "dma" and rm.time_s < rf.time_s
    # both engines also charge the shared HBM lane: ddr = dma + l2port on every load
    for a in kf.trace.body:
        if a.name.startswith("load"):
            assert a.work["ddr"] == pytest.approx(a.work["dma"] + a.work["l2port"])


def test_l2_op_bytes_must_divide_the_interleave_granularity():
    bad = HW.override({"memory.l2.op_bytes": 384})          # 1 KB stripes / 384 B does not divide
    with pytest.raises(ValueError):
        bad.lanes()
    with pytest.raises(ValueError):
        _lower_gemm(bad, 1024, 1024, 1024)
    HW.override({"memory.l2.op_bytes": 64}).lanes()       # smaller is fine as long as it divides


def _decode(hw, mode=None):
    if mode:
        hw = hw.override({"memory.dma.fetch_mode": mode})
    t = _core.AttnTile(block_m=64, block_n=64, stages=2)
    k = _core.lower_attention_decode(hw, "a", 32, 16, 1, 8192, 576, 512, "bf16", "bf16", True, t)[0]
    tot = {lane: sum(a.work.get(lane, 0) for a in k.trace.body) * k.trace.iters * k.trace.num_blocks
           for lane in ("ddr", "dma", "l2port", "sram")}
    return k, evaluate(k, hw), tot


def test_fetch_modes_route_matrix_operands_to_the_right_engine():
    """memory.dma.fetch_mode: how matrix operands (K/V, GEMM A/B) come in from HBM.
    Kimi-K2-like decode MLA, 32 sequences x 8192 tokens of 576 bf16 latent = 302 MB of KV."""
    k0, _, _ = _decode(HW)
    kv = k0.meta["kv_bytes"]
    assert k0.meta["fetch_mode"] == "dma_smart"            # auto without a buffer
    # dma_smart: DMA only, chunks sized so two per concurrent stream fit L2, multiples of 128 B,
    # aligned to the stream's own start -> HBM bytes ~= the KV actually needed
    _, r_s, smart = _decode(HW, "dma_smart")
    chunk = k0.meta["dma_chunk_bytes"][0]
    assert chunk % 128 == 0 and chunk <= HW.l2_capacity_bytes / (2 * k0.meta["fetch_streams"])
    assert smart["l2port"] == 0 and smart["dma"] == pytest.approx(kv, rel=0.03)
    # l2_port: the same bytes (tile rounded up to op_bytes), all through the L2 ports -> slower
    _, r_p, port = _decode(HW, "l2_port")
    assert port["dma"] == 0 and port["l2port"] == pytest.approx(kv, rel=0.03)
    assert r_p.bottleneck == "l2port" and r_p.time_s > r_s.time_s
    # dma_page: whole 2 MB pages into a 132 MB L2 for ~150 concurrent streams -> thrash
    _, r_g, page = _decode(HW, "dma_page")
    assert page["l2port"] == 0 and page["dma"] > 5 * kv and r_g.time_s > 5 * r_s.time_s
    # dma_page_tail_l2: a split's KV range (~1.9 MB) never covers a whole page -> all L2 ports
    _, _, tail = _decode(HW, "dma_page_tail_l2")
    assert tail["dma"] == 0 and tail["l2port"] == pytest.approx(port["l2port"])
    # ... while a long range is mostly pages with only the ends through the L2 ports
    t = TileConfig(bm=128, bn=128, bk=128)
    g = _lower_gemm(HW.override({"memory.dma.fetch_mode": "dma_page_tail_l2"}), 128, 128, 66560,
                    a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=t)[0]
    # B strip: 128 x 66560 fp8 = 8.1 MB, not page-aligned -> 3 whole pages + two partial ends
    dma, port = g.meta["hbm_dma_bytes_per_load"][1], g.meta["hbm_port_bytes_per_load"][1]
    assert dma > 0 and port > 0 and dma > 2 * port
    w = g.trace.body[1].work
    assert w["dma"] > 0 and w["l2port"] > 0


def test_b200_with_buffer_holds_whole_pages_that_l2_thrashes_on():
    """B200 has no buffer. A B200 with one (buffer = 4x the L2, L2 cut to 1/4) is the case the
    buffer exists for: DMA pages land in the buffer, which can hold one per concurrent stream,
    instead of in a hardware-managed L2 that evicts them before their stream is done."""
    b200 = HardwareSpec.load("b200")
    assert b200.sram is None
    l2 = b200.get("memory.l2.capacity_MB")
    buf = b200.override({"memory.l2.capacity_MB": l2 / 4, "memory.sram.capacity_MB": l2 * 4,
                         "memory.sram.bandwidth_TBps": 20.5, "memory.sram.latency_ns": 400})
    k_b, r_b, t_b = _decode(buf)
    kv = k_b.meta["kv_bytes"]
    assert k_b.meta["fetch_mode"] == "dma_page" and k_b.meta["dma_to_buffer"]   # auto with a buffer
    assert t_b["dma"] == pytest.approx(kv, rel=0.03)       # every page fetched once
    assert t_b["sram"] == pytest.approx(kv, rel=0.03)      # and read out of the buffer
    _, r_n, t_n = _decode(b200, "dma_page")                # same pages, no buffer: L2 thrashes
    assert t_n["dma"] > 5 * kv and r_n.time_s > 5 * r_b.time_s


def test_l2_ddr_bandwidth_is_capped_by_memory_slices():
    """L2 and DDR are not one GPU-wide pool: each memory slice owns its own L2 port and HBM
    channel (memory.addressing.l2.ports), and every tile's address routes to exactly one slice.
    A kernel only reaches the full configured aggregate bandwidth if it keeps at least as many
    blocks concurrently in flight as there are slices; fewer concurrent blocks touch fewer
    slices and reach proportionally less bandwidth, with the per-byte address routing unchanged."""
    t = TileConfig(bm=128, bn=128, bk=64)

    def kernel(n_ports):
        # the occupancy proxy applies to L2-port traffic (a DMA page/chunk spans every slice)
        hw = HW.override({"memory.addressing.l2": {"ports": n_ports, "mode": "interleave",
                                                    "granularity_KB": 1},
                          "memory.dma.fetch_mode": "l2_port"})
        k = _lower_gemm(hw, 512, 512, 8192, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=t)[0]
        return hw, k

    hw16, k16 = kernel(16)
    hw64, k64 = kernel(64)
    assert k16.meta["resident"] == 1
    concurrent_blocks = 16  # mt=nt=4, batch=1, resident=1 -> min(16, sms*1) == 16
    ddr16 = k16.trace.body[0].work["ddr"]
    ddr64 = k64.trace.body[0].work["ddr"]
    # 16 slices, 16 concurrent blocks -> every slice reached, no derating.
    # 64 slices, still only 16 concurrent blocks -> only 1/4 of the slices reached.
    assert ddr16 == pytest.approx(ddr64 * concurrent_blocks / 64)
    assert evaluate(k64, hw64).time_s > evaluate(k16, hw16).time_s
    # Enough concurrent blocks to cover the slices restores full bandwidth (no further penalty).
    hw_wide, k_wide = kernel(concurrent_blocks)  # ports == concurrent_blocks: fully saturated
    assert k_wide.trace.body[0].work["ddr"] == pytest.approx(ddr16)


def test_dma_hugepage_keys_the_l2_simulation_instead_of_rounding_after_the_fact():
    """memory.dma.hugepage_KB models a DMA that moves exactly one hugepage per operation, never
    less. Real interleaving distributes it across slices one granule at a time, round-robin; when
    the granule count doesn't divide evenly by the slice count (e.g. a 2048 KB hugepage over 12
    slices at 1 KB granules: 2048/12 = 170.67), some slices simply get one fewer granule than
    others in reality. Rather than track which slice is short, hugepage_bytes() pads up to what the
    fullest slice gets (171 granules here, i.e. an effective 2052 KB) -- never an underestimate,
    still every slice reached, still no address arithmetic to alias, unlike two earlier attempts at
    this same problem that derived slice spread from the L2 hit/miss simulation's synthetic
    per-tile address stream (which aliases on power-of-two tile strides).

    Configuring it also changes the L2 simulation's cache atom from "one raw tile" to "one
    hugepage": lower_gemm keys the A/B access stream by hugepage index instead of loop index, so a
    second access landing in an already-resident hugepage is a real hit in the simulation, and
    miss_fraction() means "how often a genuinely new hugepage is needed" -- which gload then bills
    directly as miss * hugepage_bytes. An earlier approach instead rounded each call's already-
    amortized raw-tile-miss bytes up to a whole hugepage independently every call, which the
    round-time formula's steady-state x iters multiplication then blew up ~200x on a genuinely
    compute-bound GEMM (0.315ms tc-bound -> 63.7ms, falsely ddr-bound) -- exactly what this test
    guards against staying broken."""
    valid_addr = {"ports": 16, "mode": "interleave", "granularity_KB": 1}  # unit = 16KB

    # A hugepage that doesn't divide evenly by ports x granularity is padded up, not rejected:
    # 24KB / 16KB unit = 1.5 -> padded to 2 units = 32KB.
    odd_addr = {"ports": 12, "mode": "interleave", "granularity_KB": 1}   # 2048/12 doesn't divide evenly
    k_odd = _lower_gemm(HW.override({"memory.addressing.l2": odd_addr, "memory.dma.hugepage_KB": 2048,
                                     "memory.dma.fetch_mode": "dma_page"}),
                        512, 512, 8192, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                        tile=TileConfig(bm=256, bn=128, bk=64))[0]
    assert k_odd.trace.body[0].work["ddr"] == pytest.approx(k_odd.meta["l2_miss_A"] * 171 * 12 * 1024)

    big_tile = TileConfig(bm=128, bn=256, bk=64, cluster_m=2, cta_pair=True)
    hw = HW.override({"memory.addressing.l2": valid_addr, "memory.dma.hugepage_KB": 2048,
                      "memory.dma.fetch_mode": "dma_page"})
    k = _lower_gemm(hw, 8192, 8192, 8192, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8", tile=big_tile)[0]

    # DDR work is exactly miss * hugepage_bytes now, not a per-call rounding of raw tile bytes.
    hugepage_bytes = 2048 * 1024
    assert k.trace.body[0].work["ddr"] == pytest.approx(k.meta["l2_miss_A"] * hugepage_bytes)
    assert k.trace.body[1].work["ddr"] == pytest.approx(k.meta["l2_miss_B"] * hugepage_bytes)

    # A genuinely compute-bound GEMM stays close to tc-bound: hugepage-aware reuse keeps the
    # amortized DDR cost small, unlike the earlier per-call rounding that made every kernel falsely
    # DDR-bound regardless of how reusable its access pattern actually was.
    ideal = evaluate(_lower_gemm(HW, 8192, 8192, 8192, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8",
                                 tile=big_tile)[0], HW)
    assert ideal.bottleneck == "tc"
    assert evaluate(k, hw).time_s < 3 * ideal.time_s


def test_dma_destination_changes_which_lanes_are_used():
    to_smem, to_l2, bypass = (_k(HW.override({"memory.dma.destination": d}))
                              for d in ("smem", "l2", "bypass"))
    l2_of = lambda k: sum(a.work.get("l2", 0) for a in k.trace.body)      # noqa: E731
    assert l2_of(bypass) < l2_of(to_smem) < l2_of(to_l2)
    assert evaluate(to_l2, HW).time_s > evaluate(to_smem, HW).time_s


def test_figure3_outputs_all_three_panels(tmp_path):
    import json as _json
    from tilesight.gpuTilingPerfHWModel.genResult.figure3 import figure3_html, figure3_json
    k = _k(HW)
    html = figure3_html(k, HW, "gemm")
    assert "(d)" in html and "(e)" in html and "(f)" in html
    assert "http://" not in html and "https://" not in html      # self-contained
    assert "<svg" in html and "utilisation" in html
    d = _json.loads(figure3_json(k, HW))
    assert set(d) == {"d", "e", "f"}
    assert d["d"]["actions"] and d["e"]["events"] and d["f"]["time_s"] > 0
    assert d["d"]["limiter"] in ("tc", "l2", "ddr", "smem", "latency", "cuda", "sfu", "tmem")
    (tmp_path / "f.html").write_text(html)
    assert (tmp_path / "f.html").stat().st_size > 5000


def test_sweeping_sms_scales_compute_with_it():
    from tilesight.gpuTilingPerfHWModel.model.dse.sweep import default_links
    f = default_links(HW, "sms")
    ch = f(2 * HW.sms)
    assert ch["compute.tc_dense_tflops.fp8"] == HW.get("compute.tc_dense_tflops.fp8") * 2
    assert ch["compute.cuda_fp32_tflops"] == HW.get("compute.cuda_fp32_tflops") * 2
    assert default_links(HW, "memory.ddr.bandwidth_TBps") is None
    # with the link, a compute-bound kernel gets faster when SMs double; without it, it does not
    big = dict(kernel="gemm", M=4096, N=4096, K=7168, a_dtype="fp8", b_dtype="fp8", compute_dtype="fp8")
    from tilesight.cli.server import _kernel_candidates
    base = _kernel_candidates(HW, big)[0]["time_us"]
    linked = _kernel_candidates(HW.override({"sms": 2 * HW.sms, **f(2 * HW.sms)}), big)[0]["time_us"]
    naive = _kernel_candidates(HW.override({"sms": 2 * HW.sms}), big)[0]["time_us"]
    # linked: real speedup (it stops at 1.34x here because L2 then binds, which is correct)
    assert linked < base * 0.85 and naive > base * 0.9


def test_collectives_go_hierarchical_outside_the_fast_domain():
    d = int(HW.get("network.nvlink.domain_size"))
    inside = _core.allreduce(HW, "ar", 256e6, d)[0]
    outside = _core.allreduce(HW, "ar", 256e6, 2 * d)[0]
    assert "hierarchical" in outside.meta["algo"] and "hierarchical" not in inside.meta["algo"]
    # leaving the domain costs more, but nothing like a flat ring over the slow fabric
    assert inside.trace.fixed_time_s < outside.trace.fixed_time_s < 3 * inside.trace.fixed_time_s
    # and it keeps growing slowly with more nodes
    big = _core.allreduce(HW, "ar", 256e6, 9 * d)[0]
    assert outside.trace.fixed_time_s < big.trace.fixed_time_s < 2 * outside.trace.fixed_time_s
    a2a = _core.all_to_all(HW, "a2a", 256e6, 2 * d)[0]
    assert "hierarchical" in a2a.meta["algo"] and a2a.meta["inter_s"] > 0


def test_l2_is_simulated_deterministically_per_partition():
    # tile atoms (l2_port): a DMA page/chunk is sharded over every partition by construction
    HW = HardwareSpec.load("b300").override({"memory.dma.fetch_mode": "l2_port"})
    k = _k(HW)
    assert k.meta["l2_partitions"] == HW.get("memory.l2.partitions")
    assert 0 <= k.meta["l2_hit_rate"] <= 1
    # the model can say exactly what is resident, not just a probability
    assert sum(k.meta["l2_partition_tiles"]) > 0
    assert all(u <= c for u, c in zip(k.meta["l2_partition_bytes"], k.meta["l2_partition_capacity"]))
    assert k.meta["l2_waves_simulated"] >= 1
    # deterministic: same inputs, same answer
    assert _k(HW).meta["l2_hit_rate"] == k.meta["l2_hit_rate"]
    # capacity actually binds: a tiny L2 evicts and misses more
    small = _k(HW.override({"memory.l2.effective_capacity_MB": 0.5}))
    assert small.meta["l2_hit_rate"] < k.meta["l2_hit_rate"]
    assert sum(small.meta["l2_partition_evictions"]) > 0
    # the address map decides the partition: hashing spreads the same tiles out
    hashed = _k(HW.override({"memory.addressing.l2.mode": "hash"}))
    used_plain = sum(1 for t in k.meta["l2_partition_tiles"] if t)
    used_hash = sum(1 for t in hashed.meta["l2_partition_tiles"] if t)
    assert used_hash > used_plain
    # a pure LRU sanity case: 3 tiles into a 2-tile cache, cyclic -> always miss
    r = _core.cache_simulate([1, 2, 3] * 4, [0, 4096, 8192] * 4, [4096.0] * 12, [0] * 12, 1,
                             capacity_bytes=2 * 4096, n_partitions=1, ports=1, granularity=1024,
                             policy="lru")
    assert r["hit_rate"] == 0.0


def test_cross_wave_reuse_is_visible():
    # weights re-read by a later wave can hit now (B1); simulating one wave only cannot see it
    one = _k(HW.override({"memory.l2.waves_simulated": 1}))
    two = _k(HW.override({"memory.l2.waves_simulated": 3}))
    assert two.meta["l2_waves_simulated"] > one.meta["l2_waves_simulated"]
    assert two.meta["l2_hit_rate"] >= one.meta["l2_hit_rate"]


def test_queueing_inflates_latency_near_saturation():
    off = HW.override({"memory.queueing.coef": 0.0})
    on = HW.override({"memory.queueing.coef": 1.0})
    a = evaluate(_k(off), off).time_s
    b = evaluate(_k(on), on).time_s
    assert b > a                                    # queueing can only make it slower
    assert b < 1.5 * a                              # and is capped, not unbounded


def test_overlap_modes_hide_collectives():
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig as RC
    m = ModelSpec.load("kimi_k2.hf")
    base = dict(phase="decode", batch=128, seq_len=4096, tp=2, dp=4)
    none = run_model(m, HW, RC(overlap_mode="none", **base))
    stream = run_model(m, HW, RC(overlap_mode="stream", **base))
    two = run_model(m, HW, RC(overlap_mode="two_batch", **base))
    comm = lambda r: sum(o.total_s for o in r.ops if o.op.kind in ("allreduce", "a2a"))  # noqa: E731
    assert comm(none) > 0
    assert comm(stream) <= comm(none) and comm(two) <= comm(stream) + 1e-12
    assert two.step_time_s <= stream.step_time_s <= none.step_time_s
    manual = run_model(m, HW, RC(overlap_mode="manual", comm_overlap=0.5, **base))
    assert abs(comm(manual) - 0.5 * comm(none)) / comm(none) < 1e-9


def test_capacity_loss_is_configured_and_off_by_default():
    # DEFAULT: no loss — every preset may use its full physical capacity
    for name in ("b300", "b200", "h200", "mi300x", "mi325x", "mi355x", "mi450"):
        h = HardwareSpec.load(name)
        assert h.get("memory.l2.capacity_derate", 1.0) == 1.0
        assert h.l2_capacity_bytes == h.get("memory.l2.capacity_MB") * 1024 * 1024
        assert h.get("memory.l1.capacity_derate", 1.0) == 1.0
    # a configured loss applies...
    d = HW.override({"memory.l2.capacity_derate": 0.5})
    assert abs(d.l2_capacity_bytes - HW.l2_capacity_bytes * 0.5) < 1
    # ...and a measured cliff overrides it outright
    o = HW.override({"memory.l2.effective_capacity_MB": 83})
    assert o.l2_capacity_bytes == 83 * 1024 * 1024
    # a smaller cache means more misses
    lo = _k(HW.override({"memory.l2.effective_capacity_MB": 1}))
    hi = _k(HW.override({"memory.l2.effective_capacity_MB": 200}))
    assert lo.meta["l2_hit_rate"] <= hi.meta["l2_hit_rate"]


def test_l1_is_a_modeled_level_only_for_paths_that_use_it():
    # L1 = (capacity - SMEM carve-out) x derate, x cluster size when the cluster shares it
    l1 = HW.get("memory.l1")
    expect = (l1["capacity_KB"] - l1["smem_carveout_KB"]) * 1024 * l1.get("capacity_derate", 1.0) * l1["cluster_size"]
    assert abs(HW.l1_capacity_bytes - expect) < 1
    assert "l1" in [l.name for l in HW.lanes()]
    dma = _lower_gemm(HW, 1024, 1024, 512, tile=TileConfig(64, 64, 64, load_path="tma"))[0]
    lsu = _lower_gemm(HW, 1024, 1024, 512, tile=TileConfig(64, 64, 64, load_path="lsu"))[0]
    assert dma.meta["l1_miss"] is None                     # DMA writes SMEM directly
    assert lsu.meta["l1_miss"] is not None and all(0 <= m <= 1 for m in lsu.meta["l1_miss"])
    # an L1 hit does not reach the L2 datapath
    l2_of = lambda k: sum(a.work.get("l2", 0) for a in k.trace.body)   # noqa: E731
    no_l1 = _lower_gemm(HW.override({"memory.l1.cache_global_loads": False}), 1024, 1024, 512,
                        tile=TileConfig(64, 64, 64, load_path="lsu"))[0]
    assert l2_of(lsu) < l2_of(no_l1)
    assert any(a.work.get("l1", 0) > 0 for a in lsu.trace.body)


def test_lossless_mode_turns_every_modelled_loss_off():
    ideal = HW.lossless()
    assert all(ideal.eff(k) == 1.0 for k in ("tc", "cuda", "sfu", "ddr", "l2", "smem"))
    assert ideal.get("memory.queueing.coef") == 0.0
    assert ideal.get("memory.l2.capacity_derate") == 1.0
    assert ideal.outstanding_cap("ddr") == float("inf")
    fast = evaluate(_k(ideal), ideal).time_s
    real = evaluate(_k(HW), HW).time_s
    assert fast < real                       # the ideal machine is never slower
    assert fast > real * 0.5                 # but the losses are a correction, not the model


def test_config_is_the_only_interface_between_gpu_and_model():
    import pathlib
    import re
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun import schema
    # 1. every shipped preset validates against the schema
    for name in ("b300", "b200", "h200", "mi300x", "mi325x", "mi355x", "mi450"):
        assert HardwareSpec.load(name).validate() == [], name
    # 2. the model (never a device/vendor name) — the shape-only lowering that stays Python
    # (interfaceAndRun/lower.py, attention_blocks.py, memmap.py — see CLAUDE.md) plus model/,
    # and separately gpu_top's C++ below. hardware_spec.py/schema.py/slice_config.py legitimately
    # name vendors in doc comments (they describe real hardware fields) so are not scanned.
    root = pathlib.Path(schema.__file__).parent.parent
    py_files = list((root / "model").rglob("*.py"))
    for name in ("lower.py", "attention_blocks.py", "memmap.py"):
        py_files.append(root / "interfaceAndRun" / name)
    for f in py_files:
        code = re.sub(r'(""".*?"""|#[^\n]*)', "", f.read_text(), flags=re.S)
        for word in ("b300", "h200", "mi355x", "blackwell", "hopper", "cdna"):
            assert word not in code.lower(), f"{f.name} hard-codes {word}"
    for f in (root / "model" / "gpu_top").glob("*.cpp"):
        code = re.sub(r"(/\*.*?\*/|//[^\n]*)", "", f.read_text(), flags=re.S)
        for word in ("b300", "h200", "mi355x", "blackwell", "hopper", "cdna"):
            assert word not in code.lower(), f"{f.name} hard-codes {word}"
    # 3. defaults live in the schema, not at the call sites
    assert HardwareSpec({"name": "x"}).get("occupancy.max_blocks_per_sm") == \
        schema.default_for("occupancy.max_blocks_per_sm")
    assert HardwareSpec({"name": "x"}).get("memory.queueing.coef") == 0.0     # no loss by default
    # 4. validation catches typos and missing required fields
    bad = HardwareSpec({"name": "x", "sms": 1, "memory": {"l2": {"capcity_MB": 1}}}).validate()
    assert any("unknown field" in p for p in bad) and any("missing required" in p for p in bad)
