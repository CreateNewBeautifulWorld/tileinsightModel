"""Timeline reconstruction — the paper's Figure 3(e) view.

The engine returns *when* a round ends and *what* bounds it; this module reconstructs a
concrete schedule consistent with that: one steady-state round of the K-loop body, laid
out lane by lane.

Model: actions are walked in topological order; an action starts when (a) its
dependencies are done and (b) every lane it needs is free. It then occupies each lane
`r` for u_r seconds and finishes after max_r u_r + latency. Loads are multi-buffered, so
the loads drawn in a round belong to iteration i+stages-1 while the compute belongs to
iteration i — that is exactly the software-pipeline overlap Figure 3(e) shows.

The greedy makespan can differ from the engine's round length (the engine takes the max
of a resource bound and the recurrence bounds); both are reported so the picture stays
honest: `round_s` is the number the model uses, `makespan_s` is this schedule's length.
"""
from __future__ import annotations

import math

from ..engine.reference import _longest_path, _node_w, _u
from ..gpuTilingPerfHWModel.spec import HardwareSpec
from ..ir.kernel import Kernel


def steady_timeline(k: Kernel, cur_gpu_config: HardwareSpec, max_rounds: int = 0, addr_fn=None) -> dict:
    """max_rounds = 0 -> stages+1 rounds, so a load and the MMA that consumes it both appear."""
    max_rounds = max_rounds or min(8, k.stages + 1)
    lanes = cur_gpu_config.lanes()
    resident = max(1, k.resident)
    conc = cur_gpu_config.sms * resident
    blocks = min(k.num_blocks, conc) or 1
    active = min(cur_gpu_config.sms, math.ceil(blocks / resident))
    bps = math.ceil(blocks / active)

    def is_load(a):
        return a.name.startswith("load")

    def sched(actions, offset=0.0, prefetch=False):
        items, lane_free, end = [], {}, {}
        for i, a in enumerate(actions):
            use = {l.name: _u(a, l, active) * bps for l in lanes if _u(a, l, active) > 0}
            # with multi-buffering the consumer reads a buffer filled `stages-1` rounds ago,
            # so a load edge does not constrain this round (that is the overlap Fig. 3e shows)
            deps = [d for d in a.deps if not (prefetch and is_load(actions[d]))] if prefetch else a.deps
            dep_ready = max((end[d] for d in deps), default=0.0)
            start = max([dep_ready] + [lane_free.get(n, 0.0) for n in use])
            dur = max(list(use.values()) + [0.0]) + a.latency_s
            for n, v in use.items():
                lane_free[n] = start + v
            end[i] = start + dur
            items.append({"action": a.name, "start": start + offset, "end": start + dur + offset,
                          "latency_s": a.latency_s, "recurrent": a.recurrent,
                          "lanes": {n: v for n, v in use.items()}})
        return items, (max(end.values()) if end else 0.0)

    body_items, makespan = sched(k.body, prefetch=k.stages > 1)
    # engine's numbers for the same wave
    sums = {l.name: bps * sum(_u(a, l, active) for a in k.body) for l in lanes}
    lane, rb = max(sums.items(), key=lambda kv: kv[1])
    w = [_node_w(a, lanes, active) for a in k.body]
    cp = _longest_path(k.body, w)
    cp_rec = _longest_path(k.body, [x if a.recurrent else 0.0 for a, x in zip(k.body, w)])
    lat_bound = max(cp / max(1, k.stages), cp_rec / max(1, k.consumers))
    round_s = max(rb, lat_bound)
    limiter = lane if rb >= lat_bound else "latency"

    rounds = min(max_rounds, max(1, k.iters))
    steady = []
    for r in range(rounds):
        for it in body_items:
            steady.append({**it, "start": it["start"] + r * round_s, "end": it["end"] + r * round_s,
                           "round": r,
                           # loads are prefetches for a later iteration (multi-buffering)
                           "iteration": r + k.stages - 1 if it["action"].startswith("load") else r})
    pro, pro_len = sched(k.prologue)
    epi, epi_len = sched(k.epilogue)
    full, tail = divmod(k.num_blocks, conc)
    clock = cur_gpu_config.clock_hz
    if addr_fn is not None:                       # attach the tile address each action touches
        for it in body_items + steady + pro + epi:
            a = addr_fn(it["action"], it.get("iteration", it.get("round", 0)))
            if a:
                it.update(addr=a["addr"], addr_size=a["size"], l2_slice=a["slice"], hbm_port=a["port"])
    for it in body_items + steady + pro + epi:
        it["start_cyc"] = it["start"] * clock
        it["end_cyc"] = it["end"] * clock
        it["lanes_cyc"] = {n: v * clock for n, v in it["lanes"].items()}
    return {
        "clock_hz": clock,
        "round_cyc": round_s * clock,
        "lanes": [l.name for l in lanes if any(l.name in it["lanes"] for it in body_items + pro + epi)],
        "prologue": pro, "steady": steady, "epilogue": epi,
        "round_s": round_s, "makespan_s": makespan, "rounds_shown": rounds,
        "iters": k.iters, "stages": k.stages, "consumers": k.consumers,
        "resident": resident, "blocks_per_sm": bps, "active_sms": active,
        # how the grid maps onto the machine (paper §3.4: WaveDecompose; one representative SM
        # is modeled, waves are aggregated, the tail wave gets a bigger share of L2/DDR)
        "num_blocks": k.num_blocks, "sms": cur_gpu_config.sms, "full_waves": full,
        "tail_blocks": tail, "tail_active_sms": min(cur_gpu_config.sms, math.ceil(tail / resident)) if tail else 0,
        "limiter": limiter, "resource_bound_s": rb, "latency_bound_s": lat_bound,
        "cp_s": cp, "cp_recurrent_s": cp_rec,
        "prologue_s": pro_len, "epilogue_s": epi_len,
        "total_s": pro_len + max(0.0, cp - round_s) + k.iters * round_s + epi_len,
    }


def machine_timeline(tl: dict) -> dict:
    """Wave-level view of the whole grid (paper §3.4 WaveDecompose).

    Every SM inside a wave runs the identical pipeline in this model, so drawing one row per
    SM would draw the same picture N times. Instead we report the wave structure: how many
    full waves of `SMs x resident` blocks, the tail wave and its active-SM count, and each
    wave's duration. Per-SM variation would require a load-imbalance model (docs/TASKS.md).
    """
    per_wave = tl["total_s"]
    waves = []
    t = 0.0
    for i in range(tl["full_waves"]):
        waves.append({"wave": i, "kind": "full", "blocks": tl["sms"] * tl["resident"],
                      "active_sms": tl["sms"], "start": t, "end": t + per_wave})
        t += per_wave
    if tl["tail_blocks"]:
        # the tail wave keeps the same per-block work but shares L2/DDR among fewer SMs;
        # `tail_scale` is the engine's ratio, approximated here by the active-SM ratio on
        # the bounding lane (exact number comes from the engine, this view is structural)
        waves.append({"wave": len(waves), "kind": "tail", "blocks": tl["tail_blocks"],
                      "active_sms": tl["tail_active_sms"], "start": t, "end": t + per_wave})
        t += per_wave
    return {"waves": waves, "wave_s": per_wave, "total_s": t,
            "sms": tl["sms"], "resident": tl["resident"], "blocks": tl["num_blocks"]}


def cycle_rows(tl: dict, full: bool = False):
    """Yield one row per GPU cycle: (cycle, time_ns, phase, round, {lane: (action, busy_frac)}).

    `full=False` covers the drawn rounds; `full=True` covers prologue + every K-loop
    iteration + epilogue (that is `iters` rounds, so the file can be large)."""
    clock = tl["clock_hz"]
    segs: list[tuple[str, int, list[dict], float]] = []      # (phase, round, items, offset)
    t = 0.0
    if tl["prologue"]:
        segs.append(("prologue", -1, tl["prologue"], 0.0))
        t += tl["prologue_s"]
    body = [it for it in tl["steady"] if it["round"] == 0]
    n_rounds = tl["iters"] if full else tl["rounds_shown"]
    for r in range(n_rounds):
        segs.append(("steady", r, body, t))
        t += tl["round_s"]
    if tl["epilogue"]:
        segs.append(("epilogue", -1, tl["epilogue"], t))
        t += tl["epilogue_s"]
    # lane intervals (start, end, action, phase, round)
    iv = []
    for phase, rnd, items, off in segs:
        for it in items:
            for lane, dur in it["lanes"].items():
                iv.append((it["start"] + off, it["start"] + off + dur, lane, it["action"], phase, rnd,
                           it.get("addr")))
    iv.sort()
    # segment boundaries in time, so a cycle where every unit happens to be idle still
    # reports the phase/round it belongs to
    bounds, t0 = [], 0.0
    for phase, rnd, items, off in segs:
        dur = (tl["prologue_s"] if phase == "prologue" else
               tl["epilogue_s"] if phase == "epilogue" else tl["round_s"])
        bounds.append((t0, t0 + dur, phase, rnd))
        t0 += dur
    lanes = tl["lanes"]
    total_cycles = int(math.ceil(t * clock))
    pos = 0
    live: list[tuple[float, float, str, str, str, int]] = []
    for c in range(total_cycles):
        c0, c1 = c / clock, (c + 1) / clock
        while pos < len(iv) and iv[pos][0] < c1:
            live.append(iv[pos])
            pos += 1
        live = [x for x in live if x[1] > c0]
        cur = {lane: ("", 0.0) for lane in lanes}
        addr = {lane: None for lane in lanes}
        phase, rnd = next(((p, r) for a, b, p, r in bounds if a <= c0 < b), ("idle", -1))
        for st, en, lane, act, ph, r, ad in live:
            ov = max(0.0, min(en, c1) - max(st, c0)) * clock      # fraction of this cycle
            if ov > cur.get(lane, ("", 0.0))[1]:
                cur[lane] = (act, min(1.0, cur[lane][1] + ov) if cur[lane][0] == act else min(1.0, ov))
                if ad is not None:
                    addr[lane] = ad
            elif ov > 0:
                cur[lane] = (cur[lane][0], min(1.0, cur[lane][1] + ov))
                if ad is not None and addr[lane] is None:
                    addr[lane] = ad
        yield c, c0 * 1e9, phase, rnd, cur, addr


def cycle_csv(tl: dict, full: bool = False) -> str:
    """CSV with one row per cycle and one column per hardware unit (plus a busy fraction)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    lanes = tl["lanes"]
    has_addr = any("addr" in it for it in tl["steady"])
    head = ["cycle", "time_ns", "phase", "round"] + lanes + [f"{l}_busy" for l in lanes]
    if has_addr:
        head += [f"{l}_addr" for l in lanes]
    w.writerow(head)
    for c, ns, phase, rnd, cur, addr in cycle_rows(tl, full=full):
        row = [c, f"{ns:.2f}", phase, rnd] + [cur[l][0] for l in lanes] + \
              [f"{cur[l][1]:.3f}" for l in lanes]
        if has_addr:
            row += [f"0x{addr[l]:012x}" if addr[l] is not None else "" for l in lanes]
        w.writerow(row)
    return buf.getvalue()


def timeline_text(tl: dict, width: int = 62, unit: str = "us") -> str:
    """ASCII rendering of the drawn rounds (CLI / tests).  unit: 'us' | 'cyc'."""
    cyc = unit == "cyc"
    scale = tl["clock_hz"] if cyc else 1e6
    tag = "cyc" if cyc else "us"
    span = max((it["end"] for it in tl["steady"]), default=1.0) or 1.0
    rows = []
    rows.append(f"round {tl['round_s'] * scale:.2f} {tag}  (resource {tl['resource_bound_s'] * scale:.2f} / "
                f"latency {tl['latency_bound_s'] * scale:.2f}) -> limiter {tl['limiter']}; "
                f"{tl['blocks_per_sm']} block(s)/SM, {tl['stages']} stages, {tl['consumers']} consumers")
    for lane in tl["lanes"]:
        line = [" "] * width
        for it in tl["steady"]:
            v = it["lanes"].get(lane)
            if not v:
                continue
            a = int(it["start"] / span * width)
            b = max(a + 1, int((it["start"] + v) / span * width))
            for x in range(a, min(b, width)):
                line[x] = "#"
        rows.append(f"{lane:>10s} |{''.join(line)}|")
    marks = [" "] * width
    for r in range(tl["rounds_shown"]):
        x = min(width - 1, int(r * tl["round_s"] / span * width))
        marks[x] = "|"
    rows.append(f"{'round':>10s} |{''.join(marks)}|  0 .. {span * scale:.2f} {tag}")
    return "\n".join(rows)


def trace_text(tl: dict, title: str = "", hw_name: str = "", extra: dict | None = None) -> str:
    """Full text trace: header, per-action event table (ns + cycles), ASCII gantt in both units.

    Written by `tilesight kernel --trace-out FILE` and downloadable from the web UI."""
    clock = tl["clock_hz"]
    L = ["# TileSight steady-state trace (analytical, not measured)",
         f"# kernel      : {title}",
         f"# hardware    : {hw_name}  clock {clock / 1e9:.3f} GHz  {tl['sms']} SMs",
         f"# grid        : {tl['num_blocks']} blocks -> {tl['full_waves']} full wave(s) of "
         f"{tl['sms'] * tl['resident']} + tail {tl['tail_blocks']} block(s) on {tl['tail_active_sms']} SM(s); "
         f"this trace is one representative SM in a full wave ({tl['active_sms']} active)",
         f"# schedule    : {tl['blocks_per_sm']} block(s)/SM, {tl['stages']} stages, "
         f"{tl['consumers']} consumers, {tl['iters']} K-loop iterations",
         f"# round       : {tl['round_s'] * 1e9:.2f} ns = {tl['round_cyc']:.1f} cycles "
         f"(resource {tl['resource_bound_s'] * 1e9:.2f} ns / dependency {tl['latency_bound_s'] * 1e9:.2f} ns)"
         f" -> limiter {tl['limiter']}",
         f"# kernel time : prologue {tl['prologue_s'] * 1e9:.1f} ns + {tl['iters']} rounds + "
         f"epilogue {tl['epilogue_s'] * 1e9:.1f} ns = {tl['total_s'] * 1e6:.3f} us per wave",
         "# NOTE loads shown in round r belong to iteration r+stages-1 (multi-buffered).", ""]
    for k, v in (extra or {}).items():
        L.append(f"# {k:11s}: {v}")
    if extra:
        L.append("")
    has_addr = any("addr" in it for it in tl["steady"])
    L.append(f"{'phase':9s} {'round':>5s} {'iter':>5s} {'action':26s} {'start_ns':>10s} {'dur_ns':>9s} "
             f"{'start_cyc':>10s} {'dur_cyc':>9s} " + ("{:>16s} {:>4s} {:>4s} ".format("addr", "sl", "pt")
                                                       if has_addr else "") + " lane occupancy (ns)")
    def rows(items, phase):
        for it in items:
            lanes = " ".join(f"{n}={v * 1e9:.1f}" for n, v in
                             sorted(it["lanes"].items(), key=lambda kv: -kv[1]))
            addr = (f" 0x{it['addr']:012x} {it['l2_slice']:4d} {it['hbm_port']:4d} "
                    if "addr" in it else ("" if not has_addr else " " * 27))
            L.append(f"{phase:9s} {it.get('round', 0):5d} {it.get('iteration', 0):5d} {it['action']:26s} "
                     f"{it['start'] * 1e9:10.2f} {(it['end'] - it['start']) * 1e9:9.2f} "
                     f"{it['start_cyc']:10.1f} {it['end_cyc'] - it['start_cyc']:9.1f}{addr}  {lanes}")
    rows(tl["prologue"], "prologue")
    rows(tl["steady"], "steady")
    rows(tl["epilogue"], "epilogue")
    L += ["", "# --- gantt (microseconds) ---", timeline_text(tl),
          "", "# --- gantt (cycles) ---", timeline_text(tl, unit="cyc"), ""]
    busy = {lane: sum(it["lanes"].get(lane, 0.0) for it in tl["steady"] if it.get("round") == 0)
            for lane in tl["lanes"]}
    L.append("# lane occupancy in one round:")
    for lane, v in sorted(busy.items(), key=lambda kv: -kv[1]):
        L.append(f"#   {lane:10s} {v * 1e9:9.2f} ns  {v * clock:9.1f} cyc  {v / tl['round_s']:6.1%}")
    return "\n".join(L)
