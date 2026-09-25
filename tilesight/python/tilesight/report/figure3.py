"""Figure 3 (d)(e)(f) of the paper, produced as an output artifact.

  (d) the analysis: per-action resource vectors + the inter-tile DAG + the recursive
      prologue / steady / epilogue envelope, with both bounds spelled out
  (e) the timeline: the envelope rendered over the resource lanes, software-pipelined loads
      overlapping compute, with the round boundaries marked
  (f) the per-tile report: latency, utilisation, cache hit, overlap rate

Writes one self-contained HTML file (no external resources) so it can be opened anywhere or
attached to a report.
"""
from __future__ import annotations

import html
import json

from ..engine import backend
from ..gpuTilingHWModel.spec import HardwareSpec
from ..ir.kernel import Kernel
from .timeline import steady_timeline

LANE_COLOR = {"tc": "#b4552d", "cuda": "#6b8f9c", "sfu": "#8a6fb0", "smem": "#5b9279",
              "tmem": "#c2903a", "l2": "#4f7cac", "ddr": "#b3563a", "sram": "#7f9a52",
              "net": "#7a7a7a"}


def _color(lane: str) -> str:
    return LANE_COLOR.get(lane, "#9a8f7a" if lane.startswith("path") else "#888")


def _panel_d(k: Kernel, tl: dict) -> str:
    lanes = tl["lanes"]
    rows = []
    for i, a in enumerate(k.body):
        cells = "".join(
            f'<td class="num">{(a.work.get(l, 0) * 1e9):.1f}</td>' if not l.startswith(("l2", "ddr", "sram"))
            else f'<td class="num">{a.work.get(l, 0) / 1024:.1f} KB</td>' for l in lanes)
        deps = ", ".join(k.body[d].name for d in a.deps) or "—"
        rows.append(f'<tr><td class="mono">{html.escape(a.name)}</td>{cells}'
                    f'<td class="num">{a.latency_s * tl["clock_hz"]:.0f}</td>'
                    f'<td class="mono">{html.escape(deps)}</td>'
                    f'<td>{"loop-carried" if a.recurrent else ""}</td></tr>')
    head = "".join(f"<th>{l}</th>" for l in lanes)
    env = (f"T = T_pro + T_fill + {tl['iters']} x R + T_epi = "
           f"{tl['prologue_s'] * 1e9:.0f} + {max(0.0, tl['cp_s'] - tl['round_s']) * 1e9:.0f} + "
           f"{tl['iters']} x {tl['round_s'] * 1e9:.1f} + {tl['epilogue_s'] * 1e9:.0f} ns")
    return f"""
<h2>(d) intra-tile resource vectors + inter-tile DAG &rarr; pipeline envelope</h2>
<table><thead><tr><th>action</th>{head}<th>latency (cyc)</th><th>depends on</th><th></th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">Per-SM lanes are nanoseconds of occupancy on one SM; shared lanes (l2 / ddr / sram)
are bytes. The round is <b>R = max(resource {tl['resource_bound_s'] * 1e9:.1f} ns,
dependency {tl['latency_bound_s'] * 1e9:.1f} ns)</b> = {tl['round_s'] * 1e9:.1f} ns
&rarr; limiter <b>{tl['limiter']}</b>, where the dependency term is
max(critical path / {tl['stages']} stages, loop-carried chain / {tl['consumers']} consumers).<br>
{env}, over {tl['full_waves']} full wave(s) of {tl['sms']}x{tl['resident']} blocks
+ a tail of {tl['tail_blocks']} block(s) on {tl['tail_active_sms']} SM(s).</p>"""


def _panel_e(tl: dict) -> str:
    lanes, span = tl["lanes"], max((i["end"] for i in tl["steady"]), default=1.0) or 1.0
    W, rowH, P = 880, 26, 90
    H = len(lanes) * rowH + 52
    X = lambda t: P + (W - P - 16) * t / span                       # noqa: E731
    g = []
    for r in range(tl["rounds_shown"]):
        x = X(r * tl["round_s"])
        g.append(f'<line x1="{x:.1f}" y1="16" x2="{x:.1f}" y2="{len(lanes) * rowH + 22}" '
                 f'stroke="#999" stroke-dasharray="3 3"/>')
    for i, lane in enumerate(lanes):
        y = 22 + i * rowH
        g.append(f'<text x="{P - 8}" y="{y + 13}" font-size="11" text-anchor="end">{lane}</text>')
        g.append(f'<rect x="{P}" y="{y}" width="{W - P - 16}" height="{rowH - 6}" fill="#0000000d" rx="3"/>')
        for it in tl["steady"]:
            v = it["lanes"].get(lane)
            if not v:
                continue
            x0, w = X(it["start"]), max(1.5, X(it["start"] + v) - X(it["start"]))
            g.append(f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{rowH - 6}" rx="3" '
                     f'fill="{_color(lane)}" opacity="{0.95 if it["round"] == 0 else 0.5}">'
                     f'<title>{html.escape(it["action"])} · {v * 1e9:.1f} ns · '
                     f'{v * tl["clock_hz"]:.0f} cyc · iteration {it.get("iteration", 0)}</title></rect>')
    g.append(f'<text x="{P}" y="{H - 8}" font-size="11">0</text>')
    g.append(f'<text x="{W - 16}" y="{H - 8}" font-size="11" text-anchor="end">'
             f'{span * 1e9:.0f} ns / {span * tl["clock_hz"]:.0f} cycles</text>')
    return f"""
<h2>(e) timeline of one SM: {tl['rounds_shown']} rounds of the K-loop</h2>
<svg viewBox="0 0 {W} {H}" width="100%">{''.join(g)}</svg>
<p class="note">Dashed lines are round boundaries. Loads drawn in a round belong to iteration
i+{tl['stages'] - 1} (multi-buffered) while the compute is iteration i — that offset is the
software pipeline. Hover a bar for its occupancy in ns and cycles.</p>"""


def _panel_f(res, k: Kernel, tl: dict) -> str:
    util = "".join(f'<tr><td class="mono">{l}</td>'
                   f'<td class="num">{v:.1%}</td>'
                   f'<td><div class="bar"><i style="width:{min(100, v * 100):.0f}%;'
                   f'background:{_color(l)}"></i></div></td></tr>'
                   for l, v in sorted(res.util.items(), key=lambda kv: -kv[1]))
    lim = "".join(f'<tr><td class="mono">{html.escape(n)}</td>'
                  f'<td class="num">{t * 1e9:.0f} ns</td>'
                  f'<td class="num">{t / res.time_s:.1%}</td></tr>'
                  for n, t in sorted(res.limiter_detail.items(), key=lambda kv: -kv[1]) if t > 0)
    m = k.meta
    hits = [(n, v) for n, v in m.items() if "miss" in n and isinstance(v, (int, float))]
    overlap = 1.0 - (tl["prologue_s"] + max(0.0, tl["cp_s"] - tl["round_s"]) + tl["epilogue_s"]) / tl["total_s"]
    facts = [("latency", f"{res.time_s * 1e6:.2f} µs / {res.time_s * tl['clock_hz']:.0f} cycles"),
             ("waves", f"{res.waves}"), ("occupancy", f"{m.get('resident', '?')} blocks/SM"
                                         f" (limited by {m.get('occ_limiter', '?')})"),
             ("tile", str(m.get("tile", "-"))),
             ("overlap rate", f"{overlap:.1%} of the kernel is steady-state overlap"),
             ("bottleneck", res.bottleneck)]
    facts += [(n.replace("_", " "), f"{(1 - v):.1%} hit") for n, v in hits]
    if "l2_hit_rate" in m:
        sim = m.get("l2_sim")
        facts.append(("L2 (simulated)", f"{m['l2_hit_rate']:.1%} hit over {m.get('l2_partitions', 1)} "
                                        f"partitions, {m.get('l2_policy', 'lru')}"))
        if sim is not None:
            facts.append(("L2 residency", "; ".join(
                f"p{i}: {len(p.resident)} tiles / {p.used / 1024:.0f} KB"
                for i, p in enumerate(sim.partitions))))
    return f"""
<h2>(f) per-tile report</h2>
<div class="cols">
<table><tbody>{''.join(f'<tr><td>{a}</td><td class="mono">{html.escape(str(b))}</td></tr>' for a, b in facts)}</tbody></table>
<table><thead><tr><th>lane</th><th>utilisation</th><th></th></tr></thead><tbody>{util}</tbody></table>
<table><thead><tr><th>time charged to</th><th>ns</th><th>share</th></tr></thead><tbody>{lim}</tbody></table>
</div>"""


def figure3_html(k: Kernel, cur_gpu_config: HardwareSpec, title: str = "", addr_fn=None) -> str:
    tl = steady_timeline(k, cur_gpu_config, addr_fn=addr_fn)
    res = backend.evaluate(k, cur_gpu_config)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>TileSight — {html.escape(title or k.name)}</title><style>
body{{font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;
 background:#fbfbfa;color:#1d1c1a;max-width:1000px}}
h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:14px;margin:26px 0 8px;color:#6b6a67;
 text-transform:uppercase;letter-spacing:.06em}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:8px}}
th,td{{text-align:left;padding:4px 7px;border-bottom:1px solid #e3e1dd;white-space:nowrap}}
th{{color:#6b6a67;font-size:11px;text-transform:uppercase}} td.num{{text-align:right;
 font-variant-numeric:tabular-nums}} .mono{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}}
.note{{font-size:12px;color:#6b6a67}} .cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.bar{{background:#f4ece7;border-radius:3px;height:8px;width:90px}} .bar i{{display:block;height:100%;border-radius:3px}}
svg{{background:#fff;border:1px solid #e3e1dd;border-radius:8px}}
@media(max-width:800px){{.cols{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(title or k.name)}</h1>
<div class="note">{cur_gpu_config.name} · {cur_gpu_config.sms} SMs · {cur_gpu_config.clock_hz / 1e9:.2f} GHz · analytical model, nothing was executed</div>
{_panel_d(k, tl)}{_panel_e(tl)}{_panel_f(res, k, tl)}
<p class="note">Panels follow Figure 3(d)(e)(f) of the TileSight paper. Values marked [calib] in
the hardware YAML are placeholders until the microbenchmark suite runs on real silicon.</p>
</body></html>"""


def figure3_json(k: Kernel, cur_gpu_config: HardwareSpec) -> str:
    """Same three panels as data, for pipelines that want the numbers rather than the page."""
    tl = steady_timeline(k, cur_gpu_config)
    res = backend.evaluate(k, cur_gpu_config)
    return json.dumps({
        "d": {"actions": [{"name": a.name, "work": a.work, "deps": a.deps,
                           "latency_cycles": a.latency_s * tl["clock_hz"], "recurrent": a.recurrent}
                          for a in k.body],
              "round_s": tl["round_s"], "resource_bound_s": tl["resource_bound_s"],
              "latency_bound_s": tl["latency_bound_s"], "limiter": tl["limiter"],
              "stages": tl["stages"], "consumers": tl["consumers"], "iters": tl["iters"]},
        "e": {"lanes": tl["lanes"], "events": [{k2: v for k2, v in it.items() if k2 != "lanes_cyc"}
                                               for it in tl["steady"]]},
        "f": {"time_s": res.time_s, "bottleneck": res.bottleneck, "util": res.util,
              "limiter_detail": res.limiter_detail, "waves": res.waves,
              "meta": {n: v for n, v in k.meta.items()
                       if isinstance(v, (int, float, str, bool, list, tuple))}},
    }, indent=2, default=float)
