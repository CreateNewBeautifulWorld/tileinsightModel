"""PDF report: the same numbers as the web results page, as a document you can send on."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from tilesight.gpuTilingPerfHWModel.model.engine import backend
from tilesight.gpuTilingPerfHWModel.genResult.timeline import steady_timeline

FONT = "Helvetica"
LANE_RGB = {"switch": "#6f8f6a", "tc": "#b4552d", "cuda": "#6b8f9c", "sfu": "#8a6fb0", "smem": "#5b9279",
            "tmem": "#c2903a", "l1": "#2f7d54", "l2": "#4f7cac", "ddr": "#b3563a",
            "sram": "#7f9a52", "net": "#7a7a7a"}


def _tbl(rows, widths=None, size=8):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#6b6a67")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e3e1dd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


class _Gantt(Flowable):
    """Drawing of the steady-state timeline, grouped by unit section."""

    def __init__(self, tl, width, groups):
        super().__init__()
        self.tl, self.width, self.groups = tl, width, groups

    def wrap(self, *_):
        rows = sum(1 + len(v) for v in self.groups.values() if v)
        self.height = rows * 13 + 18
        return self.width, self.height

    def draw(self):
        self._paint(self.canv, 0, 0)

    def _paint(self, canv, x, y):
        tl = self.tl
        span = max((i["end"] for i in tl["steady"]), default=1.0) or 1.0
        left, right = x + 90, x + self.width - 10
        row = y + self.wrap()[1] - 14
        for gname, lanes in self.groups.items():
            if not lanes:
                continue
            canv.setFont(FONT, 7)
            canv.setFillColor(colors.HexColor("#6b6a67"))
            canv.drawString(x, row, gname.upper())
            row -= 12
            for lane in lanes:
                canv.setFont(FONT, 7)
                canv.setFillColor(colors.HexColor("#1d1c1a"))
                canv.drawRightString(left - 6, row, lane)
                canv.setFillColor(colors.HexColor("#f2efec"))
                canv.rect(left, row - 2, right - left, 8, stroke=0, fill=1)
                for it in tl["steady"]:
                    v = it["lanes"].get(lane)
                    if not v:
                        continue
                    x0 = left + (right - left) * it["start"] / span
                    w = max(0.8, (right - left) * v / span)
                    canv.setFillColor(colors.HexColor(LANE_RGB.get(lane, "#9a8f7a")))
                    canv.rect(x0, row - 2, w, 8, stroke=0, fill=1)
                row -= 12
        canv.setFont(FONT, 7)
        canv.setFillColor(colors.HexColor("#6b6a67"))
        canv.drawString(left, y, "0")
        canv.drawRightString(right, y, f"{span * 1e9:.0f} ns / {span * tl['clock_hz']:.0f} cycles")

    def drawOn(self, canv, x, y, _sW=0):  # noqa: N803
        canv.saveState()
        canv.translate(x, y)
        self._paint(canv, 0, 0)
        canv.restoreState()


def unit_groups(lanes: list[str]) -> dict[str, list[str]]:
    """The three blocks of the model, with their units — the same split as the GPU config."""
    g = {"shader slice · cores": [], "shader slice · L1/scratchpad": [],
         "shader slice · load paths": [], "on-chip buffer": [],
         "memory · L2 ports": [], "memory · HBM": []}
    for l in lanes:
        if l in ("tc", "cuda", "sfu"):
            g["shader slice · cores"].append(l)
        elif l in ("smem", "tmem", "l1"):
            g["shader slice · L1/scratchpad"].append(l)
        elif l in ("sram", "switch"):
            g["on-chip buffer"].append(l)
        elif l.startswith("path"):
            g["shader slice · load paths"].append(l)
        elif l == "l2":
            g["memory · L2 ports"].append(l)
        else:
            g["memory · HBM"].append(l)
    return g


def write_pdf(path: str, cur_gpu_config, kernels, title: str, workload_cfg: dict | None = None,
              report=None) -> str:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName=FONT, fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=FONT, fontSize=10,
                        textColor=colors.HexColor("#6b6a67"), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["Normal"], fontName=FONT, fontSize=8,
                          textColor=colors.HexColor("#3a3835"))
    doc = SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title=title)
    W = doc.width
    story = [Paragraph(title, h1),
             Paragraph(f"{cur_gpu_config.name} · {cur_gpu_config.sms} SMs · {cur_gpu_config.clock_hz / 1e9:.2f} GHz · analytical model, "
                       f"nothing was executed on a GPU", body)]

    if report is not None:
        m = report.memory
        story += [Paragraph("summary", h2),
                  _tbl([["step time", f"{report.step_time_s * 1e3:.3f} ms"],
                        ["throughput", f"{report.tokens_per_s_per_gpu:,.0f} tok/s/GPU"],
                        ["weights / KV / activations",
                         f"{m.weights_GB:.1f} / {m.kv_cache_GB:.1f} / {m.activations_GB:.2f} GB"],
                        ["memory total", f"{m.total_GB:.1f} / {m.capacity_GB:.0f} GB "
                                         f"({'fits' if m.fits else 'OVERFLOW'})"],
                        ["top bound", next(iter(report.detail_breakdown()), "-")]], [60 * mm, 120 * mm])]
        story += [Paragraph("where the time goes", h2),
                  _tbl([["resource:tensor", "share"]] +
                       [[k, f"{v / report.step_time_s:.1%}"]
                        for k, v in list(report.detail_breakdown().items())[:10]], [90 * mm, 25 * mm])]

    if workload_cfg:
        flat = []
        def walk(d, p=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(v, p + k + ".")
                else:
                    flat.append([p + k, str(v)])
        walk(workload_cfg)
        story += [Paragraph("workload config", h2), _tbl([["field", "value"]] + flat, [70 * mm, 60 * mm])]

    for k in kernels[:3]:
        tl = steady_timeline(k, cur_gpu_config)
        res = backend.evaluate(k, cur_gpu_config)
        story += [PageBreak(), Paragraph(f"kernel: {k.name}", h2),
                  _tbl([["latency", f"{res.time_s * 1e6:.2f} us / {res.time_s * tl['clock_hz']:.0f} cycles"],
                        ["tile", str(k.meta.get("tile", "-"))],
                        ["occupancy", f"{k.meta.get('resident', '?')} blocks/SM "
                                      f"(limited by {k.meta.get('occ_limiter', '?')})"],
                        ["round", f"{tl['round_cyc']:.1f} cycles, limiter {tl['limiter']}"],
                        ["waves", f"{tl['full_waves']} full + tail {tl['tail_blocks']} block(s)"],
                        ["L2 hit (simulated)", f"{k.meta.get('l2_hit_rate', float('nan')):.1%}"
                                               if "l2_hit_rate" in k.meta else "-"],
                        ["bound", res.bottleneck]], [50 * mm, 120 * mm]),
                  Spacer(1, 6),
                  Paragraph("steady-state timeline, grouped by unit", body),
                  _Gantt(tl, W, unit_groups(tl["lanes"])),
                  Spacer(1, 8),
                  _tbl([["lane", "busy cycles / round", "occupancy"]] +
                       [[lane, f"{sum(i['lanes'].get(lane, 0) for i in tl['steady'] if i['round'] == 0) * tl['clock_hz']:.1f}",
                         f"{sum(i['lanes'].get(lane, 0) for i in tl['steady'] if i['round'] == 0) / tl['round_s']:.1%}"]
                        for lane in tl["lanes"]], [40 * mm, 45 * mm, 30 * mm])]
    doc.build(story)
    return path
