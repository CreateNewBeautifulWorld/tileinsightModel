"""Excel export of the cycle-level trace.

Layout (sheet "timeline"): one row per GPU cycle, one column per hardware unit.
A cell holds the tile that occupies that unit in that cycle, e.g. `A#7` (the A tile of
iteration 7), `B#7`, `MMA#5`. **Colour = iteration**, so the same tile keeps its colour when
it shows up on a different unit at a different time — a load that runs on `ddr`/`l2`/`path`
in one round is consumed by `tc` a few rounds later (multi-buffering), and the colour makes
that stagger visible. The stagger in cycles is written on the "legend" sheet.

A tile occupies a unit for as many cycles as the model gives it (bytes/bandwidth or
FLOPs/rate), so a wide tile can be 200–500 cycles on `ddr` and a few dozen on `tc`.

Other sheets: "legend" (colours, the stagger, every assumption) and "summary" (round
composition, lane occupancy, wave/grid structure).
"""
from __future__ import annotations

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .timeline import cycle_rows

FONT = "Arial"
# 12 distinguishable fills, cycled by iteration index
PALETTE = ["BFD7EA", "F2C6A0", "C9E4C5", "E7C6E0", "F7E3A1", "BCD0C7", "F2B5A0", "C3C7E8",
           "D9E8B8", "E8C7C0", "B8DDE8", "EAD6B8"]
IDLE = "F5F5F4"


def _tile_label(item: dict) -> str:
    a = item["action"]
    if a.startswith("load:"):
        what = a.split(":", 1)[1]
        short = {"act": "A", "weight": "B", "expert_weight": "B", "q": "Q",
                 "kv_cache(latent)": "KV", "kv_cache(K)": "K", "kv_cache(V)": "V",
                 "probs(P)": "P", "scores(S)": "S"}.get(what, what[:6])
        return f"{short}#{item.get('iteration', 0)}"
    if a.startswith("store:"):
        return f"ST#{item.get('round', 0)}"
    if a in ("mma", "gemm_qk", "gemm_pv", "softmax"):
        return f"{a.upper().replace('GEMM_', '')}#{item.get('iteration', item.get('round', 0))}"
    return f"{a}#{item.get('iteration', 0)}"


def _color(label: str) -> str:
    if "#" not in label:
        return PALETTE[0]
    try:
        it = int(label.split("#", 1)[1])
    except ValueError:
        return PALETTE[0]
    return PALETTE[it % len(PALETTE)]


def write_excel(tl: dict, path: str, title: str = "", hw_name: str = "", full: bool = False,
                max_rows: int = 20000, extra: dict | None = None) -> dict:
    """Write the cycle grid to `path`. Returns {'rows':…, 'truncated':bool}."""
    wb = Workbook()
    ws = wb.active
    ws.title = "timeline"
    from .pdfreport import unit_groups
    groups = {k: v for k, v in unit_groups(tl["lanes"]).items() if v}
    lanes = [l for v in groups.values() for l in v]          # columns ordered by section
    head = ["cycle", "time_ns", "phase", "round"] + lanes
    # row 1: section band over the unit columns, row 2: the column names
    ws.append([""] * 4 + [g for g, v in groups.items() for _ in v])
    col = 5
    band = {"shader": "F2E2D8", "L1 / scratchpad": "DCEBE2", "on-chip buffer": "E7EBD6",
            "DMA": "EDE8DE", "L2 ports": "DCE5F0", "HBM": "F2DED8"}
    for g, v in groups.items():
        if len(v) > 1:
            ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + len(v) - 1)
        for i in range(len(v)):
            c = ws.cell(row=1, column=col + i)
            c.fill = PatternFill("solid", fgColor=band.get(g, "EEEEEE"))
            c.font = Font(name=FONT, bold=True, size=8)
            c.alignment = Alignment(horizontal="center")
        col += len(v)
    ws.append(head)
    for c in range(1, len(head) + 1):
        cell = ws.cell(row=2, column=c)
        cell.font = Font(name=FONT, bold=True, size=9)
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "E3"

    # map action -> tile label per phase/round (cycle_rows gives action names only)
    label_of: dict[tuple[str, int, str], str] = {}
    for it in tl["prologue"] + tl["steady"] + tl["epilogue"]:
        label_of[(it.get("round", -1) if "round" in it else -1, it.get("iteration", -1),
                  it["action"])] = _tile_label(it)

    rows, truncated = 0, False
    for c, ns, phase, rnd, cur, _addr in cycle_rows(tl, full=full):
        if rows >= max_rows:
            truncated = True
            break
        r = rows + 3
        ws.cell(row=r, column=1, value=c).font = Font(name=FONT, size=9)
        ws.cell(row=r, column=2, value=round(ns, 2)).font = Font(name=FONT, size=9)
        ws.cell(row=r, column=3, value=phase).font = Font(name=FONT, size=9)
        ws.cell(row=r, column=4, value=rnd).font = Font(name=FONT, size=9)
        for i, lane in enumerate(lanes):
            action, busy = cur[lane]
            cell = ws.cell(row=r, column=5 + i)
            cell.font = Font(name=FONT, size=8)
            cell.alignment = Alignment(horizontal="center")
            if not action:
                cell.fill = PatternFill("solid", fgColor=IDLE)
                continue
            label = next((v for (rr, ii, aa), v in label_of.items()
                          if aa == action and (rr == rnd or rnd < 0)), action)
            cell.value = label
            cell.fill = PatternFill("solid", fgColor=_color(label))
            if busy < 0.999:
                cell.value = f"{label} ({busy:.0%})"
        rows += 1
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 7
    for i in range(len(lanes)):
        ws.column_dimensions[get_column_letter(5 + i)].width = 16

    # ---------------------------------------------------------------- addresses
    events = [it for it in tl["prologue"] + tl["steady"] + tl["epilogue"] if "addr" in it]
    if events:
        ad = wb.create_sheet("addresses")
        ad.append(["phase/round", "iteration", "action", "tile label", "address", "size B",
                   "L2 slice", "HBM port", "start_cyc"])
        for it in events:
            label = _tile_label(it)
            r = ad.max_row + 1
            ad.append([it.get("round", -1), it.get("iteration", -1), it["action"], label,
                       f"0x{it['addr']:012x}", it["addr_size"], it["l2_slice"], it["hbm_port"],
                       round(it["start_cyc"], 1)])
            ad.cell(row=r, column=4).fill = PatternFill("solid", fgColor=_color(label))
        for row in ad.iter_rows():
            for c2 in row:
                c2.font = Font(name=FONT, size=9, bold=c2.row == 1)
        for col, wdt in zip("ABCDEFGHI", (12, 10, 26, 14, 18, 10, 9, 9, 11)):
            ad.column_dimensions[col].width = wdt

    # ---------------------------------------------------------------- legend
    lg = wb.create_sheet("legend")
    lg.append(["What you are looking at"])
    lg.append(["One row = one GPU cycle. One column = one hardware unit (lane)."])
    lg.append(["Cell text = the tile occupying that unit, e.g. A#7 = A tile of iteration 7."])
    lg.append(["Colour = iteration: the same tile keeps its colour on every unit it touches,"])
    lg.append(["so you can follow it from HBM/L2 to SMEM and finally into the tensor core."])
    lg.append(["A percentage in a cell means the unit is busy only that fraction of the cycle."])
    lg.append([])
    stagger_rounds = max(0, tl["stages"] - 1)
    lg.append(["Stagger (load -> use)", f"{stagger_rounds} round(s)",
               f"{stagger_rounds * tl['round_cyc']:.1f} cycles",
               f"{stagger_rounds * tl['round_s'] * 1e9:.1f} ns"])
    lg.append(["Reason", f"{tl['stages']} software-pipeline stages: the MMA of iteration i reads the "
                         f"buffer filled {stagger_rounds} round(s) earlier"])
    lg.append([])
    lg.append(["Colour key"])
    for i, col in enumerate(PALETTE):
        c = lg.cell(row=lg.max_row + 1, column=1, value=f"iteration % {len(PALETTE)} == {i}")
        c.font = Font(name=FONT, size=9)
        lg.cell(row=lg.max_row, column=2).fill = PatternFill("solid", fgColor=col)
    lg.append([])
    lg.append(["Assumptions (all analytical — nothing was executed on a GPU)"])
    for line in [
        f"hardware: {hw_name}, clock {tl['clock_hz'] / 1e9:.3f} GHz (hw YAML, [calib] where tagged)",
        f"kernel: {title}",
        "a unit is modelled as busy contiguously from the action's start",
        "loads are multi-buffered; the round they are drawn in is not the round they are used in",
        "one representative SM is shown; every SM in a wave runs the identical pipeline",
        "addresses (sheet 'addresses') come from the tensor layout + base offset, tile-granular",
        f"grid: {tl['num_blocks']} blocks = {tl['full_waves']} full wave(s) x {tl['sms']}x{tl['resident']}"
        f" + tail {tl['tail_blocks']} block(s) on {tl['tail_active_sms']} SM(s)",
    ]:
        lg.append([line])
    for k, v in (extra or {}).items():
        lg.append([k, str(v)])
    lg.column_dimensions["A"].width = 68
    lg.column_dimensions["B"].width = 44
    for row in lg.iter_rows():
        for c in row:
            if c.value is not None:
                c.font = Font(name=FONT, size=9, bold=str(c.value).endswith(("looking at", "key",
                                                                             "GPU)")))

    # ---------------------------------------------------------------- summary
    sm = wb.create_sheet("summary")
    sm.append(["metric", "cycles", "ns", "note"])
    sm.append(["round (steady)", round(tl["round_cyc"], 1), round(tl["round_s"] * 1e9, 2),
               f"limiter: {tl['limiter']}"])
    sm.append(["resource bound", round(tl["resource_bound_s"] * tl["clock_hz"], 1),
               round(tl["resource_bound_s"] * 1e9, 2), "max over lanes of summed occupancy"])
    sm.append(["dependency bound", round(tl["latency_bound_s"] * tl["clock_hz"], 1),
               round(tl["latency_bound_s"] * 1e9, 2), "critical path / stages, recurrence / consumers"])
    sm.append(["prologue", round(tl["prologue_s"] * tl["clock_hz"], 1), round(tl["prologue_s"] * 1e9, 2), ""])
    sm.append(["epilogue", round(tl["epilogue_s"] * tl["clock_hz"], 1), round(tl["epilogue_s"] * 1e9, 2), ""])
    sm.append(["kernel (one wave)", round(tl["total_s"] * tl["clock_hz"], 1),
               round(tl["total_s"] * 1e9, 2), f"{tl['iters']} iterations"])
    sm.append([])
    sm.append(["lane", "busy cycles / round", "occupancy", ""])
    busy = {lane: sum(it["lanes"].get(lane, 0.0) for it in tl["steady"] if it["round"] == 0)
            for lane in lanes}
    for lane, v in sorted(busy.items(), key=lambda kv: -kv[1]):
        sm.append([lane, round(v * tl["clock_hz"], 1), round(v / tl["round_s"], 4), ""])
    for row in sm.iter_rows():
        for c in row:
            c.font = Font(name=FONT, size=9, bold=c.row == 1)
    sm.column_dimensions["A"].width = 22
    sm.column_dimensions["B"].width = 20
    sm.column_dimensions["C"].width = 14
    sm.column_dimensions["D"].width = 46
    wb.save(path)
    return {"rows": rows, "truncated": truncated, "lanes": lanes}
