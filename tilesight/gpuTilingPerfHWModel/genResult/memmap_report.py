"""Where every tensor lives in HBM: the per-GPU memory map as files and a page section.

`interfaceAndRun/memmap.py` allocates every tensor this GPU holds as one contiguous, 2 MB-aligned
region from `memory.addressing.base` (default 0x8000_0000), in execution order. This module
turns that allocation into something to look at:

  * one row per tensor: name, kind, layer, [base, end), size, logical shape, and how its bytes
    spread over the HBM ports (`memory.addressing.ddr`) and L2 slices (`memory.addressing.l2`) —
    computed exactly from the interleave map, not sampled;
  * per-kind totals and per-HBM-port bytes by kind;
  * CSV + Excel (sheets `regions`, `address_map` with a chart of the address space,
    `hbm_ports` with a stacked chart, `summary`) under `out/`, and a JSON form the results page
    draws.

Run it standalone: `python -m tilesight.gpuTilingPerfHWModel.genResult.memmap_report --model
kimi_k2.hf --gpu-tiling-perf-hw-model b300 --phase decode --batch 256 --seq 8192 --dp 8`
(or `tilesight memmap ... --out DIR`).
"""
from __future__ import annotations

import csv
from pathlib import Path

from tilesight.gpuTilingPerfHWModel.genResult.addressing import AddressMap

OUT_DIR = Path(__file__).resolve().parents[1] / "out"
KINDS = ("weight", "kv", "act", "workspace")
KIND_COLOR = {"weight": "4F7CAC", "kv": "B3563A", "act": "7F9A52", "workspace": "C2903A"}


def port_bytes(amap: AddressMap, addr: int, size: int) -> list[float]:
    """Exact bytes of [addr, addr + size) landing on each port of `amap`."""
    out = [0.0] * amap.ports
    if size <= 0:
        return out
    if amap.mode == "range":
        span = (1 << amap.addr_bits) // amap.ports
        a, end = addr, addr + size
        while a < end:
            p = min(amap.ports - 1, a // span)
            nxt = end if p == amap.ports - 1 else min(end, (p + 1) * span)
            out[p] += nxt - a
            a = nxt
        return out
    g = amap.granularity
    first, last = addr // g, (addr + size - 1) // g
    if amap.mode == "interleave":
        n = last - first + 1
        full, rem = divmod(n, amap.ports)
        for i in range(amap.ports):
            out[i] = full * g
        for i in range(rem):
            out[(first + i) % amap.ports] += g
    else:                                        # hash: walk the stripes (exact), capped
        n = last - first + 1
        step = max(1, n // 200_000)
        for s in range(first, last + 1, step):
            out[amap.port_of(s * g)] += g * step
    # trim the partial first / last stripe
    out[amap.port_of(first * g)] -= addr - first * g
    out[amap.port_of(last * g)] -= (last + 1) * g - (addr + size)
    return out


def memmap_rows(mm, cur_gpu_config) -> list[dict]:
    ddr, l2 = AddressMap.from_hw(cur_gpu_config, "ddr"), AddressMap.from_hw(cur_gpu_config, "l2")
    rows = []
    for r in mm.regions:
        hb = port_bytes(ddr, r.base, r.size)
        lb = port_bytes(l2, r.base, r.size)
        mean = r.size / max(1, ddr.ports)
        rows.append({
            "name": r.name, "kind": r.kind, "layer": r.layer,
            "base": r.base, "end": r.end, "size": r.size,
            "base_hex": f"0x{r.base:012x}", "end_hex": f"0x{r.end:012x}",
            "offset": r.base - mm.base,
            "shape": f"{r.rows} x {r.cols} x {r.elem_bytes:g}B" if r.rows else "",
            "hbm_port_bytes": hb, "l2_slice_bytes": lb,
            "hbm_first_port": ddr.port_of(r.base), "hbm_last_port": ddr.port_of(r.end - 1),
            "hbm_imbalance": (max(hb) / mean) if mean > 0 else 1.0,
        })
    return rows


def memmap_summary(mm, cur_gpu_config, rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else memmap_rows(mm, cur_gpu_config)
    ddr, l2 = AddressMap.from_hw(cur_gpu_config, "ddr"), AddressMap.from_hw(cur_gpu_config, "l2")
    per_port = {k: [0.0] * ddr.ports for k in KINDS}
    totals = {k: 0 for k in KINDS}
    for r in rows:
        totals[r["kind"]] = totals.get(r["kind"], 0) + r["size"]
        for p, b in enumerate(r["hbm_port_bytes"]):
            per_port.setdefault(r["kind"], [0.0] * ddr.ports)[p] += b
    cap = float(cur_gpu_config.get("memory.ddr.capacity_GB") or 0) * 1e9
    return {"base": mm.base, "base_hex": f"0x{mm.base:x}", "span": mm.total_bytes,
            "end_hex": f"0x{mm.base + mm.total_bytes:x}", "capacity": cap,
            "totals": totals, "hbm_ports": ddr.ports, "l2_slices": l2.ports,
            "hbm_map": ddr.describe(), "l2_map": l2.describe(),
            "per_port_by_kind": per_port, "regions": len(rows)}


def memmap_json(mm, cur_gpu_config, max_regions: int = 4000) -> dict:
    """What the results page draws: the summary plus a compact region list."""
    rows = memmap_rows(mm, cur_gpu_config)
    out = memmap_summary(mm, cur_gpu_config, rows)
    out["rows"] = [{"name": r["name"], "kind": r["kind"], "base": r["base_hex"], "end": r["end_hex"],
                    "offset": r["offset"], "size": r["size"], "shape": r["shape"],
                    "hbm": [round(b) for b in r["hbm_port_bytes"]],
                    "imb": round(r["hbm_imbalance"], 4)} for r in rows[:max_regions]]
    out["truncated"] = max(0, len(rows) - max_regions)
    return out


def write_csv(rows: list[dict], path: Path, n_ports: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "kind", "layer", "base", "end", "size_bytes", "shape", "hbm_imbalance"] +
                   [f"hbm_port{p}_bytes" for p in range(n_ports)])
        for r in rows:
            w.writerow([r["name"], r["kind"], r["layer"], r["base_hex"], r["end_hex"], r["size"], r["shape"],
                        f"{r['hbm_imbalance']:.4f}"] + [f"{b:.0f}" for b in r["hbm_port_bytes"]])
    return path


def write_xlsx(mm, cur_gpu_config, path: Path, title: str = "", rows: list[dict] | None = None) -> Path:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Font, PatternFill

    rows = rows if rows is not None else memmap_rows(mm, cur_gpu_config)
    summ = memmap_summary(mm, cur_gpu_config, rows)
    n_ports = summ["hbm_ports"]
    wb = Workbook()
    bold = Font(bold=True)

    ws = wb.active
    ws.title = "regions"
    head = ["region", "kind", "layer", "base", "end", "size MB", "shape", "HBM imbalance",
            "first HBM port", "last HBM port"] + [f"port{p} MB" for p in range(n_ports)]
    ws.append(head)
    for c in ws[1]:
        c.font = bold
    for r in rows:
        ws.append([r["name"], r["kind"], r["layer"], r["base_hex"], r["end_hex"], r["size"] / 1e6, r["shape"],
                   round(r["hbm_imbalance"], 4), r["hbm_first_port"], r["hbm_last_port"]] +
                  [b / 1e6 for b in r["hbm_port_bytes"]])
        ws.cell(ws.max_row, 2).fill = PatternFill("solid", fgColor=KIND_COLOR.get(r["kind"], "DDDDDD"))
    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 44
    for col in "DE":
        ws.column_dimensions[col].width = 17

    # address_map: a Gantt of the address space — invisible offset + one visible series per kind
    am = wb.create_sheet("address_map")
    am.append(["region", "offset MB (from base)"] + [f"{k} MB" for k in KINDS])
    for c in am[1]:
        c.font = bold
    for r in rows:
        am.append([r["name"], r["offset"] / 1e6] + [r["size"] / 1e6 if r["kind"] == k else 0 for k in KINDS])
    n = len(rows)
    if n:
        ch = BarChart()
        ch.type, ch.grouping, ch.overlap = "bar", "stacked", 100
        ch.title = f"HBM address space from {summ['base_hex']} ({title})".strip()
        ch.y_axis.title, ch.x_axis.title = "MB from base", "region (allocation order)"
        ch.add_data(Reference(am, min_col=2, max_col=2 + len(KINDS), min_row=1, max_row=n + 1), titles_from_data=True)
        ch.set_categories(Reference(am, min_col=1, min_row=2, max_row=n + 1))
        ch.series[0].graphicalProperties.noFill = True           # the offset bar is invisible
        ch.series[0].graphicalProperties.line.noFill = True
        for s, k in zip(ch.series[1:], KINDS):
            s.graphicalProperties.solidFill = KIND_COLOR[k]
        ch.x_axis.scaling.orientation = "maxMin"                 # first allocation on top
        ch.height, ch.width = max(8, min(60, 0.35 * n)), 26
        am.add_chart(ch, "H2")

    hp = wb.create_sheet("hbm_ports")
    hp.append(["HBM port"] + [f"{k} GB" for k in KINDS] + ["total GB"])
    for c in hp[1]:
        c.font = bold
    for p in range(n_ports):
        vals = [summ["per_port_by_kind"][k][p] / 1e9 for k in KINDS]
        hp.append([f"port{p}"] + vals + [sum(vals)])
    ch2 = BarChart()
    ch2.type, ch2.grouping, ch2.overlap = "col", "stacked", 100
    ch2.title, ch2.y_axis.title = "bytes per HBM port, by tensor kind", "GB"
    ch2.add_data(Reference(hp, min_col=2, max_col=1 + len(KINDS), min_row=1, max_row=n_ports + 1),
                 titles_from_data=True)
    ch2.set_categories(Reference(hp, min_col=1, min_row=2, max_row=n_ports + 1))
    for s, k in zip(ch2.series, KINDS):
        s.graphicalProperties.solidFill = KIND_COLOR[k]
    hp.add_chart(ch2, "H2")

    sm = wb.create_sheet("summary")
    for k, v in [("title", title), ("base", summ["base_hex"]), ("end", summ["end_hex"]),
                 ("span GB", summ["span"] / 1e9), ("HBM capacity GB", summ["capacity"] / 1e9),
                 ("regions", summ["regions"]), ("HBM map", summ["hbm_map"]), ("L2 map", summ["l2_map"])] + \
                [(f"{k} GB", summ["totals"].get(k, 0) / 1e9) for k in KINDS] + \
                [("layout", "every tensor is one contiguous region, 2 MB-aligned, allocated in execution "
                            "order: weights layer by layer, then the KV cache (one region per attention "
                            "layer), then double-buffered activations and a workspace")]:
        sm.append([k, v])
        sm.cell(sm.max_row, 1).font = bold
    sm.column_dimensions["A"].width = 18
    sm.column_dimensions["B"].width = 60
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def write_all(mm, cur_gpu_config, out_dir: Path | str | None = None, stem: str = "memmap",
              title: str = "") -> dict[str, Path]:
    """CSV + Excel + a plain-text listing under `out_dir` (default the project's out/)."""
    out_dir = Path(out_dir) if out_dir else OUT_DIR / "memmap"
    rows = memmap_rows(mm, cur_gpu_config)
    txt = out_dir / f"{stem}.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt.write_text(mm.text(limit=len(mm.regions)) + "\n")
    return {"csv": write_csv(rows, out_dir / f"{stem}.csv", AddressMap.from_hw(cur_gpu_config, "ddr").ports),
            "xlsx": write_xlsx(mm, cur_gpu_config, out_dir / f"{stem}.xlsx", title, rows),
            "txt": txt}


def main(argv=None) -> None:
    import argparse

    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import HardwareSpec
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.memmap import build_memory_map
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.model_spec import ModelSpec
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig
    ap = argparse.ArgumentParser(description="Write the per-GPU HBM memory map (CSV, Excel, text) to out/")
    ap.add_argument("--model", default="kimi_k2.hf")
    ap.add_argument("--gpu-tiling-perf-hw-model", dest="gpu", default="b300")
    ap.add_argument("--phase", default="decode")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--out", default=None, help="output directory (default out/memmap)")
    a = ap.parse_args(argv)
    hw = HardwareSpec.load(a.gpu)
    mm = build_memory_map(ModelSpec.load(a.model), RunConfig(phase=a.phase, batch=a.batch, seq_len=a.seq,
                                                              tp=a.tp, dp=a.dp),
                          base=int(hw.get("memory.addressing.base")))
    stem = f"memmap_{a.model.split('.')[0]}_{a.gpu}_{a.phase}"
    paths = write_all(mm, hw, a.out, stem, f"{a.model} on {hw.name}, {a.phase}")
    print(mm.text(limit=12))
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
