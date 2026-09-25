"""A picture of the configured GPU, drawn from the config alone.

Everything in the diagram comes from the hardware YAML — SM count, L1/SMEM/TMEM per SM, L2
partitions and ports, the optional on-chip buffer, DMA engines, HBM ports and the scale-up
link — so it doubles as a sanity check that the config says what you think it says.
"""
from __future__ import annotations

import html

from ..gpuTilingHWModel.spec import HardwareSpec

C_SM, C_L1, C_L2, C_BUF, C_DMA, C_HBM = "#b4552d", "#5b9279", "#4f7cac", "#7f9a52", "#9a8f7a", "#b3563a"


def _box(x, y, w, h, fill, label, sub="", rx=6, fs=11):
    t = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" opacity="0.18" '
         f'stroke="{fill}" stroke-width="1.2"/>'
         f'<text x="{x + w / 2}" y="{y + (h / 2 if not sub else h / 2 - 5)}" font-size="{fs}" '
         f'text-anchor="middle" dominant-baseline="middle" fill="currentColor">{html.escape(label)}</text>')
    if sub:
        t += (f'<text x="{x + w / 2}" y="{y + h / 2 + 10}" font-size="9" text-anchor="middle" '
              f'fill="currentColor" opacity=".7">{html.escape(sub)}</text>')
    return t


def arch_svg(cur_gpu_config: HardwareSpec, width: int = 860) -> str:
    g = []
    sms = cur_gpu_config.sms
    l1 = cur_gpu_config.get("memory.l1") or {}
    smem_kb = cur_gpu_config.get("memory.onchip.smem.capacity_KB")
    tmem_kb = cur_gpu_config.get("memory.onchip.tmem.capacity_KB") or 0
    parts = int(cur_gpu_config.get("memory.l2.partitions") or 1)
    blocks = int(cur_gpu_config.get("memory.l2.blocks") or 0)
    ports = int(cur_gpu_config.get("memory.ddr.ports") or 1)
    engines = int(cur_gpu_config.get("memory.dma.engines") or 1)
    sram = cur_gpu_config.sram
    cluster = int(l1.get("cluster_size", 1)) if str(l1.get("owner", "sm")) == "cluster" else 1

    y = 44
    # --- SM row (draw a handful, label the real count)
    shown = 6
    bw, gap = 112, 12
    x0 = (width - (shown * bw + (shown - 1) * gap)) / 2
    for i in range(shown):
        x = x0 + i * (bw + gap)
        g.append(_box(x, y, bw, 78, C_SM, f"SM {i}" if i < shown - 1 else f"… x{sms}",
                      "", rx=8))
        g.append(_box(x + 8, y + 26, bw - 16, 20, C_L1, f"SMEM {smem_kb:.0f} KB", "", rx=4, fs=9))
        if tmem_kb:
            g.append(_box(x + 8, y + 50, bw - 16, 20, C_L1, f"TMEM {tmem_kb:.0f} KB", "", rx=4, fs=9))
        elif l1.get("capacity_KB"):
            g.append(_box(x + 8, y + 50, bw - 16, 20, C_L1,
                          f"L1 {cur_gpu_config.l1_capacity_bytes / 1024:.0f} KB", "", rx=4, fs=9))
    g.append(f'<text x="{width / 2}" y="{y - 12}" font-size="12" text-anchor="middle" '
             f'fill="currentColor" opacity=".75">{sms} SMs @ {cur_gpu_config.clock_hz / 1e9:.2f} GHz'
             + (f" · clusters of {cluster} share L1 via DSMEM" if cluster > 1 else "") + '</text>')

    # --- load paths
    y2 = y + 96
    paths = list((cur_gpu_config.get("load_paths") or {}).keys())
    pw = min(170, (width - 60) / max(1, len(paths)) - 10)
    for i, p in enumerate(paths):
        px = 30 + i * (pw + 10)
        kind = "DMA engine" if cur_gpu_config.path_is_dma(p) else "vector loads"
        g.append(_box(px, y2, pw, 34, C_DMA, f"{p}", f"{kind} · {cur_gpu_config.path_attr(p, 'per_sm_GBps', 0)} GB/s/SM"))
    g.append(f'<text x="{width - 30}" y="{y2 + 20}" font-size="10" text-anchor="end" fill="currentColor" '
             f'opacity=".7">{engines} DMA engine(s), {"per L2 block" if cur_gpu_config.get("memory.dma.per_l2_block") else "shared"}'
             f' → {cur_gpu_config.dma_destination}</text>')

    # --- optional on-chip buffer
    y3 = y2 + 52
    if sram:
        sw = sram.get("switch_TBps") or sram.get("bandwidth_TBps", 0)
        g.append(_box(30, y3, width - 60, 26, C_DMA, f"switch  {sw} TB/s aggregate", "", rx=4, fs=10))
        y3 += 32
        pieces = min(6, max(1, int(cur_gpu_config.sms // max(1, int((cur_gpu_config.get("memory.l1") or {}).get("cluster_size", 1))))))
        pw2 = (width - 60 - (pieces - 1) * 8) / pieces
        for i in range(pieces):
            g.append(_box(30 + i * (pw2 + 8), y3, pw2, 30, C_BUF,
                          f"buffer {i}" if i < pieces - 1 else "…", "", rx=4, fs=9))
        g.append(f'<text x="{width / 2}" y="{y3 + 44}" font-size="10" text-anchor="middle" '
                 f'fill="currentColor" opacity=".7">on-chip buffer '
                 f'{sram.get("capacity_MB", 0):.0f} MB total, one piece next to each shader slice, '
                 f'{sram.get("bandwidth_TBps", 0)} TB/s · policy {sram.get("policy", "cache")}</text>')
        y3 += 58

    # --- L2 partitions
    pwid = (width - 60 - (parts - 1) * 10) / max(1, parts)
    for i in range(parts):
        g.append(_box(30 + i * (pwid + 10), y3, pwid, 46, C_L2, f"L2 partition {i}",
                      f"{cur_gpu_config.l2_capacity_bytes / parts / 1024 / 1024:.0f} MiB"))
    sub = f"L2 {cur_gpu_config.get('memory.l2.capacity_MB'):.0f} MB · {cur_gpu_config.get('memory.l2.bandwidth_TBps')} TB/s"
    if blocks:
        sub += f" · {blocks} block(s) x {cur_gpu_config.get('memory.l2.ports_per_block')} port(s)"
    sub += f" · {cur_gpu_config.get('memory.addressing.l2', {}).get('mode', 'interleave')} @ " \
           f"{cur_gpu_config.get('memory.addressing.l2', {}).get('granularity_KB', 1)} KB"
    g.append(f'<text x="{width / 2}" y="{y3 + 62}" font-size="10" text-anchor="middle" '
             f'fill="currentColor" opacity=".7">{html.escape(sub)}</text>')

    # --- HBM ports
    y4 = y3 + 78
    hw_w = (width - 60 - (ports - 1) * 8) / max(1, ports)
    for i in range(ports):
        g.append(_box(30 + i * (hw_w + 8), y4, hw_w, 34, C_HBM, f"port {i}", "", rx=4, fs=9))
    g.append(f'<text x="{width / 2}" y="{y4 + 50}" font-size="11" text-anchor="middle" fill="currentColor">'
             f'HBM {cur_gpu_config.get("memory.ddr.capacity_GB"):.0f} GB · {cur_gpu_config.get("memory.ddr.bandwidth_TBps")} TB/s · '
             f'{ports} ports · {cur_gpu_config.get("memory.ddr.latency_ns")} ns</text>')

    # --- scale-up link
    y5 = y4 + 64
    nv = cur_gpu_config.get("network.nvlink") or {}
    g.append(_box(width / 2 - 160, y5, 320, 30, C_DMA,
                  f"scale-up {nv.get('bandwidth_GBps', 0):.0f} GB/s",
                  f"domain {nv.get('domain_size', 1)} GPUs · alpha {nv.get('alpha_us', 0)} us"))
    h = y5 + 54
    return (f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">'
            f'<text x="{width / 2}" y="20" font-size="13" text-anchor="middle" fill="currentColor">'
            f'{html.escape(cur_gpu_config.name)} — as configured</text>' + "".join(g) + "</svg>")
