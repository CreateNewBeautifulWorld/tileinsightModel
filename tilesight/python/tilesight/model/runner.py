"""Evaluate a model on a hardware spec: ops -> kernels (tile policy) -> engine -> report."""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field

from ..engine import backend
from ..hw.spec import HardwareSpec
from ..ir.kernel import Kernel, KernelResult
from ..kernels import comm
from ..kernels.attention import lower_attention_decode, lower_attention_prefill
from ..kernels.gemm import lower_elementwise, lower_gemm
from ..kernels.tiles import AttnTileConfig, TileConfig, attn_search_space, gemm_search_space
from .lower import Op, lower_model
from .memory import MemoryReport, memory_report
from .run_config import RunConfig
from .spec import ModelSpec


@dataclass
class OpResult:
    op: Op
    group: str
    repeat: int
    kernels: list[KernelResult]
    tile: str

    exposed: float = 1.0           # <1 for collectives partly hidden by compute (rc.comm_overlap)

    @property
    def time_s(self) -> float:
        return sum(k.time_s for k in self.kernels) * self.exposed

    @property
    def total_s(self) -> float:
        return self.time_s * self.repeat

    @property
    def bottleneck_detail(self) -> str:
        """Finest attribution, e.g. 'ddr:load:expert_weight' or 'latency:softmax(sfu)'."""
        d: dict[str, float] = {}
        for k in self.kernels:
            for n, t in k.limiter_detail.items():
                d[n] = d.get(n, 0.0) + t
        return max(d.items(), key=lambda kv: kv[1])[0] if d else "-"

    @property
    def occupancy(self) -> str:
        """'resident/limiter' of the main kernel, e.g. '1/smem', '2/regs', '4/tmem'."""
        m = self.kernels[0].meta if self.kernels else {}
        if "resident" not in m:
            return "-"
        s = f"{m['resident']}/{m.get('occ_limiter', '?')}"
        if m.get("reg_spill_bytes"):
            s += "+spill"
        return s

    @property
    def bottleneck(self) -> str:
        lim: dict[str, float] = {}
        for k in self.kernels:
            for n, t in k.limiter_time.items():
                lim[n] = lim.get(n, 0.0) + t
        return max(lim.items(), key=lambda kv: kv[1])[0] if lim else "-"


def _apply_overlap(results: list["OpResult"], rc: RunConfig) -> None:
    """Hide collectives behind compute, per layer group (docs/DESIGN.md §5.4).

    none      : every collective is exposed (the safe default)
    stream    : a collective can hide behind the compute that FOLLOWS it in the same layer —
                what a separate comm stream buys you without restructuring the model
    two_batch : two micro-batches in flight, so a collective can hide behind the whole layer's
                compute (DeepEP / two-batch overlap); the classic MoE serving trick
    manual    : scale every collective by (1 - rc.comm_overlap)
    """
    if rc.overlap_mode == "none":
        return
    if rc.overlap_mode == "manual":
        for o in results:
            if o.op.kind in ("allreduce", "a2a"):
                o.exposed = 1.0 - rc.comm_overlap
        return
    eff = max(0.0, min(1.0, rc.overlap_efficiency))
    by_group: dict[str, list[OpResult]] = {}
    for o in results:
        by_group.setdefault(o.group, []).append(o)
    for ops in by_group.values():
        is_comm = [o.op.kind in ("allreduce", "a2a") for o in ops]
        if not any(is_comm):
            continue
        comm_total = sum(o.time_s for o, c in zip(ops, is_comm) if c)
        if comm_total <= 0:
            continue
        if rc.overlap_mode == "two_batch":
            hideable = sum(o.time_s for o, c in zip(ops, is_comm) if not c)
        else:                                   # stream: only the compute after the first comm
            first = is_comm.index(True)
            hideable = sum(o.time_s for o, c in zip(ops[first:], is_comm[first:]) if not c)
        exposed = max(0.0, comm_total - hideable * eff)
        scale = exposed / comm_total
        for o, c in zip(ops, is_comm):
            if c:
                o.exposed = scale


@dataclass
class ModelReport:
    model: str
    gpu_name: str
    rc: RunConfig
    ops: list[OpResult]
    memory: MemoryReport
    backend: str = backend.BACKEND
    notes: list[str] = field(default_factory=list)

    @property
    def step_time_s(self) -> float:
        return sum(o.total_s for o in self.ops)

    def limiter_breakdown(self) -> dict[str, float]:
        """Seconds per limiter over the whole step (tc, ddr, l2, sfu, latency, launch, net ...)."""
        out: dict[str, float] = {}
        for o in self.ops:
            for k in o.kernels:
                for n, t in k.limiter_time.items():
                    out[n] = out.get(n, 0.0) + t * o.repeat * o.exposed
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def detail_breakdown(self) -> dict[str, float]:
        """Seconds per fine-grained limiter: resource + tensor/action (ddr:load:kv_cache(K) ...)."""
        out: dict[str, float] = {}
        for o in self.ops:
            for k in o.kernels:
                for n, t in k.limiter_detail.items():
                    out[n] = out.get(n, 0.0) + t * o.repeat * o.exposed
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def by_domain(self) -> dict[str, float]:
        """Time attributed to each of the model's three blocks (shader slice / gmem / memory)."""
        from ..report.table import domain_of
        out: dict[str, float] = {}
        for k, v in self.detail_breakdown().items():
            d = domain_of(k)
            out[d] = out.get(d, 0.0) + v
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def activity_by_domain(self) -> dict[str, float]:
        """How busy each block is, whether or not it is the bottleneck.

        `by_domain()` splits the *critical* time, so a fast unit that never limits shows 0
        there. This one sums how long every lane is busy, which is what you want when asking
        "is the on-chip buffer actually doing work".
        """
        from ..hw.spec import lane_domain
        out: dict[str, float] = {}
        for o in self.ops:
            for k in o.kernels:
                for lane, u in k.util.items():
                    d = lane_domain(lane)
                    out[d] = out.get(d, 0.0) + u * k.time_s * o.repeat * o.exposed
        tot = self.step_time_s or 1.0
        return dict(sorted(((k, v / tot) for k, v in out.items()), key=lambda kv: -kv[1]))

    def by_category(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for o in self.ops:
            cat = o.op.name.split(".")[-1]
            out[cat] = out.get(cat, 0.0) + o.total_s
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def tokens_per_s_per_gpu(self) -> float:
        toks = self.rc.batch if self.rc.phase == "decode" else self.rc.batch * self.rc.seq_len
        return toks / self.step_time_s / self.rc.world


def _tile_policy(cur_gpu_config: HardwareSpec, key: str) -> str | dict:
    """compute.tile_policy.gemm/attn: "auto", a JSON string, or (already) a dict/'auto'."""
    v = cur_gpu_config.get(f"compute.tile_policy.{key}")
    return json.loads(v) if isinstance(v, str) and v != "auto" else v


def _override(cur_gpu_config: HardwareSpec, name: str) -> dict | None:
    for pat, fields in (cur_gpu_config.get("compute.tile_policy.overrides") or {}).items():
        if fnmatch.fnmatch(name, pat):
            return fields
    return None


def _eval_all(kernels: list[Kernel], cur_gpu_config) -> list[KernelResult]:
    return [backend.evaluate(k, cur_gpu_config) for k in kernels]


def _best(cands, cur_gpu_config):
    best = None
    for tile, ks in cands:
        if ks is None:
            continue
        res = _eval_all(ks, cur_gpu_config)
        t = sum(r.time_s for r in res)
        if best is None or t < best[0]:
            best = (t, tile, res)
    return best


def resolve_op(op: Op, cur_gpu_config: HardwareSpec, rc: RunConfig) -> tuple[list[KernelResult], str]:
    p = op.p
    if op.kind == "gemm":
        ov = _override(cur_gpu_config, op.name)
        gemm_tile = _tile_policy(cur_gpu_config, "gemm")
        if ov is not None:
            tiles = [TileConfig(**ov)]
        elif gemm_tile == "auto":
            tiles = gemm_search_space(p["M"], p["N"], p["K"])
        else:
            tiles = [TileConfig(**gemm_tile)]
        cands = ((t, lower_gemm(cur_gpu_config, op.name, p["M"], p["N"], p["K"], batch=p["batch"], a_dtype=p["a_dtype"],
                                b_dtype=p["b_dtype"], c_dtype=p["c_dtype"], compute_dtype=p["compute_dtype"],
                                tile=t, names=tuple(p.get("names", ("act", "weight", "out"))))) for t in tiles)
        best = _best(cands, cur_gpu_config)
        if best is None:
            raise ValueError(f"{op.name}: no legal tile among {len(tiles)} candidates")
        return best[2], best[2][0].meta.get("tile", best[1].short())

    if op.kind in ("attn_decode", "attn_prefill"):
        ov = _override(cur_gpu_config, op.name)
        attn_tile = _tile_policy(cur_gpu_config, "attn")
        if ov is not None:
            tiles = [AttnTileConfig(**ov)]
        elif cur_gpu_config.get("compute.attention_tile_m") and cur_gpu_config.get("compute.attention_tile_n"):
            # the part fixes the attention tile: search only the pipeline knobs around it
            bm, bn = int(cur_gpu_config.get("compute.attention_tile_m")), int(cur_gpu_config.get("compute.attention_tile_n"))
            tiles = [AttnTileConfig(block_m=bm, block_n=bn, stages=st, consumers=c)
                     for st in (2, 3) for c in (1, 2)]
        elif attn_tile == "auto":
            tiles = attn_search_space()
        else:
            tiles = [AttnTileConfig(**attn_tile)]
        kw = dict(B=p["B"], H=p["H"], kv_heads=p["kv_heads"], S=p["S"], d_qk=p["d_qk"], d_v=p["d_v"],
                  kv_dtype=rc.kv_dtype, compute_dtype=rc.attn_compute_dtype)
        if op.kind == "attn_decode":
            cands = ((t, lower_attention_decode(cur_gpu_config, op.name, v_in_k=p["v_in_k"], tile=t, **kw)) for t in tiles)
        else:
            cands = ((t, lower_attention_prefill(cur_gpu_config, op.name, tile=t, causal=p.get("causal", True),
                                                 window=p.get("window", 0), **kw)) for t in tiles)
        best = _best(cands, cur_gpu_config)
        if best is None:
            raise ValueError(f"{op.name}: no legal attention tile")
        return best[2], best[2][0].meta.get("tile", best[1].short())

    if op.kind == "elementwise":
        q = dict(p)
        if "names" in q:
            q["names"] = tuple(q["names"])
        return _eval_all(lower_elementwise(cur_gpu_config, op.name, **q), cur_gpu_config), "-"
    if op.kind == "allreduce":
        return _eval_all(comm.allreduce(cur_gpu_config, op.name, p["bytes"], p["group"]), cur_gpu_config), "-"
    if op.kind == "a2a":
        return _eval_all(comm.all_to_all(cur_gpu_config, op.name, p["bytes"], p["group"]), cur_gpu_config), "-"
    raise ValueError(op.kind)


def _install_sram_footprints(model: ModelSpec, rc: RunConfig, cur_gpu_config: HardwareSpec, groups) -> HardwareSpec:
    """Tell the on-chip buffer how big each tensor category is on this GPU (see resident_frac)."""
    if cur_gpu_config.sram is None:
        return cur_gpu_config
    from .memory import kv_bytes_per_seq_all
    weights = sum(op.weight_bytes * rep for _, rep, ops in groups for op in ops)
    kv = kv_bytes_per_seq_all(model, rc, rc.seq_len) * rc.seqs_per_rank
    act = max((p["M"] * p["N"] * p.get("batch", 1) * 2 if op.kind == "gemm" else
               p.get("bytes_out", 0.0) for _, _, ops in groups for op in ops for p in [op.p]),
              default=0.0)
    return cur_gpu_config.override({"memory.sram.footprint": {"weight": weights, "kv": kv, "act": act}})


@dataclass
class CurModelConfig:
    """The model's only 'what to run' input: architecture (ModelSpec) + how to run it
    (RunConfig), bundled into one object. Built at the boundary (CLI flags / web request) from
    either a full multi-layer model (`--model`) or a one-layer `workload` dict
    (model/workload.py's `to_model_spec()`/`to_run_config()`) — the model layer itself never
    sees "workload", only this."""
    spec: ModelSpec
    run: RunConfig


def run(cur_gpu_config: HardwareSpec, cur_model_config: CurModelConfig, progress=None) -> ModelReport:
    """The model's only entry point: two runtime-built configs, nothing else.

    progress(done, total, label) is called per op so UIs can show a processing state."""
    return run_model(cur_model_config.spec, cur_gpu_config, cur_model_config.run, progress=progress)


def run_model(model: ModelSpec, cur_gpu_config: HardwareSpec, rc: RunConfig, progress=None) -> ModelReport:
    """Internal engine entry point (model, cur_gpu_config, rc positional) — kernels/lowering/report code and
    existing scripts use this directly. `run(cur_gpu_config, cur_model_config)` above is the
    boundary-facing wrapper; prefer it in new CLI/server/UI code.

    progress(done, total, label) is called per op so UIs can show a processing state."""
    groups = lower_model(model, rc)
    cur_gpu_config = _install_sram_footprints(model, rc, cur_gpu_config, groups)
    results = []
    total = sum(len(ops) for _, _, ops in groups)
    done = 0
    for gname, rep, ops in groups:
        for op in ops:
            if progress:
                progress(done, total, op.name)
            done += 1
            ks, tile = resolve_op(op, cur_gpu_config, rc)
            results.append(OpResult(op, gname, rep, ks, tile, 1.0))
    _apply_overlap(results, rc)
    mem = memory_report(model, rc, groups, cur_gpu_config)
    rep = ModelReport(model.name, cur_gpu_config.name, rc, results, mem)
    if not mem.fits:
        rep.notes.append(f"DOES NOT FIT: needs {mem.total_GB:.1f} GB > {mem.capacity_GB:.1f} GB per GPU")
    return rep
