#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "../common/cache/address_map.hpp"
#include "../common/cache/cache_sim.hpp"
#include "gpu_top.hpp"

namespace nb = nanobind;
using namespace tilesight;
using namespace tilesight::gpu_top;

NB_MODULE(_core, m) {
  m.doc() = "TileSight C++ core: gpu_top (lowering + tiling search) over shader_core/"
            "shader_slice/on_chip_buffer/memory_slice/common-cache.";

  // TraceAction/TraceKernel are plain reporting records (see shader_core.hpp's doc comment).
  // Their constructors here exist for two callers only: gpu_top's own per-op-kind lowering
  // (which builds them internally, in C++) and, for testing, exercising shader_core's
  // wave/round/critical-path engine directly against a hand-built action DAG without going
  // through a real op's lowering — never a generic "build any op" surface the model itself uses.
  nb::class_<shader_core::TraceAction>(m, "TraceAction")
      .def("__init__",
          [](shader_core::TraceAction* a, std::string name, std::map<std::string, double> work,
             std::vector<int> deps, bool recurrent, double latency_s) {
            new (a) shader_core::TraceAction{std::move(name), std::move(work), std::move(deps), recurrent,
                                             latency_s};
          },
          nb::arg("name"), nb::arg("work") = std::map<std::string, double>{},
          nb::arg("deps") = std::vector<int>{}, nb::arg("recurrent") = false, nb::arg("latency_s") = 0.0)
      .def_ro("name", &shader_core::TraceAction::name)
      .def_ro("work", &shader_core::TraceAction::work)
      .def_ro("deps", &shader_core::TraceAction::deps)
      .def_ro("recurrent", &shader_core::TraceAction::recurrent)
      .def_ro("latency_s", &shader_core::TraceAction::latency);

  nb::class_<shader_core::TraceKernel>(m, "TraceKernel")
      .def("__init__",
          [](shader_core::TraceKernel* k, std::string name, std::string kind, int64_t num_blocks,
             int64_t iters, int stages, int resident, std::vector<shader_core::TraceAction> body,
             std::vector<shader_core::TraceAction> prologue, std::vector<shader_core::TraceAction> epilogue,
             int consumers) {
            new (k) shader_core::TraceKernel{std::move(name),
                                             std::move(kind),
                                             num_blocks,
                                             iters,
                                             stages,
                                             resident,
                                             consumers,
                                             std::move(body),
                                             std::move(prologue),
                                             std::move(epilogue),
                                             false,
                                             0.0,
                                             0.0,
                                             3.0};
          },
          nb::arg("name"), nb::arg("kind"), nb::arg("num_blocks"), nb::arg("iters"), nb::arg("stages"),
          nb::arg("resident"), nb::arg("body"), nb::arg("prologue") = std::vector<shader_core::TraceAction>{},
          nb::arg("epilogue") = std::vector<shader_core::TraceAction>{}, nb::arg("consumers") = 1)
      .def_ro("name", &shader_core::TraceKernel::name)
      .def_ro("kind", &shader_core::TraceKernel::kind)
      .def_ro("num_blocks", &shader_core::TraceKernel::num_blocks)
      .def_ro("iters", &shader_core::TraceKernel::iters)
      .def_ro("stages", &shader_core::TraceKernel::stages)
      .def_ro("resident", &shader_core::TraceKernel::resident)
      .def_ro("consumers", &shader_core::TraceKernel::consumers)
      .def_ro("body", &shader_core::TraceKernel::body)
      .def_ro("prologue", &shader_core::TraceKernel::prologue)
      .def_ro("epilogue", &shader_core::TraceKernel::epilogue)
      .def_ro("fixed_time_s", &shader_core::TraceKernel::fixed_time);

  nb::class_<LoweredKernel>(m, "LoweredKernel")
      .def("__init__",
          [](LoweredKernel* lk, shader_core::TraceKernel trace, nb::dict meta) {
            new (lk) LoweredKernel{std::move(trace), std::move(meta)};
          },
          nb::arg("trace"), nb::arg("meta") = nb::dict())
      .def_ro("trace", &LoweredKernel::trace)
      .def_ro("meta", &LoweredKernel::meta)
      .def_prop_ro("name", [](const LoweredKernel& k) { return k.trace.name; })
      .def_prop_ro("kind", [](const LoweredKernel& k) { return k.trace.kind; });

  nb::class_<KernelResult>(m, "KernelResult")
      .def_ro("name", &KernelResult::name)
      .def_ro("kind", &KernelResult::kind)
      .def_ro("time_s", &KernelResult::time_s)
      .def_ro("bottleneck", &KernelResult::bottleneck)
      .def_ro("util", &KernelResult::util)
      .def_ro("breakdown", &KernelResult::breakdown)
      .def_ro("limiter_time", &KernelResult::limiter_time)
      .def_ro("limiter_detail", &KernelResult::limiter_detail)
      .def_ro("waves", &KernelResult::waves)
      .def_ro("meta", &KernelResult::meta);

  nb::class_<GemmTile>(m, "GemmTile")
      .def("__init__", [](GemmTile* t, int bm, int bn, int bk, int stages, int swizzle, int split_k,
                          std::string load_path, int cluster_m, bool cta_pair) {
        new (t) GemmTile{bm, bn, bk, stages, swizzle, split_k, std::move(load_path), cluster_m, cta_pair};
      }, nb::arg("bm") = 128, nb::arg("bn") = 128, nb::arg("bk") = 64, nb::arg("stages") = 0,
         nb::arg("swizzle") = 8, nb::arg("split_k") = 1, nb::arg("load_path") = "tma",
         nb::arg("cluster_m") = 1, nb::arg("cta_pair") = false)
      .def_rw("bm", &GemmTile::bm).def_rw("bn", &GemmTile::bn).def_rw("bk", &GemmTile::bk)
      .def_rw("stages", &GemmTile::stages).def_rw("swizzle", &GemmTile::swizzle)
      .def_rw("split_k", &GemmTile::split_k).def_rw("load_path", &GemmTile::load_path)
      .def_rw("cluster_m", &GemmTile::cluster_m).def_rw("cta_pair", &GemmTile::cta_pair);

  nb::class_<AttnTile>(m, "AttnTile")
      .def("__init__", [](AttnTile* t, int block_m, int block_n, int stages, int num_splits, int consumers) {
        new (t) AttnTile{block_m, block_n, stages, num_splits, consumers};
      }, nb::arg("block_m") = 64, nb::arg("block_n") = 64, nb::arg("stages") = 2,
         nb::arg("num_splits") = 0, nb::arg("consumers") = 2)
      .def_rw("block_m", &AttnTile::block_m).def_rw("block_n", &AttnTile::block_n)
      .def_rw("stages", &AttnTile::stages).def_rw("num_splits", &AttnTile::num_splits)
      .def_rw("consumers", &AttnTile::consumers);

  m.def("lower_gemm", [](nb::object cfg, std::string name, int64_t M, int64_t N, int64_t K, int64_t batch,
                         std::string a_dtype, std::string b_dtype, std::string c_dtype,
                         std::string compute_dtype, const GemmTile& tile, std::string a_name,
                         std::string b_name, std::string c_name) {
    HwView hw(cfg);
    return lower_gemm(hw, name, M, N, K, batch, a_dtype, b_dtype, c_dtype, compute_dtype, tile,
                      a_name, b_name, c_name);
  }, nb::arg("cur_gpu_config"), nb::arg("name"), nb::arg("M"), nb::arg("N"), nb::arg("K"),
     nb::arg("batch") = 1, nb::arg("a_dtype") = "bf16", nb::arg("b_dtype") = "bf16",
     nb::arg("c_dtype") = "bf16", nb::arg("compute_dtype") = "bf16", nb::arg("tile") = GemmTile{},
     nb::arg("a_name") = "act", nb::arg("b_name") = "weight", nb::arg("c_name") = "out");

  m.def("lower_attention_decode", [](nb::object cfg, std::string name, int64_t B, int64_t H,
                                     int64_t kv_heads, int64_t S, int64_t d_qk, int64_t d_v,
                                     std::string kv_dtype, std::string compute_dtype, bool v_in_k,
                                     const AttnTile& tile) {
    HwView hw(cfg);
    return lower_attention_decode(hw, name, B, H, kv_heads, S, d_qk, d_v, kv_dtype, compute_dtype,
                                  v_in_k, tile);
  }, nb::arg("cur_gpu_config"), nb::arg("name"), nb::arg("B"), nb::arg("H"), nb::arg("kv_heads"),
     nb::arg("S"), nb::arg("d_qk"), nb::arg("d_v"), nb::arg("kv_dtype") = "bf16",
     nb::arg("compute_dtype") = "bf16", nb::arg("v_in_k") = false, nb::arg("tile") = AttnTile{});

  m.def("lower_attention_prefill", [](nb::object cfg, std::string name, int64_t B, int64_t H,
                                      int64_t kv_heads, int64_t S, int64_t d_qk, int64_t d_v,
                                      std::string kv_dtype, std::string compute_dtype, bool causal,
                                      int64_t window, const AttnTile& tile) {
    HwView hw(cfg);
    return lower_attention_prefill(hw, name, B, H, kv_heads, S, d_qk, d_v, kv_dtype, compute_dtype,
                                   causal, window, tile);
  }, nb::arg("cur_gpu_config"), nb::arg("name"), nb::arg("B"), nb::arg("H"), nb::arg("kv_heads"),
     nb::arg("S"), nb::arg("d_qk"), nb::arg("d_v"), nb::arg("kv_dtype") = "bf16",
     nb::arg("compute_dtype") = "bf16", nb::arg("causal") = true, nb::arg("window") = 0,
     nb::arg("tile") = AttnTile{});

  m.def("lower_elementwise", [](nb::object cfg, std::string name, double bytes_in, double bytes_out,
                                double flops, double sfu_ops, std::string kind, int64_t chunk,
                                std::string in_name, std::string out_name, std::optional<bool> on_chip) {
    HwView hw(cfg);
    return lower_elementwise(hw, name, bytes_in, bytes_out, flops, sfu_ops, kind, chunk, in_name, out_name,
                             on_chip);
  }, nb::arg("cur_gpu_config"), nb::arg("name"), nb::arg("bytes_in") = 0.0, nb::arg("bytes_out") = 0.0,
     nb::arg("flops") = 0.0, nb::arg("sfu_ops") = 0.0, nb::arg("kind") = "elementwise",
     nb::arg("chunk") = 32 * 1024, nb::arg("in_name") = "act", nb::arg("out_name") = "act",
     nb::arg("on_chip") = nb::none());

  m.def("allreduce", [](nb::object cfg, std::string name, double nbytes, int group, std::string algo) {
    return allreduce(HwView(cfg), name, nbytes, group, algo);
  }, nb::arg("cur_gpu_config"), nb::arg("name"), nb::arg("bytes"), nb::arg("group"), nb::arg("algo") = "auto");

  m.def("all_to_all", [](nb::object cfg, std::string name, double send_bytes, int group) {
    return all_to_all(HwView(cfg), name, send_bytes, group);
  }, nb::arg("cur_gpu_config"), nb::arg("name"), nb::arg("bytes"), nb::arg("group"));

  m.def("gemm_search_space", &gemm_search_space, nb::arg("M"), nb::arg("N"), nb::arg("K"));
  m.def("attn_search_space", &attn_search_space);

  m.def("occupancy", [](nb::object cfg, int64_t smem_bytes, int64_t acc_bytes, int threads, int extra_regs) {
    return occupancy(HwView(cfg), smem_bytes, acc_bytes, threads, extra_regs);
  }, nb::arg("cur_gpu_config"), nb::arg("smem_bytes"), nb::arg("acc_bytes"), nb::arg("threads") = 256,
     nb::arg("extra_regs") = 0);

  m.def("evaluate", [](nb::object cfg, const LoweredKernel& k) { return evaluate(HwView(cfg), k); },
        nb::arg("cur_gpu_config"), nb::arg("kernel"));

  m.def("evaluate_all", [](nb::object cfg, const std::vector<LoweredKernel>& ks) {
    HwView hw(cfg);
    std::vector<KernelResult> out;
    out.reserve(ks.size());
    for (const auto& k : ks) out.push_back(evaluate(hw, k));
    return out;
  }, nb::arg("cur_gpu_config"), nb::arg("kernels"));

  // Deterministic tile-level cache simulation (common/cache), exposed for genResult/addressing.py
  // and standalone cache-model inspection.
  m.def("cache_simulate", [](std::vector<int64_t> keys, std::vector<int64_t> addrs, std::vector<double> sizes,
                             std::vector<int> streams, int n_streams, double capacity_bytes,
                             int n_partitions, int ports, std::string mode, int64_t granularity,
                             int addr_bits, std::string policy) {
    cache::AddrCfg cfg{ports, mode, granularity, addr_bits};
    std::vector<int> part_of(addrs.size());
    for (size_t i = 0; i < addrs.size(); ++i) part_of[i] = static_cast<int>(cache::port_of(cfg, addrs[i]));
    auto r = cache::simulate(keys, addrs, sizes, streams, n_streams, capacity_bytes, n_partitions, part_of,
                             policy);
    nb::dict out;
    out["misses"] = r.misses;
    out["accesses"] = r.accesses;
    out["miss_fraction"] = r.miss_fraction();
    out["hit_rate"] = r.hit_rate;
    out["partition_tiles"] = r.partition_tiles;
    out["partition_bytes"] = r.partition_bytes;
    out["partition_capacity"] = r.partition_capacity;
    out["partition_evictions"] = r.partition_evictions;
    return out;
  }, nb::arg("keys"), nb::arg("addrs"), nb::arg("sizes"), nb::arg("streams"), nb::arg("n_streams"),
     nb::arg("capacity_bytes"), nb::arg("n_partitions"), nb::arg("ports"), nb::arg("mode") = "interleave",
     nb::arg("granularity") = 1024, nb::arg("addr_bits") = 48, nb::arg("policy") = "lru");
}
