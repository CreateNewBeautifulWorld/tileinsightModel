// gpu_top — the orchestrator: turns one op's typed shape/dtype/tile arguments into a tile
// execution plan (via shader_slice/on_chip_buffer/memory_slice/common-cache) and evaluates it
// (via shader_core). Port of engine/reference.py (the scheduler) + kernels/{gemm,attention,
// tiles,comm}.py (per-op-kind tiling/lowering) + the tile-search/lowering half of what
// lower.py/attention_blocks.py did (the shape-only "block -> op" half of those two files stays
// Python, in interfaceAndRun/, since it never touches HardwareSpec — see CLAUDE.md).
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "../shader_core/shader_core.hpp"
#include "hw_view.hpp"

namespace nb = nanobind;

namespace tilesight::gpu_top {

using shader_core::TraceAction;
using shader_core::TraceKernel;

struct GemmTile {
  int bm = 128, bn = 128, bk = 64;
  int stages = 0;
  int swizzle = 8;
  int split_k = 1;
  std::string load_path = "tma";
  int cluster_m = 1;
  bool cta_pair = false;
};

struct AttnTile {
  int block_m = 64, block_n = 64;
  int stages = 2;
  int num_splits = 0;
  int consumers = 2;
};

// A lowered kernel: the trace (fed to shader_core::evaluate_kernel) plus its reporting metadata
// (tile string, occupancy, cache stats, ...) — a plain record, not a generic op-dispatch type.
struct LoweredKernel {
  TraceKernel trace;
  nb::dict meta;
};

struct KernelResult : shader_core::KernelResult {
  nb::dict meta;
};

// ---- occupancy / register estimation (kernels/gemm.py) ---------------------------------
std::pair<int, std::string> occupancy(const HwView& hw, int64_t smem_bytes, int64_t acc_bytes, int threads,
                                      int extra_regs = 0);
std::pair<int, int64_t> reg_estimate(const HwView& hw, int64_t acc_bytes, int threads);

// ---- lowering: shape + tile -> a tile execution plan ------------------------------------
// Returns std::nullopt for an illegal tile (occupancy doesn't fit, cluster unsupported, ...),
// an empty vector for a zero-sized op.
std::optional<std::vector<LoweredKernel>> lower_gemm(const HwView& hw, const std::string& name, int64_t M,
                                                      int64_t N, int64_t K, int64_t batch,
                                                      const std::string& a_dtype, const std::string& b_dtype,
                                                      const std::string& c_dtype,
                                                      const std::string& compute_dtype, const GemmTile& tile,
                                                      const std::string& a_name, const std::string& b_name,
                                                      const std::string& c_name);

std::optional<std::vector<LoweredKernel>> lower_attention_decode(
    const HwView& hw, const std::string& name, int64_t B, int64_t H, int64_t kv_heads, int64_t S,
    int64_t d_qk, int64_t d_v, const std::string& kv_dtype, const std::string& compute_dtype, bool v_in_k,
    const AttnTile& tile);

std::optional<std::vector<LoweredKernel>> lower_attention_prefill(
    const HwView& hw, const std::string& name, int64_t B, int64_t H, int64_t kv_heads, int64_t S,
    int64_t d_qk, int64_t d_v, const std::string& kv_dtype, const std::string& compute_dtype, bool causal,
    int64_t window, const AttnTile& tile);

std::vector<LoweredKernel> lower_elementwise(const HwView& hw, const std::string& name, double bytes_in,
                                             double bytes_out, double flops, double sfu_ops,
                                             const std::string& kind, int64_t chunk,
                                             const std::string& in_name, const std::string& out_name,
                                             std::optional<bool> on_chip);

// ---- collectives (kernels/comm.py) -------------------------------------------------------
std::vector<LoweredKernel> allreduce(const HwView& hw, const std::string& name, double nbytes, int group,
                                     const std::string& algo);
std::vector<LoweredKernel> all_to_all(const HwView& hw, const std::string& name, double send_bytes,
                                      int group);

// ---- tile search spaces (kernels/tiles.py) -----------------------------------------------
std::vector<GemmTile> gemm_search_space(int64_t M, int64_t N, int64_t K);
std::vector<AttnTile> attn_search_space();

// ---- evaluate: shader_core::evaluate_kernel with this cur_gpu_config's lanes -------------
KernelResult evaluate(const HwView& hw, const LoweredKernel& k);

}  // namespace tilesight::gpu_top
