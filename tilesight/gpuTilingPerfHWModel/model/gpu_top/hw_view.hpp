// HwView — gpu_top's read-only window onto the live Python HardwareSpec.
//
// interfaceAndRun/hardware_spec.py stays the single source of hardware-derived numbers
// (dtype tables, per-unit cost formulas, path resolution, capacity/derating policy): gpu_top
// calls back into the very same, unchanged HardwareSpec object for those, instead of forking a
// second copy of that logic into C++. What lives in C++ here is the lowering/tiling-search and
// the wave/round/cache engine (shader_core, shader_slice, on_chip_buffer, memory_slice,
// common/cache) — the numerically interesting, previously Python "reference engine" part of
// gpuTilingPerfHWModel/model/.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <utility>
#include <vector>

#include "../shader_core/shader_core.hpp"

namespace nb = nanobind;

namespace tilesight::gpu_top {

class HwView {
 public:
  explicit HwView(nb::object cfg) : cfg_(std::move(cfg)) {}

  const nb::object& obj() const { return cfg_; }

  std::string fingerprint() const;
  int sms() const;
  double clock_hz() const;
  double launch_s() const;
  double eff(const std::string& key) const;
  double l1_capacity_bytes() const;
  double l2_capacity_bytes() const;
  int64_t smem_per_sm() const;
  int64_t tmem_per_sm() const;
  bool has_tmem() const;
  int tc_min_m() const;
  std::string name() const;

  // generic dotted-path getter; `has` reports whether the raw config had the key (vs. the
  // schema default), matching HardwareSpec.get()'s "None means absent" semantics closely
  // enough for the numeric fields gpu_top reads.
  nb::object get(const std::string& path) const;
  double get_num(const std::string& path, double def = 0.0) const;
  std::string get_str(const std::string& path, const std::string& def = "") const;
  bool get_bool(const std::string& path, bool def = false) const;

  std::pair<std::string, double> mma_cost(double flops, const std::string& dtype) const;
  double cuda_time_per_sm(double flops) const;
  double sfu_time_per_sm(double ops) const;
  double l1_time_per_sm(double nbytes) const;
  double smem_time_per_sm(double nbytes) const;
  double tmem_time_per_sm(double read_bytes, double write_bytes) const;
  double unit_latency_s(const std::string& unit) const;

  std::string resolve_path(const std::string& path) const;
  bool path_is_dma(const std::string& path) const;
  double path_time_per_sm(const std::string& path, double nbytes) const;
  double path_latency_s(const std::string& path) const;
  double path_attr_num(const std::string& path, const std::string& key, double def) const;
  bool path_attr_bool(const std::string& path, const std::string& key, bool def) const;

  bool has_sram() const;
  double sram_bandwidth_Bps() const;
  double sram_latency_ns() const;
  bool sram_costream() const;
  bool sram_prefetch() const;
  double sram_stage_share_blocks() const;
  double sram_capacity_for(const std::string& klass) const;
  double sram_capacity_bytes() const;
  bool sram_keeps_intermediates() const;

  double ddr_bandwidth_Bps() const;

  const std::vector<shader_core::LaneSpec>& lanes() const;

 private:
  nb::object cfg_;
  mutable bool lanes_cached_ = false;
  mutable std::vector<shader_core::LaneSpec> lanes_cache_;
};

}  // namespace tilesight::gpu_top
