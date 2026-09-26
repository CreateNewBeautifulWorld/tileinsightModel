#include "hw_view.hpp"

#include <cmath>
#include <limits>

namespace tilesight::gpu_top {

std::string HwView::fingerprint() const { return nb::cast<std::string>(cfg_.attr("fingerprint")); }
int HwView::sms() const { return nb::cast<int>(cfg_.attr("sms")); }
double HwView::clock_hz() const { return nb::cast<double>(cfg_.attr("clock_hz")); }
double HwView::launch_s() const { return nb::cast<double>(cfg_.attr("launch_s")); }
double HwView::eff(const std::string& key) const { return nb::cast<double>(cfg_.attr("eff")(key)); }
double HwView::l1_capacity_bytes() const { return nb::cast<double>(cfg_.attr("l1_capacity_bytes")); }
double HwView::l2_capacity_bytes() const { return nb::cast<double>(cfg_.attr("l2_capacity_bytes")); }
int64_t HwView::smem_per_sm() const { return nb::cast<int64_t>(cfg_.attr("smem_per_sm")); }
int64_t HwView::tmem_per_sm() const { return nb::cast<int64_t>(cfg_.attr("tmem_per_sm")); }
bool HwView::has_tmem() const { return nb::cast<bool>(cfg_.attr("has_tmem")); }
int HwView::tc_min_m() const { return nb::cast<int>(cfg_.attr("tc_min_m")); }
std::string HwView::name() const { return nb::cast<std::string>(cfg_.attr("name")); }

nb::object HwView::get(const std::string& path) const { return cfg_.attr("get")(path); }

double HwView::get_num(const std::string& path, double def) const {
  nb::object o = get(path);
  if (o.is_none()) return def;
  return nb::cast<double>(o);
}

std::string HwView::get_str(const std::string& path, const std::string& def) const {
  nb::object o = get(path);
  if (o.is_none()) return def;
  return nb::cast<std::string>(o);
}

bool HwView::get_bool(const std::string& path, bool def) const {
  nb::object o = get(path);
  if (o.is_none()) return def;
  return nb::cast<bool>(o);
}

std::pair<std::string, double> HwView::mma_cost(double flops, const std::string& dtype) const {
  nb::object r = cfg_.attr("mma_cost")(flops, dtype);
  auto t = nb::cast<nb::tuple>(r);
  return {nb::cast<std::string>(t[0]), nb::cast<double>(t[1])};
}

double HwView::cuda_time_per_sm(double flops) const {
  return nb::cast<double>(cfg_.attr("cuda_time_per_sm")(flops));
}
double HwView::sfu_time_per_sm(double ops) const {
  return nb::cast<double>(cfg_.attr("sfu_time_per_sm")(ops));
}
double HwView::l1_time_per_sm(double nbytes) const {
  return nb::cast<double>(cfg_.attr("l1_time_per_sm")(nbytes));
}
double HwView::smem_time_per_sm(double nbytes) const {
  return nb::cast<double>(cfg_.attr("smem_time_per_sm")(nbytes));
}
double HwView::tmem_time_per_sm(double read_bytes, double write_bytes) const {
  return nb::cast<double>(cfg_.attr("tmem_time_per_sm")(read_bytes, write_bytes));
}
double HwView::unit_latency_s(const std::string& unit) const {
  return nb::cast<double>(cfg_.attr("unit_latency_s")(unit));
}

std::string HwView::resolve_path(const std::string& path) const {
  return nb::cast<std::string>(cfg_.attr("resolve_path")(path));
}
bool HwView::path_is_dma(const std::string& path) const {
  return nb::cast<bool>(cfg_.attr("path_is_dma")(path));
}
double HwView::path_time_per_sm(const std::string& path, double nbytes) const {
  return nb::cast<double>(cfg_.attr("path_time_per_sm")(path, nbytes));
}
double HwView::path_latency_s(const std::string& path) const {
  return nb::cast<double>(cfg_.attr("path_latency_s")(path));
}
double HwView::path_attr_num(const std::string& path, const std::string& key, double def) const {
  nb::object o = cfg_.attr("path_attr")(path, key, def);
  if (o.is_none()) return def;
  return nb::cast<double>(o);
}
bool HwView::path_attr_bool(const std::string& path, const std::string& key, bool def) const {
  nb::object o = cfg_.attr("path_attr")(path, key, def);
  if (o.is_none()) return def;
  return nb::cast<bool>(o);
}

bool HwView::has_sram() const { return !cfg_.attr("sram").is_none(); }
double HwView::sram_bandwidth_Bps() const {
  return get_num("memory.sram.bandwidth_TBps", 0.0) * 1e12 * eff("sram");
}
double HwView::sram_latency_ns() const { return get_num("memory.sram.latency_ns", 0.0); }
bool HwView::sram_costream() const { return get_bool("memory.sram.costream", false); }
bool HwView::sram_prefetch() const { return get_bool("memory.sram.prefetch", false); }
double HwView::sram_stage_share_blocks() const { return get_num("memory.sram.stage.share_blocks", 1.0); }
double HwView::sram_capacity_for(const std::string& klass) const {
  return nb::cast<double>(cfg_.attr("sram_capacity_for")(klass));
}
double HwView::sram_capacity_bytes() const {
  return nb::cast<double>(cfg_.attr("sram_capacity_bytes"));
}
bool HwView::sram_keeps_intermediates() const {
  return nb::cast<bool>(cfg_.attr("sram_keeps_intermediates"));
}

double HwView::ddr_bandwidth_Bps() const {
  return get_num("memory.ddr.bandwidth_TBps", 0.0) * 1e12 * eff("ddr");
}

const std::vector<shader_core::LaneSpec>& HwView::lanes() const {
  if (lanes_cached_) return lanes_cache_;
  lanes_cache_.clear();
  nb::object py_lanes = cfg_.attr("lanes")();
  for (nb::handle h : py_lanes) {
    nb::object l = nb::borrow<nb::object>(h);
    shader_core::LaneSpec ls;
    ls.name = nb::cast<std::string>(l.attr("name"));
    ls.shared = nb::cast<bool>(l.attr("shared"));
    ls.total_rate = nb::cast<double>(l.attr("total_rate"));
    double cap = nb::cast<double>(l.attr("per_sm_cap"));
    ls.per_sm_cap = std::isinf(cap) ? std::numeric_limits<double>::max() : cap;
    lanes_cache_.push_back(std::move(ls));
  }
  lanes_cached_ = true;
  return lanes_cache_;
}

}  // namespace tilesight::gpu_top
