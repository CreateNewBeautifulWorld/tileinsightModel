#include "memory_slice.hpp"

namespace tilesight::memory_slice {

double mem_latency_s(double l2_latency_ns, double ddr_latency_ns, double miss, bool has_l2) {
  if (!has_l2) return ddr_latency_ns * 1e-9;
  return ((1.0 - miss) * l2_latency_ns + miss * ddr_latency_ns) * 1e-9;
}

cache::SimResult simulate_l2(const std::vector<int64_t>& keys, const std::vector<int64_t>& addrs,
                             const std::vector<double>& sizes, const std::vector<int>& streams,
                             int n_streams, double l2_capacity_bytes, int n_partitions,
                             const cache::AddrCfg& l2_addr_cfg, const std::string& policy,
                             bool page_fill) {
  std::vector<int> part_of(addrs.size());
  for (size_t i = 0; i < addrs.size(); ++i)
    part_of[i] = static_cast<int>(cache::port_of(l2_addr_cfg, addrs[i]));
  return cache::simulate(keys, addrs, sizes, streams, n_streams, l2_capacity_bytes, n_partitions, part_of,
                         policy, page_fill);
}

}  // namespace tilesight::memory_slice
