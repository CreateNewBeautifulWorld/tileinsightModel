// memory_slice — the L2 port + DMA port + HBM behind it, at tile granularity (port of
// engine/memory_model.py's L2/HBM piece).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../common/cache/address_map.hpp"
#include "../common/cache/cache_sim.hpp"

namespace tilesight::memory_slice {

// L2 hit/miss latency blend, seconds. `has_l2` false means every miss pays HBM latency (no L2
// level at all).
double mem_latency_s(double l2_latency_ns, double ddr_latency_ns, double miss, bool has_l2);

// Run a tile-access stream through the memory slices' L2s (one partition per slice).
cache::SimResult simulate_l2(const std::vector<int64_t>& keys, const std::vector<int64_t>& addrs,
                             const std::vector<double>& sizes, const std::vector<int>& streams,
                             int n_streams, double l2_capacity_bytes, int n_partitions,
                             const cache::AddrCfg& l2_addr_cfg, const std::string& policy);

}  // namespace tilesight::memory_slice
