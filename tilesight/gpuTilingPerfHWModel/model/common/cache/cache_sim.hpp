// Deterministic tile-level cache simulation (port of engine/cache_sim.py).
//
// The tile is the atom (no cache lines). Capacity is split into `n_partitions` independent
// sets of ways, each holding whole tiles; a tile goes to the partition its address maps to
// (address_map.hpp), and each partition runs an explicit replacement policy (lru|fifo|mru).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tilesight::cache {

struct SimResult {
  std::vector<double> misses;              // per stream
  std::vector<double> miss_bytes;          // per stream: sum of sizes[i] over the misses
  std::vector<int64_t> accesses;           // per stream
  double hit_rate = 1.0;
  // final-residency summary, one entry per partition
  std::vector<int64_t> partition_tiles;
  std::vector<int64_t> partition_bytes;
  std::vector<int64_t> partition_capacity;
  std::vector<int64_t> partition_evictions;

  std::vector<double> miss_fraction() const {
    std::vector<double> out(misses.size(), 0.0);
    for (size_t i = 0; i < misses.size(); ++i)
      out[i] = accesses[i] ? misses[i] / static_cast<double>(accesses[i]) : 0.0;
    return out;
  }
};

// `part_of(addr)` maps a tile address to a partition index in [0, n_partitions).
//
// `page_fill` (DMA hugepage mode): a key is a whole page spread evenly over every partition, so
// each partition holds only its own shard (`sizes[i] / n_partitions` bytes). An access looks up
// the shard in the partition its address maps to; a miss there means the DMA fetches the whole
// page again, which refills that page's shard in every partition (a partition already holding it
// just refreshes it). Each partition still evicts on its own, so some slices can hold a page's
// shard while others have already dropped theirs. `fill_mask` (optional, one flag per access)
// limits that to the accesses whose atom really is a DMA page/chunk; the others (tile-granular
// L2-port fetches, e.g. the tail of a stream in dma_page_tail_l2 mode) stay whole-tile in the one
// partition their address maps to.
SimResult simulate(const std::vector<int64_t>& keys, const std::vector<int64_t>& addrs,
                    const std::vector<double>& sizes, const std::vector<int>& streams, int n_streams,
                    double capacity_bytes, int n_partitions,
                    const std::vector<int>& part_of_addr, const std::string& policy,
                    bool page_fill = false, const std::vector<char>* fill_mask = nullptr);

}  // namespace tilesight::cache
