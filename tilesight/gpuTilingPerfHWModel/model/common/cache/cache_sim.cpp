#include "cache_sim.hpp"

#include <algorithm>
#include <list>
#include <unordered_map>
#include <utility>

namespace tilesight::cache {
namespace {

struct Partition {
  int64_t capacity;
  int64_t used = 0;
  int64_t evictions = 0;
  // ordered resident set, front = oldest / LRU-evict-first depending on policy; key -> size
  std::list<std::pair<int64_t, int64_t>> order;
  std::unordered_map<int64_t, std::list<std::pair<int64_t, int64_t>>::iterator> index;

  bool access(int64_t key, int64_t size, const std::string& policy) {
    auto it = index.find(key);
    if (it != index.end()) {
      if (policy == "lru") {
        order.splice(order.end(), order, it->second);  // move_to_end
      }
      return true;
    }
    while (used + size > capacity && !order.empty()) {
      // policy=="mru" pops from the back (last), else pops from the front (first)
      std::pair<int64_t, int64_t> victim;
      if (policy == "mru") {
        victim = order.back();
        order.pop_back();
      } else {
        victim = order.front();
        order.pop_front();
      }
      index.erase(victim.first);
      used -= victim.second;
      ++evictions;
    }
    if (size <= capacity) {
      order.push_back({key, size});
      index[key] = std::prev(order.end());
      used += size;
    }
    return false;
  }
};

}  // namespace

SimResult simulate(const std::vector<int64_t>& keys, const std::vector<int64_t>& addrs,
                    const std::vector<double>& sizes, const std::vector<int>& streams, int n_streams,
                    double capacity_bytes, int n_partitions, const std::vector<int>& part_of_addr,
                    const std::string& policy, bool page_fill) {
  n_partitions = std::max(1, n_partitions);
  std::vector<Partition> parts(n_partitions);
  int64_t cap_each = static_cast<int64_t>(capacity_bytes) / n_partitions;
  for (auto& p : parts) p.capacity = cap_each;

  SimResult res;
  res.misses.assign(n_streams, 0.0);
  res.accesses.assign(n_streams, 0);
  int64_t total = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    int st = streams[i];
    res.accesses[st] += 1;
    ++total;
    int part = ((part_of_addr[i] % n_partitions) + n_partitions) % n_partitions;
    if (!page_fill) {
      if (!parts[part].access(keys[i], static_cast<int64_t>(sizes[i]), policy)) res.misses[st] += 1.0;
      continue;
    }
    int64_t shard = static_cast<int64_t>(sizes[i]) / n_partitions;
    if (parts[part].access(keys[i], shard, policy)) continue;
    res.misses[st] += 1.0;
    for (int p = 0; p < n_partitions; ++p)
      if (p != part) parts[p].access(keys[i], shard, policy);
  }
  double miss_sum = 0.0;
  for (double m : res.misses) miss_sum += m;
  res.hit_rate = total ? 1.0 - miss_sum / static_cast<double>(total) : 1.0;
  for (auto& p : parts) {
    res.partition_tiles.push_back(static_cast<int64_t>(p.order.size()));
    res.partition_bytes.push_back(p.used);
    res.partition_capacity.push_back(p.capacity);
    res.partition_evictions.push_back(p.evictions);
  }
  return res;
}

}  // namespace tilesight::cache
