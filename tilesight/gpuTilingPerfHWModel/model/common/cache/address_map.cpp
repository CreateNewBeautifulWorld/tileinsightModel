#include "address_map.hpp"

namespace tilesight::cache {

int64_t port_of(const AddrCfg& cfg, int64_t addr) {
  if (cfg.mode == "range") {
    int64_t span = (int64_t(1) << cfg.addr_bits) / std::max(1, cfg.ports);
    int64_t p = span > 0 ? addr / span : 0;
    return std::min<int64_t>(cfg.ports - 1, std::max<int64_t>(0, p));
  }
  int64_t idx = cfg.granularity > 0 ? addr / cfg.granularity : addr;
  if (cfg.mode == "hash") {
    int64_t h = idx;
    for (int shift : {4, 8, 16}) h ^= (idx >> shift);
    idx = h;
  }
  int64_t p = idx % std::max(1, cfg.ports);
  return p < 0 ? p + cfg.ports : p;
}

}  // namespace tilesight::cache
