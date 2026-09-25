// Tile-granularity addressing: which L2 slice / HBM port a tile address lands on.
// Port of the mapping logic in genResult/addressing.py's AddressMap.port_of (the reporting/
// dump machinery there stays in Python; this is only the piece the memory-slice L2 simulation
// needs at run time).
#pragma once
#include <cstdint>
#include <string>

namespace tilesight::cache {

struct AddrCfg {
  int ports = 1;
  std::string mode = "interleave";  // interleave | range | hash
  int64_t granularity = 1024;       // bytes
  int addr_bits = 48;
};

int64_t port_of(const AddrCfg& cfg, int64_t addr);

}  // namespace tilesight::cache
