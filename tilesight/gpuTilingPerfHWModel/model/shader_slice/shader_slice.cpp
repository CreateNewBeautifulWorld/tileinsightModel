#include "shader_slice.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace tilesight::shader_slice {

std::vector<Wave> wave_plan(int64_t num_blocks, int cores, int resident) {
  resident = std::max(1, resident);
  int64_t conc = static_cast<int64_t>(cores) * resident;
  int64_t full = conc > 0 ? num_blocks / conc : 0;
  int64_t tail = conc > 0 ? num_blocks % conc : num_blocks;
  std::vector<Wave> out;
  for (auto [blocks, count] : {std::pair<int64_t, int64_t>{conc, full}, {tail, 1}}) {
    if (blocks <= 0 || count <= 0) continue;
    int active = static_cast<int>(std::min<int64_t>(cores, (blocks + resident - 1) / resident));
    active = std::max(1, active);
    int64_t bps = (blocks + active - 1) / active;
    out.push_back({blocks, count, active, bps});
  }
  return out;
}

}  // namespace tilesight::shader_slice
