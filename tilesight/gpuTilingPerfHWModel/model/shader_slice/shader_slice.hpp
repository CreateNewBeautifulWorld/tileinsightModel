// shader_slice — grid -> waves, and the slice-shared L1 (port of engine/shader.py's
// wave_plan, and kernels/gemm.py's l1_miss_fractions).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tilesight::shader_slice {

struct Wave {
  int64_t blocks = 0;   // blocks in this wave (conc for a full wave, tail otherwise)
  int64_t count = 0;    // how many such waves (full waves: >1; tail: always 1)
  int active = 0;       // active cores in this wave
  int64_t bps = 0;       // blocks per active core (ceil)
};

// [(blocks in the wave, how many such waves, active cores, blocks per active core)]
std::vector<Wave> wave_plan(int64_t num_blocks, int cores, int resident);

}  // namespace tilesight::shader_slice
