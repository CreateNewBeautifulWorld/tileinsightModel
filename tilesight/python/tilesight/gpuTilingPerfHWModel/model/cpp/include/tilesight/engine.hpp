// TileSight C++ core — must match python/tilesight/engine/reference.py and
// engine/cache.py to 1e-9 relative (tests/test_cpp_parity.py enforces it).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tilesight {

struct Lane {
  std::string name;
  bool shared = false;      // shared lanes: work in bytes, rate = min(total/active, cap)
  double total_rate = 0.0;  // bytes/s (shared lanes)
  double per_sm_cap = 0.0;  // bytes/s (shared lanes)
};

struct Action {
  std::string name;          // e.g. "load:weight", "mma", "softmax" (used for attribution)
  std::vector<double> work;  // indexed like the lane table; per-SM lanes in seconds
  std::vector<int> deps;     // indices of earlier actions in the same list
  bool recurrent = false;    // loop-carried compute (hidden only by `consumers`)
  double latency = 0.0;      // seconds
};

struct Kernel {
  std::string name;
  int64_t num_blocks = 0;
  int64_t iters = 0;
  int stages = 1;
  int resident = 1;
  int consumers = 1;
  std::vector<Action> body, prologue, epilogue;
  bool fixed = false;        // communication kernels: fixed_time given
  double fixed_time = 0.0;
  double queue_coef = 0.0;   // M/D/1-style latency inflation near saturation (0 = off)
  double queue_max = 3.0;
};

struct Result {
  double time = 0.0;
  std::string bottleneck;
  std::map<std::string, double> util;
  std::map<std::string, double> breakdown;
  std::map<std::string, double> limiter_time;
  std::map<std::string, double> limiter_detail;  // "lane:action" / "latency:action(lane)"
  int64_t waves = 0;
};

Result evaluate(const Kernel& k, const std::vector<Lane>& lanes, int sms, double launch_s);

std::vector<Result> evaluate_batch(const std::vector<Kernel>& ks, const std::vector<Lane>& lanes,
                                   int sms, double launch_s, int threads);

// Tile reuse-distance L2 model (paper Eqs. 6-10).
double hit_prob(int64_t d, int assoc, double cap_tiles);
std::vector<double> expected_misses(const std::vector<int64_t>& keys, const std::vector<int>& streams,
                                    int n_streams, int assoc, double cap_tiles);

}  // namespace tilesight
