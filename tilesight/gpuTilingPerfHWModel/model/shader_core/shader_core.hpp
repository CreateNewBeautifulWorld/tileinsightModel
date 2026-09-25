// shader_core — one core's per-lane time and the steady-state K-loop round formula
// (port of engine/reference.py + engine/shader.py + engine/gmem.py's fair-share math).
//
// `TraceAction`/`TraceKernel` are plain reporting/assembly records built by gpu_top's per-op
// lowering functions (gemm/attention/comm) — NOT a generic dispatch IR: nothing outside this
// translation unit and gpu_top ever pattern-matches on an action's "kind", they are just a
// bag of {name, per-lane seconds/bytes, dependency indices, latency} used once to run the
// round/critical-path formula below and, for reporting, handed back to Python read-only
// (genResult/figure3.py + timeline.py's Fig.3(d)(e) reconstruction).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tilesight::shader_core {

struct LaneSpec {
  std::string name;
  bool shared = false;      // shared: work is bytes, rate = min(total_rate/active, per_sm_cap)
  double total_rate = 0.0;  // bytes/s, whole GPU (shared lanes)
  double per_sm_cap = 0.0;  // bytes/s, one core's ceiling (shared lanes)
};

struct TraceAction {
  std::string name;
  std::map<std::string, double> work;  // lane name -> seconds (per-core) or bytes (shared)
  std::vector<int> deps;
  bool recurrent = false;
  double latency = 0.0;
};

struct TraceKernel {
  std::string name, kind;
  int64_t num_blocks = 0;
  int64_t iters = 0;
  int stages = 1;
  int resident = 1;
  int consumers = 1;
  std::vector<TraceAction> body, prologue, epilogue;
  bool fixed = false;
  double fixed_time = 0.0;
  double queue_coef = 0.0;
  double queue_max = 3.0;
};

struct KernelResult {
  std::string name, kind, bottleneck;
  double time_s = 0.0;
  std::map<std::string, double> util, breakdown, limiter_time, limiter_detail;
  int64_t waves = 0;
};

double lane_time(double work, const LaneSpec& lane, int active_cores);
double per_core_rate(const LaneSpec& lane, int active_cores);

double node_weight(const TraceAction& a, const std::vector<LaneSpec>& lanes, int active, double qf = 1.0);

// Longest path through actions (topological order), returns path length; if `path` is given,
// fills it with the node indices on that path (source-to-sink order).
double critical_path(const std::vector<TraceAction>& acts, const std::vector<double>& w,
                     std::vector<int>* path = nullptr);

// Evaluate a whole kernel (wave decomposition + steady round + prologue/epilogue) — the
// engine/reference.py `evaluate()` equivalent.
KernelResult evaluate_kernel(const TraceKernel& k, const std::vector<LaneSpec>& lanes, int sms,
                             double launch_s);

}  // namespace tilesight::shader_core
