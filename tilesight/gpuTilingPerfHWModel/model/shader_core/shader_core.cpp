#include "shader_core.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "../shader_slice/shader_slice.hpp"

namespace tilesight::shader_core {
namespace {

inline double get_work(const TraceAction& a, const std::string& lane) {
  auto it = a.work.find(lane);
  return it == a.work.end() ? 0.0 : it->second;
}

std::string lane_detail(const std::vector<TraceAction>& acts, const LaneSpec& lane, int active) {
  double best = -1.0;
  size_t arg = 0;
  for (size_t i = 0; i < acts.size(); ++i) {
    double v = lane_time(get_work(acts[i], lane.name), lane, active);
    if (v > best) { best = v; arg = i; }
  }
  return lane.name + ":" + acts[arg].name;
}

std::string latency_detail(const std::vector<TraceAction>& acts, const std::vector<double>& w,
                           const std::vector<LaneSpec>& lanes, int active) {
  std::vector<int> path;
  critical_path(acts, w, &path);
  if (path.empty()) return "latency:none";
  int j = path[0];
  for (int i : path) if (w[i] > w[j]) j = i;
  const TraceAction& a = acts[j];
  double best = -1.0;
  std::string lane_name = "none";
  for (const auto& l : lanes) {
    double v = lane_time(get_work(a, l.name), l, active);
    if (v > best) { best = v; lane_name = l.name; }
  }
  if (a.latency > best) lane_name = "mem-lat";
  return "latency:" + a.name + "(" + lane_name + ")";
}

// (bps * sum_o u_r(o), argmax lane index); first max wins (matches Python max()).
std::pair<double, int> resource_bound(const std::vector<TraceAction>& acts,
                                      const std::vector<LaneSpec>& lanes, int active, int64_t bps) {
  double best = -1.0;
  int arg = 0;
  for (size_t r = 0; r < lanes.size(); ++r) {
    double s = 0.0;
    for (const auto& a : acts) s += lane_time(get_work(a, lanes[r].name), lanes[r], active);
    s *= static_cast<double>(bps);
    if (s > best) { best = s; arg = static_cast<int>(r); }
  }
  return {best, arg};
}

struct Phase { double t; std::string lim, det; };

Phase phase_time(const std::vector<TraceAction>& acts, const std::vector<LaneSpec>& lanes, int active,
                 int64_t bps, double qf) {
  if (acts.empty()) return {0.0, "none", "none"};
  auto [rb, lane] = resource_bound(acts, lanes, active, bps);
  std::vector<double> w;
  w.reserve(acts.size());
  for (const auto& a : acts) w.push_back(node_weight(a, lanes, active, qf));
  double cp = critical_path(acts, w);
  if (rb >= cp) return {rb, lanes[lane].name, lane_detail(acts, lanes[lane], active)};
  return {cp, "latency", latency_detail(acts, w, lanes, active)};
}

}  // namespace

double per_core_rate(const LaneSpec& lane, int active_cores) {
  return std::min(lane.total_rate / std::max(1, active_cores), lane.per_sm_cap);
}

double lane_time(double work, const LaneSpec& lane, int active_cores) {
  if (work == 0.0) return 0.0;
  if (!lane.shared) return work;
  return work / per_core_rate(lane, active_cores);
}

double node_weight(const TraceAction& a, const std::vector<LaneSpec>& lanes, int active, double qf) {
  double m = 0.0;
  for (const auto& l : lanes) m = std::max(m, lane_time(get_work(a, l.name), l, active));
  return m + a.latency * qf;
}

double critical_path(const std::vector<TraceAction>& acts, const std::vector<double>& w,
                     std::vector<int>* path) {
  const int n = static_cast<int>(acts.size());
  std::vector<double> best(n, 0.0);
  std::vector<int> pred(n, -1);
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    int p = -1;
    for (int d : acts[i].deps)
      if (best[d] > s) { s = best[d]; p = d; }
    best[i] = s + w[i];
    pred[i] = p;
  }
  if (n == 0) return 0.0;
  int end = 0;
  for (int i = 1; i < n; ++i) if (best[i] > best[end]) end = i;
  const double len = best[end];
  if (path) {
    path->clear();
    for (int e = end; e >= 0; e = pred[e]) path->push_back(e);
    std::reverse(path->begin(), path->end());
  }
  return len;
}

KernelResult evaluate_kernel(const TraceKernel& k, const std::vector<LaneSpec>& lanes, int sms,
                             double launch) {
  KernelResult res;
  res.name = k.name;
  res.kind = k.kind;
  if (k.fixed) {
    res.time_s = k.fixed_time + launch;
    res.bottleneck = "net";
    res.util["net"] = 1.0;
    res.breakdown = {{"comm", k.fixed_time}, {"launch", launch}};
    res.limiter_time = {{"net", k.fixed_time}, {"launch", launch}};
    res.limiter_detail = {{"net:" + k.name, k.fixed_time}, {"launch", launch}};
    res.waves = 0;
    return res;
  }

  const int resident = std::max(1, k.resident);
  auto waves = shader_slice::wave_plan(k.num_blocks, sms, resident);

  double total = 0.0;
  double bp = 0, bf = 0, bs = 0, be = 0;
  std::map<std::string, double> lim{{"launch", launch}};
  std::vector<std::string> order{"launch"};
  auto add = [&](const std::string& n, double t) {
    if (t <= 0) return;
    if (!lim.count(n)) order.push_back(n);
    lim[n] += t;
  };
  std::map<std::string, double> det{{"launch", launch}};
  auto addd = [&](const std::string& n, double t) { if (t > 0) det[n] += t; };

  for (const auto& wv : waves) {
    const int active = wv.active;
    const int64_t bps = wv.bps;
    auto [rb, lane] = resource_bound(k.body, lanes, active, bps);
    double qf = 1.0;
    if (k.queue_coef > 0) {
      double best_u = 0.0;
      for (const auto& l : lanes) {
        if (!l.shared) continue;
        double s2 = 0.0;
        for (const auto& a : k.body) s2 += lane_time(get_work(a, l.name), l, active);
        best_u = std::max(best_u, s2 * static_cast<double>(bps) / std::max(rb, 1e-18));
      }
      best_u = std::min(best_u, 0.995);
      qf = std::min(k.queue_max, 1.0 + k.queue_coef * best_u / (1.0 - best_u));
    }
    std::vector<double> w, wst, wrec;
    w.reserve(k.body.size());
    wst.reserve(k.body.size());
    wrec.reserve(k.body.size());
    for (const auto& a : k.body) {
      double x = node_weight(a, lanes, active, qf);
      w.push_back(x);
      double xs = x - (a.recurrent ? 0.0 : a.latency * qf);
      wst.push_back(xs);
      wrec.push_back(a.recurrent ? xs : 0.0);
    }
    const double cp = critical_path(k.body, w);
    const double cpst = critical_path(k.body, wst);
    const double cprec = critical_path(k.body, wrec);
    const double latb = std::max(cpst / std::max(1, k.stages), cprec / std::max(1, k.consumers));
    double rnd;
    std::string limiter, dsteady;
    if (k.body.empty()) { rnd = 0.0; limiter = "none"; dsteady = "none"; }
    else if (rb >= latb) {
      rnd = rb;
      limiter = lanes[lane].name;
      dsteady = lane_detail(k.body, lanes[lane], active);
    } else {
      rnd = latb;
      limiter = "latency";
      const bool use_rec = cprec / std::max(1, k.consumers) > cpst / std::max(1, k.stages);
      dsteady = latency_detail(k.body, use_rec ? wrec : wst, lanes, active);
    }
    Phase pro = phase_time(k.prologue, lanes, active, bps, qf);
    Phase epi = phase_time(k.epilogue, lanes, active, bps, qf);
    const double tpro = pro.t, tepi = epi.t;
    const double tfill = k.iters > 0 ? std::max(0.0, cp - rnd) : 0.0;
    const double tsteady = static_cast<double>(k.iters) * rnd;
    total += static_cast<double>(wv.count) * (tpro + tfill + tsteady + tepi);
    bp += wv.count * tpro; bf += wv.count * tfill; bs += wv.count * tsteady; be += wv.count * tepi;
    add(limiter, wv.count * tsteady);
    add("latency", wv.count * tfill);
    add(pro.lim, wv.count * tpro);
    add(epi.lim, wv.count * tepi);
    addd(dsteady, wv.count * tsteady);
    addd("latency:fill", wv.count * tfill);
    addd(pro.det, wv.count * tpro);
    addd(epi.det, wv.count * tepi);
  }
  total += launch;

  for (const auto& l : lanes) {
    auto wsum = [&](const std::vector<TraceAction>& v) {
      double s = 0;
      for (const auto& a : v) s += get_work(a, l.name);
      return s;
    };
    double per_block = wsum(k.prologue) + static_cast<double>(k.iters) * wsum(k.body) + wsum(k.epilogue);
    double busy = static_cast<double>(k.num_blocks) * per_block;
    busy = l.shared ? busy / l.total_rate : busy / sms;
    if (busy > 0) res.util[l.name] = busy / total;
  }
  res.time_s = total;
  res.breakdown = {{"prologue", bp}, {"fill", bf}, {"steady", bs}, {"epilogue", be}, {"launch", launch}};
  res.limiter_time = lim;
  res.limiter_detail = det;
  double bestv = -1;
  for (const auto& n : order) if (lim[n] > bestv) { bestv = lim[n]; res.bottleneck = n; }
  const int64_t conc = static_cast<int64_t>(sms) * resident;
  const int64_t full = conc > 0 ? k.num_blocks / conc : 0;
  const int64_t tail = conc > 0 ? k.num_blocks % conc : k.num_blocks;
  res.waves = waves.empty() ? 0 : full + (tail ? 1 : 0);
  return res;
}

}  // namespace tilesight::shader_core
