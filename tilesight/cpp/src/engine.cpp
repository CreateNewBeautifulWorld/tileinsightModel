#include "tilesight/engine.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace tilesight {
namespace {

inline double u(const Action& a, const Lane& l, size_t r, int active) {
  double w = r < a.work.size() ? a.work[r] : 0.0;
  if (w == 0.0) return 0.0;
  if (l.shared) return w / std::min(l.total_rate / active, l.per_sm_cap);
  return w;
}

inline double node_w(const Action& a, const std::vector<Lane>& lanes, int active, double qf = 1.0) {
  double m = 0.0;
  for (size_t r = 0; r < lanes.size(); ++r) m = std::max(m, u(a, lanes[r], r, active));
  return m + a.latency * qf;
}

// Longest path + node list (mirrors reference._critical_path, incl. tie-breaks).
double critical_path(const std::vector<Action>& acts, const std::vector<double>& w, std::vector<int>* path) {
  const int n = static_cast<int>(acts.size());
  std::vector<double> best(n, 0.0);
  std::vector<int> pred(n, -1);
  for (int i = 0; i < n; ++i) {
    double s = 0.0; int p = -1;
    for (int d : acts[i].deps) if (best[d] > s) { s = best[d]; p = d; }
    best[i] = s + w[i]; pred[i] = p;
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

double longest_path(const std::vector<Action>& acts, const std::vector<double>& w) {
  return critical_path(acts, w, nullptr);
}

std::string lane_detail(const std::vector<Action>& acts, const std::vector<Lane>& lanes, size_t r, int active) {
  double best = -1.0; size_t arg = 0;
  for (size_t i = 0; i < acts.size(); ++i) {
    double v = u(acts[i], lanes[r], r, active);
    if (v > best) { best = v; arg = i; }
  }
  return lanes[r].name + ":" + acts[arg].name;
}

std::string latency_detail(const std::vector<Action>& acts, const std::vector<double>& w,
                           const std::vector<Lane>& lanes, int active) {
  std::vector<int> path;
  critical_path(acts, w, &path);
  if (path.empty()) return "latency:none";
  int j = path[0];
  for (int i : path) if (w[i] > w[j]) j = i;
  const Action& a = acts[j];
  double best = -1.0; std::string lane = "none";
  for (size_t r = 0; r < lanes.size(); ++r) {
    double v = u(a, lanes[r], r, active);
    if (v > best) { best = v; lane = lanes[r].name; }
  }
  if (a.latency > best) lane = "mem-lat";
  return "latency:" + a.name + "(" + lane + ")";
}

// returns (max_r bps*sum_o u_r, argmax lane index); first max wins (matches Python max()).
std::pair<double, size_t> resource_bound(const std::vector<Action>& acts, const std::vector<Lane>& lanes,
                                         int active, int64_t bps) {
  double best = -1.0;
  size_t arg = 0;
  for (size_t r = 0; r < lanes.size(); ++r) {
    double s = 0.0;
    for (const auto& a : acts) s += u(a, lanes[r], r, active);
    s *= static_cast<double>(bps);
    if (s > best) { best = s; arg = r; }
  }
  return {best, arg};
}

struct Phase { double t; std::string lim, det; };

Phase phase(const std::vector<Action>& acts, const std::vector<Lane>& lanes, int active, int64_t bps,
            double qf = 1.0) {
  if (acts.empty()) return {0.0, "none", "none"};
  auto [rb, lane] = resource_bound(acts, lanes, active, bps);
  std::vector<double> w;
  for (const auto& a : acts) w.push_back(node_w(a, lanes, active, qf));
  double cp = longest_path(acts, w);
  if (rb >= cp) return {rb, lanes[lane].name, lane_detail(acts, lanes, lane, active)};
  return {cp, "latency", latency_detail(acts, w, lanes, active)};
}

}  // namespace

Result evaluate(const Kernel& k, const std::vector<Lane>& lanes, int sms, double launch) {
  Result res;
  if (k.fixed) {
    res.time = k.fixed_time + launch;
    res.bottleneck = "net";
    res.util["net"] = 1.0;
    res.breakdown = {{"comm", k.fixed_time}, {"launch", launch}};
    res.limiter_time = {{"net", k.fixed_time}, {"launch", launch}};
    res.limiter_detail = {{"net:" + k.name, k.fixed_time}, {"launch", launch}};
    return res;
  }
  const int64_t resident = std::max(1, k.resident);
  const int64_t conc = sms * resident;
  const int64_t full = k.num_blocks / conc, tail = k.num_blocks % conc;
  std::vector<std::pair<int64_t, int64_t>> waves;
  if (full) waves.push_back({conc, full});
  if (tail) waves.push_back({tail, 1});

  double total = 0.0;
  double bp = 0, bf = 0, bs = 0, be = 0;
  std::map<std::string, double> lim{{"launch", launch}};
  std::vector<std::string> order{"launch"};   // insertion order: Python max() tie-break
  auto add = [&](const std::string& n, double t) {
    if (t <= 0) return;
    if (!lim.count(n)) order.push_back(n);
    lim[n] += t;
  };
  std::map<std::string, double> det{{"launch", launch}};
  auto addd = [&](const std::string& n, double t) { if (t > 0) det[n] += t; };

  for (auto [blocks, count] : waves) {
    const int active = static_cast<int>(std::min<int64_t>(sms, (blocks + resident - 1) / resident));
    const int64_t bps = (blocks + active - 1) / active;
    auto [rb, lane] = resource_bound(k.body, lanes, active, bps);
    double qf = 1.0;
    if (k.queue_coef > 0) {                             // M/D/1-style latency inflation
      double best_u = 0.0;
      for (size_t r = 0; r < lanes.size(); ++r) {
        if (!lanes[r].shared) continue;
        double s2 = 0.0;
        for (const auto& a : k.body) s2 += u(a, lanes[r], r, active);
        best_u = std::max(best_u, s2 * static_cast<double>(bps) / std::max(rb, 1e-18));
      }
      best_u = std::min(best_u, 0.995);
      qf = std::min(k.queue_max, 1.0 + k.queue_coef * best_u / (1.0 - best_u));
    }
    std::vector<double> w, wst, wrec;
    for (const auto& a : k.body) {
      double x = node_w(a, lanes, active, qf);
      w.push_back(x);                                   // full weights: pipeline fill
      double xs = x - (a.recurrent ? 0.0 : a.latency * qf);  // steady: only loop-carried latency
      wst.push_back(xs);
      wrec.push_back(a.recurrent ? xs : 0.0);
    }
    const double cp = longest_path(k.body, w);
    const double cpst = longest_path(k.body, wst);
    const double cprec = longest_path(k.body, wrec);
    const double latb = std::max(cpst / std::max(1, k.stages), cprec / std::max(1, k.consumers));
    double rnd;
    std::string limiter, dsteady;
    if (k.body.empty()) { rnd = 0.0; limiter = "none"; dsteady = "none"; }
    else if (rb >= latb) { rnd = rb; limiter = lanes[lane].name; dsteady = lane_detail(k.body, lanes, lane, active); }
    else {
      rnd = latb; limiter = "latency";
      const bool use_rec = cprec / std::max(1, k.consumers) > cpst / std::max(1, k.stages);
      dsteady = latency_detail(k.body, use_rec ? wrec : wst, lanes, active);
    }
    Phase pro = phase(k.prologue, lanes, active, bps, qf);
    Phase epi = phase(k.epilogue, lanes, active, bps, qf);
    const double tpro = pro.t, tepi = epi.t;
    const std::string& lpro = pro.lim; const std::string& lepi = epi.lim;
    const double tfill = k.iters > 0 ? std::max(0.0, cp - rnd) : 0.0;
    const double tsteady = static_cast<double>(k.iters) * rnd;
    total += count * (tpro + tfill + tsteady + tepi);
    bp += count * tpro; bf += count * tfill; bs += count * tsteady; be += count * tepi;
    add(limiter, count * tsteady); add("latency", count * tfill);
    add(lpro, count * tpro); add(lepi, count * tepi);
    addd(dsteady, count * tsteady); addd("latency:fill", count * tfill);
    addd(pro.det, count * tpro); addd(epi.det, count * tepi);
  }
  total += launch;

  for (size_t r = 0; r < lanes.size(); ++r) {
    auto wsum = [&](const std::vector<Action>& v) {
      double s = 0; for (const auto& a : v) s += r < a.work.size() ? a.work[r] : 0.0; return s; };
    double per_block = wsum(k.prologue) + static_cast<double>(k.iters) * wsum(k.body) + wsum(k.epilogue);
    double busy = static_cast<double>(k.num_blocks) * per_block;
    busy = lanes[r].shared ? busy / lanes[r].total_rate : busy / sms;
    if (busy > 0) res.util[lanes[r].name] = busy / total;
  }
  res.time = total;
  res.breakdown = {{"prologue", bp}, {"fill", bf}, {"steady", bs}, {"epilogue", be}, {"launch", launch}};
  res.limiter_time = lim;
  res.limiter_detail = det;
  double bestv = -1;
  for (const auto& n : order) if (lim[n] > bestv) { bestv = lim[n]; res.bottleneck = n; }
  res.waves = static_cast<int64_t>(waves.empty() ? 0 : full + (tail ? 1 : 0));
  return res;
}

std::vector<Result> evaluate_batch(const std::vector<Kernel>& ks, const std::vector<Lane>& lanes,
                                   int sms, double launch, int threads) {
  std::vector<Result> out(ks.size());
  threads = std::max(1, std::min<int>(threads, static_cast<int>(ks.size())));
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (size_t i = t; i < ks.size(); i += threads) out[i] = evaluate(ks[i], lanes, sms, launch);
    });
  for (auto& th : pool) th.join();
  return out;
}

}  // namespace tilesight
