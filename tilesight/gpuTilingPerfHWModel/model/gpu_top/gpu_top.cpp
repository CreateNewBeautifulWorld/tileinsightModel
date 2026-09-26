#include "gpu_top.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>

#include "../memory_slice/memory_slice.hpp"
#include "../on_chip_buffer/on_chip_buffer.hpp"

namespace tilesight::gpu_top {

using shader_core::LaneSpec;

// ------------------------------------------------------------------------- small helpers
namespace {

double dtype_bytes(const std::string& dt) {
  static const std::map<std::string, double> T = {
      {"fp4", 0.5}, {"nvfp4", 0.5}, {"mxfp4", 0.5}, {"int4", 0.5}, {"fp6", 0.75}, {"mxfp6", 0.75},
      {"fp8", 1.0}, {"mxfp8", 1.0}, {"int8", 1.0},  {"bf16", 2.0}, {"fp16", 2.0}, {"tf32", 4.0}, {"fp32", 4.0},
  };
  auto it = T.find(dt);
  return it == T.end() ? 2.0 : it->second;
}

std::map<std::string, double> parse_path_split(const std::string& load_path) {
  const std::string prefix = "split:";
  if (load_path.rfind(prefix, 0) != 0) return {{load_path, 1.0}};
  std::map<std::string, double> out;
  double sum = 0.0;
  std::stringstream ss(load_path.substr(prefix.size()));
  std::string part;
  while (std::getline(ss, part, ',')) {
    auto eq = part.find('=');
    std::string k = part.substr(0, eq);
    double v = std::stod(part.substr(eq + 1));
    out[k] = v;
    sum += v;
  }
  if (sum > 0) for (auto& kv : out) kv.second /= sum;
  return out;
}

std::vector<std::pair<int64_t, int64_t>> grouped_raster(int64_t mt, int64_t nt, int64_t group_m) {
  group_m = std::max<int64_t>(1, std::min(group_m, mt));
  std::vector<std::pair<int64_t, int64_t>> out;
  for (int64_t g0 = 0; g0 < mt; g0 += group_m) {
    int64_t rows = std::min(group_m, mt - g0);
    for (int64_t n = 0; n < nt; ++n)
      for (int64_t r = 0; r < rows; ++r) out.push_back({g0 + r, n});
  }
  return out;
}

cache::AddrCfg l2_addr_cfg(const HwView& hw) {
  cache::AddrCfg cfg;
  double slices = hw.get_num("memory.l2.slices", 0.0);
  double dies = hw.get_num("dies", 1.0);
  cfg.ports = slices > 0 ? static_cast<int>(slices) : std::max(1, static_cast<int>(dies) * 8);
  cfg.mode = "interleave";
  cfg.granularity = static_cast<int64_t>(hw.get_num("memory.interleave_KB", 2.0) * 1024);
  cfg.addr_bits = static_cast<int>(hw.get_num("memory.addressing.addr_bits", 48.0));
  nb::object o = hw.get("memory.addressing.l2");
  if (!o.is_none()) {
    nb::dict d = nb::cast<nb::dict>(o);
    if (d.contains("ports")) cfg.ports = nb::cast<int>(d["ports"]);
    if (d.contains("mode")) cfg.mode = nb::cast<std::string>(d["mode"]);
    if (d.contains("granularity_KB")) cfg.granularity = static_cast<int64_t>(nb::cast<double>(d["granularity_KB"]) * 1024);
    if (d.contains("addr_bits")) cfg.addr_bits = nb::cast<int>(d["addr_bits"]);
  }
  return cfg;
}

// gload: one tile load, routed shader-slice -> [L1] -> [buffer over the switch] -> memory slice.
// Mirrors kernels/gemm.py `gload`.
TraceAction gload(const HwView& hw, const std::string& name, double nbytes, double miss,
                  const std::string& load_path, const std::vector<int>& deps,
                  bool has_sram_miss, double sram_miss, bool has_l1_miss, double l1_miss,
                  double slice_frac = 1.0) {
  double l1_m = 1.0;
  std::map<std::string, double> work;
  if (has_l1_miss && hw.l1_capacity_bytes() > 0) {
    l1_m = std::max(0.0, std::min(1.0, l1_miss));
    if (l1_m < 1.0) work["l1"] = hw.l1_time_per_sm(nbytes);
  }
  double after_l1 = nbytes * l1_m;

  bool has_sram = hw.has_sram();
  double from_buffer = 0.0;
  if (has_sram) {
    from_buffer = on_chip_buffer::route_from_buffer(true, miss, has_sram_miss, sram_miss, hw.sram_costream(),
                                                     hw.sram_bandwidth_Bps(), hw.ddr_bandwidth_Bps());
    work["sram"] = after_l1 * from_buffer;
    work["switch"] = after_l1 * from_buffer;
  }
  double to_memory = after_l1 * (1.0 - from_buffer);
  work["l2"] = to_memory;
  work["ddr"] = to_memory * miss;

  double lat = 0.0;
  for (const auto& [path, frac] : parse_path_split(load_path)) {
    std::string real = hw.resolve_path(path);
    std::string key = "path:" + real;
    work[key] = work.count(key) ? work[key] : 0.0;
    work[key] += hw.path_time_per_sm(real, nbytes * frac);
    lat = std::max(lat, hw.path_latency_s(real));
    if (hw.path_is_dma(real)) {
      std::string dest = hw.get_str("memory.dma.destination", "smem");
      if (dest == "bypass") work["l2"] -= to_memory * frac;
      else if (dest == "l2") work["l2"] += to_memory * frac;
    }
    double ipc = hw.path_attr_num(real, "issue_bytes_per_clk", 0.0);
    if (ipc > 0) work["cuda"] = work.count("cuda") ? work["cuda"] : 0.0, work["cuda"] += nbytes * frac / (ipc * hw.clock_hz());
    if (!hw.path_attr_bool(real, "smem_direct", true))
      work["smem"] = (work.count("smem") ? work["smem"] : 0.0) + hw.smem_time_per_sm(nbytes * frac);
  }
  work["l2"] = std::max(0.0, work["l2"]);
  if (slice_frac < 1.0) {
    work["l2"] /= slice_frac;
    work["ddr"] /= slice_frac;
  }

  double buf_lat = has_sram ? hw.sram_latency_ns() * 1e-9 : 0.0;
  bool has_l2 = hw.l2_capacity_bytes() > 0;
  double mem_lat = memory_slice::mem_latency_s(hw.get_num("memory.l2.latency_ns", 0.0),
                                               hw.get_num("memory.ddr.latency_ns", 0.0), miss, has_l2);
  lat += from_buffer * buf_lat + (1.0 - from_buffer) * mem_lat;
  if (has_sram && hw.sram_prefetch()) lat *= std::max(0.0, 1.0 - from_buffer);

  TraceAction a;
  a.name = name;
  a.work = std::move(work);
  a.deps = deps;
  a.latency = lat;
  return a;
}

TraceAction gstore(const std::string& name, double nbytes, const std::vector<int>& deps, double ddr_frac,
                   double sram_frac, double slice_frac = 1.0) {
  TraceAction a;
  a.name = name;
  a.work["l2"] = nbytes;
  a.work["ddr"] = nbytes * ddr_frac;
  if (slice_frac < 1.0) {
    a.work["l2"] /= slice_frac;
    a.work["ddr"] /= slice_frac;
  }
  if (sram_frac != 0.0) a.work["sram"] = nbytes * sram_frac;
  a.deps = deps;
  return a;
}

constexpr int BASE_REGS = 40;

std::string tensor_class(const std::string& name) {
  std::string n = name;
  std::transform(n.begin(), n.end(), n.begin(), ::tolower);
  if (n.find("weight") != std::string::npos) return "weight";
  if (n.find("kv") != std::string::npos || n == "k" || n == "v") return "kv";
  return "act";
}

double resident_frac(const HwView& hw, double footprint_bytes, const std::string& klass) {
  if (!hw.has_sram()) return 0.0;
  nb::object o = hw.get("memory.sram.footprint." + klass);
  bool has_override = !o.is_none();
  double fp_override = has_override ? nb::cast<double>(o) : 0.0;
  double cap = hw.sram_capacity_for(klass);
  return on_chip_buffer::resident_frac(true, has_override, fp_override, cap, footprint_bytes);
}

// L2 and DDR are not one GPU-wide pool: each memory slice owns its own L2 port and HBM channel.
// `memory.dma.mega_tile_KB`, when configured, models the fixed byte granularity of one DMA
// burst: it splits — by definition, not by simulated address — into n_slices equal atoms,
// round-robined one per slice. A transfer needing fewer atoms than there are slices only ever
// reaches that many slices; one needing more finishes some slices' atoms before others but
// reaches every slice it needs at the full per-slice rate. Rounding a transfer up to whole atoms
// is the only waste charged — there is no address arithmetic to alias, unlike two earlier attempts
// at this same problem that derived slice spread from the L2 hit/miss simulation's synthetic
// per-tile address stream (tuned for cache behavior, and it aliases badly on power-of-two tile
// strides vs. the interleave granularity).
// Unconfigured (mega_tile_KB <= 0, the default), falls back to the coarser occupancy proxy: how
// many blocks are concurrently resident vs. how many slices there are — no behavior change for
// any preset that hasn't opted in.
double slice_bw_frac(const HwView& hw, int n_slices, int64_t concurrent_blocks, double nbytes) {
  n_slices = std::max(1, n_slices);
  if (nbytes <= 0) return 1.0;
  double mega = hw.get_num("memory.dma.mega_tile_KB", 0.0) * 1024.0;
  if (mega <= 0) {
    if (n_slices <= 1 || concurrent_blocks <= 0) return 1.0;
    return std::min(1.0, static_cast<double>(concurrent_blocks) / static_cast<double>(n_slices));
  }
  double atom = std::max(1.0, mega / n_slices);
  double atoms = std::max(1.0, std::ceil(nbytes / atom));
  double touched = std::min<double>(n_slices, atoms);
  double padded = atoms * atom;
  return nbytes * touched / (padded * n_slices);
}

}  // namespace

// ------------------------------------------------------------------------- occupancy
std::pair<int, int64_t> reg_estimate(const HwView& hw, int64_t acc_bytes, int threads) {
  int64_t max_rpt = static_cast<int64_t>(hw.get_num("occupancy.max_regs_per_thread", 0.0));
  int64_t acc_regs = hw.has_tmem() ? 0 : -(-acc_bytes / std::max(1, 4 * threads));
  int64_t rpt = BASE_REGS + acc_regs + (hw.has_tmem() ? 16 : 0);
  int64_t spill = std::max<int64_t>(0, rpt - max_rpt) * 4 * threads;
  return {static_cast<int>(std::min(rpt, max_rpt)), spill};
}

std::pair<int, std::string> occupancy(const HwView& hw, int64_t smem_bytes, int64_t acc_bytes, int threads,
                                      int extra_regs) {
  auto [rpt0, spill0] = reg_estimate(hw, acc_bytes, threads);
  (void)spill0;
  int64_t rpt = rpt0 + extra_regs;
  std::vector<std::pair<std::string, int64_t>> lim;
  lim.push_back({"max_blocks", static_cast<int64_t>(hw.get_num("occupancy.max_blocks_per_sm", 0.0))});
  lim.push_back({"smem", hw.smem_per_sm() / std::max<int64_t>(1, smem_bytes)});
  lim.push_back({"threads", static_cast<int64_t>(hw.get_num("occupancy.max_threads_per_sm", 0.0)) / std::max(1, threads)});
  lim.push_back({"regs", static_cast<int64_t>(hw.get_num("occupancy.regs_per_sm", 0.0)) / std::max<int64_t>(1, rpt * threads)});
  if (hw.has_tmem()) lim.push_back({"tmem", hw.tmem_per_sm() / std::max<int64_t>(1, acc_bytes)});
  int64_t val = lim[0].second;
  for (auto& kv : lim) val = std::min(val, kv.second);
  std::string name;
  for (auto& kv : lim)
    if (kv.second == val) { if (!name.empty()) name += "+"; name += kv.first; }
  return {static_cast<int>(std::max<int64_t>(0, val)), name};
}

// ------------------------------------------------------------------------- GEMM
std::optional<std::vector<LoweredKernel>> lower_gemm(const HwView& hw, const std::string& name, int64_t M,
                                                      int64_t N, int64_t K, int64_t batch,
                                                      const std::string& a_dt, const std::string& b_dt,
                                                      const std::string& c_dt, const std::string& comp_dt,
                                                      const GemmTile& t, const std::string& an,
                                                      const std::string& bn_, const std::string& cn) {
  if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) return std::vector<LoweredKernel>{};
  double ab = dtype_bytes(a_dt), bb = dtype_bytes(b_dt), cb = dtype_bytes(c_dt);
  int64_t mt = (M + t.bm - 1) / t.bm, nt = (N + t.bn - 1) / t.bn, kt = (K + t.bk - 1) / t.bk;
  int sk = std::max(1, std::min(t.split_k, static_cast<int>(kt)));
  int64_t iters = (kt + sk - 1) / sk;
  double a_tile = t.bm * t.bk * ab, b_tile = t.bn * t.bk * bb;
  int cl = std::max(1, t.cluster_m);
  if (cl > 1 && (mt < cl || !hw.get_bool("compute.cluster_multicast", false) ||
                (t.cta_pair && (cl != 2 || !hw.get_bool("compute.cta_pair", false)))))
    return std::nullopt;
  double stage = a_tile + (t.cta_pair ? b_tile / cl : b_tile);
  double epi_smem = t.bm * t.bn * cb;
  int stages = t.stages ? t.stages
                        : static_cast<int>(std::min<double>(8, (hw.smem_per_sm() - epi_smem) / std::max(1.0, stage)));
  if (stages < 2 || stages * stage + epi_smem > hw.smem_per_sm()) return std::nullopt;
  double acc = t.bm * t.bn * 4.0;
  int threads = 128 * (1 + (t.bm >= 128 ? 2 : 1));
  auto paths = parse_path_split(t.load_path);
  if (cl > 1)
    for (auto& [p, _] : paths)
      if (!hw.path_attr_bool(p, "multicast", false)) return std::nullopt;
  double extra_regs = 0;
  for (auto& [p, _] : paths) extra_regs = std::max(extra_regs, hw.path_attr_num(p, "regs_per_thread", 0.0));
  auto [resident, occ_lim] = occupancy(hw, static_cast<int64_t>(stages * stage + epi_smem),
                                       static_cast<int64_t>(acc), threads, static_cast<int>(extra_regs));
  if (resident < 1) return std::nullopt;
  auto [rpt, spill] = reg_estimate(hw, static_cast<int64_t>(acc), threads - 128);
  int64_t blocks = batch * mt * nt * sk;
  int64_t concurrent_blocks = std::min<int64_t>(blocks, static_cast<int64_t>(hw.sms()) * resident);

  double share_a = 1.0, share_b = 1.0;
  bool has_stage_cfg = hw.has_sram() && !hw.get("memory.sram.stage").is_none();
  if (has_stage_cfg) {
    double share = hw.sram_stage_share_blocks();
    share_b = std::max(1.0, std::min(share, static_cast<double>(mt)));
    share_a = std::max(1.0, std::min(share, static_cast<double>(nt)));
  }

  // ---- L2 model: deterministic tile-level simulation over several waves --------------
  int64_t conc = static_cast<int64_t>(hw.sms()) * resident;
  auto order_all = grouped_raster(mt, nt, t.swizzle);
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> full_order;  // (b,s,m,n)
  for (int64_t b_ = 0; b_ < batch; ++b_)
    for (int s_ = 0; s_ < sk; ++s_)
      for (auto [m, n] : order_all) full_order.push_back({b_, s_, m, n});
  int64_t n_waves = std::min<int64_t>(static_cast<int64_t>(hw.get_num("memory.l2.waves_simulated", 1.0)),
                                      std::max<int64_t>(1, (static_cast<int64_t>(full_order.size()) + conc - 1) / std::max<int64_t>(1, conc)));
  int64_t n_take = std::min<int64_t>(static_cast<int64_t>(full_order.size()), conc * n_waves);
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> order(full_order.begin(), full_order.begin() + n_take);
  double a_base = 0, b_base = M * K * ab * batch;
  std::vector<int64_t> keys, addrs;
  std::vector<double> sizes;
  std::vector<int> streams;
  int ksteps = static_cast<int>(std::min<int64_t>(iters, 4));
  for (int64_t w = 0; w < n_waves; ++w) {
    int64_t lo = w * conc, hi = std::min<int64_t>(order.size(), (w + 1) * conc);
    for (int kk = 0; kk < ksteps; ++kk) {
      for (int64_t idx = lo; idx < hi; ++idx) {
        auto [b_, s_, m, n] = order[idx];
        int64_t k = s_ * iters + kk;
        keys.push_back(((b_ * 4096 + m) * 65536 + k) * 2);
        addrs.push_back(static_cast<int64_t>(a_base + ((b_ * mt + m) * kt + k) * a_tile));
        sizes.push_back(a_tile);
        streams.push_back(0);
        keys.push_back(((b_ * 4096 + n) * 65536 + k) * 2 + 1);
        addrs.push_back(static_cast<int64_t>(b_base + ((b_ * kt + k) * nt + n) * b_tile));
        sizes.push_back(b_tile);
        streams.push_back(1);
      }
    }
  }
  cache::AddrCfg lmap = l2_addr_cfg(hw);
  auto slice_frac = [&](double nbytes) { return slice_bw_frac(hw, lmap.ports, concurrent_blocks, nbytes); };
  int n_part = static_cast<int>(hw.get_num("memory.l2.partitions", 1.0));
  std::string policy = hw.get_str("memory.l2.policy", "lru");
  bool through_l1 = false;
  for (auto& [p, _] : paths)
    if (!hw.path_attr_bool(p, "smem_direct", true)) through_l1 = true;
  cache::SimResult sim = memory_slice::simulate_l2(keys, addrs, sizes, streams, 2, hw.l2_capacity_bytes(),
                                                    n_part, lmap, policy);
  auto mf = sim.miss_fraction();
  double fa = mf[0], fb = mf[1];

  std::optional<cache::SimResult> l1_sim;
  double l1_miss_a = 0, l1_miss_b = 0;
  bool has_l1_miss = false;
  if (through_l1 && hw.l1_capacity_bytes() > 0 && !keys.empty()) {
    int64_t share = std::max<int64_t>(1, resident);
    if (hw.get_str("memory.l1.owner", "sm") == "cluster")
      share *= static_cast<int64_t>(hw.get_num("memory.l1.cluster_size", 1.0));
    int64_t blocks_in_wave = static_cast<int64_t>(order.size()) / std::max<int64_t>(1, n_waves);
    int64_t step = std::max<int64_t>(1, blocks_in_wave / share);
    std::vector<int64_t> sk_, sa_;
    std::vector<double> ss_;
    std::vector<int> st_;
    for (size_t i = 0; i < keys.size(); ++i)
      if ((static_cast<int64_t>(i) / 2) % step == 0) { sk_.push_back(keys[i]); sa_.push_back(addrs[i]); ss_.push_back(sizes[i]); st_.push_back(streams[i]); }
    if (!sk_.empty()) {
      cache::AddrCfg zero_cfg;
      zero_cfg.mode = "interleave";
      zero_cfg.ports = 1;
      zero_cfg.granularity = 1;
      std::vector<int> part1(sa_.size(), 0);
      cache::SimResult l1r = cache::simulate(sk_, sa_, ss_, st_, 2, hw.l1_capacity_bytes(), 1, part1,
                                             hw.get_str("memory.l1.policy", "lru"));
      auto l1mf = l1r.miss_fraction();
      l1_miss_a = l1mf[0];
      l1_miss_b = l1mf[1];
      has_l1_miss = true;
    }
  }

  bool has_sram = hw.has_sram();
  bool has_sa = false, has_sb = false;
  double sa = 0, sb = 0;
  if (has_sram) {
    // Same principle as attention's KV: capacity competes with whatever's concurrently resident
    // right now (stages-deep x one operand tile x concurrently-active blocks), not the whole
    // A/B matrix. Reuse across several M/N-tiles from that one resident copy is share_a/share_b's
    // job (an already-computed, searched staging factor) applied separately below — it says how
    // many times a resident tile gets reused, not how big the resident footprint is.
    double a_active = static_cast<double>(stages) * a_tile * static_cast<double>(concurrent_blocks);
    double b_active = static_cast<double>(stages) * b_tile * static_cast<double>(concurrent_blocks);
    double ra = resident_frac(hw, a_active, tensor_class(an));
    double rb_ = resident_frac(hw, b_active, tensor_class(bn_));
    sa = fa * (1 - ra) / share_a;
    sb = fb * (1 - rb_) / share_b;
    has_sa = has_sb = true;
  }

  double bm_c = std::max<double>(t.bm, hw.tc_min_m());
  auto [mma_lane, mma_t] = hw.mma_cost(2.0 * bm_c * t.bn * t.bk, comp_dt);

  std::vector<TraceAction> body;
  body.push_back(gload(hw, "load:" + an, a_tile, fa, t.load_path, {}, has_sa, sa, has_l1_miss, l1_miss_a, slice_frac(a_tile)));
  body.push_back(gload(hw, "load:" + bn_, b_tile / cl, fb, t.load_path, {}, has_sb, sb, has_l1_miss, l1_miss_b, slice_frac(b_tile / cl)));
  {
    TraceAction mma;
    mma.name = "mma";
    mma.work[mma_lane] = mma_t;
    mma.work["smem"] = hw.smem_time_per_sm(a_tile + (t.cta_pair ? b_tile / cl : b_tile));
    mma.deps = {0, 1};
    mma.latency = hw.unit_latency_s(mma_lane) + hw.unit_latency_s("smem");
    body.push_back(mma);
  }
  if (spill) {
    TraceAction sp;
    sp.name = "spill:regs";
    sp.work["l2"] = 2.0 * spill / std::max<int64_t>(1, iters) / slice_frac(a_tile);
    sp.deps = {2};
    body.push_back(sp);
  }
  double out_bytes = t.bm * t.bn * (sk > 1 ? 4.0 : cb);
  std::vector<TraceAction> epilogue;
  {
    TraceAction acc_a;
    acc_a.name = hw.has_tmem() ? "acc_read" : "acc_regs";
    acc_a.work["tmem"] = hw.tmem_time_per_sm(acc, 0.0);
    acc_a.work["cuda"] = hw.cuda_time_per_sm(2.0 * t.bm * t.bn);
    acc_a.latency = hw.unit_latency_s(hw.has_tmem() ? "tmem" : "cuda");
    epilogue.push_back(acc_a);
    epilogue.push_back(gstore(sk == 1 ? "store:" + cn : "store:partials", out_bytes, {0}, 1.0, 0.0, slice_frac(out_bytes)));
  }

  double useful = 2.0 * M * N * K * batch;
  double pad_eff = useful / (2.0 * batch * mt * t.bm * nt * t.bn * kt * t.bk);
  std::ostringstream tile_s;
  tile_s << t.bm << "x" << t.bn << "x" << t.bk << "/s" << stages;
  if (sk > 1) tile_s << "/sk" << sk;
  if (cl > 1) tile_s << "/c" << cl << (t.cta_pair ? "p" : "");

  TraceKernel k;
  k.name = name;
  k.kind = "gemm";
  k.num_blocks = blocks;
  k.iters = iters;
  k.stages = stages;
  k.resident = resident;
  k.body = body;
  k.epilogue = epilogue;
  k.queue_coef = hw.get_num("memory.queueing.coef", 0.0);
  k.queue_max = hw.get_num("memory.queueing.max_factor", 3.0);

  nb::dict meta;
  meta["M"] = M; meta["N"] = N; meta["K"] = K; meta["batch"] = batch;
  meta["tile"] = tile_s.str(); meta["stages"] = stages; meta["resident"] = resident;
  meta["flops"] = useful; meta["pad_eff"] = pad_eff; meta["weight_bytes"] = K * N * bb * batch;
  meta["l2_miss_A"] = fa; meta["l2_miss_B"] = fb;
  meta["l2_hit_rate"] = sim.hit_rate; meta["l2_partitions"] = n_part; meta["l2_policy"] = policy;
  meta["dtype"] = a_dt + "x" + b_dt + "->" + comp_dt;
  meta["occ_limiter"] = occ_lim; meta["regs_per_thread"] = rpt; meta["reg_spill_bytes"] = spill;
  meta["smem_per_block"] = static_cast<int64_t>(stages * stage + epi_smem);
  meta["l2_waves_simulated"] = n_waves;
  meta["stage_share"] = std::make_tuple(share_a, share_b);
  meta["l1_capacity_KB"] = hw.l1_capacity_bytes() / 1024.0;
  meta["l1_miss"] = has_l1_miss ? nb::cast(std::vector<double>{l1_miss_a, l1_miss_b}) : nb::none();
  meta["l2_partition_tiles"] = sim.partition_tiles;
  meta["l2_partition_bytes"] = sim.partition_bytes;
  meta["l2_partition_capacity"] = sim.partition_capacity;
  meta["l2_partition_evictions"] = sim.partition_evictions;

  std::vector<LoweredKernel> out;
  out.push_back({std::move(k), std::move(meta)});
  if (sk > 1) {
    auto ew = lower_elementwise(hw, name + ".splitk_reduce", sk * M * N * 4.0 * batch, M * N * cb * batch,
                                sk * M * N * batch, 0.0, "elementwise", 32 * 1024, "partials", cn, std::nullopt);
    for (auto& e : ew) out.push_back(std::move(e));
  }
  return out;
}

// ------------------------------------------------------------------------- elementwise
std::vector<LoweredKernel> lower_elementwise(const HwView& hw, const std::string& name, double bytes_in,
                                             double bytes_out, double flops, double sfu_ops,
                                             const std::string& kind, int64_t chunk,
                                             const std::string& in_name, const std::string& out_name,
                                             std::optional<bool> on_chip) {
  if (bytes_in + bytes_out <= 0) return {};
  int64_t blocks = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::max(bytes_in, bytes_out) / chunk)));
  double per = 1.0 / blocks;
  bool oc = on_chip.has_value() ? *on_chip : hw.sram_keeps_intermediates();
  // Same fix as GEMM/attention: an elementwise op streams each chunk once (no reuse), so the
  // buffer only ever holds the chunks concurrently in flight right now, not the whole tensor.
  // resident=8 is the fixed occupancy this lowering uses below (no real occupancy() call here).
  const int64_t elementwise_resident = 8;
  int64_t concurrent_blocks = std::min<int64_t>(blocks, static_cast<int64_t>(hw.sms()) * elementwise_resident);
  double active_bytes = std::max(bytes_in, bytes_out) * per * static_cast<double>(concurrent_blocks);
  double res = oc ? resident_frac(hw, active_bytes, "act") : 0.0;
  int n_slices = l2_addr_cfg(hw).ports;
  std::vector<TraceAction> body;
  body.push_back(gload(hw, "load:" + in_name, bytes_in * per, 1.0, "lsu", {}, true, 1.0 - res, false, 0.0,
                       slice_bw_frac(hw, n_slices, concurrent_blocks, bytes_in * per)));
  TraceAction comp;
  comp.name = "compute";
  comp.work["cuda"] = hw.cuda_time_per_sm(flops * per);
  comp.work["sfu"] = hw.sfu_time_per_sm(sfu_ops * per);
  comp.deps = {0};
  comp.latency = hw.unit_latency_s(sfu_ops > 0 ? "sfu" : "cuda");
  body.push_back(comp);
  body.push_back(gstore("store:" + out_name, bytes_out * per, {1}, 1.0 - res, hw.has_sram() ? res : 0.0,
                        slice_bw_frac(hw, n_slices, concurrent_blocks, bytes_out * per)));

  TraceKernel k;
  k.name = name;
  k.kind = kind;
  k.num_blocks = blocks;
  k.iters = 1;
  k.stages = 1;
  k.resident = elementwise_resident;
  k.body = body;
  nb::dict meta;
  meta["bytes"] = bytes_in + bytes_out;
  meta["flops"] = flops;
  meta["occ_limiter"] = std::string("-");
  meta["on_chip_frac"] = res;
  return {LoweredKernel{std::move(k), std::move(meta)}};
}

namespace {

TraceAction softmax_action(const HwView& hw, double rows, double cols, const std::vector<int>& deps) {
  double s_bytes = rows * cols * 4.0;
  TraceAction a;
  a.name = "softmax";
  a.work["sfu"] = hw.sfu_time_per_sm(rows * cols);
  a.work["cuda"] = hw.cuda_time_per_sm(6.0 * rows * cols);
  a.work["tmem"] = hw.tmem_time_per_sm(s_bytes, s_bytes * 0.5);
  a.deps = deps;
  a.recurrent = true;
  a.latency = hw.unit_latency_s("sfu") + hw.unit_latency_s("tmem");
  return a;
}

}  // namespace

// ------------------------------------------------------------------------- attention decode
std::optional<std::vector<LoweredKernel>> lower_attention_decode(
    const HwView& hw, const std::string& name, int64_t B, int64_t H, int64_t kv_heads, int64_t S,
    int64_t d_qk, int64_t d_v, const std::string& kv_dtype, const std::string& compute_dtype, bool v_in_k,
    const AttnTile& tile) {
  double kvb = dtype_bytes(kv_dtype);
  int64_t qpk = H / std::max<int64_t>(1, kv_heads);
  int64_t hb = std::min<int64_t>(qpk, tile.block_m);
  int64_t groups = (qpk + hb - 1) / std::max<int64_t>(1, hb);
  int64_t units = B * kv_heads * groups;
  int64_t ntiles = (S + tile.block_n - 1) / tile.block_n;
  double k_tile = tile.block_n * d_qk * kvb;
  double v_tile = v_in_k ? 0.0 : tile.block_n * d_v * kvb;
  double smem = tile.stages * (k_tile + v_tile) + hb * d_qk * 2.0;
  if (smem > hw.smem_per_sm()) return std::nullopt;
  int threads = 128 * (1 + tile.consumers);
  auto [resident, occ_lim] = occupancy(hw, static_cast<int64_t>(smem),
                                       static_cast<int64_t>(hb * (d_v + tile.block_n) * 4), threads);
  if (resident < 1) return std::nullopt;
  auto [rpt, spill] = reg_estimate(hw, static_cast<int64_t>(hb * (d_v + tile.block_n) * 4), threads - 128);
  int64_t splits = tile.num_splits ? tile.num_splits
                                   : std::max<int64_t>(1, std::min<int64_t>(ntiles, (static_cast<int64_t>(hw.sms()) * resident + units - 1) / std::max<int64_t>(1, units)));
  int64_t iters = (ntiles + splits - 1) / splits;
  int64_t blocks = units * splits;
  int64_t concurrent_blocks = std::min<int64_t>(blocks, static_cast<int64_t>(hw.sms()) * resident);

  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> order;
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < kv_heads; ++h)
      for (int64_t s = 0; s < splits; ++s)
        for (int64_t g = 0; g < groups; ++g) order.push_back({b, h, g, s});
  int64_t cap = static_cast<int64_t>(hw.sms()) * resident;
  if (static_cast<int64_t>(order.size()) > cap) order.resize(cap);
  std::vector<int64_t> keys, addrs;
  std::vector<double> sizes;
  std::vector<int> streams;
  double tile_b = k_tile + v_tile;
  int itn = static_cast<int>(std::min<int64_t>(iters, 4));
  for (int it = 0; it < itn; ++it)
    for (auto [b, h, g, s] : order) {
      (void)g;
      keys.push_back((((b * 256 + h) * 4096 + s) * 4096 + it));
      streams.push_back(0);
    }
  for (size_t i = 0; i < keys.size(); ++i) { addrs.push_back(static_cast<int64_t>(i * tile_b)); sizes.push_back(tile_b); }
  cache::AddrCfg lmap = l2_addr_cfg(hw);
  int n_part = static_cast<int>(hw.get_num("memory.l2.partitions", 1.0));
  cache::SimResult sim = memory_slice::simulate_l2(keys, addrs, sizes, streams, 1, hw.l2_capacity_bytes(),
                                                    n_part, lmap, hw.get_str("memory.l2.policy", "lru"));
  double f = sim.miss_fraction()[0];
  bool has_fs = false;
  double fs = 0.0;
  if (hw.has_sram()) {
    // The buffer isn't a cache and doesn't hold the whole layer's KV at once — it's a
    // deterministic capacity share of whatever's actively streaming through the pipeline right
    // now: `stages`-deep multi-buffering x one tile's K/V bytes x however many blocks are
    // concurrently resident GPU-wide. Comparing that (not B*kv_heads*S, the entire sequence)
    // against sram_capacity_for("kv") is what "only the part currently needed" means.
    double kv_active = static_cast<double>(tile.stages) * (k_tile + v_tile) * static_cast<double>(concurrent_blocks);
    fs = f * (1 - resident_frac(hw, kv_active, "kv"));
    has_fs = true;
  }

  double hb_c = std::max<double>(hb, hw.tc_min_m());
  auto [qk_lane, qk_t] = hw.mma_cost(2.0 * hb_c * d_qk * tile.block_n, compute_dtype);
  auto [pv_lane, pv_t] = hw.mma_cost(2.0 * hb_c * tile.block_n * d_v, compute_dtype);

  std::vector<TraceAction> body;
  body.push_back(gload(hw, v_in_k ? "load:kv_cache(latent)" : "load:kv_cache(K)", k_tile, f, "tma", {}, has_fs, fs,
                       false, 0.0, slice_bw_frac(hw, lmap.ports, concurrent_blocks, k_tile)));
  int iv = 0;
  if (!v_in_k) {
    body.push_back(gload(hw, "load:kv_cache(V)", v_tile, f, "tma", {}, has_fs, fs, false, 0.0,
                         slice_bw_frac(hw, lmap.ports, concurrent_blocks, v_tile)));
    iv = 1;
  }
  {
    TraceAction qk;
    qk.name = "gemm_qk";
    qk.work[qk_lane] = qk_t;
    qk.work["smem"] = hw.smem_time_per_sm(k_tile + hb * d_qk * 2.0);
    qk.deps = {0};
    qk.recurrent = true;
    qk.latency = hw.unit_latency_s(qk_lane) + hw.unit_latency_s("smem");
    body.push_back(qk);
  }
  body.push_back(softmax_action(hw, static_cast<double>(hb), static_cast<double>(tile.block_n),
                                {static_cast<int>(body.size()) - 1}));
  {
    TraceAction pv;
    pv.name = "gemm_pv";
    pv.work[pv_lane] = pv_t;
    pv.work["smem"] = hw.smem_time_per_sm(tile.block_n * d_v * kvb);
    pv.deps = {static_cast<int>(body.size()) - 1, iv};
    pv.recurrent = true;
    pv.latency = hw.unit_latency_s(pv_lane) + hw.unit_latency_s("smem");
    body.push_back(pv);
  }
  if (spill) {
    TraceAction sp;
    sp.name = "spill:regs";
    sp.work["l2"] = 2.0 * spill / std::max<int64_t>(1, iters);
    sp.deps = {static_cast<int>(body.size()) - 1};
    body.push_back(sp);
  }
  std::vector<TraceAction> prologue{gload(hw, "load:q", hb * d_qk * 2.0, 1.0, "tma", {}, false, 0.0, false, 0.0)};
  double out_b = splits > 1 ? 4.0 : 2.0;
  std::vector<TraceAction> epilogue;
  {
    TraceAction o_read;
    o_read.name = "o_read";
    o_read.work["tmem"] = hw.tmem_time_per_sm(hb * d_v * 4.0, 0.0);
    epilogue.push_back(o_read);
    epilogue.push_back(gstore(splits == 1 ? "store:out" : "store:partials", hb * d_v * out_b, {0}, 1.0, 0.0));
  }

  double kv_bytes = B * kv_heads * S * (d_qk + (v_in_k ? 0 : d_v)) * kvb;
  TraceKernel k;
  k.name = name;
  k.kind = "attention";
  k.num_blocks = blocks;
  k.iters = iters;
  k.stages = tile.stages;
  k.resident = resident;
  k.consumers = tile.consumers;
  k.body = body;
  k.prologue = prologue;
  k.epilogue = epilogue;
  k.queue_coef = hw.get_num("memory.queueing.coef", 0.0);
  k.queue_max = hw.get_num("memory.queueing.max_factor", 3.0);

  nb::dict meta;
  meta["B"] = B; meta["H"] = H; meta["S"] = S;
  std::ostringstream tile_s;
  tile_s << "hb" << hb << "x" << tile.block_n << "/s" << tile.stages << "/sp" << splits;
  meta["tile"] = tile_s.str(); meta["resident"] = resident; meta["occ_limiter"] = occ_lim;
  meta["regs_per_thread"] = rpt; meta["reg_spill_bytes"] = spill;
  meta["smem_per_block"] = static_cast<int64_t>(smem);
  meta["l2_hit_rate"] = sim.hit_rate; meta["l2_partitions"] = n_part;
  meta["flops"] = 2.0 * B * H * S * (d_qk + d_v); meta["kv_bytes"] = kv_bytes; meta["l2_miss_KV"] = f;
  meta["splits"] = splits;

  std::vector<LoweredKernel> out;
  out.push_back({std::move(k), std::move(meta)});
  if (splits > 1) {
    auto ew = lower_elementwise(hw, name + ".combine", B * H * splits * (d_v + 1) * 4.0, B * H * d_v * 2.0,
                                3.0 * B * H * splits * d_v, static_cast<double>(B * H * splits), "elementwise",
                                32 * 1024, "partials", "out", std::nullopt);
    for (auto& e : ew) out.push_back(std::move(e));
  }
  return out;
}

// ------------------------------------------------------------------------- attention prefill
std::optional<std::vector<LoweredKernel>> lower_attention_prefill(
    const HwView& hw, const std::string& name, int64_t B, int64_t H, int64_t kv_heads, int64_t S,
    int64_t d_qk, int64_t d_v, const std::string& kv_dtype, const std::string& compute_dtype, bool causal,
    int64_t window, const AttnTile& tile) {
  double kvb = dtype_bytes(kv_dtype);
  int64_t bm = tile.block_m, bn = tile.block_n;
  int64_t qt = (S + bm - 1) / bm;
  std::vector<int64_t> iters_list;
  int64_t sum_iters = 0;
  for (int64_t i = 0; i < qt; ++i) {
    int64_t hi = causal ? std::min<int64_t>(S, (i + 1) * bm) : S;
    int64_t lo = window > 0 ? std::max<int64_t>(0, i * bm - window + 1) : 0;
    int64_t it = std::max<int64_t>(1, (hi + bn - 1) / bn - lo / bn);
    iters_list.push_back(it);
    sum_iters += it;
  }
  int64_t iters = std::max<int64_t>(1, std::llround(static_cast<double>(sum_iters) / std::max<int64_t>(1, qt)));
  double k_tile = bn * d_qk * kvb, v_tile = bn * d_v * kvb, q_tile = bm * d_qk * 2.0;
  double smem = tile.stages * (k_tile + v_tile) + q_tile;
  if (smem > hw.smem_per_sm()) return std::nullopt;
  int threads = 128 * (1 + tile.consumers);
  auto [resident, occ_lim] = occupancy(hw, static_cast<int64_t>(smem), static_cast<int64_t>(bm * (d_v + bn) * 4), threads);
  if (resident < 1) return std::nullopt;
  auto [rpt, spill] = reg_estimate(hw, static_cast<int64_t>(bm * (d_v + bn) * 4), threads - 128);
  int64_t blocks = B * H * qt;
  int64_t concurrent_blocks = std::min<int64_t>(blocks, static_cast<int64_t>(hw.sms()) * resident);

  std::vector<std::tuple<int64_t, int64_t, int64_t>> order;
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t q = 0; q < qt; ++q) order.push_back({b, h, q});
  int64_t cap = static_cast<int64_t>(hw.sms()) * resident;
  if (static_cast<int64_t>(order.size()) > cap) order.resize(cap);
  int64_t qpk = std::max<int64_t>(1, H / std::max<int64_t>(1, kv_heads));
  std::vector<int64_t> keys, addrs;
  std::vector<double> sizes;
  std::vector<int> streams;
  int itn = static_cast<int>(std::min<int64_t>(iters, 4));
  for (int it = 0; it < itn; ++it)
    for (auto [b, h, q] : order) {
      if (it < iters_list[q]) {
        keys.push_back((((b * 256 + h / qpk) * 65536 + it) * 2));
        streams.push_back(0);
        keys.push_back((((b * 256 + h / qpk) * 65536 + it) * 2 + 1));
        streams.push_back(0);
      }
    }
  double tb = (k_tile + v_tile) / 2.0;
  for (size_t i = 0; i < keys.size(); ++i) { addrs.push_back(static_cast<int64_t>(i * tb)); sizes.push_back(tb); }
  cache::AddrCfg lmap = l2_addr_cfg(hw);
  int n_part = static_cast<int>(hw.get_num("memory.l2.partitions", 1.0));
  cache::SimResult sim = memory_slice::simulate_l2(keys, addrs, sizes, streams, 1, hw.l2_capacity_bytes(),
                                                    n_part, lmap, hw.get_str("memory.l2.policy", "lru"));
  double f = sim.miss_fraction()[0];
  bool has_fs = false;
  double fs = 0.0;
  if (hw.has_sram()) {
    // Same reasoning as lower_attention_decode: the buffer holds whatever's actively streaming
    // right now (stages-deep x one K/V tile x concurrently-resident blocks), not the whole
    // sequence's KV.
    double kv_active = static_cast<double>(tile.stages) * (k_tile + v_tile) * static_cast<double>(concurrent_blocks);
    fs = f * (1 - resident_frac(hw, kv_active, "kv"));
    has_fs = true;
  }

  double bm_c = std::max<double>(bm, hw.tc_min_m());
  auto [qk_lane, qk_t] = hw.mma_cost(2.0 * bm_c * bn * d_qk, compute_dtype);
  auto [pv_lane, pv_t] = hw.mma_cost(2.0 * bm_c * bn * d_v, compute_dtype);

  std::vector<TraceAction> body;
  body.push_back(gload(hw, "load:kv_cache(K)", k_tile, f, "tma", {}, has_fs, fs, false, 0.0,
                       slice_bw_frac(hw, lmap.ports, concurrent_blocks, k_tile)));
  body.push_back(gload(hw, "load:kv_cache(V)", v_tile, f, "tma", {}, has_fs, fs, false, 0.0,
                       slice_bw_frac(hw, lmap.ports, concurrent_blocks, v_tile)));
  {
    TraceAction qk;
    qk.name = "gemm_qk";
    qk.work[qk_lane] = qk_t;
    qk.work["smem"] = hw.smem_time_per_sm(k_tile + q_tile);
    qk.deps = {0};
    qk.recurrent = true;
    qk.latency = hw.unit_latency_s(qk_lane) + hw.unit_latency_s("smem");
    body.push_back(qk);
  }
  body.push_back(softmax_action(hw, static_cast<double>(bm), static_cast<double>(bn), {2}));
  {
    TraceAction pv;
    pv.name = "gemm_pv";
    pv.work[pv_lane] = pv_t;
    pv.work["smem"] = hw.smem_time_per_sm(v_tile);
    pv.deps = {3, 1};
    pv.recurrent = true;
    pv.latency = hw.unit_latency_s(pv_lane) + hw.unit_latency_s("smem");
    body.push_back(pv);
  }
  if (spill) {
    TraceAction sp;
    sp.name = "spill:regs";
    sp.work["l2"] = 2.0 * spill / std::max<int64_t>(1, iters);
    sp.deps = {4};
    body.push_back(sp);
  }
  std::vector<TraceAction> prologue{gload(hw, "load:q", q_tile, 1.0, "tma", {}, false, 0.0, false, 0.0)};
  std::vector<TraceAction> epilogue;
  {
    TraceAction o_read;
    o_read.name = "o_read";
    o_read.work["tmem"] = hw.tmem_time_per_sm(bm * d_v * 4.0, 0.0);
    epilogue.push_back(o_read);
    epilogue.push_back(gstore("store:out", bm * d_v * 2.0, {0}, 1.0, 0.0));
  }

  int64_t pairs = 0;
  for (int64_t i = 0; i < qt; ++i) {
    int64_t rows = std::min<int64_t>(bm, S - i * bm);
    int64_t hi = causal ? std::min<int64_t>(S, (i + 1) * bm) : S;
    int64_t lo = window > 0 ? std::max<int64_t>(0, i * bm - window + 1) : 0;
    pairs += rows * (hi - lo) - (causal ? rows * (rows - 1) / 2 : 0);
  }

  TraceKernel k;
  k.name = name;
  k.kind = "attention";
  k.num_blocks = blocks;
  k.iters = iters;
  k.stages = tile.stages;
  k.resident = resident;
  k.consumers = tile.consumers;
  k.body = body;
  k.prologue = prologue;
  k.epilogue = epilogue;
  k.queue_coef = hw.get_num("memory.queueing.coef", 0.0);
  k.queue_max = hw.get_num("memory.queueing.max_factor", 3.0);

  nb::dict meta;
  meta["B"] = B; meta["H"] = H; meta["S"] = S;
  meta["resident"] = resident; meta["occ_limiter"] = occ_lim;
  meta["regs_per_thread"] = rpt; meta["reg_spill_bytes"] = spill;
  meta["smem_per_block"] = static_cast<int64_t>(smem);
  meta["l2_hit_rate"] = sim.hit_rate; meta["l2_partitions"] = n_part;
  meta["flops"] = 2.0 * B * H * pairs * (d_qk + d_v); meta["l2_miss_KV"] = f;
  std::ostringstream tile_s2;
  tile_s2 << bm << "x" << bn << "/s" << tile.stages << "/sp" << tile.num_splits << "/c" << tile.consumers;
  meta["tile"] = tile_s2.str();

  return std::vector<LoweredKernel>{LoweredKernel{std::move(k), std::move(meta)}};
}

// ------------------------------------------------------------------------- comm (alpha-beta)
namespace {
std::pair<double, double> link(const HwView& hw, int group) {
  double dom = hw.get_num("network.nvlink.domain_size", 1.0);
  if (group <= dom)
    return {hw.get_num("network.nvlink.alpha_us", 0.0) * 1e-6, hw.get_num("network.nvlink.bandwidth_GBps", 1.0) * 1e9};
  return {hw.get_num("network.scaleout.alpha_us", 0.0) * 1e-6, hw.get_num("network.scaleout.bandwidth_GBps", 1.0) * 1e9};
}

std::pair<double, std::string> flat_time(double a, double bw, double nbytes, int group, const std::string& algo) {
  if (group <= 1) return {0.0, "none"};
  double ring = 2 * (group - 1) * a + 2.0 * (group - 1) / group * nbytes / bw;
  double rd = std::ceil(std::log2(static_cast<double>(group))) * (a + nbytes / bw);
  if (algo == "auto") return ring <= rd ? std::make_pair(ring, std::string("ring"))
                                        : std::make_pair(rd, std::string("recursive_doubling"));
  return algo == "ring" ? std::make_pair(ring, std::string("ring")) : std::make_pair(rd, std::string("recursive_doubling"));
}

LoweredKernel comm_kernel(const std::string& name, double fixed_time, nb::dict meta) {
  TraceKernel k;
  k.name = name;
  k.kind = "comm";
  k.stages = 1;
  k.resident = 1;
  k.fixed = true;
  k.fixed_time = fixed_time;
  return {std::move(k), std::move(meta)};
}
}  // namespace

std::vector<LoweredKernel> allreduce(const HwView& hw, const std::string& name, double nbytes, int group,
                                     const std::string& algo) {
  if (group <= 1 || nbytes <= 0) return {};
  int d = static_cast<int>(hw.get_num("network.nvlink.domain_size", 1.0));
  if (group <= d) {
    auto [a, bw] = link(hw, group);
    auto [t, used] = flat_time(a, bw, nbytes, group, algo);
    nb::dict meta;
    meta["bytes"] = nbytes; meta["group"] = group; meta["algo"] = used;
    return {comm_kernel(name, t, std::move(meta))};
  }
  int nodes = (group + d - 1) / d;
  auto [ai, bwi] = link(hw, d);
  auto [ao, bwo] = link(hw, group);
  double rs = (d - 1) * ai + static_cast<double>(d - 1) / d * nbytes / bwi;
  auto [inter, _u] = flat_time(ao, bwo, nbytes / d, nodes, "auto");
  double ag = rs;
  nb::dict meta;
  meta["bytes"] = nbytes; meta["group"] = group;
  meta["algo"] = "hierarchical(" + std::to_string(d) + "x" + std::to_string(nodes) + ")";
  meta["intra_s"] = rs + ag; meta["inter_s"] = inter;
  return {comm_kernel(name, rs + inter + ag, std::move(meta))};
}

std::vector<LoweredKernel> all_to_all(const HwView& hw, const std::string& name, double send_bytes, int group) {
  if (group <= 1 || send_bytes <= 0) return {};
  int d = static_cast<int>(hw.get_num("network.nvlink.domain_size", 1.0));
  if (group <= d) {
    auto [a, bw] = link(hw, group);
    double t = a * std::ceil(std::log2(static_cast<double>(group))) + send_bytes * (group - 1) / group / bw;
    nb::dict meta;
    meta["bytes"] = send_bytes; meta["group"] = group; meta["algo"] = "a2a";
    return {comm_kernel(name, t, std::move(meta))};
  }
  int nodes = (group + d - 1) / d;
  auto [ai, bwi] = link(hw, d);
  auto [ao, bwo] = link(hw, group);
  double intra = ai * std::ceil(std::log2(static_cast<double>(d))) + send_bytes * (d - 1) / group / bwi;
  double inter = ao + send_bytes * (group - d) / group / bwo;
  nb::dict meta;
  meta["bytes"] = send_bytes; meta["group"] = group;
  meta["algo"] = "a2a-hierarchical(" + std::to_string(d) + "x" + std::to_string(nodes) + ")";
  meta["intra_s"] = intra; meta["inter_s"] = inter;
  return {comm_kernel(name, intra + inter, std::move(meta))};
}

// ------------------------------------------------------------------------- tile search spaces
std::vector<GemmTile> gemm_search_space(int64_t M, int64_t N, int64_t K) {
  std::vector<GemmTile> out;
  for (int bm : {64, 128})
    for (int bn : {64, 128, 256})
      for (int bk : {64, 128})
        for (int sk : {1, 2, 4, 8}) {
          if (sk > 1 && K / sk < 2 * bk) continue;
          GemmTile t;
          t.bm = bm; t.bn = bn; t.bk = bk; t.split_k = sk;
          out.push_back(t);
          if (sk == 1 && M >= 2 * bm) {
            GemmTile t2 = t; t2.split_k = 1; t2.cluster_m = 2;
            out.push_back(t2);
            GemmTile t3 = t2; t3.cta_pair = true;
            out.push_back(t3);
          }
        }
  return out;
}

std::vector<AttnTile> attn_search_space() {
  std::vector<AttnTile> out;
  for (int bm : {64, 128})
    for (int bn : {32, 64, 128})
      for (int st : {2, 3}) {
        AttnTile t;
        t.block_m = bm; t.block_n = bn; t.stages = st;
        out.push_back(t);
      }
  return out;
}

// ------------------------------------------------------------------------- evaluate
KernelResult evaluate(const HwView& hw, const LoweredKernel& lk) {
  KernelResult r;
  static_cast<shader_core::KernelResult&>(r) = shader_core::evaluate_kernel(lk.trace, hw.lanes(), hw.sms(), hw.launch_s());
  r.meta = lk.meta;
  if (lk.trace.fixed) {
    // comm kernels: label uses the algorithm, same as the old Python engine did
    std::string algo = "comm";
    if (lk.meta.contains("algo")) algo = nb::cast<std::string>(lk.meta["algo"]);
    r.limiter_detail.clear();
    r.limiter_detail["net:" + algo] = lk.trace.fixed_time;
    if (r.limiter_time.count("launch")) r.limiter_detail["launch"] = r.limiter_time.at("launch");
  }
  return r;
}

}  // namespace tilesight::gpu_top
