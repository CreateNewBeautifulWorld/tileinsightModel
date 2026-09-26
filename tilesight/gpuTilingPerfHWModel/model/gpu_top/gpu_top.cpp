#include "gpu_top.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
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

// Physical address the first tensor starts at (memory.addressing.base, default 0x8000_0000).
double addr_base(const HwView& hw) { return hw.get_num("memory.addressing.base", 2147483648.0); }

// HBM bytes per load of one matrix operand, from plan_fetch below (set = false: no plan, the
// load's own miss fraction goes through the L2 ports).
// buffer_frac >= 0: the DMA destination is the on-chip buffer and `dma` already counts the
// buffer's own misses; that fraction of the load's bytes is read out of the buffer.
struct HbmFetch {
  bool set = false;
  double dma = 0.0, port = 0.0;
  double buffer_frac = -1.0;
};

// gload: one tile load, routed shader-slice -> [L1] -> [buffer over the switch] -> memory slice.
// Mirrors kernels/gemm.py `gload`.
TraceAction gload(const HwView& hw, const std::string& name, double nbytes, double miss,
                  const std::string& load_path, const std::vector<int>& deps,
                  bool has_sram_miss, double sram_miss, bool has_l1_miss, double l1_miss,
                  double slice_frac = 1.0, const HbmFetch& fetch = HbmFetch{}) {
  double l1_m = 1.0;
  std::map<std::string, double> work;
  if (has_l1_miss && hw.l1_capacity_bytes() > 0) {
    l1_m = std::max(0.0, std::min(1.0, l1_miss));
    if (l1_m < 1.0) work["l1"] = hw.l1_time_per_sm(nbytes);
  }
  double after_l1 = nbytes * l1_m;

  bool has_sram = hw.has_sram();
  double from_buffer = 0.0;
  bool dma_to_buffer = has_sram && fetch.set && fetch.buffer_frac >= 0;
  if (dma_to_buffer) {
    from_buffer = fetch.buffer_frac;
    work["sram"] = after_l1 * from_buffer;
    work["switch"] = after_l1 * from_buffer;
  } else if (has_sram) {
    from_buffer = on_chip_buffer::route_from_buffer(true, miss, has_sram_miss, sram_miss, hw.sram_costream(),
                                                     hw.sram_bandwidth_Bps(), hw.ddr_bandwidth_Bps());
    work["sram"] = after_l1 * from_buffer;
    work["switch"] = after_l1 * from_buffer;
  }
  double to_memory = after_l1 * (1.0 - from_buffer);
  work["l2"] = to_memory;
  // HBM leg. With a fetch plan (matrix operands: GEMM A/B, attention K/V), `fetch` already holds
  // the real HBM bytes per load from the fetch-atom-keyed L2 simulation (a DMA page / smart chunk
  // or an op_bytes-rounded tile on a miss, amortized over the hits in between), split into what
  // the DMA engines move and what the L2 ports fetch. Without one (Q, activations, elementwise)
  // it is the plain "miss fraction of this tile", fetched by the L2 ports. Either way, whatever
  // the on-chip buffer already answered never reaches HBM at all.
  double keep = dma_to_buffer ? 1.0 : 1.0 - from_buffer;
  double dma_b = fetch.set ? fetch.dma * keep : 0.0;
  double port_b = fetch.set ? fetch.port * keep : to_memory * miss;
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
  // The L2 leg (every access, hit or miss) and the L2-port HBM leg pay the occupancy-derived
  // slice-spread proxy. A DMA page/chunk spans every slice by construction, so it does not.
  if (slice_frac < 1.0) {
    work["l2"] /= slice_frac;
    port_b /= slice_frac;
  }
  // HBM bandwidth is one pool shared by both paths (`ddr`); each path also has its own ceiling:
  // the DMA engines (`dma`) and the L2 ports' outstanding-entry-limited fetch rate (`l2port`).
  work["dma"] = dma_b;
  work["l2port"] = port_b;
  work["ddr"] = dma_b + port_b;

  double buf_lat = has_sram ? hw.sram_latency_ns() * 1e-9 : 0.0;
  bool has_l2 = hw.l2_capacity_bytes() > 0;
  double mem_lat = memory_slice::mem_latency_s(hw.get_num("memory.l2.latency_ns", 0.0),
                                               hw.get_num("memory.ddr.latency_ns", 0.0), miss, has_l2);
  // Fill latency of the path a miss comes back on: a DMA beat lands in the buffer/L2 in
  // memory.dma.fill_latency_cycles, an L2-port beat in memory.l2.ddr_fill_latency_cycles.
  {
    double tot = dma_b + port_b;
    double dma_share = tot > 0 ? dma_b / tot : 0.0;
    double fill_cyc = dma_share * hw.get_num("memory.dma.fill_latency_cycles", 0.0) +
                      (1.0 - dma_share) * hw.get_num("memory.l2.ddr_fill_latency_cycles", 0.0);
    mem_lat += miss * fill_cyc / hw.clock_hz();
  }
  lat += from_buffer * buf_lat + (1.0 - from_buffer) * mem_lat;
  if (has_sram && hw.sram_prefetch()) lat *= std::max(0.0, 1.0 - from_buffer);

  TraceAction a;
  a.name = name;
  a.work = std::move(work);
  a.deps = deps;
  a.latency = lat;
  return a;
}

// gstore: write-back to HBM. A matrix output (GEMM C, attention O: a contiguous tile) is written
// by the DMA engines, like matrix loads; everything else (elementwise outputs, split partials)
// through the L2 ports, whose outstanding entries reads and writes share.
TraceAction gstore(const std::string& name, double nbytes, const std::vector<int>& deps, double ddr_frac,
                   double sram_frac, double slice_frac = 1.0, bool by_dma = false) {
  TraceAction a;
  a.name = name;
  a.work["l2"] = nbytes;
  a.work["ddr"] = nbytes * ddr_frac;
  if (slice_frac < 1.0) {
    a.work["l2"] /= slice_frac;
    if (!by_dma) a.work["ddr"] /= slice_frac;
  }
  a.work[by_dma ? "dma" : "l2port"] = a.work["ddr"];
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

// Buffer capacity one kernel's own access trace competes for. A shared buffer (no alloc/pin) is
// split across tensor classes by footprint only for what stays resident *between* kernels
// (resident_frac); while a kernel runs, the trace replays from cold and its streams are the only
// ones being touched, so LRU would give them the whole buffer. A static alloc/pin share is a
// dedicated carve-out and stays the class's share.
double trace_buffer_cap(const HwView& hw, const std::string& klass) {
  bool static_alloc = !hw.get("memory.sram.alloc").is_none() || !hw.get("memory.sram.pin").is_none();
  return static_alloc ? hw.sram_capacity_for(klass) : hw.sram_capacity_bytes();
}

// The on-chip buffer as the DMA destination: DMA'd pages/chunks land there (not in L2) and are
// read out of it, so plan_fetch simulates them in the buffer. Only for a shared buffer used as a
// plain cache: a static alloc/pin share (pre-staged, resident-by-assumption) and the explicit
// `memory.sram.stage` model keep their closed-form treatment, with DMA traffic going via L2.
double dma_buffer_cap(const HwView& hw, const std::string& klass) {
  if (!hw.has_sram() || !hw.get("memory.sram.stage").is_none()) return 0.0;
  bool static_alloc = !hw.get("memory.sram.alloc").is_none() || !hw.get("memory.sram.pin").is_none();
  if (static_alloc) return 0.0;
  return trace_buffer_cap(hw, klass);
}

// L2 and DDR are not one GPU-wide pool: each memory slice owns its own L2 port and HBM channel.
// Without a DMA hugepage configured, this is the only signal available for how spread a kernel's
// traffic is: how many blocks are concurrently resident vs. how many slices there are. With one
// configured, hugepage_bytes()/the hugepage-keyed L2 simulation supersede this entirely (a
// hugepage spans every slice evenly by construction — see hugepage_bytes below), so this proxy is
// used only for kernels/lanes that never call simulate_l2's hugepage path (matrix operand loads
// do; Q/output/spill do not).
double slice_bw_frac(int n_slices, int64_t concurrent_blocks) {
  n_slices = std::max(1, n_slices);
  if (n_slices <= 1 || concurrent_blocks <= 0) return 1.0;
  return std::min(1.0, static_cast<double>(concurrent_blocks) / static_cast<double>(n_slices));
}

// A DMA moves exactly one hugepage per operation, never less. Real interleaving distributes it
// across slices one granule at a time, round-robin; when the granule count doesn't divide evenly
// by the slice count, some slices simply end up with one fewer granule than others (still every
// slice reached, still no address arithmetic to alias, unlike two earlier attempts at this same
// problem that derived slice spread from the L2 hit/miss simulation's synthetic per-tile address
// stream, which aliases on power-of-two tile strides). Rather than track which slice is the short
// one, this pads up to what the fullest slice gets (never an underestimate of the real transfer):
// `per_slice_granules = ceil(granules / n_slices)`, hugepage size = `per_slice_granules * n_slices
// * granularity`. E.g. a 2048 KB hugepage over 12 slices at 1 KB granules doesn't split evenly
// (2048 / 12 = 170.67); every slice is modeled as getting the 171 granules the fullest one does.
// Returns 0 (disabled) when unset, or when there is no L2 to key by hugepage at all: with zero L2
// capacity nothing can ever stay resident, so every access would wrongly cost a fresh hugepage
// instead of falling back to the plain per-tile miss cost a no-L2 config already models correctly.
double hugepage_bytes(const HwView& hw, const cache::AddrCfg& cfg) {
  if (hw.l2_capacity_bytes() <= 0) return 0.0;
  double hp = hw.get_num("memory.dma.hugepage_KB", 0.0) * 1024.0;
  if (hp <= 0) return 0.0;
  int64_t granularity = std::max<int64_t>(1, cfg.granularity);
  int n_slices = std::max(1, cfg.ports);
  int64_t granules = static_cast<int64_t>(std::ceil(hp / static_cast<double>(granularity)));
  int64_t per_slice = (granules + n_slices - 1) / n_slices;
  return static_cast<double>(per_slice) * n_slices * granularity;
}

// How much of a kernel the L2/L1/buffer simulations replay. By default the whole kernel: every
// wave and every K step, because capacity effects (does the working set fit the L2 / the buffer
// all the shader cores share) and hugepage reuse (the first K step into a page pays for it, the
// following ones are free) only show up over a long enough trace — a short window sees only the
// compulsory misses and extrapolates them. `memory.l2.waves_simulated` / `.ksteps_simulated`
// (0 = all) can shorten it on purpose; `memory.l2.sim_max_accesses` caps the cost: over budget,
// waves are dropped first (they are close to statistically alike) and K steps only last (cutting
// K is what biases a hugepage trace). `coverage` = simulated / total accesses, reported in meta.
struct SimWindow {
  int64_t waves = 1, ksteps = 1;
  double coverage = 1.0;
};

SimWindow sim_window(const HwView& hw, int64_t units, int64_t conc, int64_t iters, int per_step) {
  conc = std::max<int64_t>(1, conc);
  units = std::max<int64_t>(1, units);
  iters = std::max<int64_t>(1, iters);
  int64_t total_waves = (units + conc - 1) / conc;
  int64_t w_cfg = static_cast<int64_t>(hw.get_num("memory.l2.waves_simulated", 0.0));
  int64_t k_cfg = static_cast<int64_t>(hw.get_num("memory.l2.ksteps_simulated", 0.0));
  SimWindow w;
  w.waves = w_cfg > 0 ? std::min(w_cfg, total_waves) : total_waves;
  w.ksteps = k_cfg > 0 ? std::min(k_cfg, iters) : iters;
  auto accesses = [&](int64_t wv, int64_t ks) {
    return static_cast<double>(std::min(units, conc * wv)) * static_cast<double>(ks) * per_step;
  };
  double budget = hw.get_num("memory.l2.sim_max_accesses", 0.0);
  if (budget > 0) {
    double per_wave = static_cast<double>(std::min(units, conc)) * static_cast<double>(w.ksteps) * per_step;
    if (accesses(w.waves, w.ksteps) > budget)
      w.waves = std::max<int64_t>(1, std::min(w.waves, static_cast<int64_t>(budget / per_wave)));
    if (accesses(w.waves, w.ksteps) > budget)
      w.ksteps = std::max<int64_t>(1, static_cast<int64_t>(budget / (static_cast<double>(std::min(units, conc)) * per_step)));
  }
  w.coverage = accesses(w.waves, w.ksteps) / (static_cast<double>(units) * static_cast<double>(iters) * per_step);
  return w;
}

// ---- fetch plan: how a matrix operand (GEMM A/B, attention K/V) comes in from HBM ----------
// memory.dma.fetch_mode picks the atom a miss brings in, and which engine moves it:
//   dma_page          every miss is one whole DMA page (memory.dma.hugepage_KB, padded), by DMA.
//   dma_page_tail_l2  pages fully inside the stream's needed range by DMA; the partial page at
//                     either end of that range comes through the L2 ports, one tile rounded up
//                     to memory.l2.op_bytes at a time.
//   l2_port           everything through the L2 ports, tile rounded up to memory.l2.op_bytes.
//   dma_smart         DMA, but the transfer is sized to what is needed: the largest multiple of
//                     memory.dma.port_bytes (128 B) that is <= the page, <= the stream's needed
//                     range, and small enough that one chunk per concurrently-walked stream fits
//                     the destination (L2, or the buffer when there is one) — never below a tile.
//   auto (default)    dma_page when the DMA lands in a shared on-chip buffer that can hold one
//                     page per concurrent stream (it keeps a page until its stream is done),
//                     dma_smart otherwise.
// Every access carries the stream's needed range [lo, hi) — the contiguous bytes that one block
// walks over the K loop — which dma_smart and dma_page_tail_l2 need.
struct FetchAcc {
  int64_t addr, tkey;
  double tile;
  int op;
  int64_t lo, hi;
  bool wave0;  // part of the first wave: used to count concurrently-walked streams
};

struct FetchPlan {
  std::string mode;
  double page = 0.0;             // padded DMA page (0 = none)
  std::vector<double> chunk;     // per op: bytes one DMA miss moves
  std::vector<double> port_atom; // per op: bytes one L2-port miss moves
  std::vector<double> miss;      // per op: fraction of loads that miss L2
  std::vector<HbmFetch> fetch;   // per op: HBM bytes per load, by engine
  std::vector<int64_t> keys;     // the L2 atoms (also replayed by the shared-buffer simulation)
  std::vector<double> sizes;
  std::vector<int> op_of;        // per access: operand index
  int64_t streams_concurrent = 0;
  bool via_buffer = false;       // DMA atoms simulated in the on-chip buffer instead of L2
  double buffer_hit_rate = 1.0;
  cache::SimResult sim;          // the L2 simulation
};

std::string fetch_mode(const HwView& hw) {
  std::string m = hw.get_str("memory.dma.fetch_mode", "auto");
  if (m == "auto") return m;
  if (m != "dma_page" && m != "dma_page_tail_l2" && m != "l2_port" && m != "dma_smart")
    throw std::invalid_argument("memory.dma.fetch_mode must be auto | dma_page | dma_page_tail_l2 | "
                                "l2_port | dma_smart, got " + m);
  return m;
}

// L2 operation (= eviction) granularity. It has to tile the L2 interleave stripe exactly, or one
// L2 op would straddle two slices.
int64_t l2_op_bytes(const HwView& hw, const cache::AddrCfg& cfg) {
  int64_t op = static_cast<int64_t>(hw.get_num("memory.l2.op_bytes", 256.0));
  if (op <= 0 || cfg.granularity % op != 0)
    throw std::invalid_argument("memory.l2.op_bytes (" + std::to_string(op) +
                                ") must divide the L2 interleave granularity (" +
                                std::to_string(cfg.granularity) + " B)");
  return op;
}

double round_up(double x, double unit) { return unit > 0 ? std::ceil(x / unit - 1e-9) * unit : x; }
double round_down(double x, double unit) { return unit > 0 ? std::floor(x / unit + 1e-9) * unit : x; }

FetchPlan plan_fetch(const HwView& hw, const std::vector<FetchAcc>& acc, int n_ops, const cache::AddrCfg& lmap,
                     int n_part, double buffer_cap = 0.0) {
  FetchPlan fp;
  // where the DMA lands: the buffer when it is the DMA destination, else L2
  double dest_cap = buffer_cap > 0 ? buffer_cap : hw.l2_capacity_bytes();
  fp.mode = fetch_mode(hw);
  int64_t op_b = l2_op_bytes(hw, lmap);
  double dma_port = std::max(1.0, hw.get_num("memory.dma.port_bytes", 128.0));
  fp.page = hugepage_bytes(hw, lmap);
  // concurrently-walked streams (distinct needed ranges in the first wave), and the largest
  // tile / needed range per operand
  std::vector<double> tile(n_ops, 0.0), extent(n_ops, 0.0);
  {
    std::set<std::pair<int64_t, int64_t>> uniq;
    for (const auto& a : acc) {
      if (a.wave0) uniq.insert({a.lo, a.hi});
      tile[a.op] = std::max(tile[a.op], a.tile);
      extent[a.op] = std::max(extent[a.op], static_cast<double>(a.hi - a.lo));
    }
    fp.streams_concurrent = static_cast<int64_t>(uniq.size());
  }
  // auto: whole pages when they land in a buffer that holds one per concurrent stream (it keeps
  // a page until its stream is done), else a smart DMA sized to fit
  if (fp.mode == "auto")
    fp.mode = (buffer_cap > 0 && fp.page > 0 &&
               fp.page * static_cast<double>(std::max<int64_t>(1, fp.streams_concurrent)) <= buffer_cap)
                  ? "dma_page" : "dma_smart";
  bool page_mode = fp.mode == "dma_page" || fp.mode == "dma_page_tail_l2";
  fp.chunk.assign(n_ops, 0.0);
  fp.port_atom.assign(n_ops, 0.0);
  for (int o = 0; o < n_ops; ++o) {
    fp.port_atom[o] = round_up(tile[o], static_cast<double>(op_b));
    double t128 = round_up(tile[o], dma_port);
    if (page_mode && fp.page > 0) {
      fp.chunk[o] = fp.page;
    } else if (fp.mode == "dma_smart" && dest_cap > 0) {
      double cap = fp.page > 0 ? fp.page : hw.get_num("memory.dma.hugepage_KB", 2048.0) * 1024.0;
      cap = std::min(cap, round_up(extent[o], dma_port));
      // two chunks per stream: the one being consumed and the next one the DMA is filling
      if (fp.streams_concurrent > 0 && dest_cap > 0)
        cap = std::min(cap, round_down(dest_cap / (2.0 * static_cast<double>(fp.streams_concurrent)), dma_port));
      fp.chunk[o] = std::max(t128, cap);
    } else {
      fp.chunk[o] = t128;  // nowhere to keep a page (no L2/buffer): one tile per DMA
    }
  }
  // atoms: key low 2 bits = kind (0 page/chunk by DMA, 2 tile by L2 port)
  std::vector<int64_t> addrs;
  std::vector<int> streams;
  std::vector<char> fill;
  fp.keys.reserve(acc.size());
  for (const auto& a : acc) {
    bool by_port = fp.mode == "l2_port";
    if (fp.mode == "dma_page_tail_l2" && fp.page > 0) {
      int64_t pg = static_cast<int64_t>(fp.page);
      int64_t p0 = (a.addr / pg) * pg;
      by_port = !(p0 >= a.lo && p0 + pg <= a.hi);
    } else if (page_mode && fp.page <= 0) {
      by_port = fp.mode == "dma_page_tail_l2";
    }
    int64_t key;
    double size;
    if (by_port) {
      key = a.tkey * 4 + 2;
      size = fp.port_atom[a.op];
    } else if (fp.mode == "dma_smart") {
      // a smart DMA starts where the stream needs data and stops where its need ends: chunks are
      // aligned to the stream's own start, the last one trimmed to the remaining range
      int64_t c = std::max<int64_t>(1, static_cast<int64_t>(fp.chunk[a.op]));
      int64_t idx = (a.addr - a.lo) / c;
      int64_t c0 = a.lo + idx * c;
      key = (((a.lo / 128) * 1000003 + idx) * 8 + a.op) * 4 + 1;
      size = std::max(round_up(a.tile, dma_port),
                      round_up(static_cast<double>(std::min<int64_t>(c, a.hi - c0)), dma_port));
    } else {
      int64_t c = std::max<int64_t>(1, static_cast<int64_t>(fp.chunk[a.op]));
      key = (a.addr / c) * 4;
      size = fp.chunk[a.op];
    }
    fp.keys.push_back(key);
    fp.sizes.push_back(size);
    fp.op_of.push_back(a.op);
    addrs.push_back(a.addr);
    streams.push_back(a.op * 2 + (by_port ? 1 : 0));
    fill.push_back(by_port ? 0 : 1);
  }
  // With a buffer as the DMA destination, the DMA atoms live in the buffer (one LRU over its
  // whole capacity), not in L2; only the L2-port atoms go through the L2 simulation.
  fp.via_buffer = buffer_cap > 0;
  std::vector<double> buf_miss(n_ops, 0.0), buf_acc(n_ops, 0.0), buf_mb(n_ops, 0.0);
  if (fp.via_buffer) {
    std::vector<int64_t> bk, ba;
    std::vector<double> bs;
    std::vector<int> bst, one;
    std::vector<int64_t> lk, la;
    std::vector<double> ls;
    std::vector<int> lst;
    std::vector<char> lf;
    for (size_t i = 0; i < fp.keys.size(); ++i) {
      if (fill[i]) {
        bk.push_back(fp.keys[i]); ba.push_back(addrs[i]); bs.push_back(fp.sizes[i]); bst.push_back(fp.op_of[i]);
        one.push_back(0);
      } else {
        lk.push_back(fp.keys[i]); la.push_back(addrs[i]); ls.push_back(fp.sizes[i]); lst.push_back(streams[i]);
        lf.push_back(0);
      }
    }
    cache::SimResult br = cache::simulate(bk, ba, bs, bst, n_ops, buffer_cap, 1, one, "lru");
    for (int o = 0; o < n_ops; ++o) {
      buf_miss[o] = br.misses[o];
      buf_mb[o] = br.miss_bytes[o];
      buf_acc[o] = static_cast<double>(br.accesses[o]);
    }
    fp.buffer_hit_rate = br.hit_rate;
    fp.sim = memory_slice::simulate_l2(lk, la, ls, lst, 2 * n_ops, hw.l2_capacity_bytes(), n_part, lmap,
                                       hw.get_str("memory.l2.policy", "lru"), false, &lf);
  } else {
    fp.sim = memory_slice::simulate_l2(fp.keys, addrs, fp.sizes, streams, 2 * n_ops, hw.l2_capacity_bytes(), n_part,
                                       lmap, hw.get_str("memory.l2.policy", "lru"), fp.mode != "l2_port", &fill);
  }
  fp.miss.assign(n_ops, 0.0);
  fp.fetch.assign(n_ops, HbmFetch{});
  for (int o = 0; o < n_ops; ++o) {
    double m_port = fp.sim.misses[2 * o + 1];
    double m_dma = fp.via_buffer ? buf_miss[o] : fp.sim.misses[2 * o];
    double n_dma = fp.via_buffer ? buf_acc[o] : static_cast<double>(fp.sim.accesses[2 * o]);
    double loads = n_dma + static_cast<double>(fp.sim.accesses[2 * o + 1]);
    fp.fetch[o].set = true;
    if (loads <= 0) continue;
    fp.miss[o] = (m_dma + m_port) / loads;
    fp.fetch[o].dma = (fp.via_buffer ? buf_mb[o] : fp.sim.miss_bytes[2 * o]) / loads;
    fp.fetch[o].port = fp.sim.miss_bytes[2 * o + 1] / loads;
    if (fp.via_buffer) fp.fetch[o].buffer_frac = n_dma / loads;
  }
  return fp;
}

void fetch_meta(nb::dict& meta, const FetchPlan& fp) {
  meta["fetch_mode"] = fp.mode;
  meta["dma_hugepage_bytes"] = fp.page;
  meta["dma_chunk_bytes"] = fp.chunk;
  meta["l2_port_atom_bytes"] = fp.port_atom;
  meta["fetch_streams"] = fp.streams_concurrent;
  meta["dma_to_buffer"] = fp.via_buffer;
  if (fp.via_buffer) meta["buffer_hit_rate"] = fp.buffer_hit_rate;
  std::vector<double> d, p;
  for (const auto& f : fp.fetch) { d.push_back(f.dma); p.push_back(f.port); }
  meta["hbm_dma_bytes_per_load"] = d;
  meta["hbm_port_bytes_per_load"] = p;
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

  // ---- L2 model: deterministic simulation of the fetch atoms over the whole kernel --------
  cache::AddrCfg lmap = l2_addr_cfg(hw);
  double occ_frac = slice_bw_frac(lmap.ports, concurrent_blocks);
  int64_t conc = static_cast<int64_t>(hw.sms()) * resident;
  auto order_all = grouped_raster(mt, nt, t.swizzle);
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> full_order;  // (b,s,m,n)
  for (int64_t b_ = 0; b_ < batch; ++b_)
    for (int s_ = 0; s_ < sk; ++s_)
      for (auto [m, n] : order_all) full_order.push_back({b_, s_, m, n});
  SimWindow win = sim_window(hw, static_cast<int64_t>(full_order.size()), conc, iters, 2);
  int64_t n_waves = win.waves;
  int64_t n_take = std::min<int64_t>(static_cast<int64_t>(full_order.size()), conc * n_waves);
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> order(full_order.begin(), full_order.begin() + n_take);
  // Layout (tile-major, "hierarchy Z"): A is [batch][M-tile][K-tile], B is [batch][N-tile][K-tile]
  // (K contiguous per N strip, like an nn.Linear weight [out][in]). So each block walks one
  // contiguous range of A and one of B over its K loop: those are the streams a DMA can fetch.
  double a_base = addr_base(hw), b_base = a_base + round_up(M * K * ab * batch, 4096.0);
  std::vector<FetchAcc> facc;
  std::vector<int64_t> tkeys, addrs;
  std::vector<double> tsizes;
  std::vector<int> streams;
  int64_t ksteps = win.ksteps;
  for (int64_t w = 0; w < n_waves; ++w) {
    int64_t lo = w * conc, hi = std::min<int64_t>(order.size(), (w + 1) * conc);
    for (int64_t kk = 0; kk < ksteps; ++kk) {
      for (int64_t idx = lo; idx < hi; ++idx) {
        auto [b_, s_, m, n] = order[idx];
        int64_t k0 = s_ * iters, k1 = std::min<int64_t>(kt, k0 + iters);
        int64_t k = k0 + kk;
        if (k >= kt) continue;
        int64_t a_row = static_cast<int64_t>(a_base + (b_ * mt + m) * kt * a_tile);
        int64_t b_row = static_cast<int64_t>(b_base + (b_ * nt + n) * kt * b_tile);
        int64_t a_addr = a_row + static_cast<int64_t>(k * a_tile);
        int64_t b_addr = b_row + static_cast<int64_t>(k * b_tile);
        int64_t a_key = ((b_ * 4096 + m) * 65536 + k) * 2, b_key = ((b_ * 4096 + n) * 65536 + k) * 2 + 1;
        facc.push_back({a_addr, a_key, a_tile, 0, a_row + static_cast<int64_t>(k0 * a_tile),
                       a_row + static_cast<int64_t>(k1 * a_tile), w == 0});
        facc.push_back({b_addr, b_key, b_tile, 1, b_row + static_cast<int64_t>(k0 * b_tile),
                       b_row + static_cast<int64_t>(k1 * b_tile), w == 0});
        tkeys.push_back(a_key); addrs.push_back(a_addr); tsizes.push_back(a_tile); streams.push_back(0);
        tkeys.push_back(b_key); addrs.push_back(b_addr); tsizes.push_back(b_tile); streams.push_back(1);
      }
    }
  }
  int n_part = static_cast<int>(hw.get_num("memory.l2.partitions", 1.0));
  std::string policy = hw.get_str("memory.l2.policy", "lru");
  bool through_l1 = false;
  for (auto& [p, _] : paths)
    if (!hw.path_attr_bool(p, "smem_direct", true)) through_l1 = true;
  FetchPlan fp = plan_fetch(hw, facc, 2, lmap, n_part, dma_buffer_cap(hw, tensor_class(bn_)));
  const cache::SimResult& sim = fp.sim;
  const std::vector<int64_t>& keys = fp.keys;
  const std::vector<double>& sizes = fp.sizes;
  double fa = fp.miss[0], fb = fp.miss[1];

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
    for (size_t i = 0; i < tkeys.size(); ++i)
      if ((static_cast<int64_t>(i) / 2) % step == 0) { sk_.push_back(tkeys[i]); sa_.push_back(addrs[i]); ss_.push_back(tsizes[i]); st_.push_back(streams[i]); }
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

  // A real tile-keyed LRU trace of the buffer, exactly like the L1 one above and simulate_l2
  // itself: the buffer is explicitly managed, but which tile occupies it right now is still a
  // deterministic question given the real access order and a real capacity, not something a
  // closed-form ratio should have to approximate. Reuses the very same A/B access stream already
  // built for the L2 simulation, filtered to one operand's own accesses, against that operand's
  // own class capacity. Only applies to a *shared* buffer, where tiles genuinely compete for space
  // via real access order (memory.sram.alloc/pin unset): a static per-class share (alloc/pin) means
  // that capacity is a dedicated, pre-staged carve-out, not something this kernel's own access
  // order fills from cold — charging this simulation's compulsory first-touch misses against it
  // would be wrong, so that case keeps the closed-form resident_frac. Also falls back there when
  // the class has no buffer capacity to simulate at all.
  bool sram_is_static_alloc = !hw.get("memory.sram.alloc").is_none() || !hw.get("memory.sram.pin").is_none();
  auto buf_miss = [&](const std::string& klass, int stream) -> std::optional<double> {
    double cap = trace_buffer_cap(hw, klass);
    if (cap <= 0 || sram_is_static_alloc) return std::nullopt;
    std::vector<int64_t> sk_, sa_;
    std::vector<double> ss_;
    std::vector<int> st_;
    for (size_t i = 0; i < keys.size(); ++i)
      if (fp.op_of[i] == stream) { sk_.push_back(keys[i]); sa_.push_back(addrs[i]); ss_.push_back(sizes[i]); st_.push_back(0); }
    if (sk_.empty()) return std::nullopt;
    std::vector<int> part1(sa_.size(), 0);
    cache::SimResult r = cache::simulate(sk_, sa_, ss_, st_, 1, cap, 1, part1, hw.get_str("memory.sram.policy", "lru"));
    return r.miss_fraction()[0];
  };

  bool has_sram = hw.has_sram();
  bool has_sa = false, has_sb = false;
  double sa = 0, sb = 0;
  if (has_sram && !fp.via_buffer) {
    // Same principle as attention's KV: capacity competes with whatever's concurrently resident
    // right now (stages-deep x one operand tile x concurrently-active blocks), not the whole
    // A/B matrix. Reuse across several M/N-tiles from that one resident copy is share_a/share_b's
    // job (an already-computed, searched staging factor) applied separately below — it says how
    // many times a resident tile gets reused, not how big the resident footprint is.
    auto ra_miss = buf_miss(tensor_class(an), 0);
    auto rb_miss = buf_miss(tensor_class(bn_), 1);
    double a_active = static_cast<double>(stages) * a_tile * static_cast<double>(concurrent_blocks);
    double b_active = static_cast<double>(stages) * b_tile * static_cast<double>(concurrent_blocks);
    double ra = ra_miss ? (1.0 - *ra_miss) : resident_frac(hw, a_active, tensor_class(an));
    double rb_ = rb_miss ? (1.0 - *rb_miss) : resident_frac(hw, b_active, tensor_class(bn_));
    sa = fa * (1 - ra) / share_a;
    sb = fb * (1 - rb_) / share_b;
    has_sa = has_sb = true;
  }

  double bm_c = std::max<double>(t.bm, hw.tc_min_m());
  auto [mma_lane, mma_t] = hw.mma_cost(2.0 * bm_c * t.bn * t.bk, comp_dt);

  std::vector<TraceAction> body;
  // (a cluster's blocks share one B fetch: the simulation already sees the second access as a hit)
  const HbmFetch& fetch_b = fp.fetch[1];
  body.push_back(gload(hw, "load:" + an, a_tile, fa, t.load_path, {}, has_sa, sa, has_l1_miss, l1_miss_a, occ_frac,
                       fp.fetch[0]));
  body.push_back(gload(hw, "load:" + bn_, b_tile / cl, fb, t.load_path, {}, has_sb, sb, has_l1_miss, l1_miss_b,
                       occ_frac, fetch_b));
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
    sp.work["l2"] = 2.0 * spill / std::max<int64_t>(1, iters) / occ_frac;
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
    epilogue.push_back(gstore(sk == 1 ? "store:" + cn : "store:partials", out_bytes, {0}, 1.0, 0.0, occ_frac,
                              sk == 1));
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
  meta["l2_waves_simulated"] = n_waves; meta["l2_ksteps_simulated"] = ksteps;
  meta["sim_coverage"] = win.coverage;
  fetch_meta(meta, fp);
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
  double ew_occ_frac = slice_bw_frac(l2_addr_cfg(hw).ports, concurrent_blocks);
  std::vector<TraceAction> body;
  body.push_back(gload(hw, "load:" + in_name, bytes_in * per, 1.0, "lsu", {}, true, 1.0 - res, false, 0.0,
                       ew_occ_frac));
  TraceAction comp;
  comp.name = "compute";
  comp.work["cuda"] = hw.cuda_time_per_sm(flops * per);
  comp.work["sfu"] = hw.sfu_time_per_sm(sfu_ops * per);
  comp.deps = {0};
  comp.latency = hw.unit_latency_s(sfu_ops > 0 ? "sfu" : "cuda");
  body.push_back(comp);
  body.push_back(gstore("store:" + out_name, bytes_out * per, {1}, 1.0 - res, hw.has_sram() ? res : 0.0,
                        ew_occ_frac));

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
  int64_t conc = static_cast<int64_t>(hw.sms()) * resident;
  SimWindow win = sim_window(hw, static_cast<int64_t>(order.size()), conc, iters, 1);
  int64_t n_take = std::min<int64_t>(static_cast<int64_t>(order.size()), conc * win.waves);
  cache::AddrCfg lmap = l2_addr_cfg(hw);
  double occ_frac = slice_bw_frac(lmap.ports, concurrent_blocks);
  // The KV cache as it is laid out: [B][kv_heads][ntiles] tiles of tile_b bytes (one tile's K and
  // V adjacent). The GQA groups of one KV head read the very same tiles, so they share addresses
  // and keys — that is the reuse this simulation is for. Each split walks its own contiguous
  // range of `iters` tiles (the stream a DMA fetches). Replayed wave by wave, every split
  // stepping through its tiles in lockstep, like lower_gemm's K loop.
  double tile_b = k_tile + v_tile;
  int64_t kv_base = static_cast<int64_t>(addr_base(hw));
  std::vector<FetchAcc> acc;
  std::vector<int64_t> addrs;
  for (int64_t w = 0; w < win.waves; ++w) {
    int64_t lo = w * conc, hi = std::min<int64_t>(n_take, (w + 1) * conc);
    for (int64_t it = 0; it < win.ksteps; ++it)
      for (int64_t idx = lo; idx < hi; ++idx) {
        auto [b, h, g, s] = order[idx];
        (void)g;
        int64_t t_idx = s * iters + it;
        if (t_idx >= ntiles) continue;
        int64_t row = (b * kv_heads + h) * ntiles;
        int64_t tile_id = row + t_idx;
        int64_t addr = kv_base + static_cast<int64_t>(tile_id * tile_b);
        int64_t r_lo = kv_base + static_cast<int64_t>((row + s * iters) * tile_b);
        int64_t r_hi = kv_base + static_cast<int64_t>((row + std::min<int64_t>(ntiles, (s + 1) * iters)) * tile_b);
        acc.push_back({addr, tile_id, tile_b, 0, r_lo, r_hi, w == 0});
        addrs.push_back(addr);
      }
  }
  int n_part = static_cast<int>(hw.get_num("memory.l2.partitions", 1.0));
  FetchPlan fp = plan_fetch(hw, acc, 1, lmap, n_part, dma_buffer_cap(hw, "kv"));
  const cache::SimResult& sim = fp.sim;
  const std::vector<int64_t>& keys = fp.keys;
  const std::vector<double>& sizes = fp.sizes;
  std::vector<int> streams(keys.size(), 0);
  double f = fp.miss[0];
  // K and V are billed as separate gload calls but share one combined fetch above, so each gets
  // its proportional share of it (they sum back to exactly the one fetch per real miss).
  double k_share = tile_b > 0 ? k_tile / tile_b : 0.0, v_share = tile_b > 0 ? v_tile / tile_b : 0.0;
  auto share = [&](double x) { HbmFetch h = fp.fetch[0]; h.dma *= x; h.port *= x; return h; };
  bool has_fs = false;
  double fs = 0.0;
  if (hw.has_sram() && !fp.via_buffer) {
    bool sram_is_static_alloc = !hw.get("memory.sram.alloc").is_none() || !hw.get("memory.sram.pin").is_none();
    double cap = trace_buffer_cap(hw, "kv");
    std::optional<double> kv_miss;
    if (cap > 0 && !sram_is_static_alloc && !keys.empty()) {
      // Same real tile-keyed LRU trace as lower_gemm's buf_miss, over the same combined K+V
      // access stream already built for the L2 simulation above.
      std::vector<int> part1(addrs.size(), 0);
      cache::SimResult r = cache::simulate(keys, addrs, sizes, streams, 1, cap, 1, part1,
                                           hw.get_str("memory.sram.policy", "lru"));
      kv_miss = r.miss_fraction()[0];
    }
    if (kv_miss) {
      fs = f * (*kv_miss);
    } else {
      // The buffer isn't a cache and doesn't hold the whole layer's KV at once — it's a
      // deterministic capacity share of whatever's actively streaming through the pipeline right
      // now: `stages`-deep multi-buffering x one tile's K/V bytes x however many blocks are
      // concurrently resident GPU-wide. Comparing that (not B*kv_heads*S, the entire sequence)
      // against sram_capacity_for("kv") is what "only the part currently needed" means.
      double kv_active = static_cast<double>(tile.stages) * (k_tile + v_tile) * static_cast<double>(concurrent_blocks);
      fs = f * (1 - resident_frac(hw, kv_active, "kv"));
    }
    has_fs = true;
  }

  double hb_c = std::max<double>(hb, hw.tc_min_m());
  auto [qk_lane, qk_t] = hw.mma_cost(2.0 * hb_c * d_qk * tile.block_n, compute_dtype);
  auto [pv_lane, pv_t] = hw.mma_cost(2.0 * hb_c * tile.block_n * d_v, compute_dtype);

  std::vector<TraceAction> body;
  body.push_back(gload(hw, v_in_k ? "load:kv_cache(latent)" : "load:kv_cache(K)", k_tile, f, "tma", {}, has_fs, fs,
                       false, 0.0, occ_frac, share(k_share)));
  int iv = 0;
  if (!v_in_k) {
    body.push_back(gload(hw, "load:kv_cache(V)", v_tile, f, "tma", {}, has_fs, fs, false, 0.0,
                         occ_frac, share(v_share)));
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
    epilogue.push_back(gstore(splits == 1 ? "store:out" : "store:partials", hb * d_v * out_b, {0}, 1.0, 0.0, 1.0,
                              splits == 1));
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
  meta["sim_coverage"] = win.coverage;
  fetch_meta(meta, fp);
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
  int64_t conc = static_cast<int64_t>(hw.sms()) * resident;
  int64_t max_iters = 1;
  for (auto it : iters_list) max_iters = std::max(max_iters, it);
  SimWindow win = sim_window(hw, static_cast<int64_t>(order.size()), conc, max_iters, 2);
  int64_t n_take = std::min<int64_t>(static_cast<int64_t>(order.size()), conc * win.waves);
  int64_t qpk = std::max<int64_t>(1, H / std::max<int64_t>(1, kv_heads));
  int64_t kv_tiles = (S + bn - 1) / bn;
  cache::AddrCfg lmap = l2_addr_cfg(hw);
  double occ_frac = slice_bw_frac(lmap.ports, concurrent_blocks);
  // The KV cache as laid out: [B][kv_heads][kv_tiles], each tile's K then V. Every query tile of a
  // head (and every head of a GQA group) walks the same KV tiles, so they share addresses/keys;
  // a block's stream is the contiguous KV range [first, first + its iters) it walks. Replayed
  // wave by wave, all blocks stepping through their KV tiles in lockstep.
  double kv_total = k_tile + v_tile;
  int64_t kv_base = static_cast<int64_t>(addr_base(hw));
  std::vector<FetchAcc> acc;
  std::vector<int64_t> addrs;
  for (int64_t w = 0; w < win.waves; ++w) {
    int64_t lo_i = w * conc, hi_i = std::min<int64_t>(n_take, (w + 1) * conc);
    for (int64_t it = 0; it < win.ksteps; ++it)
      for (int64_t idx = lo_i; idx < hi_i; ++idx) {
        auto [b, h, q] = order[idx];
        if (it >= iters_list[q]) continue;
        int64_t first = window > 0 ? std::max<int64_t>(0, q * bm - window + 1) / bn : 0;
        int64_t row = (b * kv_heads + h / qpk) * kv_tiles;
        int64_t tile_id = row + std::min<int64_t>(first + it, kv_tiles - 1);
        int64_t k_addr = kv_base + static_cast<int64_t>(tile_id * kv_total);
        int64_t v_addr = k_addr + static_cast<int64_t>(k_tile);
        int64_t r_lo = kv_base + static_cast<int64_t>((row + first) * kv_total);
        int64_t r_hi = kv_base + static_cast<int64_t>((row + std::min<int64_t>(kv_tiles, first + iters_list[q])) * kv_total);
        acc.push_back({k_addr, tile_id * 2, k_tile, 0, r_lo, r_hi, w == 0});
        acc.push_back({v_addr, tile_id * 2 + 1, v_tile, 1, r_lo, r_hi, w == 0});
        addrs.push_back(k_addr);
        addrs.push_back(v_addr);
      }
  }
  int n_part = static_cast<int>(hw.get_num("memory.l2.partitions", 1.0));
  FetchPlan fp = plan_fetch(hw, acc, 2, lmap, n_part, dma_buffer_cap(hw, "kv"));
  const cache::SimResult& sim = fp.sim;
  const std::vector<int64_t>& keys = fp.keys;
  const std::vector<double>& sizes = fp.sizes;
  std::vector<int> streams(keys.size(), 0);
  double f = (fp.sim.misses[0] + fp.sim.misses[1] + fp.sim.misses[2] + fp.sim.misses[3]) /
             std::max<double>(1.0, static_cast<double>(fp.sim.accesses[0] + fp.sim.accesses[1] +
                                                        fp.sim.accesses[2] + fp.sim.accesses[3]));
  bool has_fs = false;
  double fs = 0.0;
  if (hw.has_sram() && !fp.via_buffer) {
    bool sram_is_static_alloc = !hw.get("memory.sram.alloc").is_none() || !hw.get("memory.sram.pin").is_none();
    double cap = trace_buffer_cap(hw, "kv");
    std::optional<double> kv_miss;
    if (cap > 0 && !sram_is_static_alloc && !keys.empty()) {
      // Same real tile-keyed LRU trace as lower_gemm's buf_miss, over the same combined K+V
      // access stream already built for the L2 simulation above.
      std::vector<int> part1(addrs.size(), 0);
      cache::SimResult r = cache::simulate(keys, addrs, sizes, streams, 1, cap, 1, part1,
                                           hw.get_str("memory.sram.policy", "lru"));
      kv_miss = r.miss_fraction()[0];
    }
    if (kv_miss) {
      fs = f * (*kv_miss);
    } else {
      // Same reasoning as lower_attention_decode: the buffer holds whatever's actively streaming
      // right now (stages-deep x one K/V tile x concurrently-resident blocks), not the whole
      // sequence's KV.
      double kv_active = static_cast<double>(tile.stages) * (k_tile + v_tile) * static_cast<double>(concurrent_blocks);
      fs = f * (1 - resident_frac(hw, kv_active, "kv"));
    }
    has_fs = true;
  }

  double bm_c = std::max<double>(bm, hw.tc_min_m());
  auto [qk_lane, qk_t] = hw.mma_cost(2.0 * bm_c * bn * d_qk, compute_dtype);
  auto [pv_lane, pv_t] = hw.mma_cost(2.0 * bm_c * bn * d_v, compute_dtype);

  std::vector<TraceAction> body;
  body.push_back(gload(hw, "load:kv_cache(K)", k_tile, fp.miss[0], "tma", {}, has_fs, fs, false, 0.0,
                       occ_frac, fp.fetch[0]));
  body.push_back(gload(hw, "load:kv_cache(V)", v_tile, fp.miss[1], "tma", {}, has_fs, fs, false, 0.0,
                       occ_frac, fp.fetch[1]));
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
    epilogue.push_back(gstore("store:out", bm * d_v * 2.0, {0}, 1.0, 0.0, 1.0, true));
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
  meta["sim_coverage"] = win.coverage;
  fetch_meta(meta, fp);
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
