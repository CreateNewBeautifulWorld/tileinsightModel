#include "on_chip_buffer.hpp"

#include <algorithm>

namespace tilesight::on_chip_buffer {

double resident_frac(bool has_sram, bool has_fp_override, double fp_override, double capacity_for_class,
                     double footprint_bytes) {
  if (!has_sram) return 0.0;
  double fp = (has_fp_override && fp_override > 0) ? fp_override : footprint_bytes;
  if (fp > 0 && capacity_for_class > 0) return std::min(1.0, capacity_for_class / fp);
  return 0.0;
}

double route_from_buffer(bool has_sram, double miss, bool has_sram_miss, double sram_miss, bool costream,
                         double buffer_bps, double ddr_bps) {
  if (!has_sram) return 0.0;
  double to_memory_frac = has_sram_miss ? std::min(miss, sram_miss) : miss;
  double x;
  if (miss > 0) x = 1.0 - to_memory_frac / miss;
  else x = has_sram_miss ? (1.0 - sram_miss) : 0.0;
  double from_buffer = std::max(0.0, std::min(1.0, x));
  if (costream && from_buffer > 0 && miss > 0 && (buffer_bps + ddr_bps) > 0) {
    double S = from_buffer, D = 1.0 - from_buffer;
    double dx = std::min(std::max((S * ddr_bps - D * buffer_bps) / (buffer_bps + ddr_bps), 0.0), S);
    from_buffer -= dx;
  }
  return from_buffer;
}

}  // namespace tilesight::on_chip_buffer
