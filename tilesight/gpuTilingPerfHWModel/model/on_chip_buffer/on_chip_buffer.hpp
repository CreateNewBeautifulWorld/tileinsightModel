// on_chip_buffer — the optional staging SRAM between the shader slices and the memory slices,
// and its "switch" share of gmem's per-core fair-share formula (port of the buffer-residency
// and buffer-routing pieces of kernels/gemm.py's `gload` + `resident_frac`).
#pragma once

namespace tilesight::on_chip_buffer {

// Fraction of a tensor class that is pre-allocated on the buffer (deterministic capacity share).
double resident_frac(bool has_sram, bool has_fp_override, double fp_override, double capacity_for_class,
                     double footprint_bytes);

// Fraction of one tile load's bytes answered by the buffer instead of reaching the memory slice
// (the "from_buffer" computation in gload, incl. the costream bandwidth split).
double route_from_buffer(bool has_sram, double miss, bool has_sram_miss, double sram_miss, bool costream,
                         double buffer_bps, double ddr_bps);

}  // namespace tilesight::on_chip_buffer
