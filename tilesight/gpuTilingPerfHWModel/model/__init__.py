"""The computation: pure C++ (nanobind extension `tilesight._core`), reached only through
gpuTilingPerfHWModel/interfaceAndRun/runner.py — never a public entry point of its own.

  gpu_top/          orchestrator: lowering (shape+tile -> a tile execution plan) and the
                     wave-decomposition scheduler, per op kind (gemm, attention, comm)
  shader_core/       one core's per-lane time; the steady-state K-loop round formula
  shader_slice/      grid -> waves; the slice-shared L1
  on_chip_buffer/    the optional staging SRAM between the shader slices and the memory slices
  memory_slice/      the L2 port + DMA port + HBM behind it
  common/cache/       the deterministic tile-level cache simulation shared by the two above

`dse/` (sweeps a config many times) and `memory.py` (HBM capacity accounting) stay Python: they
orchestrate many runs or do standalone capacity math, not part of one run's execution.
"""
