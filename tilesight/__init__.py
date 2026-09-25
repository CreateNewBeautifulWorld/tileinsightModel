"""TileSight re-implementation: tile-centric analytical GPU performance model.

The model has exactly two inputs, built at the boundary (CLI flags / web request) and never
mixed with anything else once built:
  cur_gpu_config    a HardwareSpec — what the part is, including how it tiles (compute.tile_policy.*)
  cur_model_config  a CurModelConfig — what runs on it (architecture + run settings)

Quick start:
    from tilesight import CurModelConfig, HardwareSpec, ModelSpec, RunConfig, run
    cur_gpu_config = HardwareSpec.load("b300")
    cur_model_config = CurModelConfig(spec=ModelSpec.load("kimi_k2.hf"),
                                      run=RunConfig(phase="decode", batch=256, seq_len=8192, tp=1, dp=8))
    rep = run(cur_gpu_config, cur_model_config)

`run_model(model, cur_gpu_config, rc)` is the lower-level, three-argument engine entry point
that `run()` wraps; existing code and internal callers keep using it directly.
"""
def _ensure_core_built() -> None:
    """Build tilesight._core on first use if it isn't there yet, instead of failing on import.

    Only ever runs once per checkout: once build/_core*.so exists next to this file, the
    plain `import tilesight._core` below succeeds immediately and this function is a no-op.
    """
    try:
        import tilesight._core  # noqa: F401
        return
    except ImportError:
        pass

    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent
    build_dir = root / "build"
    print("tilesight: C++ core not built yet, building it now (one-time, ~1-2 min)...",
          file=sys.stderr)
    try:
        subprocess.run(
            ["cmake", "-S", str(root), "-B", str(build_dir),
             f"-DPython_EXECUTABLE={sys.executable}"],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build_dir), "-j"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ImportError(
            "tilesight: auto-build of the C++ core failed. Install a C++17 compiler and cmake, "
            "or build it yourself with `cmake -S . -B build && cmake --build build -j` from the "
            f"project root ({root}). Underlying error: {e}"
        ) from e

    import importlib
    importlib.invalidate_caches()
    import tilesight._core  # noqa: F401


_ensure_core_built()

from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import HardwareSpec
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.runner import CurModelConfig, run, run_model
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.model_spec import ModelSpec

__all__ = ["CurModelConfig", "HardwareSpec", "ModelSpec", "RunConfig", "run", "run_model"]
