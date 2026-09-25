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
from .gpuTilingPerfHWModel.spec import HardwareSpec
from .model.run_config import RunConfig
from .model.runner import CurModelConfig, run, run_model
from .model.spec import ModelSpec

__all__ = ["CurModelConfig", "HardwareSpec", "ModelSpec", "RunConfig", "run", "run_model"]
