"""TileSight re-implementation: tile-centric analytical GPU performance model.

Quick start:
    from tilesight import HardwareSpec, ModelSpec, RunConfig, run_model
    rep = run_model(ModelSpec.load("kimi_k2.hf"), HardwareSpec.load("b300"),
                    RunConfig(phase="decode", batch=256, seq_len=8192, tp=1, dp=8))
"""
from .hw.spec import HardwareSpec
from .model.run_config import RunConfig
from .model.runner import run_model
from .model.spec import ModelSpec

__all__ = ["HardwareSpec", "ModelSpec", "RunConfig", "run_model"]
