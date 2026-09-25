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
import sys

# `python -m tilesight.cli.cli serve` (and the `tilesight` console script) import this package
# BEFORE any code in cli.py runs — that's how Python resolves a dotted module path, there is no
# way for cli.py to "get in first". So `serve`'s whole point (bind the socket, show build
# progress in the browser, build in the background) has to be decided right here, from the raw
# process argv, not from inside cli.py. `argv[1]` is always this CLI's subcommand name, and
# "serve" isn't a value any other subcommand takes there, so this is unambiguous in practice.
_DEFERRED_BUILD = len(sys.argv) > 1 and sys.argv[1] == "serve"

from tilesight import _core_builder  # noqa: E402

if not _DEFERRED_BUILD:
    _core_builder.ensure_core_built()

from tilesight.gpuTilingPerfHWModel.interfaceAndRun.hardware_spec import HardwareSpec  # noqa: E402
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.run_config import RunConfig  # noqa: E402
from tilesight.gpuTilingPerfHWModel.interfaceAndRun.model_spec import ModelSpec  # noqa: E402

if _DEFERRED_BUILD:
    # cli/server_boot.py builds tilesight._core itself, in a background thread, once the socket
    # is already bound; only after that does anything actually need CurModelConfig/run, at which
    # point cli/server.py imports them straight from runner.py (not from here) and they work.
    CurModelConfig = run = run_model = None
else:
    from tilesight.gpuTilingPerfHWModel.interfaceAndRun.runner import CurModelConfig, run, run_model  # noqa: E402

__all__ = ["CurModelConfig", "HardwareSpec", "ModelSpec", "RunConfig", "run", "run_model"]
