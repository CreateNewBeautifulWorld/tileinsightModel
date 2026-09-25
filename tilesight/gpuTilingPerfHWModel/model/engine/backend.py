"""Backend dispatch: C++ core (`tilesight._core`) if built, else the Python reference.

Select explicitly with env TILESIGHT_BACKEND=python|cpp.
"""
from __future__ import annotations

import os

from tilesight.gpuTilingPerfHWModel.model.engine import cache as _pycache
from tilesight.gpuTilingPerfHWModel.model.engine import reference as _pyref

_core = None
if os.environ.get("TILESIGHT_BACKEND", "auto") != "python":
    try:
        from tilesight import _core  # type: ignore  # compiled by CMake/scikit-build
    except ImportError:
        if os.environ.get("TILESIGHT_BACKEND") == "cpp":
            raise

BACKEND = "cpp" if _core is not None else "python"


def evaluate(kernel, cur_gpu_config):
    if _core is not None:
        from tilesight.gpuTilingPerfHWModel.model.engine.cpp_bridge import evaluate_cpp
        return evaluate_cpp(_core, kernel, cur_gpu_config)
    return _pyref.evaluate(kernel, cur_gpu_config)


def expected_misses(keys, streams, n_streams, assoc, cap_tiles):
    if _core is not None:
        return list(_core.expected_misses(keys, streams, n_streams, assoc, cap_tiles))
    return _pycache.expected_misses(keys, streams, n_streams, assoc, cap_tiles)
