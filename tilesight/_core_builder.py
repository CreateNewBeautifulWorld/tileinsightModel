"""Builds tilesight._core (the nanobind C++ extension) if it isn't there yet.

Standalone on purpose (stdlib only at module scope): this file is loaded two ways —
  1. normally, as tilesight._core_builder, by tilesight/__init__.py — any `import tilesight`
     triggers this synchronously, once, the first time.
  2. by file path (importlib, bypassing tilesight/__init__.py entirely) from
     cli/server_boot.py, so the web server can bind its socket and start answering
     /api/build_status BEFORE the build even starts, instead of the build blocking
     before the server exists to tell anyone about it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_core_built(status_cb=None) -> None:
    """status_cb(stage, message), stage one of building | ready | error."""
    def report(stage: str, message: str = "") -> None:
        if status_cb is not None:
            status_cb(stage, message)

    try:
        import tilesight._core  # noqa: F401
        report("ready")
        return
    except ImportError:
        pass

    build_dir = ROOT / "build"
    msg = "building the C++ core (first run, ~1-2 min)..."
    print(f"tilesight: {msg}", file=sys.stderr)
    report("building", msg)
    try:
        subprocess.run(
            ["cmake", "-S", str(ROOT), "-B", str(build_dir),
             f"-DPython_EXECUTABLE={sys.executable}"],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build_dir), "-j"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        err = (
            "C++ build failed. Install a C++17 compiler and cmake, or build manually with "
            f"`cmake -S . -B build && cmake --build build -j` from {ROOT}. Underlying error: {e}"
        )
        report("error", err)
        raise ImportError(f"tilesight: {err}") from e

    import importlib
    importlib.invalidate_caches()
    try:
        import tilesight._core  # noqa: F401
    except ImportError as e:
        report("error", str(e))
        raise
    report("ready")
