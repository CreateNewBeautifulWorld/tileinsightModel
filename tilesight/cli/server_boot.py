"""`tilesight serve` entry point — binds the socket and serves html/ immediately, and only then
builds tilesight._core in a background thread, so the browser sees a live page (with a
"building the C++ core..." banner, or a build error) instead of the terminal hanging before the
server exists at all.

Deliberately does not import the `tilesight` package at module scope (that would trigger the
build synchronously, defeating the point) or `tilesight.cli.server` (same reason, it imports the
package). Once the background build finishes, the real Handler (all the /api/* routes) is swapped
in and this module's role is done.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "html"
_CORE_BUILDER_PATH = Path(__file__).resolve().parent.parent / "_core_builder.py"

_STATUS_LOCK = threading.Lock()
_STATUS = {"stage": "building", "message": "starting..."}


def _set_status(stage: str, message: str = "") -> None:
    with _STATUS_LOCK:
        _STATUS["stage"] = stage
        _STATUS["message"] = message


def _load_core_builder():
    spec = importlib.util.spec_from_file_location("tilesight_core_builder_boot", _CORE_BUILDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CTYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
           ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml"}


class _BootHandler(BaseHTTPRequestHandler):
    server_version = "tilesight-boot"

    def log_message(self, fmt, *args):                      # quiet, same as the real server
        pass

    def _send(self, code: int, body: bytes, ctype="application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):                                        # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/build_status":
            with _STATUS_LOCK:
                return self._json(200, dict(_STATUS))
        if path in ("/", "/app", "/app.html"):
            return self._send(200, (WEB_DIR / "app.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/expert", "/index.html"):
            return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        # any other static asset under html/, same as the real server
        fpath = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR not in fpath.parents or not fpath.is_file():
            return self._json(404, {"error": "not found"})
        return self._send(200, fpath.read_bytes(), _CTYPES.get(fpath.suffix, "application/octet-stream"))

    def do_POST(self):                                       # noqa: N802
        with _STATUS_LOCK:
            st = dict(_STATUS)
        self._json(503, {"error": "tilesight core is still building, try again shortly",
                          "build_status": st})


def serve_with_autobuild(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), _BootHandler)
    print(f"tilesight: serving http://{host}:{port} (open it now — the C++ core builds in the "
          "background and the page will show progress)", file=sys.stderr)

    def _build_then_swap():
        core_builder = _load_core_builder()
        try:
            core_builder.ensure_core_built(status_cb=_set_status)
        except Exception as e:                                # noqa: BLE001
            _set_status("error", str(e))
            return
        from tilesight.cli.server import Handler as RealHandler
        httpd.RequestHandlerClass = RealHandler
        _set_status("ready")
        print("tilesight: C++ core ready, full API is live.", file=sys.stderr)

    threading.Thread(target=_build_then_swap, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
