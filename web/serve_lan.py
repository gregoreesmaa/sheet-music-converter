#!/usr/bin/env python3
"""Pure static LAN file server for the 100% client-side Sheet Music Converter web app.

ZERO backend inference or conversion endpoints — strictly serves static assets
from ``web/`` with Cross-Origin Isolation headers (COOP/COEP) required for
multi-threaded ``onnxruntime-web`` WASM + WebGPU, plus proper ``.wasm`` / ``.onnx``
MIME types.

Usage:
    python web/serve_lan.py              # serves web/ on http://0.0.0.0:8080
    python web/serve_lan.py --port 8088
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class StaticIsolatedHandler(SimpleHTTPRequestHandler):
    """Static-only handler with COOP/COEP headers and WASM/ONNX MIME types."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".mjs": "application/javascript; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".mscz": "application/octet-stream",
        ".musicxml": "application/vnd.recordare.musicxml+xml; charset=utf-8",
    }

    def end_headers(self) -> None:
        # Required for SharedArrayBuffer (multi-threaded ORT WASM) + WebGPU isolation.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        path_lower = self.path.split("?", 1)[0].lower()
        if path_lower.endswith((".onnx", ".wasm")):
            self.send_header("Cache-Control", "public, max-age=3600")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def detect_lan_ip() -> str:
    """Best-effort LAN IPv4 address without sending any network packets."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the 100% client-side Sheet Music Converter web app over LAN."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Static directory to serve (default: web/)",
    )
    args = parser.parse_args()

    serve_dir = args.dir.resolve()
    if not serve_dir.is_dir():
        print(f"Error: directory does not exist: {serve_dir}", file=sys.stderr)
        sys.exit(1)

    os.chdir(serve_dir)
    lan_ip = detect_lan_ip()

    server = ThreadingHTTPServer((args.host, args.port), StaticIsolatedHandler)
    print("Sheet Music Converter — 100% Client-Side Static Server")
    print(f"  Serving:   {serve_dir}")
    print(f"  Local URL: http://127.0.0.1:{args.port}/")
    print(f"  LAN URL:   http://{lan_ip}:{args.port}/")
    print("  Headers:   COOP=same-origin, COEP=require-corp (WebGPU + threaded WASM)")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
