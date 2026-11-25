#!/usr/bin/env python3
"""Pure static LAN file server (HTTP + HTTPS) for the 100% client-side Sheet Music Converter.

ZERO backend inference or conversion endpoints — strictly serves static assets
from ``web/`` with Cross-Origin Isolation headers (COOP/COEP/CORP) required for
multi-threaded ``onnxruntime-web`` WASM, SharedArrayBuffer, WebGPU, and mobile
camera access over LAN HTTPS.

Usage:
    python web/serve_lan.py                              # HTTP :8080 + HTTPS :8443
    python web/serve_lan.py --port 8081 --https-port 8444
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import ssl
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

_LOG_LOCK = threading.Lock()
_CLIENT_LOG_PATH: Path | None = None


class StaticIsolatedHandler(SimpleHTTPRequestHandler):
    """Static-only handler with COOP/COEP/CORP headers and WASM/ONNX MIME types."""

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

    def _is_forbidden_path(self) -> bool:
        raw_path = unquote(self.path.split("?", 1)[0]).replace("\\", "/").lower()
        parts = [p for p in raw_path.split("/") if p]
        if "certs" in parts:
            return True
        if raw_path.endswith((".key", ".pem", ".pfx", ".p12")):
            return True
        return False

    def do_GET(self) -> None:
        if self._is_forbidden_path():
            self.send_error(403, "Forbidden")
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._is_forbidden_path():
            self.send_error(403, "Forbidden")
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        """Crude Rollbar-like client telemetry/error sink (POST /__log)."""
        clean_path = self.path.split("?", 1)[0]
        if clean_path != "/__log":
            self.send_error(404, "Not Found")
            return
        try:
            length = min(65536, max(0, int(self.headers.get("Content-Length", "0"))))
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            except Exception:
                payload = {"msg": raw.decode("utf-8", errors="replace")[:2000]}
            if not isinstance(payload, dict):
                payload = {"msg": str(payload)[:2000]}
            rec = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "ip": self.client_address[0] if self.client_address else "unknown",
                **payload,
            }
            level = str(rec.get("level", "info")).upper()
            src = str(rec.get("src", "ui"))
            msg = str(rec.get("msg", ""))[:500]
            print(f"[CLIENT {level} | {rec['ip']} | {src}] {msg}", flush=True)
            if _CLIENT_LOG_PATH is not None:
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with _LOG_LOCK:
                    with _CLIENT_LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(line)
        except Exception as exc:
            print(f"[CLIENT LOG ERR] {exc}", file=sys.stderr, flush=True)
        self.send_response(204)
        self.end_headers()

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


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


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
        description="Serve the 100% client-side Sheet Music Converter web app over LAN (HTTP + HTTPS)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument(
        "--https-port",
        type=int,
        default=8443,
        help="HTTPS port (default: 8443; set 0 to disable HTTPS)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Static directory to serve (default: web/)",
    )
    parser.add_argument(
        "--cert",
        type=Path,
        default=None,
        help="TLS certificate PEM path (default: <dir>/certs/telkutaga.crt)",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=None,
        help="TLS private key PEM path (default: <dir>/certs/telkutaga.key)",
    )
    args = parser.parse_args()

    serve_dir = args.dir.resolve()
    if not serve_dir.is_dir():
        print(f"Error: directory does not exist: {serve_dir}", file=sys.stderr)
        sys.exit(1)

    cert_path = (args.cert or (serve_dir / "certs" / "telkutaga.crt")).resolve()
    key_path = (args.key or (serve_dir / "certs" / "telkutaga.key")).resolve()

    global _CLIENT_LOG_PATH
    _CLIENT_LOG_PATH = serve_dir / "client_logs.jsonl"
    os.chdir(serve_dir)
    lan_ip = detect_lan_ip()

    servers: list[tuple[str, ThreadingHTTPServer]] = []
    threads: list[threading.Thread] = []

    http_server = ReusableThreadingHTTPServer((args.host, args.port), StaticIsolatedHandler)
    servers.append(("HTTP", http_server))

    https_server: ThreadingHTTPServer | None = None
    if args.https_port > 0:
        if cert_path.is_file() and key_path.is_file():
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
            https_server = ReusableThreadingHTTPServer(
                (args.host, args.https_port), StaticIsolatedHandler
            )
            https_server.socket = ctx.wrap_socket(https_server.socket, server_side=True)
            servers.append(("HTTPS", https_server))
        else:
            print(
                f"Warning: TLS cert/key not found ({cert_path}, {key_path}); HTTPS disabled.",
                file=sys.stderr,
            )

    for label, srv in servers:
        t = threading.Thread(target=srv.serve_forever, name=f"serve-{label}", daemon=True)
        t.start()
        threads.append(t)

    print("Sheet Music Converter — 100% Client-Side Static Server", flush=True)
    print(f"  Serving:     {serve_dir}", flush=True)
    if https_server is not None:
        print(f"  HTTPS (LAN): https://telkutaga.local:{args.https_port}/", flush=True)
        print(f"  HTTPS (IP):  https://{lan_ip}:{args.https_port}/", flush=True)
        print(f"  TLS Cert:    {cert_path}", flush=True)
    print(f"  HTTP  (LAN): http://telkutaga.local:{args.port}/", flush=True)
    print(f"  HTTP  (IP):  http://{lan_ip}:{args.port}/", flush=True)
    print("  Headers:     COOP=same-origin, COEP=require-corp, CORP=same-origin", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        for _, srv in servers:
            srv.shutdown()
            srv.server_close()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
