from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the static WebGPU INR app.")
    parser.add_argument(
        "--host",
        default=os.environ.get("INR_HOST", "127.0.0.1"),
        help="Host/interface to bind. Defaults to INR_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("INR_PORT", "8000")),
        help="Port to bind. Defaults to INR_PORT or 8000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), StaticHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {ROOT} at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
