from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/style.css": "style.css",
    "/script.js": "script.js",
}


def run_server(experiment: Any, host: str, port: int, root_dir: Path) -> None:
    root_dir = Path(root_dir).resolve()
    handler = _make_handler(experiment, root_dir)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"Serving interactive INR prototype at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        experiment.stop()
        server.server_close()


def _make_handler(experiment: Any, root_dir: Path) -> type[BaseHTTPRequestHandler]:
    class INRRequestHandler(BaseHTTPRequestHandler):
        server_version = "INRPrototype/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/state":
                self._send_json(experiment.get_state())
                return

            if path == "/api/images":
                self._send_json({"images": experiment.get_state().get("images", [])})
                return

            if path == "/media/reference.png":
                self._send_file(experiment.reference_path, "image/png", cache=False)
                return

            if path == "/media/latest.png":
                self._send_file(experiment.latest_path, "image/png", cache=False)
                return

            if path == "/media/latest.raw":
                body, width, height, revision = experiment.get_latest_raw()
                self._send_bytes(
                    body,
                    "application/octet-stream",
                    cache=False,
                    extra_headers={
                        "X-Image-Width": str(width),
                        "X-Image-Height": str(height),
                        "X-Image-Revision": str(revision),
                    },
                )
                return

            if path.startswith("/media/thumb/"):
                image_name = unquote(path.removeprefix("/media/thumb/"))
                try:
                    thumb_path = experiment.get_thumbnail_path(image_name)
                except Exception:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_file(thumb_path, "image/jpeg", cache=True)
                return

            if path in STATIC_ROUTES:
                file_path = root_dir / STATIC_ROUTES[path]
                content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                self._send_file(file_path, content_type, cache=True)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            payload = self._read_json()

            if path == "/api/start":
                state = experiment.start(
                    payload.get("epochs_per_preview", payload.get("epochs_per_update", 1))
                )
                self._send_json(state)
                return

            if path == "/api/stop":
                state = experiment.stop()
                self._send_json(state)
                return

            if path == "/api/step":
                state = experiment.step(payload.get("steps", 1))
                self._send_json(state)
                return

            if path == "/api/reset":
                state = experiment.reset()
                self._send_json(state)
                return

            if path == "/api/select":
                try:
                    state = experiment.select_image(payload.get("image", ""))
                except Exception as exc:
                    state = experiment.get_state()
                    state["error"] = str(exc)
                    self._send_json(state, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(state)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, file_path: Path, content_type: str, cache: bool) -> None:
            file_path = Path(file_path)
            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            body = file_path.read_bytes()
            self._send_bytes(body, content_type, cache)

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            cache: bool,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            if cache:
                self.send_header("Cache-Control", "public, max-age=60")
            else:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return INRRequestHandler
