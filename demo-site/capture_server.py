"""Local-only sink that saves canvas PNGs posted by the demo page during capture.

Bound to loopback; accepts a fixed whitelist of filenames so a stray request
cannot write outside docs/img.
"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "img")
ALLOWED = {"fig-arch.png", "fig-moe-step1.png", "fig-moe-step900.png", "fig-heat.png"}


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        name = self.path.lstrip("/")
        if name not in ALLOWED:
            self.send_response(403)
            self._cors()
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(b"ok")
        print(f"saved {name} {len(data)} bytes", flush=True)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    HTTPServer(("127.0.0.1", 4174), H).serve_forever()