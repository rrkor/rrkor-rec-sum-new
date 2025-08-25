import os
import socket
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtNetwork import QNetworkProxyFactory
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication
from flask import send_from_directory, jsonify

import main as backend

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-proxy-server")

DEFAULT_HOST = "127.0.0.1"
FRONT_ENV = os.environ.get("FRONT_BUILD_DIR", "").strip()

def find_frontend_dir(base: Path) -> Path:
    candidates = []
    if FRONT_ENV:
        candidates.append(Path(FRONT_ENV))
    candidates += [
        base / "static",
        base / "web",
        base / "frontend" / "dist",
        base / "frontend" / "build",
        base / "dist",
        base / "build",
    ]
    for p in candidates:
        if (p / "index.html").exists():
            return p
    raise RuntimeError("Нет сборки фронта: положи полный build (index.html + assets/*) в ./static")

def find_free_port(start=5000, end=5099) -> int:
    for port in range(start, end + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((DEFAULT_HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("Нет свободного порта 5000–5099")

def _route_exists(app, rule: str) -> bool:
    try:
        for r in app.url_map.iter_rules():
            if r.rule == rule:
                return True
    except Exception:
        pass
    return False

def patch_backend_static(app, web_dir: Path):
    app.static_folder = str(web_dir)

    def serve_index():
        return send_from_directory(app.static_folder, "index.html")

    if not _route_exists(app, "/"):
        app.add_url_rule("/", "index_desktop_launcher", serve_index, methods=["GET"])

    @app.route("/<path:filename>", methods=["GET"])
    def _static_catch_all__desktop_launcher(filename: str):
        candidate = Path(app.static_folder) / filename
        if candidate.is_file():
            return send_from_directory(app.static_folder, filename)
        if filename.startswith("api"):
            return jsonify(error="Not found"), 404
        return serve_index()

def run_flask(app, port: int):
    app.run(host=DEFAULT_HOST, port=port, debug=False, use_reloader=False, threaded=True)

def wait_http_ok(url: str, timeout_sec: int = 10):
    import urllib.request, urllib.error
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            with urllib.request.urlopen(url) as resp:
                if resp.status < 500:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(0.2)
    return False

def main():
    base_dir = Path(backend.__file__).parent.resolve()
    web_dir = find_frontend_dir(base_dir)
    print(f"[desktop] frontend dir: {web_dir}")
    patch_backend_static(backend.app, web_dir)

    port = find_free_port()
    t = threading.Thread(target=run_flask, args=(backend.app, port), daemon=True)
    t.start()

    url = f"http://{DEFAULT_HOST}:{port}/"
    wait_http_ok(url, timeout_sec=10)

    QNetworkProxyFactory.setUseSystemConfiguration(False)
    qt_app = QApplication(sys.argv)
    window = QWebEngineView()
    window.setWindowTitle("Giga Chat Scribe")
    window.resize(1200, 800)
    window.load(QUrl(url))
    window.show()
    sys.exit(qt_app.exec_())

if __name__ == "__main__":
    main()