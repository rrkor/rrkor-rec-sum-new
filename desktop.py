# desktop.py
import os
import sys
import time
import socket
import threading
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtNetwork import QNetworkProxyFactory

from flask import send_from_directory, jsonify, request
import main as backend  # НЕ трогаем вашу бизнес-логику, только статику

# без прокси для локалки
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-proxy-server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

# фиксированный путь к билду фронта
BASE_DIR = Path(backend.__file__).parent.resolve()
FRONT_DIR = BASE_DIR / "front" / "dist"   # <— ТОЛЬКО здесь ищем фронт

def patch_static_and_spa(app):
    """
    Статически отдаём front/dist, плюс SPA-fallback.
    /api/* не трогаем.
    """
    if not FRONT_DIR.exists() or not (FRONT_DIR / "index.html").exists():
        raise RuntimeError(f"Не найден фронт-билд: {FRONT_DIR}. Собери его в front/dist.")

    app.static_folder = str(FRONT_DIR)

    def serve_index():
        return send_from_directory(app.static_folder, "index.html")

    # Корень
    if "/" not in {r.rule for r in app.url_map.iter_rules()}:
        app.add_url_rule("/", "index_desktop_launcher", serve_index, methods=["GET"])

    # Отдача ассетов и SPA-фоллбек
    @app.route("/<path:filename>", methods=["GET"])
    def _static_or_spa(filename: str):
        # если файл существует — отдать как статику
        candidate = FRONT_DIR / filename
        if candidate.is_file():
            return send_from_directory(app.static_folder, filename)
        # API — не перехватываем
        if filename.startswith("api"):
            return jsonify(error="Not found"), 404
        # иначе это SPA-роут
        return serve_index()

def run_flask(app):
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False, use_reloader=False, threaded=True)

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
    # подменяем только статику/SPA. вся ваша логика и роуты остаются как есть
    patch_static_and_spa(backend.app)

    t = threading.Thread(target=run_flask, args=(backend.app,), daemon=True)
    t.start()

    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/"
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