# desktop_app.py
import os
import sys
import time
import signal
import subprocess
import contextlib
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtGui import QIcon

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "8000"
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/"

def wait_backend(url: str, timeout_sec: int = 20) -> bool:
    import socket
    host_port = url.split("://", 1)[-1].split("/", 1)[0]
    if ":" in host_port:
        host, port_s = host_port.split(":", 2)
        port = int(port_s)
    else:
        host, port = host_port, 80
    started = time.time()
    while time.time() - started < timeout_sec:
        s = socket.socket()
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            time.sleep(0.2)
    return False

def start_backend() -> subprocess.Popen:
    # запускаем ваш FastAPI (main.py содержит uvicorn.run("main:app", ...))
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", BACKEND_HOST, "--port", BACKEND_PORT]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def main():
    # 1) поднимаем бэкенд, если не запущен
    need_kill = False
    proc = None
    if not wait_backend(BACKEND_URL, timeout_sec=1):
        proc = start_backend()
        need_kill = True
        if not wait_backend(BACKEND_URL, timeout_sec=25):
            out = ""
            with contextlib.suppress(Exception):
                if proc and proc.stdout:
                    out = proc.stdout.read().decode("utf-8", errors="ignore")
            QMessageBox.critical(None, "Ошибка", f"Бэкенд не стартовал на {BACKEND_URL}\n\n{out}")
            if need_kill and proc:
                with contextlib.suppress(Exception):
                    proc.terminate()
            sys.exit(1)

    # 2) окно с QWebEngineView
    app = QApplication(sys.argv)
    app.setApplicationName("Recorder & Summarizer")
    app.setOrganizationName("rrkor")
    icon_path = Path(__file__).with_name("app.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    view = QWebEngineView()
    view.setWindowTitle("Recorder & Summarizer")
    view.resize(1280, 800)

    # правильная настройка атрибутов через класс QWebEngineSettings
    s = view.settings()
    # большинство включено по умолчанию, но оставлю явно:
    s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
    s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    s.setAttribute(QWebEngineSettings.WebAttribute.ErrorPageEnabled, True)

    view.load(QUrl(BACKEND_URL))
    view.show()

    def cleanup():
        if proc and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
                deadline = time.time() + 3
                while proc.poll() is None and time.time() < deadline:
                    time.sleep(0.1)
                if proc.poll() is None:
                    proc.kill()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()