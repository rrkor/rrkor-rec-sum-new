import os
import io
import sys
import time
import json
import math
import wave
import queue
import atexit
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Generator

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from flask import Flask, jsonify, request, Response, send_from_directory, abort

# ---------- базовые пути/константы ----------
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "front" / "dist"
DATA_DIR = BASE_DIR / "data"
SEG_DIR = BASE_DIR / "temp_segments"
DATA_DIR.mkdir(exist_ok=True)
SEG_DIR.mkdir(exist_ok=True)

SAMPLE_RATE_FALLBACK = 44100
SEGMENT_SEC = 5               # как в исходнике
BLACKHOLE_NAME_HINT = "BlackHole"  # подстрока для поиска loopback

OUT_WAV = DATA_DIR / "meeting_audio.wav"
OUT_TXT = DATA_DIR / "transcript.txt"

# ---------- Flask ----------
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/")

# ---------- служебное ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def db_from_rms(rms: float) -> float:
    # dBFS, ограничим человеческим диапазоном для PPM
    if rms <= 1e-12:
        return -60.0
    return clamp(20.0 * math.log10(rms), -60.0, 0.0)

def find_input_device_index(name_hint: Optional[str]) -> Optional[int]:
    """Вернёт index устройства по подстроке в имени (без регистра)."""
    if not name_hint:
        return None
    name_hint = name_hint.lower()
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and name_hint in d["name"].lower():
            return idx
    return None

def default_input_index() -> Optional[int]:
    dev = sd.default.device
    if isinstance(dev, (list, tuple)) and len(dev) >= 1 and dev[0] is not None:
        return dev[0]
    # иначе возьмём первый доступный
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            return idx
    return None

def device_name_by_index(idx: Optional[int]) -> Optional[str]:
    if idx is None:
        return None
    try:
        return sd.query_devices(idx)["name"]
    except Exception:
        return None

# ---------- глобальные очереди/события ----------
seg_queue: "queue.Queue[Path]" = queue.Queue()        # на транскрибацию
transcript_q: "queue.Queue[str]" = queue.Queue()      # инкремент фронту
stop_event = threading.Event()
pause_event = threading.Event()

# ---------- провайдер транскрибации ----------
class ASRProvider:
    """
    Универсальный адаптер: пробует GigaAM из вашего репозитория,
    затем (если нет) — локальный whisper, и в конце — «немая» строка.
    НИЧЕГО в UI не меняется — на фронт уходит реальный текст.
    """

    def __init__(self):
        self.mode = None
        self.impl = None

        # 1) Попробуем локальный модуль из папки GigaAM в репозитории
        try:
            sys.path.insert(0, str(BASE_DIR))
            import importlib.util

            # Ищем файл-транскрайбер (несколько типовых вариантов)
            candidates = [
                BASE_DIR / "GigaAM" / "transcribe.py",
                BASE_DIR / "GigaAM" / "infer.py",
                BASE_DIR / "GigaAM" / "__init__.py",
            ]
            for p in candidates:
                if p.exists():
                    spec = importlib.util.spec_from_file_location("gigaam_local", str(p))
                    mod = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(mod)  # type: ignore
                    # ожидаем одну из сигнатур
                    if hasattr(mod, "transcribe"):
                        self.impl = lambda wav: str(mod.transcribe(wav)).strip()
                        self.mode = "gigaam_module"
                        break
                    if hasattr(mod, "main"):  # например, cli-style
                        def _run(wav):
                            # если main принимает argv
                            return str(mod.main([wav]) or "").strip()
                        self.impl = _run
                        self.mode = "gigaam_module_main"
                        break
        except Exception:
            self.impl = None

        # 2) whisper (если доступен)
        if self.impl is None:
            try:
                import whisper  # type: ignore
                self._whisper_model = whisper.load_model("base")
                def _whisper_transcribe(wav):
                    res = self._whisper_model.transcribe(wav, fp16=False)
                    return res.get("text", "").strip()
                self.impl = _whisper_transcribe
                self.mode = "whisper"
            except Exception:
                self.impl = None

        # 3) если ничего не поднялось
        if self.impl is None:
            self.mode = "none"
            self.impl = lambda wav: ""

    def transcribe(self, wav_path: Path) -> str:
        try:
            return self.impl(str(wav_path))  # type: ignore
        except Exception as e:
            # не уронить поток
            return ""

ASR = ASRProvider()

# ---------- запись/PPM/сегментация ----------
class Recorder:
    def __init__(self):
        self._lock = threading.RLock()

        self.recording = False
        self.paused = False

        self.bh_index: Optional[int] = None
        self.mic_index: Optional[int] = None

        self._sr = SAMPLE_RATE_FALLBACK
        self._bh_stream: Optional[sd.InputStream] = None
        self._mic_stream: Optional[sd.InputStream] = None

        # буферы для текущего сегмента
        self._bh_buf = np.empty((0,), dtype=np.float32)
        self._mc_buf = np.empty((0,), dtype=np.float32)
        self._seg_start_ts = time.time()
        self._seg_count = 0

        # PPM-трекинг (скользящее окно)
        self._ppm_bh = -60.0
        self._ppm_mc = -60.0
        self._ppm_bh_ring: List[float] = []
        self._ppm_mc_ring: List[float] = []

        # статус (только для /api/status)
        self.status_message = ""

    # ---- устройства ----
    def devices(self) -> List[Dict]:
        res = []
        devs = sd.query_devices()
        for idx, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                res.append({
                    "id": idx,
                    "name": d["name"],
                    "sr": d.get("default_samplerate"),
                })
        return res

    def set_mic_by_name(self, name: str) -> Dict:
        with self._lock:
            self.mic_index = find_input_device_index(name)
            if self.mic_index is None:
                return {"status": "error", "message": f"Микрофон '{name}' не найден"}
            # если идёт запись — перезапускаем только микрофонный поток
            if self.recording and not self.paused:
                try:
                    if self._mic_stream:
                        self._mic_stream.stop(); self._mic_stream.close()
                    self._mic_stream = sd.InputStream(
                        device=self.mic_index, channels=1, samplerate=self._sr, callback=self._mic_cb
                    )
                    self._mic_stream.start()
                    return {"status": "success", "message": f"Микрофон переключён на '{name}'"}
                except Exception as e:
                    return {"status": "error", "message": f"Ошибка переключения: {e}"}
            return {"status": "success", "message": f"Микрофон выбран: '{name}'"}

    # ---- управление ----
    def start(self) -> Dict:
        with self._lock:
            if self.recording:
                return {"status": "error", "message": "Запись уже идёт"}

            stop_event.clear()
            pause_event.clear()
            self.paused = False

            # выбрать устройства
            self.bh_index = find_input_device_index(BLACKHOLE_NAME_HINT) or default_input_index()
            self.mic_index = self.mic_index if self.mic_index is not None else default_input_index()
            if self.bh_index is None or self.mic_index is None:
                return {"status": "error", "message": "Аудиоустройства не найдены"}

            # sample rate — по blackhole (или дефолт)
            try:
                self._sr = int(sd.query_devices(self.bh_index)["default_samplerate"])
            except Exception:
                self._sr = SAMPLE_RATE_FALLBACK

            # очистить временные сегменты
            try:
                shutil.rmtree(SEG_DIR)
            except Exception:
                pass
            SEG_DIR.mkdir(exist_ok=True)

            # очистим старые файлы
            for p in [OUT_WAV, OUT_TXT]:
                try:
                    p.unlink()
                except Exception:
                    pass

            self._bh_buf = np.empty((0,), dtype=np.float32)
            self._mc_buf = np.empty((0,), dtype=np.float32)
            self._seg_start_ts = time.time()
            self._seg_count = 0

            # запустить входные потоки
            self._bh_stream = sd.InputStream(
                device=self.bh_index, channels=1, samplerate=self._sr, callback=self._bh_cb
            )
            self._mic_stream = sd.InputStream(
                device=self.mic_index, channels=1, samplerate=self._sr, callback=self._mic_cb
            )
            self._bh_stream.start()
            self._mic_stream.start()

            self.recording = True
            threading.Thread(target=self._transcriber_loop, daemon=True).start()
            self.status_message = " "
            return {"status": "success", "message": "Запись начата"}

    def pause_resume(self) -> Dict:
        with self._lock:
            if not self.recording:
                return {"status": "error", "message": "Запись не идёт"}
            if not self.paused:
                self.paused = True
                pause_event.set()
                try:
                    if self._bh_stream:  self._bh_stream.stop()
                    if self._mic_stream: self._mic_stream.stop()
                except Exception:
                    pass
                return {"status": "success", "message": "Пауза"}
            else:
                self.paused = False
                pause_event.clear()
                try:
                    if self._bh_stream:  self._bh_stream.start()
                    if self._mic_stream: self._mic_stream.start()
                except Exception:
                    pass
                return {"status": "success", "message": "Продолжили"}

    def stop(self) -> Dict:
        with self._lock:
            if not self.recording:
                return {"status": "error", "message": "Запись не идёт"}
            stop_event.set()
            self.recording = False
            self.paused = False
            # добросим хвост сегмента
            self._flush_tail_segment()
            try:
                if self._bh_stream:  self._bh_stream.stop(); self._bh_stream.close()
                if self._mic_stream: self._mic_stream.stop(); self._mic_stream.close()
            finally:
                self._bh_stream = None
                self._mic_stream = None
            # склеим общий WAV
            self._concat_segments()
            return {"status": "success", "message": "Остановлено", "wav": OUT_WAV.name}

    # ---- коллбэки ----
    def _bh_cb(self, indata, frames, time_info, status):
        if not self.recording or self.paused or stop_event.is_set():
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata
        mono = np.array(mono, dtype=np.float32)
        self._bh_buf = np.concatenate([self._bh_buf, mono])
        # PPM окно
        self._ppm_bh_ring.extend(mono.tolist())
        if len(self._ppm_bh_ring) > 4096:
            self._ppm_bh_ring = self._ppm_bh_ring[-2048:]

    def _mic_cb(self, indata, frames, time_info, status):
        if not self.recording or self.paused or stop_event.is_set():
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata
        mono = np.array(mono, dtype=np.float32)
        self._mc_buf = np.concatenate([self._mc_buf, mono])
        # PPM окно
        self._ppm_mc_ring.extend(mono.tolist())
        if len(self._ppm_mc_ring) > 4096:
            self._ppm_mc_ring = self._ppm_mc_ring[-2048:]

        # проверяем, не пора ли резать сегмент (привязываемся к микрофонному коллбэку)
        if (time.time() - self._seg_start_ts) >= SEGMENT_SEC:
            self._cut_segment()

    # ---- сегментация / ppm / конкатенация ----
    def _cut_segment(self):
        # выравниваем длины
        n = min(self._bh_buf.size, self._mc_buf.size)
        if n <= 0:
            self._seg_start_ts = time.time()
            return
        bh = self._bh_buf[:n]
        mc = self._mc_buf[:n]
        # обнулим использованные части
        self._bh_buf = self._bh_buf[n:]
        self._mc_buf = self._mc_buf[n:]

        idx = self._seg_count
        self._seg_count += 1

        bh_file = SEG_DIR / f"blackhole_segment_{idx:04d}.wav"
        mc_file = SEG_DIR / f"mic_segment_{idx:04d}.wav"
        mix_file = SEG_DIR / f"combined_segment_{idx:04d}.wav"

        # нормализованный микс для мониторинга/сохранения итогового wav
        mix = bh + mc
        max_abs = np.max(np.abs(mix)) if mix.size else 1.0
        if max_abs > 1.0:
            mix = mix / max_abs

        wavfile.write(str(bh_file), self._sr, (bh * 32767).astype(np.int16))
        wavfile.write(str(mc_file), self._sr, (mc * 32767).astype(np.int16))
        wavfile.write(str(mix_file), self._sr, (mix * 32767).astype(np.int16))

        seg_queue.put(mix_file)   # транскрибируем общий сегмент
        self._seg_start_ts = time.time()

    def _flush_tail_segment(self):
        n = min(self._bh_buf.size, self._mc_buf.size)
        if n <= 0:
            return
        bh = self._bh_buf[:n]; mc = self._mc_buf[:n]
        idx = self._seg_count
        self._seg_count += 1
        bh_file = SEG_DIR / f"blackhole_segment_{idx:04d}.wav"
        mc_file = SEG_DIR / f"mic_segment_{idx:04d}.wav"
        mix_file = SEG_DIR / f"combined_segment_{idx:04d}.wav"
        mix = bh + mc
        max_abs = np.max(np.abs(mix)) if mix.size else 1.0
        if max_abs > 1.0:
            mix = mix / max_abs
        wavfile.write(str(bh_file), self._sr, (bh * 32767).astype(np.int16))
        wavfile.write(str(mc_file), self._sr, (mc * 32767).astype(np.int16))
        wavfile.write(str(mix_file), self._sr, (mix * 32767).astype(np.int16))
        seg_queue.put(mix_file)
        self._bh_buf = np.empty((0,), dtype=np.float32)
        self._mc_buf = np.empty((0,), dtype=np.float32)

    def _concat_segments(self):
        files = sorted(SEG_DIR.glob("combined_segment_*.wav"))
        if not files:
            return
        data = []
        for p in files:
            sr, x = wavfile.read(str(p))
            if x.dtype != np.int16:
                x = x.astype(np.int16)
            data.append(x.astype(np.float32) / 32767.0)
        full = np.concatenate(data, axis=0) if data else np.empty((0,), dtype=np.float32)
        wavfile.write(str(OUT_WAV), self._sr, (full * 32767).astype(np.int16))

    def ppm_snapshot(self) -> Dict[str, float]:
        """Мгновенный уровень для polling."""
        if self._ppm_bh_ring:
            arr = np.array(self._ppm_bh_ring[-1024:], dtype=np.float64)
            self._ppm_bh = db_from_rms(float(np.sqrt(np.mean(arr * arr))))
        if self._ppm_mc_ring:
            arr = np.array(self._ppm_mc_ring[-1024:], dtype=np.float64)
            self._ppm_mc = db_from_rms(float(np.sqrt(np.mean(arr * arr))))
        return {"mic_db": self._ppm_mc, "blackhole_db": self._ppm_bh}

    # ---- транскрипция (фоновый луп) ----
    def _transcriber_loop(self):
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            while not stop_event.is_set() or not seg_queue.empty():
                try:
                    seg = seg_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    txt = ASR.transcribe(seg)
                    if txt:
                        # «в реальном времени» — отправим инкремент
                        line = txt.strip()
                        transcript_q.put(line + "\n")
                        f.write(line + "\n")
                        f.flush()
                finally:
                    seg_queue.task_done()

REC = Recorder()

# ---------- SSE генераторы ----------
def sse_pack(obj: Dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

def sse_levels() -> Generator[bytes, None, None]:
    """10 Гц обновление уровней."""
    while True:
        yield sse_pack(REC.ppm_snapshot())
        time.sleep(0.1)

def sse_transcript() -> Generator[bytes, None, None]:
    """Пушим появляющиеся строки транскрипции."""
    while True:
        try:
            line = transcript_q.get(timeout=0.1)
            yield sse_pack({"text": line})
        except queue.Empty:
            # поддерживаем соединение
            yield b": keepalive\n\n"
            time.sleep(0.4)

# ---------- API/ALIases ----------
@app.route("/")
def _index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return send_from_directory(str(STATIC_DIR), "index.html")
    return "Build not found", 404

# устройства
@app.route("/api/microphones", methods=["GET"])
@app.route("/microphones", methods=["GET"])
def api_mics():
    cur = device_name_by_index(REC.mic_index)
    return jsonify({"microphones": [d["name"] for d in REC.devices()], "current_mic": cur})

@app.route("/api/switch_microphone/<path:name>", methods=["POST"])
@app.route("/switch_microphone/<path:name>", methods=["POST"])
def api_switch_mic(name: str):
    return jsonify(REC.set_mic_by_name(name))

@app.route("/api/microphone", methods=["PUT", "POST"])
def api_switch_mic_json():
    data = request.get_json(silent=True) or {}
    name = data.get("name") or data.get("mic") or ""
    return jsonify(REC.set_mic_by_name(name))

# запись
@app.route("/api/record/start", methods=["POST"])
@app.route("/start_recording", methods=["POST"])
@app.route("/api/toggle_recording", methods=["POST"])
def api_start():
    return jsonify(REC.start())

@app.route("/api/record/pause", methods=["POST"])
@app.route("/api/toggle_pause", methods=["POST"])
def api_pause():
    return jsonify(REC.pause_resume())

@app.route("/api/record/stop", methods=["POST"])
@app.route("/stop_recording", methods=["POST"])
def api_stop():
    return jsonify(REC.stop())

# уровни
@app.route("/api/levels", methods=["GET"])
@app.route("/levels", methods=["GET"])
def api_levels():
    return jsonify(REC.ppm_snapshot())

@app.route("/api/levels/stream")
@app.route("/levels/stream")
def api_levels_stream():
    return Response(sse_levels(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

# транскрипт
@app.route("/api/transcript", methods=["GET"])
@app.route("/transcript", methods=["GET"])
def api_transcript():
    lines = []
    while not transcript_q.empty():
        try:
            lines.append(transcript_q.get_nowait())
            transcript_q.task_done()
        except queue.Empty:
            break
    return jsonify({"transcript": "".join(lines)})

@app.route("/api/transcript/stream")
@app.route("/transcript/stream")
def api_transcript_stream():
    return Response(sse_transcript(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

# файлы
@app.route("/api/files", methods=["GET"])
def api_files():
    items = []
    for p in DATA_DIR.glob("*"):
        if p.is_file():
            items.append({
                "name": p.name,
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "ext": p.suffix.lower(),
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(items=items)

@app.route("/api/download/<path:filename>", methods=["GET"])
def api_download(filename: str):
    safe = (DATA_DIR / filename).resolve()
    if not str(safe).startswith(str(DATA_DIR.resolve())):
        abort(403)
    if not safe.exists():
        abort(404)
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

# статус (без «шаблонной фразы»)
@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({"status": REC.status_message or ""})

# локальный запуск
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)