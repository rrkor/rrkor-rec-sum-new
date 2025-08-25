import os
import queue
import shutil
import threading
import time
from pathlib import Path
from threading import Lock

import gigaam
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory, abort
from gigachat import GigaChat

# -----------------------------
# Базовые настройки / константы
# -----------------------------
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR
TEMP_DIR = BASE_DIR / "temp_segments"

SAMPLE_RATE = 44100
SEGMENT_DURATION = 5  # вернул как в исходном коде
OUTPUT_WAV = str(DATA_DIR / "meeting_audio.wav")
OUTPUT_TXT = str(DATA_DIR / "transcript.txt")

BLACKHOLE_DEVICE = "BlackHole 2ch"
DEFAULT_SILENCE_THRESHOLD = 0.01
DEFAULT_GAIN = 1.0

# Очереди/события для потоков
audio_queue: "queue.Queue[tuple[str,str,str,float]]" = queue.Queue()
transcript_queue: "queue.Queue[str]" = queue.Queue()
stop_event = threading.Event()
pause_event = threading.Event()
status_lock = Lock()

# Каталоги
TEMP_DIR.mkdir(exist_ok=True)

# Flask (desktop.py подменяет static при необходимости)
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')


# -----------------------------
# Вспомогательные функции
# -----------------------------
def db_from_rms(rms: float) -> float:
    """RMS -> dBFS, ограничиваем диапазон [-60; 0] для PPM."""
    db = 20.0 * np.log10(max(rms, 1e-12))
    return float(max(-60.0, min(0.0, db)))


# -----------------------------
# Класс записи/ASR
# -----------------------------
class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.paused = False

        self.recording_thread: threading.Thread | None = None
        self.transcribe_thread: threading.Thread | None = None

        self.blackhole_stream: sd.InputStream | None = None
        self.mic_stream: sd.InputStream | None = None

        self.stream_lock = Lock()

        self.mic_gain = DEFAULT_GAIN
        self.blackhole_gain = DEFAULT_GAIN
        self.mic_silence_threshold = DEFAULT_SILENCE_THRESHOLD
        self.blackhole_silence_threshold = DEFAULT_SILENCE_THRESHOLD

        self.current_mic = ""
        mics = self.get_available_microphones()
        if mics:
            self.current_mic = mics[0]

        # буферы для расчёта PPM
        self.blackhole_buffer: list[float] = []
        self.mic_buffer: list[float] = []
        self.blackhole_level_val = -60.0
        self.mic_level_val = -60.0

        # текст/саммари
        self.transcript = ""
        self.summary = ""

        # статус (показывает фронт)
        self.status_message = ""  # убрал «шаблонную фразу» при старте

        # инициализация клиентов
        load_dotenv()
        giga_auth = os.getenv("GIGACHAT_CREDENTIALS")
        scope = os.getenv("GIGACHAT_SCOPE")
        # SSL проверку отключаем как было в вашем коде
        self.gigachat = GigaChat(credentials=giga_auth, scope=scope, model="GigaChat-2-max", verify_ssl_certs=False)

    # ---- служебные ----
    def log_status(self, message: str):
        with status_lock:
            self.status_message = message
        print(message)

    def get_available_microphones(self) -> list[str]:
        devices = sd.query_devices()
        return [dev['name'] for dev in devices if dev.get('max_input_channels', 0) > 0]

    def is_silent(self, audio_data: np.ndarray, threshold: float) -> bool:
        if audio_data.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio_data.astype(np.float64)))))
        return rms < threshold

    def update_levels(self):
        """
        Обновляем PPM по последним ~1024 сэмплам с каждого входа.
        Вызывается из /api/levels (poll со стороны фронта).
        """
        if self.blackhole_buffer and self.mic_buffer:
            try:
                bh = self.blackhole_buffer[-1024:] if len(self.blackhole_buffer) >= 1024 else self.blackhole_buffer
                mc = self.mic_buffer[-1024:] if len(self.mic_buffer) >= 1024 else self.mic_buffer
                if bh and mc:
                    bh_rms = float(np.sqrt(np.mean(np.square(np.array(bh, dtype=np.float64)))))
                    mc_rms = float(np.sqrt(np.mean(np.square(np.array(mc, dtype=np.float64)))))
                    self.blackhole_level_val = db_from_rms(bh_rms)
                    self.mic_level_val = db_from_rms(mc_rms)
            except Exception as e:
                self.log_status(f"Ошибка обновления уровней: {e}")

    # ---- управление устройствами ----
    def switch_microphone(self, mic_name: str) -> dict:
        """
        Горячая смена микрофона даже во время записи.
        """
        self.current_mic = mic_name
        with self.stream_lock:
            if self.recording and not self.paused and self.mic_stream is not None:
                try:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                except Exception:
                    pass
                # Запускаем новый поток ввода на новый девайс
                try:
                    self.mic_stream = sd.InputStream(
                        samplerate=SAMPLE_RATE, channels=1, device=mic_name, callback=self._mic_callback
                    )
                    self.mic_stream.start()
                    self.log_status(f"Микрофон переключен на {mic_name}")
                    return {"status": "success", "message": f"Микрофон переключен на {mic_name}"}
                except Exception as e:
                    self.log_status(f"Ошибка переключения микрофона: {e}")
                    return {"status": "error", "message": f"Ошибка переключения микрофона: {e}"}
        self.log_status(f"Микрофон переключен на {mic_name}")
        return {"status": "success", "message": f"Микрофон переключен на {mic_name}"}

    # ---- запись/пауза/стоп ----
    def toggle_recording(self) -> dict:
        if self.recording:
            return {"status": "error", "message": "Запись уже идет"}
        try:
            stop_event.clear()
            pause_event.clear()
            self.recording = True
            self.paused = False

            # очистим предыдущие сегменты
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(parents=True, exist_ok=True)

            self.transcript = ""
            # старт потоков: запись + транскрибирование
            self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.transcribe_thread = threading.Thread(target=self._transcribe_segments, daemon=True)
            self.recording_thread.start()
            self.transcribe_thread.start()
            self.log_status("Запись началась")
            return {"status": "success", "message": "Запись началась"}
        except Exception as e:
            self.recording = False
            self.log_status(f"Ошибка начала записи: {e}")
            return {"status": "error", "message": f"Ошибка начала записи: {e}"}

    def toggle_pause(self) -> dict:
        if not self.recording:
            return {"status": "error", "message": "Невозможно поставить на паузу"}
        if not self.paused:
            self.paused = True
            pause_event.set()
            with self.stream_lock:
                try:
                    if self.blackhole_stream:
                        self.blackhole_stream.stop()
                    if self.mic_stream:
                        self.mic_stream.stop()
                except Exception:
                    pass
            self.log_status("Запись на паузе")
            return {"status": "success", "message": "Запись на паузе"}
        else:
            self.paused = False
            pause_event.clear()
            with self.stream_lock:
                try:
                    if self.blackhole_stream:
                        self.blackhole_stream.start()
                    if self.mic_stream:
                        self.mic_stream.start()
                except Exception:
                    pass
            self.log_status("Запись возобновлена")
            return {"status": "success", "message": "Запись возобновлена"}

    def stop_recording(self) -> dict:
        if not self.recording:
            return {"status": "error", "message": "Запись не идёт"}
        self.recording = False
        stop_event.set()
        self.log_status("Остановка записи...")
        with self.stream_lock:
            try:
                if self.blackhole_stream:
                    self.blackhole_stream.stop()
                    self.blackhole_stream.close()
                if self.mic_stream:
                    self.mic_stream.stop()
                    self.mic_stream.close()
            except Exception:
                pass
            finally:
                self.blackhole_stream = None
                self.mic_stream = None

        # склеить остатки буферов в финальный WAV (на случай незамкнувшегося сегмента)
        self._flush_tail_to_queue()
        self._concat_all_segments_to_wav()
        return {"status": "success", "message": "Запись остановлена", "wav": Path(OUTPUT_WAV).name}

    # ---- аудио-потоки ----
    def _blackhole_callback(self, indata, frames, time_info, status):
        if self.recording and not self.paused and not stop_event.is_set():
            # накопление сегмента
            self._bh_recording.append((indata.copy() * self.blackhole_gain).astype(np.float32))
            # PPM-буфер
            ch = indata[:, 0] if indata.ndim > 1 else indata
            self.blackhole_buffer.extend(map(float, ch))
            if len(self.blackhole_buffer) > 4096:
                self.blackhole_buffer = self.blackhole_buffer[-2048:]

    def _mic_callback(self, indata, frames, time_info, status):
        if self.recording and not self.paused and not stop_event.is_set():
            self._mic_recording.append((indata.copy() * self.mic_gain).astype(np.float32))
            ch = indata[:, 0] if indata.ndim > 1 else indata
            self.mic_buffer.extend(map(float, ch))
            if len(self.mic_buffer) > 4096:
                self.mic_buffer = self.mic_buffer[-2048:]

    def _record_audio(self):
        # найти BlackHole
        blackhole_device = None
        for dev in sd.query_devices():
            if dev.get('max_input_channels', 0) > 0 and BLACKHOLE_DEVICE in dev['name']:
                blackhole_device = dev['name']
                break
        if not blackhole_device:
            self.log_status(f"Устройство {BLACKHOLE_DEVICE} не найдено")
            self.recording = False
            stop_event.set()
            return

        # инициализация буферов сегмента
        self._bh_recording: list[np.ndarray] = []
        self._mic_recording: list[np.ndarray] = []
        segment_count = 0
        segment_started_at = time.time()

        # открыть два входных потока
        try:
            with self.stream_lock:
                self.blackhole_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, device=blackhole_device, callback=self._blackhole_callback
                )
                self.mic_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, device=self.current_mic or None, callback=self._mic_callback
                )
                self.blackhole_stream.start()
                self.mic_stream.start()

            while self.recording and not stop_event.is_set():
                # раз в 5 секунд резать сегмент
                if (time.time() - segment_started_at) >= SEGMENT_DURATION and not pause_event.is_set():
                    if self._bh_recording and self._mic_recording:
                        bh_seg = np.concatenate(self._bh_recording, axis=0)
                        mic_seg = np.concatenate(self._mic_recording, axis=0)
                        min_len = min(len(bh_seg), len(mic_seg))
                        if min_len > 0:
                            bh_seg = bh_seg[:min_len]
                            mic_seg = mic_seg[:min_len]

                            bh_file = str(TEMP_DIR / f"blackhole_segment_{segment_count:04d}.wav")
                            mic_file = str(TEMP_DIR / f"mic_segment_{segment_count:04d}.wav")
                            comb_file = str(TEMP_DIR / f"combined_segment_{segment_count:04d}.wav")

                            # комбинированный сегмент для отладки/потребности
                            comb = (bh_seg + mic_seg) * 0.5
                            # сохраняем сегменты
                            wavfile.write(bh_file, SAMPLE_RATE, (bh_seg * 32767.0).astype(np.int16))
                            wavfile.write(mic_file, SAMPLE_RATE, (mic_seg * 32767.0).astype(np.int16))
                            wavfile.write(comb_file, SAMPLE_RATE, (comb * 32767.0).astype(np.int16))

                            # отправляем в очередь на транскрибацию
                            audio_queue.put((bh_file, mic_file, comb_file, segment_started_at))
                            segment_count += 1
                            self._bh_recording.clear()
                            self._mic_recording.clear()
                            segment_started_at = time.time()
                            self.log_status(f"Записан сегмент {segment_count}")
                sd.sleep(50)

            # по завершению добросить хвост в очередь
            self._flush_tail_to_queue()

        except Exception as e:
            self.log_status(f"Ошибка записи: {e}")
            self.recording = False
            stop_event.set()

    def _flush_tail_to_queue(self):
        if self._bh_recording and self._mic_recording:
            bh_seg = np.concatenate(self._bh_recording, axis=0)
            mic_seg = np.concatenate(self._mic_recording, axis=0)
            min_len = min(len(bh_seg), len(mic_seg))
            if min_len > 0:
                bh_seg = bh_seg[:min_len]
                mic_seg = mic_seg[:min_len]
                idx = len(list(TEMP_DIR.glob("blackhole_segment_*.wav")))
                bh_file = str(TEMP_DIR / f"blackhole_segment_{idx:04d}.wav")
                mic_file = str(TEMP_DIR / f"mic_segment_{idx:04d}.wav")
                comb_file = str(TEMP_DIR / f"combined_segment_{idx:04d}.wav")
                comb = (bh_seg + mic_seg) * 0.5
                wavfile.write(bh_file, SAMPLE_RATE, (bh_seg * 32767.0).astype(np.int16))
                wavfile.write(mic_file, SAMPLE_RATE, (mic_seg * 32767.0).astype(np.int16))
                wavfile.write(comb_file, SAMPLE_RATE, (comb * 32767.0).astype(np.int16))
                audio_queue.put((bh_file, mic_file, comb_file, time.time()))
            self._bh_recording.clear()
            self._mic_recording.clear()

    def _concat_all_segments_to_wav(self):
        segs = sorted(TEMP_DIR.glob("combined_segment_*.wav"))
        if not segs:
            return
        # склеим в финальный WAV
        data_all: list[np.ndarray] = []
        for p in segs:
            _, data = wavfile.read(str(p))
            # нормализуем к float32
            if data.dtype != np.int16:
                data = data.astype(np.int16)
            data_all.append(data.astype(np.float32) / 32767.0)
        if data_all:
            full = np.concatenate(data_all, axis=0)
            wavfile.write(OUTPUT_WAV, SAMPLE_RATE, (full * 32767.0).astype(np.int16))
            self.log_status(f"WAV сохранён: {OUTPUT_WAV}")

    # ---- транскрибация ----
    def _transcribe_segments(self):
        self.log_status("Загрузка модели GigaAM v2 RNNT...")
        model = gigaam.load_model("v2_rnnt")
        try:
            with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
                while self.recording or not audio_queue.empty():
                    try:
                        bh_file, mic_file, comb_file, seg_t = audio_queue.get(timeout=1.0)
                    except queue.Empty:
                        time.sleep(0.25)
                        continue

                    try:
                        self.log_status(f"Транскрибируется сегмент: {Path(bh_file).name}")

                        bh_text = ""
                        if os.path.getsize(bh_file) > 0:
                            _, bh_data = wavfile.read(bh_file)
                            if not self.is_silent(bh_data, self.blackhole_silence_threshold):
                                bh_text = model.transcribe(bh_file).strip()
                                if bh_text and bh_text != "Продолжение следует...":
                                    line = f"Собеседник: {bh_text}\n"
                                    transcript_queue.put(line)
                                    f.write(line)

                        mic_text = ""
                        if os.path.getsize(mic_file) > 0:
                            _, mc_data = wavfile.read(mic_file)
                            if not self.is_silent(mc_data, self.mic_silence_threshold):
                                mic_text = model.transcribe(mic_file).strip()
                                if mic_text and mic_text != "Продолжение следует...":
                                    line = f"Вы: {mic_text}\n"
                                    transcript_queue.put(line)
                                    f.write(line)

                        f.flush()
                    finally:
                        audio_queue.task_done()
            self.log_status("Транскрибация завершена")
        except Exception as e:
            self.log_status(f"Ошибка транскрипции: {e}")

    # ---- работа с текстом ----
    def get_transcript(self) -> str:
        """
        Возвращает инкремент с момента прошлого запроса (для «реального времени» через polling).
        """
        collected = []
        while not transcript_queue.empty():
            try:
                collected.append(transcript_queue.get_nowait())
                transcript_queue.task_done()
            except queue.Empty:
                break
        if collected:
            piece = "".join(collected)
            self.transcript += piece
            return piece
        return ""

    def save_transcript(self) -> dict:
        if not self.transcript:
            self.log_status("Транскрипт пуст!")
            return {"status": "error", "message": "Транскрипт пуст!"}
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write(self.transcript)
        self.log_status("Транскрипт сохранен")
        return {"status": "success", "message": "Транскрипт сохранен"}

    def summarize_transcript(self) -> dict:
        if self.recording or self.paused:
            return {"status": "error", "message": "Нельзя пересказать во время записи"}
        if not self.transcript:
            return {"status": "error", "message": "Транскрипт пуст для пересказа"}
        try:
            response = self.gigachat.chat(f"Перескажи следующий диалог: {self.transcript}")
            self.summary = response.choices[0].message.content
            return {"status": "success", "message": "Саммари готово", "summary": self.summary}
        except Exception as e:
            return {"status": "error", "message": f"Ошибка пересказа: {e}"}


recorder = AudioRecorder()


# -----------------------------
# РОУТЫ (имена как в исходнике)
# -----------------------------

@app.route('/')
def index():
    # desktop.py отдаёт index.html из билд-папки; этот маршрут нужен, если запустят напрямую main.py
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return send_from_directory(str(STATIC_DIR), 'index.html')
    abort(404)

@app.route('/api/microphones', methods=['GET'])
def get_microphones():
    return jsonify({"microphones": recorder.get_available_microphones(), "current_mic": recorder.current_mic})

@app.route('/api/switch_microphone/<mic_name>', methods=['POST'])
def switch_microphone(mic_name):
    return jsonify(recorder.switch_microphone(mic_name))

@app.route('/api/update_mic_gain/<float:value>', methods=['POST'])
def update_mic_gain(value: float):
    recorder.mic_gain = float(value)
    return jsonify({"status": "success", "message": f"Усиление микрофона установлено: {value}"})

@app.route('/api/update_blackhole_gain/<float:value>', methods=['POST'])
def update_blackhole_gain(value: float):
    recorder.blackhole_gain = float(value)
    return jsonify({"status": "success", "message": f"Усиление BlackHole установлено: {value}"})

@app.route('/api/update_mic_gate/<float:value>', methods=['POST'])
def update_mic_gate(value: float):
    recorder.mic_silence_threshold = float(value)
    return jsonify({"status": "success", "message": f"Порог тишины микрофона: {value}"})

@app.route('/api/update_blackhole_gate/<float:value>', methods=['POST'])
def update_blackhole_gate(value: float):
    recorder.blackhole_silence_threshold = float(value)
    return jsonify({"status": "success", "message": f"Порог тишины BlackHole: {value}"})

@app.route('/api/levels', methods=['GET'])
def get_levels():
    recorder.update_levels()
    return jsonify({
        "mic_level": recorder.mic_level_val,
        "blackhole_level": recorder.blackhole_level_val
    })

@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    return jsonify(recorder.toggle_recording())

@app.route('/api/toggle_pause', methods=['POST'])
def toggle_pause():
    return jsonify(recorder.toggle_pause())

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    return jsonify(recorder.stop_recording())

@app.route('/api/transcript', methods=['GET'])
def get_transcript():
    # возвращает только инкремент с прошлого запроса (для «реального времени»)
    return jsonify({"transcript": recorder.get_transcript()})

@app.route('/api/save_transcript', methods=['POST'])
def save_transcript():
    return jsonify(recorder.save_transcript())

@app.route('/api/summarize', methods=['POST'])
def summarize():
    return jsonify(recorder.summarize_transcript())

@app.route('/api/status', methods=['GET'])
def get_status():
    # пустая строка при старте — убрал шаблонную фразу
    return jsonify({"status": recorder.status_message})


# Прямой запуск (для отладки без desktop.py)
if __name__ == "__main__":
    # слушаем только loopback
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)