import os
import time
import ssl
import json
import threading
import queue
import shutil
import logging
import warnings
from dotenv import load_dotenv

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from gigachat import GigaChat

# GigaAM import
try:
    import gigaam
except ImportError:
    gigaam = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ================== ЛОГИ/ПРЕДУПРЕЖДЕНИЯ ==================
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("rec-sum-backend")

# ================== КОНСТАНТЫ ==================
SAMPLE_RATE = 44100
SEGMENT_DURATION = 5  # Изменено с 10 на 5 секунд
OUTPUT_WAV = "meeting_audio.wav"
OUTPUT_TXT = "transcript.txt"
TEMP_DIR = "temp_segments"
BLACKHOLE_DEVICE = "BlackHole 2ch"

DEFAULT_SILENCE_THRESHOLD = 0.01
DEFAULT_GAIN = 1.0

FRONT_DIST = os.path.join(os.path.dirname(__file__), "front", "dist")

# ================== УТИЛИТЫ АУДИО ==================
def get_available_microphones():
    devices = sd.query_devices()
    return [dev["name"] for dev in devices if dev["max_input_channels"] > 0]

def is_silent(audio_data: np.ndarray, threshold: float) -> bool:
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    return rms < threshold

def db_from_buffer(buf: list[float], current_db: float = -60.0, smoothing: float = 0.8) -> float:
    if not buf:
        return current_db
    
    # Берем только последние 512 сэмплов для быстрого обновления PPM
    arr = np.array(buf[-512:], dtype=np.float64)
    
    # Используем пиковые значения для PPM (как в Google Meet/Discord)
    peak = np.max(np.abs(arr))
    if peak == 0:
        # Если нет сигнала, очень быстро уменьшаем уровень до -60dB
        return current_db * 0.1 + (-60.0) * 0.9
    
    # Конвертируем в dB с улучшенной чувствительностью
    new_db = 20.0 * np.log10(peak)
    
    # Улучшенная чувствительность для низких уровней (как в профессиональных PPM)
    # Нормализуем диапазон для PPM: -60 до 0 dB
    if new_db < -60:
        new_db = -60.0
    elif new_db > 0:
        new_db = 0.0
    
    # Увеличиваем чувствительность для низких уровней
    # Это делает PPM более отзывчивым к тихим звукам
    if new_db > -40:
        # Для уровней выше -40 dB оставляем как есть
        pass
    elif new_db > -50:
        # Для уровней -50 до -40 dB увеличиваем чувствительность
        new_db = new_db + 5
    else:
        # Для уровней ниже -50 dB еще больше увеличиваем чувствительность
        new_db = new_db + 10
    
    # Применяем экспоненциальное сглаживание (как AnalyserNode.smoothingTimeConstant)
    # smoothing = 0.4 означает быстрый отклик (60% нового значения, 40% старого)
    smoothed_db = current_db * (1 - smoothing) + new_db * smoothing
    
    # Логируем для отладки PPM
    if len(buf) > 0 and int(time.time()) % 5 == 0: # каждые 5 секунд для лучшей отладки
        logger.info(f"PPM Debug: buffer_len={len(buf)}, peak={peak:.6f}, new_db={new_db:.1f}, smoothed_db={smoothed_db:.1f}, current_db={current_db:.1f}")
    
    return float(smoothed_db)


# ================== ГЛОБАЛЬНОЕ СОСТОЯНИЕ ==================
class State:
    def __init__(self):
        self.recording = False
        self.paused = False

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self.audio_queue: "queue.Queue[tuple[str,str,str,float]]" = queue.Queue()
        self.transcript_queue: "queue.Queue[str]" = queue.Queue()

        self.blackhole_stream = None
        self.mic_stream = None

        self.mic_gain = DEFAULT_GAIN
        self.blackhole_gain = DEFAULT_GAIN
        self.mic_silence_threshold = DEFAULT_SILENCE_THRESHOLD
        self.blackhole_silence_threshold = DEFAULT_SILENCE_THRESHOLD

        # текущий микрофон
        mics = get_available_microphones()
        self.current_mic = mics[0] if mics else ""

        # буферы для уровней
        self.blackhole_buffer = []
        self.mic_buffer = []

        # Текущие уровни с экспоненциальным сглаживанием
        self.mic_db = -60.0
        self.blackhole_db = -60.0
        
        # Коэффициенты сглаживания для PPM (как в Google Meet/Discord)
        self.mic_smoothing = 0.2  # 0.2 = очень быстрый отклик, 0.4 = быстрый
        self.blackhole_smoothing = 0.2
        
        # Флаг для перезапуска аудио потоков при ошибках
        self.audio_stream_error = False

        # текст
        self.transcript_text = ""

        # потоки
        self.recording_thread: threading.Thread | None = None
        self.transcribe_thread: threading.Thread | None = None
        self.levels_thread: threading.Thread | None = None
        self.pump_thread: threading.Thread | None = None

        # GigaChat
        load_dotenv()
        giga_auth = os.getenv("GIGACHAT_CREDENTIALS")
        scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_B2B")
        self.gigachat = GigaChat(
            credentials=giga_auth,
            scope=scope,
            model="GigaChat-2-Max",
            verify_ssl_certs=False,
        )

        # GigaAM модель лениво загружаем в треде транскрипции
        self.gigaam_model = None

        # менеджеры WS
        self.ws_transcript_clients: set[WebSocket] = set()
        self.ws_levels_clients: set[WebSocket] = set()

        # служебное
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

    def reset_temp(self):
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)

STATE = State()



# ================== WS BROADCAST ==================
def ws_broadcast_transcript(line: str):
    dead = []
    payload = json.dumps({"type": "line", "text": line, "ts": time.time(),
                          "role": "peer" if line.startswith("Собеседник:") else "you" if line.startswith("Вы:") else "raw"})
    for ws in list(STATE.ws_transcript_clients):
        try:
            # send_text is async; run in thread-safe way
            ws._loop.call_soon_threadsafe(asyncio.ensure_future, ws.send_text(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.ws_transcript_clients.discard(ws)

def ws_broadcast_levels(mic_db: float, blackhole_db: float):
    dead = []
    payload = json.dumps({"mic_db": mic_db, "blackhole_db": blackhole_db})
    for ws in list(STATE.ws_levels_clients):
        try:
            ws._loop.call_soon_threadsafe(asyncio.ensure_future, ws.send_text(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.ws_levels_clients.discard(ws)

# ================== ПОТОКИ ==================
def pump_transcript_thread():
    """Перекачивает строки из очереди в общий текст и WS."""
    while True:
        try:
            line = STATE.transcript_queue.get(timeout=0.2)
        except queue.Empty:
            if not STATE.recording and STATE.audio_queue.empty():
                time.sleep(0.2)
            else:
                pass
        else:
            STATE.transcript_text += line
            ws_broadcast_transcript(line)
            STATE.transcript_queue.task_done()

def levels_thread():
    """Считает PPM-уровни и шлет WS ~60 FPS."""
    while True:
        try:
            # Обновляем уровни всегда, но с разной частотой
            if STATE.recording:
                old_mic_db = STATE.mic_db
                old_blackhole_db = STATE.blackhole_db
                
                STATE.mic_db = db_from_buffer(STATE.mic_buffer, STATE.mic_db, STATE.mic_smoothing)
                STATE.blackhole_db = db_from_buffer(STATE.blackhole_buffer, STATE.blackhole_db, STATE.blackhole_smoothing)
                
                # Логируем каждое изменение уровня во время записи
                if abs(STATE.mic_db - old_mic_db) > 0.1:
                    logger.info(f"Levels Thread: MIC level changed: {old_mic_db:.1f}dB -> {STATE.mic_db:.1f}dB")
                if abs(STATE.blackhole_db - old_blackhole_db) > 0.1:
                    logger.info(f"Levels Thread: BLACKHOLE level changed: {old_blackhole_db:.1f}dB -> {STATE.blackhole_db:.1f}dB")
                
                ws_broadcast_levels(STATE.mic_db, STATE.blackhole_db)
                time.sleep(0.016)  # ~60 Гц во время записи
            else:
                # Во время простоя обновляем реже
                STATE.mic_db = db_from_buffer(STATE.mic_buffer, STATE.mic_db, STATE.mic_smoothing)
                STATE.blackhole_db = db_from_buffer(STATE.blackhole_buffer, STATE.blackhole_db, STATE.blackhole_smoothing)
                ws_broadcast_levels(STATE.mic_db, STATE.blackhole_db)
                time.sleep(0.05)  # 20 Гц в режиме ожидания
                
            # Отладочная информация каждые 5 секунд
            if int(time.time()) % 5 == 0:
                logger.info(f"Levels Thread: mic={STATE.mic_db:.1f}dB, blackhole={STATE.blackhole_db:.1f}dB, mic_buffer_len={len(STATE.mic_buffer)}, blackhole_buffer_len={len(STATE.blackhole_buffer)}")
                logger.info(f"Levels Thread: recording={STATE.recording}, paused={STATE.paused}, stop_event={STATE.stop_event.is_set()}")
            
            # Проверяем флаг ошибки аудио потоков
            if STATE.audio_stream_error:
                logger.warning("Levels Thread: Audio stream error detected, resetting buffers")
                STATE.audio_stream_error = False
                STATE.mic_buffer.clear()
                STATE.blackhole_buffer.clear()
                STATE.mic_db = -60.0
                STATE.blackhole_db = -60.0
        except Exception as e:
            logger.error(f"Levels thread error: {e}")
            time.sleep(0.1)

def recording_thread():
    st = STATE
    # ищем устройство BlackHole
    blackhole_device = None
    for dev in sd.query_devices():
        if BLACKHOLE_DEVICE in dev["name"] and dev["max_input_channels"] > 0:
            blackhole_device = dev["name"]
            break
    if not blackhole_device:
        logger.error(f"Устройство {BLACKHOLE_DEVICE} не найдено.")
        st.recording = False
        st.stop_event.set()
        return

    blackhole_recording = []
    mic_recording = []
    segment_count = 0
    segment_start = time.time()

    # Глобальные callback функции для потоков
    def blackhole_callback(indata, frames, time_info, status):
        try:
            if st.recording and not st.paused and not st.stop_event.is_set():
                blackhole_recording.append(indata.copy() * st.blackhole_gain)
            
            # Всегда обновляем буфер для уровней - быстро и без блокировок
            last = indata[:, 0] if indata.ndim > 1 else indata
            STATE.blackhole_buffer.extend(last.tolist())
            
            # Ограничиваем размер буфера для быстрого обновления PPM
            if len(STATE.blackhole_buffer) > 2048:
                del STATE.blackhole_buffer[:-512]
            
            # Логируем для отладки каждые 2 секунды
            if int(time.time()) % 2 == 0:
                logger.info(f"BlackHole Callback: frames={frames}, data_shape={indata.shape}, buffer_len={len(STATE.blackhole_buffer)}, max_amplitude={np.max(np.abs(last)):.6f}")
                
        except Exception as e:
            # Логируем ошибки PortAudio для отладки
            logger.warning(f"BlackHole Callback error: {e}")
            # Очищаем буфер при ошибке
            STATE.blackhole_buffer.clear()
            # Устанавливаем флаг для перезапуска потока
            STATE.audio_stream_error = True

    def mic_callback(indata, frames, time_info, status):
        try:
            if st.recording and not st.paused and not st.stop_event.is_set():
                mic_recording.append(indata.copy() * st.mic_gain)
            
            # Всегда обновляем буфер для уровней - быстро и без блокировок
            last = indata[:, 0] if indata.ndim > 1 else indata
            STATE.mic_buffer.extend(last.tolist())
            
            # Ограничиваем размер буфера для быстрого обновления PPM
            if len(STATE.mic_buffer) > 2048:
                del STATE.mic_buffer[:-512]
            
            # Логируем для отладки каждые 2 секунды
            if int(time.time()) % 2 == 0:
                logger.info(f"Mic Callback: frames={frames}, data_shape={indata.shape}, buffer_len={len(STATE.mic_buffer)}, max_amplitude={np.max(np.abs(last)):.6f}")
                
        except Exception as e:
            # Логируем ошибки PortAudio для отладки
            logger.warning(f"Mic Callback error: {e}")
            # Очищаем буфер при ошибке
            STATE.mic_buffer.clear()
            # Устанавливаем флаг для перезапуска потока
            STATE.audio_stream_error = True

    try:
        st.blackhole_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, device=blackhole_device, callback=blackhole_callback
        )
        st.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, device=st.current_mic, callback=mic_callback
        )

        with st.blackhole_stream, st.mic_stream:
            st.blackhole_stream.start()
            st.mic_stream.start()

            while st.recording and not st.stop_event.is_set():
                if not st.pause_event.is_set():
                    if (time.time() - segment_start) >= SEGMENT_DURATION:
                        if blackhole_recording and mic_recording:
                            blackhole_segment = np.concatenate(blackhole_recording, axis=0)
                            mic_segment = np.concatenate(mic_recording, axis=0)

                            min_len = min(len(blackhole_segment), len(mic_segment))
                            blackhole_segment = blackhole_segment[:min_len]
                            mic_segment = mic_segment[:min_len]

                            blackhole_file = os.path.join(TEMP_DIR, f"blackhole_segment_{segment_count:04d}.wav")
                            mic_file = os.path.join(TEMP_DIR, f"mic_segment_{segment_count:04d}.wav")
                            wavfile.write(blackhole_file, SAMPLE_RATE, blackhole_segment)
                            wavfile.write(mic_file, SAMPLE_RATE, mic_segment)

                            combined = (blackhole_segment + mic_segment) / 2.0
                            combined_file = os.path.join(TEMP_DIR, f"combined_segment_{segment_count:04d}.wav")
                            wavfile.write(combined_file, SAMPLE_RATE, combined)

                            STATE.audio_queue.put((blackhole_file, mic_file, combined_file, segment_start))
                            segment_count += 1
                            blackhole_recording.clear()
                            mic_recording.clear()
                            segment_start = time.time()
                            logger.info(f"Записан сегмент {segment_count}")
                sd.sleep(100)

    except Exception as e:
        logger.error(f"Ошибка записи: {e}")
        st.recording = False
        st.stop_event.set()
        return

    # доброслив хвоста при остановке
    if blackhole_recording and mic_recording:
        blackhole_segment = np.concatenate(blackhole_recording, axis=0)
        mic_segment = np.concatenate(mic_recording, axis=0)
        min_len = min(len(blackhole_segment), len(mic_segment))
        blackhole_segment = blackhole_segment[:min_len]
        mic_segment = mic_segment[:min_len]

        blackhole_file = os.path.join(TEMP_DIR, f"blackhole_segment_{segment_count:04d}.wav")
        mic_file = os.path.join(TEMP_DIR, f"mic_segment_{segment_count:04d}.wav")
        wavfile.write(blackhole_file, SAMPLE_RATE, blackhole_segment)
        wavfile.write(mic_file, SAMPLE_RATE, mic_segment)

        combined = (blackhole_segment + mic_segment) / 2.0
        combined_file = os.path.join(TEMP_DIR, f"combined_segment_{segment_count:04d}.wav")
        wavfile.write(combined_file, SAMPLE_RATE, combined)
        STATE.audio_queue.put((blackhole_file, mic_file, combined_file, segment_start))

    # склейка в один WAV
    all_segments = []
    for i in range(segment_count + 1):
        combined_file = os.path.join(TEMP_DIR, f"combined_segment_{i:04d}.wav")
        if os.path.exists(combined_file):
            _, data = wavfile.read(combined_file)
            all_segments.append(data)
    if all_segments:
        full_audio = np.concatenate(all_segments, axis=0)
        wavfile.write(OUTPUT_WAV, SAMPLE_RATE, full_audio)


def transcribe_thread():
    logger.info("Загрузка модели GigaAM v2 RNNT...")
    try:
        if gigaam is None:
            logger.error("GigaAM не установлен. Установите: pip install gigaam")
            logger.info("Попробуйте: pip install gigaam")
            return
        
        # Проверяем доступность модели
        try:
            STATE.gigaam_model = gigaam.load_model("v2_rnnt")
            logger.info("Модель GigaAM v2 RNNT успешно загружена")
        except Exception as model_error:
            logger.error(f"Ошибка загрузки модели GigaAM: {model_error}")
            logger.info("Попробуйте другие модели: gigaam.list_models()")
            return
            
    except Exception as e:
        logger.error(f"Не удалось загрузить модель GigaAM: {e}")
        return

    try:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            while STATE.recording or not STATE.audio_queue.empty():
                try:
                    blackhole_file, mic_file, combined_file, segment_time = STATE.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    time.sleep(0.2)
                    continue

                logger.info(f"Транскрибируем: {os.path.basename(blackhole_file)}")

                # BlackHole → "Собеседник"
                if os.path.getsize(blackhole_file) > 0:
                    _, bh_data = wavfile.read(blackhole_file)
                    if not is_silent(bh_data, STATE.blackhole_silence_threshold):
                        txt = STATE.gigaam_model.transcribe(blackhole_file).strip()
                        if txt and txt != "Продолжение следует...":
                            line = f"Собеседник: {txt}\n"
                            STATE.transcript_queue.put(line)
                            f.write(line)

                # MIC → "Вы"
                if os.path.getsize(mic_file) > 0:
                    _, mic_data = wavfile.read(mic_file)
                    if not is_silent(mic_data, STATE.mic_silence_threshold):
                        txt = STATE.gigaam_model.transcribe(mic_file).strip()
                        if txt and txt != "Продолжение следует...":
                            line = f"Вы: {txt}\n"
                            STATE.transcript_queue.put(line)
                            f.write(line)

                f.flush()
                STATE.audio_queue.task_done()

        logger.info("Транскрибация завершена")
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")

# ================== FASTAPI APP ==================
import asyncio
from pydantic import BaseModel

app = FastAPI(title="Recorder Backend", version="1.0.0")

# CORS — обычно не нужен, так как фронт с того же origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# статика фронта - монтируем в конце, чтобы не перехватывать API маршруты
if not os.path.isdir(FRONT_DIST):
    logger.warning(f"Внимание: не найдена папка фронта {FRONT_DIST}. Соберите фронт в front/dist.")

# --------- Модели запросов ---------
class MicrophoneSelect(BaseModel):
    name: str

class SettingsPatch(BaseModel):
    mic_gain: float | None = None
    blackhole_gain: float | None = None
    mic_silence_threshold: float | None = None
    blackhole_silence_threshold: float | None = None

class SummarizeBody(BaseModel):
    text: str | None = None

# --------- Вспомогательные ---------
def ensure_blackhole():
    for dev in sd.query_devices():
        if BLACKHOLE_DEVICE in dev["name"] and dev["max_input_channels"] > 0:
            return True
    return False

# --------- REST ---------
@app.get("/api/devices")
def api_devices():
    return {
        "microphones": get_available_microphones(),
        "selected": STATE.current_mic,
        "blackhole_present": ensure_blackhole(),
        "blackhole_name": BLACKHOLE_DEVICE,
    }

@app.post("/api/microphone")
def api_select_mic(body: MicrophoneSelect):
    mics = get_available_microphones()
    if body.name not in mics:
        raise HTTPException(status_code=400, detail="Микрофон не найден")
    
    old_mic = STATE.current_mic
    STATE.current_mic = body.name
    
    # если идет запись и не на паузе — пересоздадим MIC stream на лету
    if STATE.recording and not STATE.paused and STATE.mic_stream:
        try:
            logger.info(f"API: Recreating microphone stream during recording...")
            
            # Останавливаем и закрываем старый стрим
            STATE.mic_stream.stop()
            STATE.mic_stream.close()
            
            # Создаем новый стрим с новым микрофоном и тем же callback
            STATE.mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                device=STATE.current_mic,
                callback=mic_callback  # Используем глобальную функцию из recording_thread
            )
            STATE.mic_stream.start()
            
            logger.info(f"Микрофон переключен с '{old_mic}' на '{STATE.current_mic}' во время записи")
        except Exception as e:
            logger.error(f"Ошибка переключения микрофона: {e}")
            # Возвращаем старый микрофон в случае ошибки
            STATE.current_mic = old_mic
            raise HTTPException(status_code=500, detail=str(e))
    
    return {"ok": True, "selected": STATE.current_mic}

@app.get("/api/state")
def api_state():
    return {
        "recording": STATE.recording,
        "paused": STATE.paused,
        "mic_gain": STATE.mic_gain,
        "blackhole_gain": STATE.blackhole_gain,
        "mic_silence_threshold": STATE.mic_silence_threshold,
        "blackhole_silence_threshold": STATE.blackhole_silence_threshold,
        "mic_db": STATE.mic_db,
        "blackhole_db": STATE.blackhole_db,
        "selected_mic": STATE.current_mic,
    }

@app.post("/api/record/start")
def api_start():
    logger.info("API: Starting recording...")
    if STATE.recording:
        logger.info("API: Recording already in progress")
        return {"ok": True, "status": "already_recording"}

    if not ensure_blackhole():
        logger.error(f"API: BlackHole device {BLACKHOLE_DEVICE} not found")
        raise HTTPException(status_code=400, detail=f"Не найдено устройство {BLACKHOLE_DEVICE}")

    logger.info("API: Resetting temp directory and clearing events")
    STATE.reset_temp()
    STATE.stop_event.clear()
    STATE.pause_event.clear()
    STATE.recording = True
    STATE.paused = False

    # чистим текущий текст
    STATE.transcript_text = ""

    # запускаем потоки
    logger.info("API: Starting recording and transcribe threads...")
    STATE.recording_thread = threading.Thread(target=recording_thread, daemon=True)
    STATE.transcribe_thread = threading.Thread(target=transcribe_thread, daemon=True)

    if STATE.pump_thread is None or not STATE.pump_thread.is_alive():
        logger.info("API: Starting pump thread...")
        STATE.pump_thread = threading.Thread(target=pump_transcript_thread, daemon=True)
        STATE.pump_thread.start()

    if STATE.levels_thread is None or not STATE.levels_thread.is_alive():
        logger.info("API: Starting levels thread...")
        STATE.levels_thread = threading.Thread(target=levels_thread, daemon=True)
        STATE.levels_thread.start()

    STATE.recording_thread.start()
    STATE.transcribe_thread.start()
    
    logger.info("API: Recording started successfully")
    return {"ok": True}

@app.post("/api/record/pause")
def api_pause():
    if not STATE.recording or STATE.paused:
        return {"ok": True}
    STATE.paused = True
    STATE.pause_event.set()
    if STATE.blackhole_stream:
        STATE.blackhole_stream.stop()
    if STATE.mic_stream:
        STATE.mic_stream.stop()
    return {"ok": True}

@app.post("/api/record/resume")
def api_resume():
    if not STATE.recording or not STATE.paused:
        return {"ok": True}
    STATE.paused = False
    STATE.pause_event.clear()
    if STATE.blackhole_stream:
        STATE.blackhole_stream.start()
    if STATE.mic_stream:
        STATE.mic_stream.start()
    return {"ok": True}

@app.post("/api/record/stop")
def api_stop():
    if not STATE.recording:
        return {"ok": True}
    
    try:
        STATE.recording = False
        STATE.stop_event.set()

        # корректно закрываем стримы
        try:
            if STATE.blackhole_stream:
                STATE.blackhole_stream.stop()
                STATE.blackhole_stream.close()
                STATE.blackhole_stream = None
            if STATE.mic_stream:
                STATE.mic_stream.stop()
                STATE.mic_stream.close()
                STATE.mic_stream = None
        except Exception as e:
            logger.warning(f"Закрытие стримов: {e}")

        # дождемся очередей
        # (не блокируем HTTP — фоновые потоки сами дольют и завершатся)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Ошибка остановки записи: {e}")
        return {"ok": False, "error": str(e)}

@app.patch("/api/settings")
def api_update_settings(body: SettingsPatch):
    if body.mic_gain is not None:
        STATE.mic_gain = float(body.mic_gain)
    if body.blackhole_gain is not None:
        STATE.blackhole_gain = float(body.blackhole_gain)
    if body.mic_silence_threshold is not None:
        STATE.mic_silence_threshold = float(body.mic_silence_threshold)
    if body.blackhole_silence_threshold is not None:
        STATE.blackhole_silence_threshold = float(body.blackhole_silence_threshold)
    return {"ok": True, "settings": {
        "mic_gain": STATE.mic_gain,
        "blackhole_gain": STATE.blackhole_gain,
        "mic_silence_threshold": STATE.mic_silence_threshold,
        "blackhole_silence_threshold": STATE.blackhole_silence_threshold,
    }}

@app.get("/api/transcript")
def api_get_transcript():
    return {"text": STATE.transcript_text}

@app.post("/api/transcript/save")
def api_save_transcript():
    text = STATE.transcript_text
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    return {"ok": True, "path": OUTPUT_TXT, "bytes": len(text.encode("utf-8"))}

@app.post("/api/summarize")
def api_summarize(body: SummarizeBody):
    logger.info("API: Summarize request received")
    logger.info(f"API: Body: {body}")
    logger.info(f"API: Current transcript length: {len(STATE.transcript_text) if STATE.transcript_text else 0}")
    
    if STATE.recording or STATE.paused:
        logger.warning("API: Cannot summarize during recording/pause")
        raise HTTPException(status_code=400, detail="Идет запись/пауза — дождитесь завершения")

    transcript = body.text if (body and body.text) else STATE.transcript_text
    logger.info(f"API: Using transcript: {transcript[:100] if transcript else 'None'}...")
    
    if not transcript or not transcript.strip():
        logger.warning("API: Empty transcript")
        raise HTTPException(status_code=400, detail="Транскрипт пуст")

    try:
        # Проверяем, что есть что суммировать
        if len(transcript.strip()) < 10:
            logger.warning(f"API: Transcript too short: {len(transcript.strip())} chars")
            raise HTTPException(status_code=400, detail="Транскрипт слишком короткий для создания саммари")
            
        logger.info("API: Calling GigaChat API...")
        # Используем простой API вызов - передаем строку напрямую
        response = STATE.gigachat.chat(f"Перескажи следующий диалог кратко и по существу: {transcript}")
        summary = response.choices[0].message.content
        logger.info(f"API: Summary generated successfully, length: {len(summary)}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"API: GigaChat summarize error: {e}")
        logger.error(f"API: Error type: {type(e)}")
        logger.error(f"API: Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --------- WebSockets ---------
@app.websocket("/ws/transcript")
async def ws_transcript(ws: WebSocket):
    await ws.accept()
    STATE.ws_transcript_clients.add(ws)
    try:
        while True:
            # читаем вход — не требуется; держим соединение
            await ws.receive_text()
    except WebSocketDisconnect:
        STATE.ws_transcript_clients.discard(ws)
    except Exception:
        STATE.ws_transcript_clients.discard(ws)

@app.websocket("/ws/levels")
async def ws_levels(ws: WebSocket):
    await ws.accept()
    STATE.ws_levels_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        STATE.ws_levels_clients.discard(ws)
    except Exception:
        STATE.ws_levels_clients.discard(ws)

# --------- health ---------
@app.get("/api/health")
def api_health():
    return {"ok": True}

# монтируем статику фронта в конце, чтобы не перехватывать API маршруты
if os.path.isdir(FRONT_DIST):
    app.mount("/", StaticFiles(directory=FRONT_DIST, html=True), name="front")

# ========= Точка входа =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)