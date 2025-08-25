#!/usr/bin/env python3
"""
Тестовый скрипт для проверки взаимодействия фронтенда с бэкендом
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000/api"

def test_api():
    print("🧪 Тестирование API бэкенда...")
    
    # Тест health
    print("\n1. Тест health endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест devices
    print("\n2. Тест devices endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/devices")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Microphones: {data.get('microphones', [])}")
        print(f"   Selected: {data.get('selected', 'N/A')}")
        print(f"   BlackHole present: {data.get('blackhole_present', False)}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест state
    print("\n3. Тест state endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/state")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Recording: {data.get('recording', False)}")
        print(f"   Paused: {data.get('paused', False)}")
        print(f"   Mic gain: {data.get('mic_gain', 0)}")
        print(f"   BlackHole gain: {data.get('blackhole_gain', 0)}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест transcript
    print("\n4. Тест transcript endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/transcript")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Text length: {len(data.get('text', ''))}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест settings update
    print("\n5. Тест settings update:")
    try:
        settings = {
            "mic_gain": 1.2,
            "blackhole_gain": 0.8
        }
        response = requests.post(f"{BASE_URL}/settings", json=settings)
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Response: {data}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

def test_websocket():
    print("\n🔌 Тестирование WebSocket соединений...")
    
    try:
        import websocket
        import threading
        
        def on_message(ws, message):
            print(f"   📨 Получено сообщение: {message[:100]}...")
        
        def on_error(ws, error):
            print(f"   ❌ WebSocket ошибка: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("   🔌 WebSocket соединение закрыто")
        
        def on_open(ws):
            print("   ✅ WebSocket соединение установлено")
        
        # Тест transcript WebSocket
        print("\n1. Тест transcript WebSocket:")
        ws_transcript = websocket.WebSocketApp(
            "ws://127.0.0.1:8000/ws/transcript",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        wst = threading.Thread(target=ws_transcript.run_forever)
        wst.daemon = True
        wst.start()
        
        time.sleep(2)
        ws_transcript.close()
        
        # Тест levels WebSocket
        print("\n2. Тест levels WebSocket:")
        ws_levels = websocket.WebSocketApp(
            "ws://127.0.0.1:8000/ws/levels",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        wsl = threading.Thread(target=ws_levels.run_forever)
        wsl.daemon = True
        wsl.start()
        
        time.sleep(2)
        ws_levels.close()
        
    except ImportError:
        print("   ⚠️  websocket-client не установлен. Установите: pip install websocket-client")
    except Exception as e:
        print(f"   ❌ Ошибка WebSocket: {e}")

if __name__ == "__main__":
    print("🚀 Запуск тестов взаимодействия фронтенда с бэкендом")
    print("=" * 60)
    
    test_api()
    test_websocket()
    
    print("\n" + "=" * 60)
    print("✅ Тестирование завершено!")
    print("\n📝 Для полного тестирования:")
    print("1. Откройте http://127.0.0.1:8000 в браузере")
    print("2. Проверьте загрузку микрофонов")
    print("3. Попробуйте изменить настройки")
    print("4. Проверьте WebSocket соединения в DevTools")
