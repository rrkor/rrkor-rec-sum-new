#!/bin/bash

echo "🚀 Запуск RRKOR™ Rec&Sum..."

# Проверяем, что мы в правильной директории
if [ ! -f "main.py" ]; then
    echo "❌ Ошибка: main.py не найден. Запустите скрипт из корневой директории проекта."
    exit 1
fi

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Ошибка: python3 не найден. Установите Python 3.8+"
    exit 1
fi

# Проверяем наличие Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Ошибка: node не найден. Установите Node.js 18+"
    exit 1
fi

# Проверяем наличие npm
if ! command -v npm &> /dev/null; then
    echo "❌ Ошибка: npm не найден. Установите npm"
    exit 1
fi

echo "✅ Проверка зависимостей завершена"

# Проверяем, собран ли фронтенд
if [ ! -d "front/dist" ]; then
    echo "📦 Сборка фронтенда..."
    cd front
    npm install
    npm run build
    cd ..
    echo "✅ Фронтенд собран"
else
    echo "✅ Фронтенд уже собран"
fi

# Проверяем, установлены ли Python зависимости
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "📦 Установка Python зависимостей..."
    pip3 install -r requirements.txt
    echo "✅ Python зависимости установлены"
else
    echo "✅ Python зависимости уже установлены"
fi

# Проверяем, запущен ли сервер
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Сервер уже запущен на порту 8000"
    echo "🔄 Перезапуск сервера..."
    lsof -ti:8000 | xargs kill -9
    sleep 2
fi

# Запускаем сервер
echo "🚀 Запуск сервера..."
python3 main.py &
SERVER_PID=$!

# Ждем запуска сервера
echo "⏳ Ожидание запуска сервера..."
sleep 5

# Проверяем, что сервер запустился
if curl -s http://127.0.0.1:8000/api/health > /dev/null; then
    echo "✅ Сервер успешно запущен!"
    echo ""
    echo "🌐 Откройте браузер и перейдите по адресу:"
    echo "   http://127.0.0.1:8000"
    echo ""
    echo "🛑 Для остановки сервера нажмите Ctrl+C"
    echo ""
    
    # Открываем браузер
    open http://127.0.0.1:8000
    
    # Ждем сигнала завершения
    wait $SERVER_PID
else
    echo "❌ Ошибка: сервер не запустился"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
