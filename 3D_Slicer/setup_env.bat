@echo off
REM Создание виртуального окружения и установка зависимостей
REM Запускать из папки проекта: setup_env.bat

echo [1/3] Создаю виртуальное окружение...
python -m venv .venv

echo [2/3] Активирую окружение...
call .venv\Scripts\activate.bat

echo [3/3] Устанавливаю зависимости (с поддержкой CUDA)...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo.
echo === Готово ===
echo Активировать окружение: .venv\Scripts\activate.bat
echo Запустить пайплайн:     python main.py --help
pause
