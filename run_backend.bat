@echo off
setlocal
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv\Scripts\activate.bat
    echo Please create it and install requirements via:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r backend_server\requirements.txt
    pause
    exit /b 1
)
call .venv\Scripts\activate
python -m uvicorn backend_server.main_server:app --host 0.0.0.0 --port 8000
pause
