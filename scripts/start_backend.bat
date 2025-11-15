@echo off
setlocal
set REPO_ROOT=%~dp0\..
set VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
  echo Python virtualenv not found at %VENV_PY%
  exit /b 1
)
if not exist "%REPO_ROOT%logs" mkdir "%REPO_ROOT%logs"
set ASR_ALLOW_FAKE=1
"%VENV_PY%" -u -m uvicorn backend_server.main_server:app --host 0.0.0.0 --port 8000 > "%REPO_ROOT%logs\backend_out.log" 2> "%REPO_ROOT%logs\backend_err.log"
