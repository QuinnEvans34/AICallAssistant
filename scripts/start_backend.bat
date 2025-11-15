@echo off
setlocal
set REPO_ROOT=%~dp0\..
set VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
  echo Python virtualenv not found at %VENV_PY%
  exit /b 1
)
if not exist "%REPO_ROOT%logs" mkdir "%REPO_ROOT%logs"

rem Ensure native build artifacts are on PATH and PYTHONPATH for this process
set BUILD_RELEASE=%REPO_ROOT%\cpp_asr\build\Release
if exist "%BUILD_RELEASE%" (
  set PATH=%BUILD_RELEASE%;%PATH%
  set PYTHONPATH=%BUILD_RELEASE%;%PYTHONPATH%
)

set ASR_ALLOW_FAKE=1
"%VENV_PY%" -u -m uvicorn backend_server.main_server:app --host 0.0.0.0 --port 8000 > "%REPO_ROOT%logs\backend_out.log" 2> "%REPO_ROOT%logs\backend_err.log"
