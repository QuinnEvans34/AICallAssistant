@echo off
setlocal ENABLEEXTENSIONS
set "REPO_ROOT=%~dp0.."
rem Normalize path
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] Python virtualenv not found at %VENV_PY%
  exit /b 1
)
set "LOGS_DIR=%REPO_ROOT%\logs"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
set "BUILD_RELEASE=%REPO_ROOT%\cpp_asr\build\Release"
if exist "%BUILD_RELEASE%" (
  set PATH=%BUILD_RELEASE%;%PATH%
  set PYTHONPATH=%BUILD_RELEASE%;%PYTHONPATH%
)
set CALL_ASSISTANT_BACKEND_URL=http://localhost:8000/
set CALL_ASSISTANT_WS_URL=ws://localhost:8000
"%VENV_PY%" -u -m uvicorn backend_server.main_server:app --host 0.0.0.0 --port 8000 1>>"%LOGS_DIR%\backend_out.log" 2>>"%LOGS_DIR%\backend_err.log"
