@echo off
setlocal

set REPO_ROOT=%~dp0
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

echo [cpp_asr] Checking for Visual Studio Build Tools...
if exist %VSWHERE% (
    for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -requires Microsoft.Component.MSBuild -property installationPath`) do (
        set VSINSTALL=%%i
    )
) else (
    echo Could not find vswhere.exe. Ensure Visual Studio 2022 or Build Tools are installed.
    exit /b 1
)

if not defined VSINSTALL (
    echo Visual Studio Build Tools were not detected. Install VS 2022 with C++ workload and retry.
    exit /b 1
)

set PYTHON_EXE=%REPO_ROOT%.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)
echo [cpp_asr] Using Python interpreter: "%PYTHON_EXE%"

for /f "usebackq tokens=*" %%p in (`"%PYTHON_EXE%" -c "import pybind11; print(pybind11.get_cmake_dir())"`) do (
    set PYBIND_DIR=%%p
)
if not defined PYBIND_DIR (
    echo Failed to locate pybind11. Run "pip install pybind11" in your environment.
    exit /b 1
)

echo [cpp_asr] Configuring project...
cmake -S "%REPO_ROOT%cpp_asr" -B "%REPO_ROOT%cpp_asr\build" -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release ^
    -DCPP_ASR_ENABLE_PYBIND11=ON ^
    -DPython3_EXECUTABLE="%PYTHON_EXE%" ^
    -Dpybind11_DIR="%PYBIND_DIR%"
if errorlevel 1 (
    echo Configuration failed.
    exit /b 1
)

echo [cpp_asr] Building Release DLL...
cmake --build "%REPO_ROOT%cpp_asr\build" --config Release
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

set DLL_PATH=%REPO_ROOT%cpp_asr\build\Release\cpp_asr.dll
set PYD_PATH=%REPO_ROOT%cpp_asr\build\Release\cpp_asr_native.pyd
if exist "%DLL_PATH%" (
    echo [cpp_asr] Build succeeded: "%DLL_PATH%"
) else (
    echo [cpp_asr] Build finished but cpp_asr.dll was not found at "%DLL_PATH%".
    exit /b 1
)

if exist "%PYD_PATH%" (
    echo [cpp_asr] PyBind11 module ready: "%PYD_PATH%"
) else (
    echo [cpp_asr] Build finished but cpp_asr_native.pyd was not found at "%PYD_PATH%".
    exit /b 1
)

exit /b 0
