# Building the C++ ASR DLL + PyBind11 Module

The Python backend loads the native engine through `cpp_asr_native.pyd`, which itself links against `cpp_asr.dll`. Both artifacts are produced side-by-side under `cpp_asr/build/Release/`.

## Prerequisites
- Visual Studio 2022 (Desktop development with C++)
- CMake 3.20 or newer (ships with VS 2022)
- Python 3.8+ (the same interpreter you use for the backend)
- `pybind11` installed in that interpreter (`pip install pybind11`)
- Git Bash/PowerShell/cmd as preferred shell

## 1. Build via helper scripts

### Windows CMD
```
build_asr.bat
```
The script:
1. Verifies Visual Studio Build Tools via `vswhere.exe`
2. Detects your Python interpreter (prefers `.venv\Scripts\python.exe`) and ensures `pybind11` is installed
3. Configures CMake with `CPP_ASR_ENABLE_PYBIND11=ON`, wiring the selected interpreter + `pybind11_DIR`
4. Runs `cmake --build cpp_asr/build --config Release`
5. Emits both the DLL and `.pyd` paths on success

### PowerShell
```
.\build_asr.ps1
```
Same behavior as the batch script, with richer error handling.

## 2. Build from VS Developer Command Prompt
```
cmake -S cpp_asr -B cpp_asr/build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCPP_ASR_ENABLE_PYBIND11=ON ^
      -DPython3_EXECUTABLE=C:\Path\To\python.exe ^
      -Dpybind11_DIR=C:\Path\To\site-packages\pybind11\share\cmake
cmake --build cpp_asr/build --config Release
```
Tip: `python -c "import pybind11; print(pybind11.get_cmake_dir())"` prints the `pybind11_DIR` value.

## 3. Build using CMake Presets / Visual Studio GUI
1. Run `cmake --preset Release` (generates the VS solution under `cpp_asr/build/`)
2. Open `cpp_asr/build/cpp_asr.sln` in Visual Studio
3. Select **Release | x64**
4. Build the `cpp_asr` and `cpp_asr_native` targets

## Output
- `cpp_asr/build/Release/cpp_asr.dll`
- `cpp_asr/build/Release/cpp_asr_native.pyd`

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| **Missing compiler / cmake errors** | Ensure Visual Studio 2022 with Desktop C++ workload is installed. Run from the "x64 Native Tools Command Prompt for VS 2022". |
| **Missing Windows SDK** | Install the latest Windows 10/11 SDK via Visual Studio Installer. |
| **`cpp_asr.dll` not found after build** | Verify you built the `Release` configuration. Delete `cpp_asr/build` and run the scripts again. |
| **Missing whisper.cpp or third-party libs** | Confirm that all required third-party headers/libraries referenced by `whisper_wrapper.*` are available and included in the project. Add their include/library paths to `cpp_asr/CMakeLists.txt` if necessary. |
| **DLL export issues** | The build enables `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` for shared libraries. If custom export macros are needed, add them to the headers. |
| **Python backend cannot load the DLL/PYD** | Confirm both `cpp_asr.dll` and `cpp_asr_native.pyd` exist in `cpp_asr/build/Release/`. Set `PYTHONPATH` to include that directory (the helper run scripts do this). Also ensure `ASR_MODEL_PATH` points at a valid Whisper model. |

Once the DLL exists, re-run `run_backend.bat` or `run_backend.sh` and the Python service will detect it automatically. Set `ASR_DLL_PATH` explicitly if you move the file elsewhere.
