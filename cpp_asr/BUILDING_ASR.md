# Building the C++ ASR DLL

The Python backend expects `cpp_asr.dll` at `cpp_asr/build/Release/cpp_asr.dll`. Use the instructions below to generate it with CMake + MSVC.

## Prerequisites
- Visual Studio 2022 (Desktop development with C++)
- CMake 3.20 or newer (ships with VS 2022)
- Git Bash/PowerShell/cmd as preferred shell

## 1. Build via helper scripts

### Windows CMD
```
build_asr.bat
```
The script:
1. Verifies Visual Studio Build Tools via `vswhere.exe`
2. Runs `cmake -S cpp_asr -B cpp_asr/build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release`
3. Runs `cmake --build cpp_asr/build --config Release`
4. Emits the resulting DLL path on success

### PowerShell
```
.\build_asr.ps1
```
Same behavior as the batch script, with richer error handling.

## 2. Build from VS Developer Command Prompt
```
cmake -S cpp_asr -B cpp_asr/build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_asr/build --config Release
```

## 3. Build using CMake Presets / Visual Studio GUI
1. Run `cmake --preset Release` (generates the VS solution under `cpp_asr/build/`)
2. Open `cpp_asr/build/cpp_asr.sln` in Visual Studio
3. Select **Release | x64**
4. Build the `cpp_asr` project

## Output
`cpp_asr/build/Release/cpp_asr.dll`

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| **Missing compiler / cmake errors** | Ensure Visual Studio 2022 with Desktop C++ workload is installed. Run from the "x64 Native Tools Command Prompt for VS 2022". |
| **Missing Windows SDK** | Install the latest Windows 10/11 SDK via Visual Studio Installer. |
| **`cpp_asr.dll` not found after build** | Verify you built the `Release` configuration. Delete `cpp_asr/build` and run the scripts again. |
| **Missing whisper.cpp or third-party libs** | Confirm that all required third-party headers/libraries referenced by `whisper_wrapper.*` are available and included in the project. Add their include/library paths to `cpp_asr/CMakeLists.txt` if necessary. |
| **DLL export issues** | The build enables `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` for shared libraries. If custom export macros are needed, add them to the headers. |
| **Python backend cannot load the DLL** | The backend auto-discovers `cpp_asr/build/Release/cpp_asr.dll`. Alternatively set `ASR_DLL_PATH` to point to the DLL. Ensure the Whisper model path is available or set `ASR_MODEL_PATH`. |

Once the DLL exists, re-run `run_backend.bat` or `run_backend.sh` and the Python service will detect it automatically. Set `ASR_DLL_PATH` explicitly if you move the file elsewhere.
