# Backend Setup Guide

## 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r backend_server/requirements.txt
```

## 2. Configure ASR Resources

- **C++ DLL + PyBind module**: Run `build_asr.bat` (or `.\build_asr.ps1`) from the repo root. The script detects your Python interpreter, enables `CPP_ASR_ENABLE_PYBIND11`, and produces both `cpp_asr.dll` and `cpp_asr_native.pyd` under `cpp_asr/build/Release/`.  
  - The backend automatically prepends `cpp_asr/build/Release/` to `PYTHONPATH`, so leave the artifacts in place.  
  - Override the DLL lookup path via `ASR_DLL_PATH` if you relocate it.
- **Model file**: Provide the Whisper model path required by the DLL via `ASR_MODEL_PATH` (e.g., `C:\models\ggml-base.en.bin`).

Environment variables respected by the backend:

| Variable | Purpose |
| --- | --- |
| `ASR_DLL_PATH` | Absolute/relative path to the compiled ASR DLL. |
| `ASR_MODEL_PATH` | Path to the Whisper model file loaded by the DLL. |
| `CALL_ASSISTANT_BACKEND_URL` | Optional override for the REST base URL used by the C# UI. |
| `CALL_ASSISTANT_WS_URL` | Optional override for the WebSocket base URL used by the C# UI. |

Ensure `qa_script.json`, `persona.txt`, and the `logs/` directory remain in the repo root; they are loaded via relative paths.

## 3. Run the Backend for Debugging

```bash
python -m uvicorn backend_server.main_server:app --host 0.0.0.0 --port 8000
```

Or use the helper scripts:

- Windows: `run_backend.bat`
- Linux/macOS: `bash run_backend.sh`

Leave the console open to view ASR/LLM logging output.

## 4. PyInstaller Packaging

Build a folder-based executable:

```bash
pyinstaller --noconfirm --onedir --console backend_server/main_server.py
```

After PyInstaller finishes, copy the following into the generated `dist/main_server/` directory (or equivalent):

- Compiled ASR DLL (`cpp_asr.dll`)
- Whisper model file required by the DLL
- `qa_script.json`
- `persona.txt`
- `logs/` directory (or create it empty so per-call folders can be generated)

Set `ASR_DLL_PATH` and `ASR_MODEL_PATH` for the packaged executable if the DLL/model aren’t located beside the binary.

## 5. Connecting the C# UI

- Start the backend using any method above; it must listen on the REST/WebSocket host/port expected by the UI.  
- Launch the WPF app (`dotnet run --project CallAssistantUI/CallAssistantUI.csproj`).  
- The UI issues `POST /start_call`, opens `/ws/*` streams, and polls `GET /current_transcript`. When done, it sends `POST /end_call`.  
- Override UI endpoints by setting `CALL_ASSISTANT_BACKEND_URL` and `CALL_ASSISTANT_WS_URL` before launching the C# application.
