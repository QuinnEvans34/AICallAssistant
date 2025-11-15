from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# Try to import the native pybind11 module; if not found, extend sys.path to common build output dirs
_native_loaded = False
try:
    import cpp_asr_native as native  # type: ignore
    _native_loaded = True
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        repo_root / "cpp_asr" / "build" / "Release",
        repo_root / "build" / "Release",
        repo_root,  # if copied next to backend
    ]
    for p in candidate_paths:
        if p.exists():
            sys.path.insert(0, str(p))
    try:
        import cpp_asr_native as native  # type: ignore
        _native_loaded = True
    except Exception:
        native = None  # type: ignore
        _native_loaded = False


class _FakeNative:
    def __init__(self):
        self._initialized = False

    def init_asr_engine(self, model_path: str):  # noqa: D401
        self._initialized = True

    def push_audio(self, samples):
        return None

    def poll_transcript(self):
        return ""

    def reset_call(self):
        return None

    def shutdown_asr_engine(self):
        self._initialized = False


if native is None and os.getenv("ASR_ALLOW_FAKE", ""):  # allow fake engine for dev/CI
    native = _FakeNative()  # type: ignore

if native is None:
    raise RuntimeError(
        "Failed to import cpp_asr_native. Ensure the .pyd is built and on PYTHONPATH, or set ASR_ALLOW_FAKE=1 to run without ASR."
    )


class ASREngine:
    """
    Thin wrapper around the native C++ ASR engine exposed through the pybind11 module.

    Handles module import, initialization, push/poll helpers, and graceful
    shutdown semantics so the rest of the backend can remain engine-agnostic.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._repo_root = Path(__file__).resolve().parents[1]
        self._model_path = self._resolve_model_path(model_path)
        self._lock = threading.Lock()
        self._initialized = False

    def _resolve_model_path(self, explicit_path: Optional[str]) -> str:
        if explicit_path:
            p = Path(explicit_path).expanduser()
            if p.exists():
                return str(p)
            raise FileNotFoundError(f"ASR model not found at {p}")

        env_path = os.getenv("ASR_MODEL_PATH")
        if env_path:
            p = Path(env_path).expanduser()
            if p.exists():
                return str(p)
            # In fake mode, allow missing model
            if os.getenv("ASR_ALLOW_FAKE", ""):
                return env_path
            raise FileNotFoundError(f"ASR model not found at {p}")

        models_dir = self._repo_root / "models"
        if models_dir.exists():
            for ext in (".bin", ".ggml", ".gguf"):
                for candidate in models_dir.rglob(f"*{ext}"):
                    if candidate.is_file():
                        return str(candidate)

        if os.getenv("ASR_ALLOW_FAKE", ""):
            return ""

        raise FileNotFoundError(
            "ASR model file not found. Place a model under models/ or set ASR_MODEL_PATH."
        )

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            # In fake mode, model path may be empty
            model_path = self._model_path or ""
            native.init_asr_engine(model_path)  # type: ignore[attr-defined]
            self._initialized = True

    def reset_call(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            native.reset_call()

    def push_audio(self, samples) -> None:
        if not self._initialized:
            raise RuntimeError("ASR engine not initialized")
        if samples is None:
            return

        # Accept numpy arrays or Python lists and convert to float32 buffer
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        if samples.size == 0:
            return

        native.push_audio(samples)

    def poll_transcript(self) -> Optional[str]:
        if not self._initialized:
            raise RuntimeError("ASR engine not initialized")
        text = native.poll_transcript()
        if text:
            text = text.strip()
            return text if text else None
        return None

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            native.shutdown_asr_engine()
            self._initialized = False
