import os

import sounddevice as sd
from faster_whisper import WhisperModel

from asr import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    SAMPLE_RATE,
    ensure_model_downloaded,
)


def pick_input_device() -> int:
    """Return a usable input device index for sounddevice."""
    try:
        default_input = sd.default.device[0]
        if default_input is not None and default_input >= 0:
            info = sd.query_devices(default_input)
            if info["max_input_channels"] > 0:
                return default_input
    except Exception:
        pass

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            return idx
    raise RuntimeError("No input audio device with capture channels is available.")


def record_and_transcribe(duration: int = 3):
    samplerate = SAMPLE_RATE
    frames = int(samplerate * duration)
    device_index = pick_input_device()

    print(f"Recording {duration}s from device {device_index} ...")
    audio = sd.rec(
        frames,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    mono_audio = audio.flatten()

    model_name = os.environ.get("CALLASSIST_MODEL_NAME", DEFAULT_MODEL_NAME)
    ensure_model_downloaded(model_name, DEFAULT_CACHE_DIR)
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(DEFAULT_CACHE_DIR),
    )

    print("Transcribing...")
    segments, _ = model.transcribe(mono_audio, language="en")
    text = " ".join(seg.text for seg in segments).strip()
    print(f"Transcribed: {text or '[no speech detected]'}")


if __name__ == "__main__":
    record_and_transcribe()
