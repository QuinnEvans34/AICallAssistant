from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from call_transcript_recorder import TranscriptRecorder
from exception_logger import exception_logger

# Support both package and script execution
try:
    from .asr_engine import ASREngine  # type: ignore
except Exception:  # pragma: no cover
    from asr_engine import ASREngine  # type: ignore


class TranscriptStream:
    """
    Handles microphone capture and transcript polling for the native ASR engine.
    """

    def __init__(
        self,
        asr_engine: ASREngine,
        transcript_recorder: TranscriptRecorder,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[dict], None]] = None,
    ):
        self.asr_engine = asr_engine
        self.transcript_recorder = transcript_recorder
        self.on_transcript = on_transcript or (lambda _: None)
        self.on_status = on_status or (lambda _: None)

        self.stop_event = threading.Event()
        self.poll_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        self._last_tail = ""
        self.dedup_window = 40
        self.sample_rate = 16000
        self.frame_samples = int(self.sample_rate * 0.03)

    def start(self):
        self.asr_engine.initialize()
        self.stop_event.clear()
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.poll_thread.start()
        self.audio_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        if self.poll_thread:
            self.poll_thread.join(timeout=1.0)

    def reset_for_call(self, call_id: Optional[str] = None):
        if call_id is None:
            import time

            call_id = str(int(time.time()))
        self._last_tail = ""
        self.asr_engine.reset_call(call_id)

    # ------------------------------------------------------------------ audio capture
    def _audio_loop(self):
        def callback(indata, frames, time_info, status):
            try:
                chunk = np.copy(indata[:, 0])
                self.asr_engine.push_audio(chunk)
            except Exception as exc:  # pragma: no cover
                exception_logger.log_exception(exc, "transcript_stream", "push_audio failure")
                raise sd.CallbackStop

            if self.stop_event.is_set():
                raise sd.CallbackStop

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_samples,
                callback=callback,
            ):
                while not self.stop_event.wait(0.1):
                    pass
        except sd.CallbackStop:
            return
        except Exception as exc:  # pragma: no cover
            exception_logger.log_exception(exc, "transcript_stream", "audio loop failure")

    # ------------------------------------------------------------------ transcript polling
    def _poll_loop(self):
        while not self.stop_event.is_set():
            start = time.time()
            text = None
            try:
                text = self.asr_engine.poll_transcript()
            except Exception as exc:  # pragma: no cover
                exception_logger.log_exception(exc, "transcript_stream", "poll_transcript failure")

            latency_ms = (time.time() - start) * 1000.0
            status_payload = {
                "asr_latency_ms": round(latency_ms, 2),
                "queue_depth": 0,
                "errors": 0,
                "processing": True,
            }
            self.on_status(status_payload)

            if text:
                fragment = self._deduplicate(text)
                if fragment:
                    self.transcript_recorder.log(fragment)
                    self.on_transcript(fragment)
            time.sleep(0.05)

    def _deduplicate(self, text: str) -> Optional[str]:
        """
        Deduplicate overlapping transcripts using trailing window logic from the
        legacy StreamASR implementation.
        """
        if not text:
            return None

        if not self._last_tail:
            self._last_tail = text[-self.dedup_window :]
            return text

        tail = text[-self.dedup_window :]
        overlap = self._last_tail
        self._last_tail = tail

        if overlap and text.startswith(overlap):
            return text[len(overlap) :].strip()
        return text
