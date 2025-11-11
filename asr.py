import os
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    webrtcvad = None

try:
    from faster_whisper import download_model as _download_model
except ImportError:  # pragma: no cover - older faster-whisper versions
    _download_model = None

DEFAULT_MODEL_NAME = "small"
DEFAULT_CACHE_DIR = Path(__file__).with_name("models")
SAMPLE_RATE = 16000
CHUNK_SEC = 1.0
OVERLAP_SEC = 0.5
VAD_SILENCE_SEC = 1.5
VAD_FRAME_MS = 30  # webrtcvad only accepts 10/20/30
VAD_AGGRESSIVENESS = 2
ENERGY_WINDOW_SEC = 0.3  # fallback silence detector when webrtcvad is unavailable
ENERGY_SILENCE_THRESHOLD = 1e-3
ENERGY_SPEECH_THRESHOLD = 1.5e-3
MAX_BUFFER_SEC = 10.0
MIN_TRANSCRIBE_SEC = 0.5
BEAM_SIZE = 5
QUEUE_MAX_SIZE = 32


def ensure_model_downloaded(model_name: str = DEFAULT_MODEL_NAME, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """
    Ensure the requested model is present locally so first transcription
    does not block on a download.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _download_model:
        _download_model(model_name, cache_dir)


class ASRManager:
    def __init__(self, callback=None):
        self.callback = callback or (lambda text: None)
        self.sample_rate = SAMPLE_RATE
        self.chunk_sec = CHUNK_SEC
        self.overlap_sec = OVERLAP_SEC
        self.channels = 1
        self.vad_silence_sec = VAD_SILENCE_SEC
        self.max_buffer_sec = MAX_BUFFER_SEC
        self.min_transcribe_samples = int(self.sample_rate * MIN_TRANSCRIBE_SEC)
        self.overlap_samples = int(self.sample_rate * self.overlap_sec)

        self.audio_q = Queue(maxsize=QUEUE_MAX_SIZE)
        self.vad_q = Queue(maxsize=QUEUE_MAX_SIZE)
        self.current_buffer = np.zeros(0, dtype=np.float32)
        self.previous_overlap = np.zeros(self.overlap_samples, dtype=np.float32)
        self.buffer_has_speech = False

        self.listening = False
        self.stop_event = threading.Event()
        self.record_thread = None
        self.segment_thread = None
        self.transcribe_thread = None

        self.energy_window_samples = int(self.sample_rate * ENERGY_WINDOW_SEC)

        if webrtcvad:
            self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            self.vad_frame_samples = int(self.sample_rate * VAD_FRAME_MS / 1000)
            self.vad_frame_duration_sec = VAD_FRAME_MS / 1000.0
        else:
            self.vad = None
            self.vad_frame_samples = 0
            self.vad_frame_duration_sec = 0.0
            print(
                "webrtcvad not available; falling back to energy-based silence detection."
            )
        self.language = os.environ.get("CALLASSIST_ASR_LANG", "en").strip() or None

        model_name = os.environ.get("CALLASSIST_MODEL_NAME", DEFAULT_MODEL_NAME)
        ensure_model_downloaded(model_name, DEFAULT_CACHE_DIR)
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(DEFAULT_CACHE_DIR),
        )
        self._warm_up_model()

    def start_listening(self):
        if not self.listening:
            self.listening = True
            self.stop_event.clear()
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.segment_thread = threading.Thread(target=self._segment_loop, daemon=True)
            self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
            self.record_thread.start()
            self.segment_thread.start()
            self.transcribe_thread.start()

    def stop_listening(self):
        self.listening = False
        self.stop_event.set()
        if self.record_thread:
            self.record_thread.join(timeout=1.5)
            self.record_thread = None
        if self.segment_thread:
            self.segment_thread.join(timeout=1.5)
            self.segment_thread = None
        if self.transcribe_thread:
            self.transcribe_thread.join(timeout=1.5)
            self.transcribe_thread = None

    def _record_loop(self):
        block_samples = int(self.sample_rate * self.chunk_sec)

        def callback(indata, frames, time_info, status):
            if status:  # pragma: no cover - logging only
                print(f"Input stream status: {status}")
            chunk = np.copy(indata[:, 0])
            self._enqueue_chunk(chunk)
            if self.stop_event.is_set():
                raise sd.CallbackStop

        try:
            with sd.InputStream(
                device=5,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=block_samples,
                callback=callback,
            ):
                while not self.stop_event.wait(0.1):
                    pass
        except sd.CallbackStop:
            pass
        except Exception as exc:  # pragma: no cover - diagnostic logging
            print(f"Audio input stream error: {exc}")

    def _enqueue_chunk(self, chunk: np.ndarray) -> None:
        try:
            self.audio_q.put(chunk, timeout=0.1)
        except Full:
            # Drop the oldest chunk to keep live audio moving.
            try:
                _ = self.audio_q.get_nowait()
            except Empty:
                pass
            try:
                self.audio_q.put(chunk, timeout=0.1)
            except Full:
                pass

    def _segment_loop(self):
        while not self.stop_event.is_set() or not self.audio_q.empty():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except Empty:
                continue
            self.current_buffer = np.concatenate((self.current_buffer, chunk))
            if self._chunk_contains_speech(chunk):
                self.buffer_has_speech = True
            elif len(self.current_buffer) > int(self.sample_rate * self.max_buffer_sec):
                # keep buffer bounded even if we're only seeing noise
                self.current_buffer = self.current_buffer[-self.overlap_samples :]
            if self._should_flush_buffer():
                self._flush_buffer()

        self._flush_buffer(force=True)

    def _transcribe_loop(self):
        while not self.stop_event.is_set() or not self.vad_q.empty():
            try:
                segment = self.vad_q.get(timeout=0.1)
            except Empty:
                continue
            self._transcribe_audio(segment)

        # Flush any trailing speech once recording stops.
        # Note: In new pipeline, vad_loop handles flushing.

    def _should_flush_buffer(self) -> bool:
        if len(self.current_buffer) < self.min_transcribe_samples:
            return False
        buffer_sec = len(self.current_buffer) / self.sample_rate
        trailing_silence = self._has_trailing_silence(self.current_buffer)
        buffer_timeout = buffer_sec >= self.max_buffer_sec
        return (trailing_silence or buffer_timeout) and self.buffer_has_speech

    def _flush_buffer(self, force: bool = False) -> None:
        if len(self.current_buffer) == 0:
            return
        if not force:
            if not self.buffer_has_speech:
                return
            if len(self.current_buffer) < self.min_transcribe_samples:
                return

        segment = np.concatenate((self.previous_overlap, self.current_buffer))
        if len(self.current_buffer) >= self.overlap_samples:
            overlap_slice = self.current_buffer[-self.overlap_samples :]
        else:
            pad = np.zeros(self.overlap_samples - len(self.current_buffer), dtype=np.float32)
            overlap_slice = np.concatenate((pad, self.current_buffer))
        self.previous_overlap = overlap_slice
        self.current_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_has_speech = False
        self._queue_segment(segment)

    def _queue_segment(self, segment: np.ndarray) -> None:
        try:
            self.vad_q.put(segment, timeout=0.1)
        except Full:
            try:
                _ = self.vad_q.get_nowait()
            except Empty:
                pass
            try:
                self.vad_q.put(segment, timeout=0.1)
            except Full:
                pass

    def _transcribe_audio(self, audio: np.ndarray) -> None:
        if len(audio) == 0:
            return
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        start = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio,
            beam_size=BEAM_SIZE,
            vad_filter=False,
            condition_on_previous_text=True,
            language=self.language,
        )
        elapsed = time.perf_counter() - start
        full_text = "".join(seg.text for seg in segments).strip()
        duration = len(audio) / self.sample_rate
        print(f"Transcribed {duration:.2f}s audio in {elapsed:.2f}s -> '{full_text}'")
        if full_text and len(full_text) > 3:
            self.callback(full_text)

    def _has_trailing_silence(self, audio: np.ndarray) -> bool:
        if self.vad:
            if len(audio) < self.vad_frame_samples:
                return False
            pcm16 = self._float_to_pcm16(audio)
            usable = len(pcm16) - (len(pcm16) % self.vad_frame_samples)
            if usable <= 0:
                return False
            pcm16 = pcm16[:usable]
            frames = pcm16.reshape(-1, self.vad_frame_samples)
            silence = 0.0
            for frame in frames[::-1]:
                frame_bytes = frame.tobytes()
                if self.vad.is_speech(frame_bytes, self.sample_rate):
                    break
                silence += self.vad_frame_duration_sec
                if silence >= self.vad_silence_sec:
                    return True
            return False
        return self._energy_based_silence(audio)

    def _energy_based_silence(self, audio: np.ndarray) -> bool:
        if len(audio) < self.energy_window_samples:
            return False
        tail = audio[-self.energy_window_samples * 5 :]  # inspect roughly the last second
        usable = len(tail) - (len(tail) % self.energy_window_samples)
        if usable <= 0:
            return False
        windows = tail[-usable:].reshape(-1, self.energy_window_samples)
        silent_windows = 0
        for window in windows[::-1]:
            energy = np.mean(np.abs(window))
            if energy < ENERGY_SILENCE_THRESHOLD:
                silent_windows += 1
                if silent_windows * ENERGY_WINDOW_SEC >= self.vad_silence_sec:
                    return True
            else:
                break
        return False

    def _chunk_contains_speech(self, audio: np.ndarray) -> bool:
        if self.vad:
            if len(audio) < self.vad_frame_samples:
                return False
            pcm16 = self._float_to_pcm16(audio)
            usable = len(pcm16) - (len(pcm16) % self.vad_frame_samples)
            if usable <= 0:
                return False
            pcm16 = pcm16[:usable]
            frames = pcm16.reshape(-1, self.vad_frame_samples)
            return any(self.vad.is_speech(frame.tobytes(), self.sample_rate) for frame in frames)
        energy = np.mean(np.abs(audio))
        return energy >= ENERGY_SPEECH_THRESHOLD

    @staticmethod
    def _float_to_pcm16(audio: np.ndarray) -> np.ndarray:
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def _warm_up_model(self) -> None:
        try:
            dummy = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            self.model.transcribe(
                dummy,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                language=self.language,
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"Warm-up inference failed: {exc}")
