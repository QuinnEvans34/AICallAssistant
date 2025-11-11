import threading
from queue import Empty, Queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

CHUNK_SEC = 1.0
OVERLAP_SEC = 0.0
SAMPLE_RATE = 16000
BEAM_SIZE = 1  # Faster for partials

class StreamASR:
    def __init__(self, callback):
        self.callback = callback
        self.sample_rate = SAMPLE_RATE
        self.chunk_sec = CHUNK_SEC
        self.overlap_sec = OVERLAP_SEC
        self.channels = 1
        self.overlap_samples = int(self.sample_rate * self.overlap_sec)
        self.chunk_samples = int(self.sample_rate * self.chunk_sec)

        self.audio_q = Queue(maxsize=32)
        self.previous_overlap = np.zeros(self.overlap_samples, dtype=np.float32)
        self.full_transcript = ""

        self.listening = False
        self.stop_event = threading.Event()
        self.record_thread = None
        self.transcribe_thread = None

        self.model = WhisperModel("small", device="cpu", compute_type="int8")

    def start_listening(self):
        if not self.listening:
            self.listening = True
            self.stop_event.clear()
            self.previous_overlap = np.zeros(self.overlap_samples, dtype=np.float32)
            self.full_transcript = ""
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
            self.record_thread.start()
            self.transcribe_thread.start()

    def stop_listening(self):
        self.listening = False
        self.stop_event.set()
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        if self.transcribe_thread:
            self.transcribe_thread.join(timeout=1.0)

    def _record_loop(self):
        def callback(indata, frames, time_info, status):
            chunk = np.copy(indata[:, 0])
            try:
                self.audio_q.put(chunk, timeout=0.1)
            except:
                pass  # Drop if full
            if self.stop_event.is_set():
                raise sd.CallbackStop

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_samples,
                callback=callback,
            ):
                while not self.stop_event.wait(0.1):
                    pass
        except sd.CallbackStop:
            pass

    def _transcribe_loop(self):
        while not self.stop_event.is_set() or not self.audio_q.empty():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except Empty:
                continue

            if not self._chunk_contains_speech(chunk):
                continue  # Skip silent chunks

            # Combine with previous overlap
            audio = np.concatenate((self.previous_overlap, chunk))
            # Update overlap for next
            if len(chunk) >= self.overlap_samples:
                self.previous_overlap = chunk[-self.overlap_samples:]
            else:
                self.previous_overlap = np.concatenate((self.previous_overlap[- (self.overlap_samples - len(chunk)):], chunk))

            # Transcribe
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            segments, _ = self.model.transcribe(
                audio,
                beam_size=BEAM_SIZE,
                vad_filter=False,
                condition_on_previous_text=False,
                language="en",
            )
            text = "".join(seg.text for seg in segments).strip()
            if text:
                self.full_transcript = text
                self.callback(self.full_transcript)

    @staticmethod
    def _chunk_contains_speech(audio: np.ndarray) -> bool:
        energy = np.mean(np.abs(audio))
        return energy >= 1.5e-3  # Threshold for speech
