"""
Real-Time Streaming Automatic Speech Recognition (ASR)

Provides continuous speech-to-text transcription with speaker detection,
rolling audio buffers, and intelligent text deduplication. Uses Faster-Whisper
for high-performance local speech recognition with minimal latency.

Key Features:
- Rolling circular audio buffer for continuous processing
- Speaker detection using energy analysis and WebRTC VAD
- Overlapping audio snapshots for smooth transcription
- Text deduplication to avoid repeated content
- Thread-safe operation with background processing
- Low-latency streaming transcription

Architecture:
- Audio capture runs in dedicated thread with callback processing
- Rolling buffer stores recent audio for snapshot extraction
- Speaker detection analyzes audio energy patterns
- Transcription uses thread pool for concurrent processing
- Text deduplication prevents duplicate transcript output

Dependencies:
- faster_whisper: Local speech recognition model
- sounddevice: Audio capture and playback
- numpy: Audio buffer processing
- webrtcvad: Voice activity detection (optional)
- concurrent.futures: Thread pool for async transcription

Author: Quinn Evans
"""

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

try:  # Optional dependency for better VAD accuracy
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover
    webrtcvad = None

try:
    import librosa
except ImportError:
    librosa = None

# Audio processing constants
SAMPLE_RATE = 16000  # Audio sample rate in Hz (16kHz standard for speech)
BUFFER_SEC = 5.0     # Rolling buffer duration in seconds
SNAPSHOT_SEC = 2.5   # Audio snapshot length for transcription
SNAPSHOT_INTERVAL = 0.5  # Time between transcription snapshots
DEDUP_WINDOW = 40    # Character window for text deduplication

# Voice Activity Detection (VAD) parameters
VAD_FRAME_MS = 30    # VAD frame size in milliseconds
SPEAKER_COOLDOWN = 2.0  # Cooldown after speaker detection
ENERGY_VARIANCE_THRESHOLD = 4e-4  # Energy variance for speaker detection
SILENCE_THRESHOLD = 1.5e-3  # Minimum energy for speech detection


class StreamASR:
    """
    Continuous Automatic Speech Recognition with speaker detection.

    Manages real-time audio capture, processing, and transcription using
    a rolling buffer approach. Detects speaker changes and provides
    streaming text output with deduplication.

    The system maintains a circular audio buffer and periodically extracts
    overlapping snapshots for transcription. Speaker detection uses energy
    analysis and optional WebRTC VAD for distinguishing customer speech
    from agent speech or silence.

    Attributes:
        partial_callback (callable): Function called with partial transcript text
        speaker_callback (callable): Function called when speaker changes
        sample_rate (int): Audio sample rate in Hz
        buffer_samples (int): Size of rolling audio buffer in samples
        snapshot_samples (int): Size of transcription snapshots in samples
        audio_buffer (np.ndarray): Circular buffer for recent audio
        executor (ThreadPoolExecutor): Thread pool for transcription tasks
        model (WhisperModel): Faster-Whisper model instance
        current_speaker (str): Current detected speaker ("customer" or "waiting")
    """

    def __init__(self, transcript_callback, speaker_callback=None, transcript_recorder=None):
        """
        Initialize the streaming ASR system.

        Sets up audio buffers, Whisper model, and VAD components.
        Prepares the system for audio capture and processing.

        Args:
            transcript_callback (callable): Function to call with transcript text
            speaker_callback (callable, optional): Function to call on speaker changes
            transcript_recorder (TranscriptRecorder, optional): Recorder for logging transcripts
        """
        self.partial_callback = transcript_callback
        self.speaker_callback = speaker_callback
        self.transcript_recorder = transcript_recorder

        # Audio buffer configuration
        self.sample_rate = SAMPLE_RATE
        self.buffer_samples = int(BUFFER_SEC * SAMPLE_RATE)
        self.snapshot_samples = int(SNAPSHOT_SEC * SAMPLE_RATE)
        self.frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

        # Circular audio buffer for continuous recording
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self.buffer_index = 0  # Current write position in buffer
        self.samples_written = 0  # Total samples written (up to buffer size)
        self.buffer_lock = threading.Lock()  # Thread-safe buffer access

        # Thread pool for concurrent transcription
        self.executor: Optional[ThreadPoolExecutor] = None

        # Initialize Whisper model with optimized settings for speed
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

        # Thread control
        self.stop_event = threading.Event()
        self.record_thread = None
        self.snapshot_thread = None

        # Speaker detection state
        self.vad = webrtcvad.Vad(2) if webrtcvad else None  # Voice activity detector
        self.energy_history = deque(maxlen=5)  # Rolling energy measurements
        self.customer_cooldown_until = 0.0  # Speaker detection cooldown
        self.current_speaker = "waiting"  # Current speaker state

        # Text deduplication state
        self._last_tail = ""  # Last portion of transcribed text

        # Heartbeat and backlog tracking
        self.call_id: Optional[str] = None
        self.current_call_dir: Optional[str] = None
        self.heartbeat_log_file: Optional[str] = None
        self.heartbeat_lock = threading.Lock()
        self.snapshot_backlog = 0
        self.backlog_lock = threading.Lock()
        self.suppress_speaker_indicator = False

    def _init_executor(self):
        """Initialize the transcription executor."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")

    def start_listening(self):
        """
        Start the audio capture and processing pipeline.

        Initializes thread pool, resets audio buffers, and starts background
        threads for recording and transcription. Clears all previous state.

        This method is non-blocking and returns immediately while processing
        continues in background threads.
        """
        self._init_executor()

        # Reset stop signal
        self.stop_event.clear()

        # Reset audio buffer state
        with self.buffer_lock:
            self.audio_buffer[:] = 0
            self.buffer_index = 0
            self.samples_written = 0

        # Reset text processing state
        self._last_tail = ""
        self.energy_history.clear()
        self.customer_cooldown_until = 0.0
        self._set_speaker("waiting")

        # Start background threads
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self.record_thread.start()
        self.snapshot_thread.start()

    def reset_for_call(self, call_id: str, call_dir: str):
        """
        Reset ASR state for a new call session.

        Args:
            call_id (str): Identifier for the call (e.g., call_YYYYMMDD_HHMMSS)
            call_dir (str): Directory where call artifacts are stored
        """
        self.call_id = call_id
        self.current_call_dir = call_dir
        heartbeat_dir = os.path.join("logs", "heartbeat")
        os.makedirs(heartbeat_dir, exist_ok=True)
        self.heartbeat_log_file = os.path.join(heartbeat_dir, f"{call_id}.jsonl")

        with self.buffer_lock:
            self.audio_buffer[:] = 0
            self.buffer_index = 0
            self.samples_written = 0

        self._last_tail = ""
        self.energy_history.clear()
        self.customer_cooldown_until = 0.0
        self.current_speaker = "waiting"
        self.snapshot_backlog = 0
        self.stop_event.clear()
        self.suppress_speaker_indicator = False

    def stop_listening(self):
        """
        Stop the audio capture and processing pipeline.

        Signals all threads to stop, waits for clean shutdown, and cleans up
        resources. This method blocks until threads terminate or timeout.
        """
        self.stop_event.set()

        # Wait for threads to finish
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        if self.snapshot_thread:
            self.snapshot_thread.join(timeout=1.0)

        # Clean up thread pool
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None

    # ------------------------------------------------------------------ Audio Capture

    def _record_loop(self):
        """
        Main audio recording loop.

        Runs in a dedicated thread to continuously capture audio from the
        microphone. Uses sounddevice's callback-based streaming for low-latency
        audio capture. Processes each audio chunk for speaker detection.
        """
        def callback(indata, frames, time_info, status):
            """
            Audio callback function called by sounddevice.

            Processes incoming audio data, adds it to the rolling buffer,
            and updates speaker detection state.

            Args:
                indata (np.ndarray): Audio data from microphone
                frames (int): Number of frames in this chunk
                time_info: Timing information (unused)
                status: Stream status (unused)

            Raises:
                sd.CallbackStop: Signals stream to stop
            """
            # Extract mono channel from audio input
            chunk = np.copy(indata[:, 0])

            # Add chunk to rolling buffer
            self._append_samples(chunk)

            # Update speaker detection with new audio
            self._update_speaker_detection(chunk)

            # Check for stop signal
            if self.stop_event.is_set():
                raise sd.CallbackStop

        try:
            # Start audio input stream with callback
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,  # Mono audio
                dtype="float32",  # 32-bit float format
                blocksize=self.frame_samples,  # Small chunks for responsiveness
                callback=callback,
            ):
                # Keep stream alive until stop signal
                while not self.stop_event.wait(0.1):
                    pass
        except sd.CallbackStop:
            return  # Normal shutdown

    def _append_samples(self, samples: np.ndarray):
        """
        Append audio samples to the rolling circular buffer.

        Adds new audio data to the circular buffer, overwriting oldest data
        when the buffer is full. Thread-safe operation.

        Args:
            samples (np.ndarray): New audio samples to add
        """
        if samples.size == 0:
            return

        with self.buffer_lock:
            # Limit samples to buffer size (prevent overflow)
            write = samples[-self.buffer_samples:]
            length = write.size

            # Calculate write positions in circular buffer
            end = (self.buffer_index + length) % self.buffer_samples

            # Write samples, handling buffer wraparound
            if end > self.buffer_index:
                # Single contiguous write
                self.audio_buffer[self.buffer_index:end] = write
            else:
                # Split write across buffer boundary
                first = self.buffer_samples - self.buffer_index
                self.audio_buffer[self.buffer_index:] = write[:first]
                self.audio_buffer[:end] = write[first:]

            # Update buffer state
            self.buffer_index = end
            self.samples_written = min(self.samples_written + length, self.buffer_samples)

    def _snapshot_loop(self):
        """
        Main snapshot processing loop.

        Runs in a dedicated thread to periodically extract audio snapshots
        from the rolling buffer and submit them for transcription.
        """
        while not self.stop_event.is_set():
            self._take_snapshot()
            time.sleep(SNAPSHOT_INTERVAL)

    def _take_snapshot(self):
        """
        Extract and transcribe an audio snapshot.

        Takes a recent segment of audio from the rolling buffer and submits
        it for transcription if speaker conditions are met. Only processes
        audio when a customer is detected as speaking.
        """
        with self.buffer_lock:
            # Check if we have enough audio and customer is speaking
            if (
                self.samples_written < self.snapshot_samples
                or self.current_speaker != "customer"
            ):
                return

            # Extract recent audio segment from circular buffer
            end = self.buffer_index
            start = (end - self.snapshot_samples) % self.buffer_samples

            # Handle buffer wraparound
            if start < end:
                snapshot = self.audio_buffer[start:end].copy()
            else:
                snapshot = np.concatenate(
                    (self.audio_buffer[start:], self.audio_buffer[:end])
                )

        # Normalize audio to prevent clipping
        peak = np.max(np.abs(snapshot))
        if peak > 0:
            snapshot = snapshot / peak

        # Submit for transcription in thread pool
        if self.executor is None:
            self._init_executor()

        enqueue_time = time.time()
        self._increment_snapshot_backlog()
        future = self.executor.submit(self._transcribe_snapshot, snapshot, enqueue_time)
        future.add_done_callback(lambda _: self._decrement_snapshot_backlog())

    def _increment_snapshot_backlog(self):
        """Track outstanding snapshot tasks and log backpressure if needed."""
        with self.backlog_lock:
            self.snapshot_backlog += 1
            backlog = self.snapshot_backlog
        if backlog > 5:
            self._log_backpressure_event(backlog)

    def _decrement_snapshot_backlog(self):
        with self.backlog_lock:
            self.snapshot_backlog = max(0, self.snapshot_backlog - 1)

    def _log_backpressure_event(self, queue_size: int):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "BACKPRESSURE",
            "queue_size": queue_size,
        }
        self._write_heartbeat_event(event)

    def _record_heartbeat(self, chunk_duration_ms: float, queue_latency_ms: float, processing_latency_ms: float):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "chunk_duration_ms": round(chunk_duration_ms, 2),
            "queue_latency_ms": round(queue_latency_ms, 2),
            "processing_latency_ms": round(processing_latency_ms, 2),
        }
        self._write_heartbeat_event(event)

    def _write_heartbeat_event(self, payload: dict):
        if not self.heartbeat_log_file:
            return
        try:
            with self.heartbeat_lock:
                with open(self.heartbeat_log_file, "a", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                    f.write("\n")
        except Exception as exc:
            print(f"[ASR] Heartbeat log error: {exc}")

    # ------------------------------------------------------------------ Speaker Detection

    def _chunk_has_speech(self, chunk: np.ndarray) -> bool:
        """
        Determine if an audio chunk contains speech.

        Uses energy-based detection as primary method, with WebRTC VAD
        as secondary confirmation when available.

        Args:
            chunk (np.ndarray): Audio chunk to analyze

        Returns:
            bool: True if speech is detected in the chunk
        """
        # Primary: Energy-based detection
        energy = float(np.mean(np.abs(chunk)))
        if energy >= SILENCE_THRESHOLD:
            return True

        # Secondary: WebRTC VAD (if available)
        if not self.vad:
            return False

        # Convert float32 to int16 for VAD
        pcm16 = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16).tobytes()

        # Analyze frames with VAD
        frame_len = int(self.sample_rate * VAD_FRAME_MS / 1000)
        step = frame_len * 2  # 16-bit samples
        speech_frames = 0
        total = 0

        for start in range(0, len(pcm16) - step + 1, step):
            frame = pcm16[start: start + step]
            total += 1
            try:
                if self.vad.is_speech(frame, SAMPLE_RATE):
                    speech_frames += 1
            except Exception:
                break

        # Require majority of frames to contain speech
        return total > 0 and (speech_frames / total) >= 0.5

    def _update_speaker_detection(self, chunk: np.ndarray):
        """
        Update speaker detection state based on new audio chunk.

        Analyzes audio energy patterns and speech detection to determine
        if the customer is currently speaking. Implements cooldown periods
        to prevent rapid speaker switching.

        Args:
            chunk (np.ndarray): New audio chunk for analysis
        """
        # Calculate and store audio energy
        energy = float(np.mean(np.square(chunk)))
        self.energy_history.append(energy)

        # Check for speech in current chunk
        speech = self._chunk_has_speech(chunk)

        # Calculate energy variance for pattern detection
        variance = np.var(self.energy_history) if len(self.energy_history) >= 5 else 0.0
        now = time.time()

        # Determine speaker state
        if speech and (variance > ENERGY_VARIANCE_THRESHOLD or now >= self.customer_cooldown_until):
            # Customer detected speaking
            self.customer_cooldown_until = now + SPEAKER_COOLDOWN
            self._set_speaker("customer")
        elif now >= self.customer_cooldown_until:
            # Cooldown expired, switch to waiting
            self._set_speaker("waiting")

    def _set_speaker(self, speaker: str):
        """
        Update the current speaker state and notify callback.

        Changes the speaker state and calls the speaker callback if the
        state actually changed.

        Args:
            speaker (str): New speaker state ("customer" or "waiting")
        """
        if speaker == self.current_speaker:
            return

        self.current_speaker = speaker

        # Notify callback of speaker change
        if self.suppress_speaker_indicator:
            return

        if callable(self.speaker_callback):
            try:
                self.speaker_callback(speaker)
            except Exception:
                pass  # Ignore callback errors

    # ------------------------------------------------------------------ Transcription

    def _transcribe_snapshot(self, audio: np.ndarray, enqueue_time: float):
        """
        Transcribe an audio snapshot using Whisper.

        Processes the audio through the Whisper model and handles the
        resulting text with deduplication before calling the callback.

        Args:
            audio (np.ndarray): Audio data to transcribe
            start_time (float): Timestamp when snapshot was taken
        """
        processing_start = time.time()
        queue_latency_ms = max(0.0, (processing_start - enqueue_time) * 1000.0)
        chunk_duration_ms = (len(audio) / float(self.sample_rate)) * 1000.0
        try:
            # Run Whisper transcription
            segments, _ = self.model.transcribe(
                audio,
                beam_size=1,  # Fast decoding
                temperature=0.0,  # Deterministic output
                vad_filter=False,  # We handle VAD ourselves
                condition_on_previous_text=False,  # Independent segments
                language="en",  # English-only for speed
            )

            # Combine all segment text
            text = "".join(seg.text for seg in segments).strip()
            if not text:
                return

            # Remove duplicate content from previous transcriptions
            fragment = self._deduplicate(text)
            if fragment:
                # Log to transcript recorder if available
                if self.transcript_recorder:
                    self.transcript_recorder.log(fragment)

                # Call callback with new text
                self.partial_callback(fragment)

                # Log latency for performance monitoring
                latency = time.time() - enqueue_time
                print(f"[ASR] snapshot latency {latency:.2f}s | text='{fragment}'")

        except Exception as exc:
            print(f"Transcription error: {exc}")
        finally:
            processing_end = time.time()
            processing_latency_ms = max(0.0, (processing_end - processing_start) * 1000.0)
            self._record_heartbeat(chunk_duration_ms, queue_latency_ms, processing_latency_ms)

    def playback(self, audio_file_path: str):
        """
        Process an audio file in playback mode instead of live audio.

        Loads the audio file and processes it in chunks as if it were live audio,
        but without playing sound or using the microphone. Useful for testing
        and analysis of recorded calls.

        Args:
            audio_file_path (str): Path to the audio file to process

        Raises:
            ImportError: If librosa is not available
            FileNotFoundError: If audio file doesn't exist
            Exception: For other audio processing errors
        """
        if not librosa:
            raise ImportError("librosa is required for audio file playback. Install with: pip install librosa")

        print(f"[ASR] Starting playback mode for: {audio_file_path}")

        self.stop_event.clear()
        self._init_executor()
        self.suppress_speaker_indicator = True

        try:
            # Load audio file
            audio, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
            print(f"[ASR] Loaded audio: {len(audio)} samples at {sr}Hz")

            # Convert to the format expected by our processing
            audio = audio.astype(np.float32)

            # Calculate chunk size for processing
            chunk_samples = int(SNAPSHOT_SEC * self.sample_rate)

            # Process audio in chunks
            for i in range(0, len(audio), chunk_samples):
                if self.stop_event.is_set():
                    break

                chunk = audio[i:i + chunk_samples]

                # Pad last chunk if necessary
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')

                # Process chunk as if it were live audio
                self._process_playback_chunk(chunk)

                # Sleep to simulate real-time processing
                time.sleep(SNAPSHOT_SEC)

            print("[ASR] Playback completed")

        except Exception as e:
            print(f"[ASR] Playback error: {e}")
            raise
        finally:
            self.suppress_speaker_indicator = False
            self._set_speaker("waiting")

    def _process_playback_chunk(self, chunk: np.ndarray):
        """
        Process a chunk of audio from playback mode.

        Simulates the live audio processing pipeline for file playback.

        Args:
            chunk (np.ndarray): Audio chunk to process
        """
        # Add chunk to rolling buffer (for consistency)
        self._append_samples(chunk)

        # Always assume customer is speaking during playback
        # (could be enhanced with VAD if needed)
        self._set_speaker("customer")

        # Take a snapshot for transcription
        self._take_snapshot()

    def _deduplicate(self, new_text: str) -> str:
        """
        Remove duplicate text from overlapping transcriptions.

        Compares the new text against the tail of previously transcribed
        text to avoid repeating content from overlapping audio segments.

        Args:
            new_text (str): New transcribed text to deduplicate

        Returns:
            str: Deduplicated text fragment, empty if no new content
        """
        if not self._last_tail:
            # First text received
            self._last_tail = new_text[-200:]  # Keep last 200 chars
            return new_text

        # Compare against recent text tail
        tail = self._last_tail[-DEDUP_WINDOW:]
        max_overlap = min(len(tail), len(new_text))
        overlap = 0

        # Find largest overlapping suffix/prefix
        for size in range(max_overlap, 0, -1):
            if tail[-size:] == new_text[:size]:
                overlap = size
                break

        # Extract non-overlapping portion
        fragment = new_text[overlap:].strip()

        if fragment:
            # Update tail with combined text
            combined = f"{self._last_tail} {fragment}".strip()
            self._last_tail = combined[-200:]  # Keep last 200 chars

        return fragment
