"""
Call Transcript Recorder

Records raw ASR segments with timestamps for call analysis and reporting.
Provides thread-safe logging of transcription data during live calls.

Features:
- Timestamped transcript segments
- Thread-safe buffer management
- JSON export functionality
- Automatic folder creation

Author: Quinn Evans
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class TranscriptRecorder:
    """
    Records and manages transcript segments for call analysis.

    Maintains a thread-safe buffer of ASR transcript segments with timestamps.
    Provides methods to start recording, log segments, and save to JSON file.

    Attributes:
        call_dir (str): Directory path for this call's logs
        buffer (List[Dict]): Thread-safe buffer of transcript segments
        lock (threading.Lock): Protects buffer access
        is_recording (bool): Whether recording is active
    """

    def __init__(self):
        """
        Initialize the transcript recorder.

        Sets up internal state but does not start recording until start() is called.
        """
        self.call_dir = None
        self.buffer: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.is_recording = False

    def start(self, call_dir: str):
        """
        Start recording transcripts for a new call.

        Args:
            call_dir (str): Directory path where transcript will be saved
        """
        self.call_dir = call_dir
        self.buffer = []
        self.is_recording = True

        # Ensure directory exists
        Path(call_dir).mkdir(parents=True, exist_ok=True)

    def log(self, text: str):
        """
        Log a transcript segment with current timestamp.

        Thread-safe method that adds a new transcript entry to the buffer.

        Args:
            text (str): Raw ASR transcript text to log
        """
        if not self.is_recording or not text.strip():
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "text": text.strip()
        }

        with self.lock:
            self.buffer.append(entry)

    def save(self):
        """
        Save the transcript buffer to transcript.json file.

        Writes all buffered transcript segments to the call directory.
        Should be called at the end of a call.
        """
        if not self.call_dir or not self.buffer:
            return

        transcript_file = os.path.join(self.call_dir, "transcript.json")

        with self.lock:
            try:
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(self.buffer, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving transcript: {e}")

    def stop(self):
        """
        Stop recording and save the transcript.

        Marks recording as inactive and saves any remaining buffered data.
        """
        self.is_recording = False
        self.save()

    def get_transcript_count(self) -> int:
        """
        Get the number of transcript segments recorded.

        Returns:
            int: Number of segments in the buffer
        """
        with self.lock:
            return len(self.buffer)

    def get_word_count(self) -> int:
        """
        Get the total word count of all transcript segments.

        Returns:
            int: Total number of words transcribed
        """
        with self.lock:
            total_words = 0
            for entry in self.buffer:
                total_words += len(entry["text"].split())
            return total_words