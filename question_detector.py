"""
Real-Time Question Detection from Speech Transcripts

Detects and extracts customer questions from streaming speech-to-text output.
Uses linguistic analysis and pattern matching to identify interrogative content
in real-time, with intelligent buffering and flushing for conversational flow.

Key Features:
- Streaming text processing with rolling buffer management
- Question pattern recognition using keywords and intent phrases
- Conjunction-based question splitting (e.g., "How much and when?")
- Partial question handling with timeout-based flushing
- Thread-safe operation with separate ingest and dispatch threads
- Configurable sensitivity for different conversational contexts

Algorithm:
- Accumulates text chunks in a rolling buffer
- Splits text into sentences and analyzes each for question characteristics
- Uses keyword detection, punctuation analysis, and intent phrase matching
- Handles compound questions connected by conjunctions
- Flushes partial questions after configurable timeouts
- Maintains conversation flow by avoiding premature question emission

Threading Architecture:
- Ingest thread: Processes incoming text chunks and manages buffer
- Dispatch thread: Handles question extraction and callback emission
- Thread-safe queues for inter-thread communication
- Graceful shutdown with proper thread cleanup

Dependencies:
- re: Regular expressions for text processing and pattern matching
- threading: Multi-threaded processing for real-time operation
- queue: Thread-safe communication between processing threads
- time: Timeout management for partial question flushing

Author: Quinn Evans
"""

import re
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import List, Dict, Any, Optional

from exception_logger import exception_logger


class QuestionDetector:
    """
    Real-time question detection and extraction from speech transcripts.

    Processes streaming text input to identify customer questions using
    linguistic patterns, keywords, and conversational analysis. Manages
    text buffering, question splitting, and timeout-based flushing for
    optimal conversational AI performance.

    The detector uses a two-thread architecture: one thread ingests and
    buffers text, while another extracts and dispatches questions. This
    ensures real-time processing without blocking the speech recognition
    pipeline.

    Attributes:
        question_callback (callable): Function called with detected questions
        buffer (str): Rolling text buffer for accumulating transcript chunks
        question_keywords (set): Words that typically start questions
        intent_phrases (tuple): Phrases indicating questioning intent
        leading_fillers (tuple): Words to strip when analyzing question starts
        conjunction_pattern (re.Pattern): Regex for splitting compound questions
        partial_max_wait (float): Seconds to wait before flushing partial questions
        flush_word_limit (int): Word count threshold for forced buffer flush
        max_buffer_chars (int): Maximum characters in rolling buffer

    Threading:
        Uses two daemon threads for processing and dispatch, ensuring
        non-blocking operation and clean shutdown.
    """

    def __init__(self, question_callback):
        """
        Initialize the question detector with callback and processing threads.

        Args:
            question_callback (callable): Function to call with detected questions.
                Receives payload dict with "questions" list and "asr_conf" score.
        """
        self.question_callback = question_callback

        # Text accumulation buffer
        self.buffer = ""

        # Question detection patterns
        self.question_keywords = {
            "how", "what", "why", "when", "where", "can", "do", "does",
            "is", "are", "will", "would", "should", "could", "who", "which"
        }
        self.leading_keywords = set(self.question_keywords) | {"curious", "wondering"}
        self.intent_phrases = (
            "i'm curious", "im curious", "i am curious",
            "i'm wondering", "im wondering", "i am wondering",
            "i also want to know", "i also want to",
            "i also need to know", "i also need to",
            "i also wonder"
        )
        self.leading_fillers = ("and", "so", "well", "also", "but", "then", "plus")

        # Text processing patterns
        self.conjunction_pattern = re.compile(r"\b(and|also|what about|tell me about)\b", re.IGNORECASE)

        # Thread communication queues
        self.question_q = Queue()  # For extracted questions
        self.text_q = Queue()      # For incoming text chunks

        # Thread control
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.processor = threading.Thread(target=self._ingest_loop, daemon=True)
        self.worker.start()
        self.processor.start()

        # Buffer management parameters
        self.partial_max_wait = 0.8     # Reduced for faster response
        self.partial_wait_start = None
        self.flush_word_limit = 15      # Reduced for quicker flushing
        self.max_buffer_chars = 2000    # Shorter rolling window for real-time

        # Reporting and logging buffers
        self.detected_questions_buffer: List[Dict[str, Any]] = []
        self.alignment_buffer: List[Dict[str, Any]] = []
        self.recent_segments: List[str] = []  # Sliding window for alignment
        self.max_recent_segments = 8  # Keep last N segments for alignment
        self.buffer_lock = threading.Lock()
        self.force_flush_event = threading.Event()
        self._activity_lock = threading.Lock()
        self._active_dispatch = 0

        # Backpressure tracking
        self.backpressure_count = 0

    def add_text(self, text: str):
        """
        Add text chunk to the processing queue.

        Accepts streaming text from speech recognition and queues it for
        question detection processing. Also maintains recent segments buffer
        for alignment mapping.

        Args:
            text (str): Text chunk from speech transcript
        """
        if not text:
            return

        cleaned = text.strip()
        if cleaned:
            self.text_q.put(cleaned)

            # Track detector backpressure when queue grows too large
            if self.text_q.qsize() > 5:
                self.backpressure_count += 1

            # Maintain recent segments buffer for alignment
            with self.buffer_lock:
                self.recent_segments.append(cleaned)
                if len(self.recent_segments) > self.max_recent_segments:
                    self.recent_segments.pop(0)

    def reset_for_call(self):
        """
        Reset detector state for a new call session.

        Clears buffers, recent segments, and reporting data so each call
        produces isolated logs.
        """
        self.buffer = ""
        self.partial_wait_start = None
        self.force_flush_event.clear()

        with self.buffer_lock:
            self.detected_questions_buffer.clear()
            self.alignment_buffer.clear()
            self.recent_segments.clear()

        self.backpressure_count = 0
        self._drain_queue(self.text_q)
        self._drain_queue(self.question_q)

    def flush_pending(self, timeout: float = 1.0):
        """
        Request an immediate flush of buffered text.

        Signals the ingest loop to treat any remaining buffer contents
        as complete so reporting captures trailing fragments.
        """
        self.force_flush_event.set()
        deadline = time.time() + timeout
        while self.force_flush_event.is_set() and time.time() < deadline:
            time.sleep(0.01)

    def wait_until_idle(self, timeout: float = 2.0) -> bool:
        """
        Wait until detector queues and dispatch loop are idle.

        Args:
            timeout (float): Maximum seconds to wait

        Returns:
            bool: True if idle before timeout, False otherwise.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.text_q.empty() and self.question_q.empty():
                with self._activity_lock:
                    if self._active_dispatch == 0:
                        return True
            time.sleep(0.05)
        return False

    def _drain_queue(self, queue_obj: Queue):
        """Remove all items from the provided queue without blocking."""
        while True:
            try:
                queue_obj.get_nowait()
            except Empty:
                break

    def stop(self):
        """
        Stop processing threads and clean up resources.

        Signals threads to stop, waits for completion, and ensures
        any remaining buffered content is processed.
        """
        self.stop_event.set()
        self.text_q.put(None)  # Sentinel for ingest thread

        if self.processor.is_alive():
            self.processor.join(timeout=1.0)

        self.question_q.put(None)  # Sentinel for dispatch thread
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)

    # ------------------------------------------------------------------ Text Ingestion

    def _ingest_loop(self):
        """
        Main text ingestion and processing loop.

        Runs in dedicated thread to continuously process incoming text chunks,
        manage the rolling buffer, and extract questions as they are detected.
        """
        while not self.stop_event.is_set():
            try:
                chunk = self.text_q.get(timeout=0.1)
            except Empty:
                # Check for stale partial questions during idle time
                self._flush_stale_partial()
                if self.force_flush_event.is_set():
                    self._process_buffer(force=True)
                    self.force_flush_event.clear()
                continue

            if chunk is None:  # Shutdown sentinel
                break

            # Add chunk to buffer and process
            self._append_chunk(chunk)
            force = self._should_flush(self.buffer)
            self._process_buffer(force=force)
            if self.force_flush_event.is_set():
                self._process_buffer(force=True)
                self.force_flush_event.clear()
            self._flush_stale_partial()

        # Process any remaining buffer content before exit
        self._process_buffer(force=True)

    def _append_chunk(self, chunk: str):
        """
        Append normalized text chunk to the rolling buffer.

        Args:
            chunk (str): Text chunk to add
        """
        normalized = self._normalize_chunk(chunk)
        if not normalized:
            return

        # Append with space separation
        if self.buffer:
            self.buffer = f"{self.buffer} {normalized}"
        else:
            self.buffer = normalized

        # Maintain buffer size limit
        if len(self.buffer) > self.max_buffer_chars:
            self.buffer = self.buffer[-self.max_buffer_chars:]

    def _normalize_chunk(self, chunk: str) -> str:
        """
        Normalize text chunk by collapsing whitespace.

        Args:
            chunk (str): Raw text chunk

        Returns:
            str: Normalized text with consistent spacing
        """
        return re.sub(r"\s+", " ", chunk).strip()

    # ------------------------------------------------------------------ Question Dispatch

    def _dispatch_loop(self):
        """
        Question dispatch loop running in dedicated thread.

        Processes extracted questions from the queue and calls the
        question callback with appropriate payload formatting.
        Also logs detected questions and creates alignment mappings.
        """
        while not self.stop_event.is_set():
            try:
                questions = self.question_q.get(timeout=0.1)
            except Empty:
                continue

            if questions is None:  # Shutdown sentinel
                break

            if questions:
                with self._activity_lock:
                    self._active_dispatch += 1
                try:
                    # Log detected questions for reporting
                    self._log_detected_questions(questions)

                    # Create alignment mapping
                    self._create_alignment_mapping(questions)

                    # Format payload for response manager
                    payload = {"questions": questions, "asr_conf": 0.8}
                    try:
                        self.question_callback(payload)
                    except Exception as e:
                        exception_logger.log_exception(e, "question_detector", "Failed to call question callback")
                finally:
                    with self._activity_lock:
                        self._active_dispatch = max(0, self._active_dispatch - 1)

    # ------------------------------------------------------------------ Buffer Processing

    def _process_buffer(self, force: bool = False):
        """
        Process the current buffer to extract questions.

        Analyzes buffered text for complete questions and dispatches them.
        Handles partial questions based on timeout and force flags.

        Args:
            force (bool): Force flush of any remaining buffer content
        """
        if not self.buffer:
            return

        # Extract questions from current buffer
        question_groups = self._extract_questions()
        emitted = False

        # Dispatch each question group
        for group in question_groups:
            if group:
                self.question_q.put(group)
                emitted = True

        if emitted:
            # Reset partial question timer on successful emission
            self.partial_wait_start = None
        elif self.buffer and self.partial_wait_start is None:
            # Start timer for partial question
            self.partial_wait_start = time.time()

        # Force flush remaining content if requested
        if force and self.buffer:
            leftover = self.buffer.strip()
            if leftover:
                self.question_q.put([leftover])
            self.buffer = ""
            self.partial_wait_start = None

    def _flush_stale_partial(self):
        """
        Flush partial questions that have exceeded the wait timeout.

        Ensures questions aren't held indefinitely in the buffer,
        maintaining conversational responsiveness.
        """
        if not self.buffer:
            self.partial_wait_start = None
            return

        now = time.time()
        if self.partial_wait_start is None:
            self.partial_wait_start = now
            return

        # Check if partial question has timed out
        if now - self.partial_wait_start >= self.partial_max_wait:
            leftover = self.buffer.strip()
            if leftover:
                self.question_q.put([leftover])
            self.buffer = ""
            self.partial_wait_start = None

    # ------------------------------------------------------------------ Question Extraction

    def _extract_questions(self) -> list[list[str]]:
        """
        Extract complete questions from the current buffer.

        Splits text into sentences, analyzes each for question characteristics,
        and handles compound questions connected by conjunctions.

        Returns:
            list[list[str]]: Groups of questions extracted from buffer
        """
        if not self.buffer:
            return []

        # Normalize buffer text
        normalized = re.sub(r"\s+", " ", self.buffer).strip()
        if not normalized:
            self.buffer = ""
            return []

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        pending = ""
        results = []

        # Process each sentence
        for idx, sentence in enumerate(sentences):
            if not sentence:
                continue

            is_last = idx == len(sentences) - 1
            if is_last and not self._has_terminal_punctuation(sentence):
                # Last sentence is incomplete, save for later
                pending = sentence
                break

            # Extract questions from complete sentence
            splits = self._split_into_questions(sentence)
            if splits:
                results.append(splits)

        # Handle remaining partial content
        leftover = pending.strip()
        if leftover:
            conjunction_splits = self._split_conjunction_questions(leftover)
            if len(conjunction_splits) > 1:
                # Process compound questions
                complete_segments = []
                partial_segment = ""
                for segment in conjunction_splits:
                    if self._is_complete_question(segment):
                        complete_segments.append(segment)
                    else:
                        partial_segment = segment

                if complete_segments:
                    results.append(complete_segments)
                leftover = partial_segment
            elif self._looks_like_question(leftover) and self._should_flush(leftover):
                # Flush likely question
                results.append([leftover])
                leftover = ""

        # Update buffer with unprocessed content
        self.buffer = leftover
        return results

    def _split_into_questions(self, text: str) -> list[str]:
        """
        Split text containing multiple questions separated by question marks.

        Args:
            text (str): Text that may contain multiple questions

        Returns:
            list[str]: Individual question segments
        """
        segments = []
        parts = [part.strip() for part in text.split("?") if part.strip()]
        for part in parts:
            if self._looks_like_question(part):
                segments.append(part)
        return segments

    def _split_conjunction_questions(self, text: str) -> list[str]:
        """
        Split compound questions connected by conjunctions.

        Handles questions like "How much does it cost and when is installation?"

        Args:
            text (str): Text with conjunction-connected questions

        Returns:
            list[str]: Split question segments
        """
        splits = []
        start = 0

        # Find conjunction boundaries
        for match in self.conjunction_pattern.finditer(text):
            segment = text[start: match.start()].strip(" ,")
            if self._looks_like_question(segment):
                splits.append(segment)
            start = match.end()

        # Add remaining text
        tail = text[start:].strip(" ,")
        if tail:
            splits.append(tail)

        return splits

    # ------------------------------------------------------------------ Text Analysis

    def _has_terminal_punctuation(self, text: str) -> bool:
        """
        Check if text ends with terminal punctuation.

        Args:
            text (str): Text to check

        Returns:
            bool: True if text has proper sentence-ending punctuation
        """
        trimmed = text.rstrip()
        if trimmed.endswith("..."):
            return False  # Ellipsis indicates continuation
        return trimmed.endswith("?") or trimmed.endswith(".") or trimmed.endswith("!")

    def _should_flush(self, text: str) -> bool:
        """
        Determine if buffer should be flushed based on content length.

        Args:
            text (str): Text to evaluate

        Returns:
            bool: True if text exceeds word limit threshold
        """
        if not text:
            return False
        words = len(text.split())
        return words >= self.flush_word_limit

    def _looks_like_question(self, text: str) -> bool:
        """
        Analyze text to determine if it appears to be a question.

        Uses multiple heuristics: punctuation, intent phrases, and keywords.

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if text appears to be interrogative
        """
        if not text:
            return False

        stripped = text.strip()
        lowered = stripped.lower().lstrip()
        lowered = self._remove_leading_fillers(lowered)

        # Explicit question mark
        if stripped.endswith("?"):
            return True

        # Intent-indicating phrases
        for phrase in self.intent_phrases:
            if lowered.startswith(phrase):
                return True

        # Question-starting keywords
        words = lowered.split()
        if not words:
            return False
        return words[0] in self.leading_keywords

    def _remove_leading_fillers(self, lowered: str) -> str:
        """
        Remove leading filler words that don't indicate question intent.

        Args:
            lowered (str): Lowercase text to clean

        Returns:
            str: Text with leading fillers removed
        """
        if not lowered:
            return lowered

        changed = True
        while changed:
            changed = False
            for filler in self.leading_fillers:
                prefix = f"{filler} "
                if lowered.startswith(prefix):
                    lowered = lowered[len(prefix):].lstrip()
                    changed = True

        return lowered

    def save_logs(self, call_dir: str):
        """
        Save detected questions and alignment data to files.

        Args:
            call_dir (str): Directory to save log files
        """
        import os
        import json

        # Save detected questions
        questions_file = os.path.join(call_dir, "detected_questions.json")
        with self.buffer_lock:
            try:
                with open(questions_file, 'w', encoding='utf-8') as f:
                    json.dump(self.detected_questions_buffer, f, indent=2, ensure_ascii=False)
            except Exception as e:
                exception_logger.log_exception(e, "question_detector", "Failed to save detected questions")

        # Save alignment data
        alignment_file = os.path.join(call_dir, "alignment.json")
        with self.buffer_lock:
            try:
                with open(alignment_file, 'w', encoding='utf-8') as f:
                    json.dump(self.alignment_buffer, f, indent=2, ensure_ascii=False)
            except Exception as e:
                exception_logger.log_exception(e, "question_detector", "Failed to save alignment data")

    def _log_detected_questions(self, questions: List[str]):
        """
        Log detected questions for reporting.

        Args:
            questions (List[str]): List of detected question texts
        """
        timestamp = datetime.now().isoformat()

        with self.buffer_lock:
            for question in questions:
                # Estimate confidence based on question characteristics
                confidence = self._estimate_question_confidence(question)

                self.detected_questions_buffer.append({
                    "timestamp": timestamp,
                    "question": question,
                    "confidence": confidence
                })

    def _create_alignment_mapping(self, questions: List[str]):
        """
        Create alignment mapping between detected questions and source segments.

        Args:
            questions (List[str]): List of detected question texts
        """
        with self.buffer_lock:
            for question in questions:
                if self.recent_segments:
                    # Calculate alignment score using fuzzy matching
                    combined_segments = " ".join(self.recent_segments)
                    alignment_score = self._calculate_alignment_score(question, combined_segments)

                    self.alignment_buffer.append({
                        "detected_question": question,
                        "confidence": self._estimate_question_confidence(question),
                        "source_segments": self.recent_segments.copy(),
                        "alignment_score": alignment_score
                    })

    def _estimate_question_confidence(self, question: str) -> float:
        """
        Estimate confidence score for a detected question.

        Args:
            question (str): Question text to evaluate

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not question:
            return 0.0

        confidence = 0.5  # Base confidence

        # Higher confidence for explicit question marks
        if question.strip().endswith("?"):
            confidence += 0.2

        # Higher confidence for question keywords
        words = question.lower().split()
        if words and words[0] in self.leading_keywords:
            confidence += 0.2

        # Higher confidence for longer questions
        if len(words) >= 4:
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_alignment_score(self, question: str, segments_text: str) -> float:
        """
        Calculate alignment score between question and source segments.

        Args:
            question (str): Detected question
            segments_text (str): Combined source segments text

        Returns:
            float: Alignment score between 0.0 and 100.0
        """
        try:
            from rapidfuzz import fuzz
            return fuzz.partial_ratio(question.lower(), segments_text.lower())
        except ImportError:
            # Fallback if rapidfuzz not available
            return 50.0
