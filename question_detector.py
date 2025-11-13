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
from queue import Empty, Queue


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

    def add_text(self, text: str):
        """
        Add text chunk to the processing queue.

        Accepts streaming text from speech recognition and queues it for
        question detection processing.

        Args:
            text (str): Text chunk from speech transcript
        """
        if not text:
            return

        cleaned = text.strip()
        if cleaned:
            self.text_q.put(cleaned)

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
                continue

            if chunk is None:  # Shutdown sentinel
                break

            # Add chunk to buffer and process
            self._append_chunk(chunk)
            force = self._should_flush(self.buffer)
            self._process_buffer(force=force)
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
        """
        while not self.stop_event.is_set():
            try:
                questions = self.question_q.get(timeout=0.1)
            except Empty:
                continue

            if questions is None:  # Shutdown sentinel
                break

            if questions:
                # Format payload for response manager
                payload = {"questions": questions, "asr_conf": 0.8}
                self.question_callback(payload)

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

    def _is_complete_question(self, text: str) -> bool:
        """
        Determine if text forms a complete, meaningful question.

        Requires minimum length and question-like characteristics.

        Args:
            text (str): Text to evaluate

        Returns:
            bool: True if text is a complete question
        """
        if not text:
            return False

        stripped = text.strip()
        if not stripped:
            return False

        # Explicit question mark indicates completeness
        if stripped.endswith("?"):
            return True

        # Minimum length and question-like structure required
        words = stripped.split()
        return len(words) >= 3 and self._looks_like_question(stripped)
