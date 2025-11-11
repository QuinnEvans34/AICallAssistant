import re
import threading
import time
from queue import Empty, Queue


class QuestionDetector:
    def __init__(self, question_callback):
        self.question_callback = question_callback
        self.buffer = ""
        self.question_keywords = {
            "how",
            "what",
            "why",
            "when",
            "where",
            "can",
            "do",
            "does",
            "is",
            "are",
            "will",
            "would",
            "should",
            "could",
            "who",
            "which",
        }
        self.leading_keywords = set(self.question_keywords) | {"curious", "wondering"}
        self.intent_phrases = (
            "i'm curious",
            "im curious",
            "i am curious",
            "i'm wondering",
            "im wondering",
            "i am wondering",
            "i also want to know",
            "i also want to",
            "i also need to know",
            "i also need to",
            "i also wonder",
        )
        self.leading_fillers = ("and", "so", "well", "also", "but", "then", "plus")
        self.conjunction_pattern = re.compile(r"\b(and|also|what about|tell me about)\b", re.IGNORECASE)
        self.question_q = Queue()
        self.text_q = Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.processor = threading.Thread(target=self._ingest_loop, daemon=True)
        self.worker.start()
        self.processor.start()
        self.partial_max_wait = 1.2
        self.partial_wait_start = None
        self.flush_word_limit = 24
        self.max_buffer_chars = 4000

    def add_text(self, text):
        if not text:
            return
        cleaned = text.strip()
        if cleaned:
            self.text_q.put(cleaned)

    def stop(self):
        self.stop_event.set()
        self.text_q.put(None)
        if self.processor.is_alive():
            self.processor.join(timeout=1.0)
        self.question_q.put(None)
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)

    def _ingest_loop(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.text_q.get(timeout=0.1)
            except Empty:
                self._flush_stale_partial()
                continue

            if chunk is None:
                break

            self._append_chunk(chunk)
            force = self._should_flush(self.buffer)
            self._process_buffer(force=force)
            self._flush_stale_partial()

        # Drain remainder before exiting.
        self._process_buffer(force=True)

    def _append_chunk(self, chunk):
        normalized = self._normalize_chunk(chunk)
        if not normalized:
            return
        if self.buffer:
            self.buffer = f"{self.buffer} {normalized}"
        else:
            self.buffer = normalized
        if len(self.buffer) > self.max_buffer_chars:
            self.buffer = self.buffer[-self.max_buffer_chars :]

    def _normalize_chunk(self, chunk):
        return re.sub(r"\s+", " ", chunk).strip()

    def _dispatch_loop(self):
        while not self.stop_event.is_set():
            try:
                questions = self.question_q.get(timeout=0.1)
            except Empty:
                continue
            if questions is None:
                break
            if questions:
                self.question_callback(questions)

    def _process_buffer(self, force=False):
        if not self.buffer:
            return

        question_groups = self._extract_questions()
        emitted = False
        for group in question_groups:
            if group:
                self.question_q.put(group)
                emitted = True

        if emitted:
            self.partial_wait_start = None
        elif self.buffer and self.partial_wait_start is None:
            self.partial_wait_start = time.time()

        if force and self.buffer:
            leftover = self.buffer.strip()
            if leftover:
                self.question_q.put([leftover])
            self.buffer = ""
            self.partial_wait_start = None

    def _flush_stale_partial(self):
        if not self.buffer:
            self.partial_wait_start = None
            return
        now = time.time()
        if self.partial_wait_start is None:
            self.partial_wait_start = now
            return
        if now - self.partial_wait_start >= self.partial_max_wait:
            leftover = self.buffer.strip()
            if leftover:
                self.question_q.put([leftover])
            self.buffer = ""
            self.partial_wait_start = None

    def _extract_questions(self):
        if not self.buffer:
            return []

        normalized = re.sub(r"\s+", " ", self.buffer).strip()
        if not normalized:
            self.buffer = ""
            return []

        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        pending = ""
        results = []

        for idx, sentence in enumerate(sentences):
            if not sentence:
                continue
            is_last = idx == len(sentences) - 1
            if is_last and not self._has_terminal_punctuation(sentence):
                pending = sentence
                break

            splits = self._split_into_questions(sentence)
            if splits:
                results.append(splits)

        leftover = pending.strip()
        if leftover:
            conjunction_splits = self._split_conjunction_questions(leftover)
            if len(conjunction_splits) > 1:
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
                results.append([leftover])
                leftover = ""

        self.buffer = leftover
        return results

    def _split_into_questions(self, text):
        segments = []
        parts = [part.strip() for part in text.split("?") if part.strip()]
        for part in parts:
            if self._looks_like_question(part):
                segments.append(part)
        return segments

    def _split_conjunction_questions(self, text):
        splits = []
        start = 0
        for match in self.conjunction_pattern.finditer(text):
            segment = text[start : match.start()].strip(" ,")
            if self._looks_like_question(segment):
                splits.append(segment)
            start = match.end()
        tail = text[start:].strip(" ,")
        if tail:
            splits.append(tail)
        return splits

    def _has_terminal_punctuation(self, text):
        trimmed = text.rstrip()
        if trimmed.endswith("..."):
            return False
        return trimmed.endswith("?") or trimmed.endswith(".") or trimmed.endswith("!")

    def _should_flush(self, text):
        if not text:
            return False
        words = len(text.split())
        return words >= self.flush_word_limit

    def _looks_like_question(self, text):
        if not text:
            return False
        stripped = text.strip()
        lowered = stripped.lower().lstrip()
        lowered = self._remove_leading_fillers(lowered)
        if stripped.endswith("?"):
            return True
        for phrase in self.intent_phrases:
            if lowered.startswith(phrase):
                return True
        words = lowered.split()
        if not words:
            return False
        return words[0] in self.leading_keywords

    def _remove_leading_fillers(self, lowered):
        if not lowered:
            return lowered
        changed = True
        while changed:
            changed = False
            for filler in self.leading_fillers:
                prefix = f"{filler} "
                if lowered.startswith(prefix):
                    lowered = lowered[len(prefix) :].lstrip()
                    changed = True
        return lowered

    def _is_complete_question(self, text):
        if not text:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        words = stripped.split()
        return len(words) >= 3 and self._looks_like_question(stripped)
