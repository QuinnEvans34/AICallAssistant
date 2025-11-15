"""
Response Management and Question Processing System

Handles intelligent question detection, intent classification, LLM response generation,
and response caching for the CallAssist system. Manages conversation flow with
duplicate detection, context awareness, and customer name personalization.

Key Features:
- Real-time question extraction from speech transcripts
- Intent classification using sentence transformers
- Semantic and exact-match response caching
- Duplicate question detection with cooldown periods
- Customer name personalization with usage limits
- Streaming LLM responses with fallback handling
- Confidence scoring and logging for analytics

Architecture:
- Threaded processing loop for asynchronous question handling
- Multi-level caching (exact match + semantic similarity)
- Intent-based routing with confidence thresholds
- Context-aware response generation
- Thread-safe operations with proper locking

Dependencies:
- sentence_transformers: For intent classification and semantic caching
- rapidfuzz: For fuzzy string matching in deduplication
- numpy: For confidence score calculations
- json: For intent logging and configuration
- re: For question parsing and text normalization

Author: Quinn Evans
"""

import json
import re
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime
from typing import List, Dict, Any
from queue import Empty, Queue

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from exception_logger import exception_logger


class IntentClassifier:
    """
    Classifies user questions into predefined intent categories.

    Uses sentence transformer embeddings to classify questions into solar panel
    related categories like pricing, installation, warranty, etc. Provides
    confidence scores for routing decisions.

    Attributes:
        labels (list): Predefined intent categories
        model (SentenceTransformer): Embedding model for classification
        label_vectors (np.ndarray): Pre-computed embeddings for intent labels
    """

    def __init__(self):
        """
        Initialize the intent classifier with predefined categories.

        Loads the sentence transformer model and pre-computes embeddings
        for all intent labels. Gracefully handles model loading failures.
        """
        self.labels = ["pricing", "installation", "warranty", "battery", "financing", "general"]

        try:
            # Load lightweight sentence transformer model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            # Pre-compute embeddings for all intent labels
            self.label_vectors = self.model.encode(self.labels, normalize_embeddings=True)
        except Exception as exc:  # pragma: no cover
            print(f"Intent classifier disabled: {exc}")
            self.model = None
            self.label_vectors = None

    def classify(self, question: str) -> tuple[str, float]:
        """
        Classify a question into an intent category with confidence score.

        Args:
            question (str): The question text to classify

        Returns:
            tuple[str, float]: (intent_label, confidence_score)
                - intent_label: One of the predefined categories
                - confidence_score: Float between 0.0 and 1.0

        Note:
            Returns ("general", 0.5) if classification is unavailable
        """
        if not self.model or not question:
            return "general", 0.5

        # Encode question and compute similarities
        q_vec = self.model.encode(question, normalize_embeddings=True)
        sims = util.cos_sim(q_vec, self.label_vectors)

        # Find most similar intent
        top_idx = int(np.argmax(sims))
        intent = self.labels[top_idx]
        confidence = float(sims[0][top_idx])

        return intent, confidence


class ResponseManager:
    """
    Orchestrates question processing, response generation, and conversation management.

    Manages the complete pipeline from raw speech input to final response output,
    including question extraction, deduplication, intent classification, LLM
    generation, and caching. Handles customer personalization and maintains
    conversation context.

    Key responsibilities:
    - Question extraction and normalization from speech transcripts
    - Duplicate detection with configurable cooldown periods
    - Intent classification and confidence scoring
    - LLM response generation with streaming support
    - Multi-level response caching (exact + semantic)
    - Customer name personalization with usage limits
    - Conversation context management
    - Thread-safe operation with proper synchronization

    Attributes:
        matcher: Q&A knowledge base matcher for finding relevant answers
        llm: Language model interface for response generation
        response_callback (callable): Function called with generated responses
        confidence_callback (callable): Function called with confidence scores
        use_llm (bool): Whether to use LLM for responses (vs. fallback)
        customer_name_provider (callable): Function to get customer name

    Threading:
        Uses a dedicated processing thread for asynchronous question handling.
        All shared state is protected with appropriate locks for thread safety.
    """

    def __init__(
        self,
        matcher,
        llm,
        response_callback,
        use_llm=True,
        customer_name_provider=None,
        confidence_callback=None,
    ):
        """
        Initialize the response manager with required components.

        Args:
            matcher: Q&A matcher instance for knowledge base lookup
            llm: LLM interface for response generation
            response_callback (callable): Called with response payloads
            use_llm (bool): Enable LLM responses (default: True)
            customer_name_provider (callable): Optional customer name getter
            confidence_callback (callable): Optional confidence score callback
        """
        # Core components
        self.matcher = matcher
        self.llm = llm
        self.response_callback = response_callback
        self.confidence_callback = confidence_callback
        self.use_llm = use_llm
        self.customer_name_provider = customer_name_provider

        # Question processing queue and thread
        self.question_q = Queue()
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        self._active_processing = 0
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

        # Conversation context (limited history)
        self.context = []
        self.context_lock = threading.Lock()
        self.context_limit = 2

        # Question parsing patterns and keywords
        self.conjunction_pattern = re.compile(r"\b(and|also|what about|tell me about)\b", re.IGNORECASE)
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

        # Duplicate question tracking
        self.recent_questions = {}  # normalized_question -> timestamp
        self.question_order = deque()  # For LRU eviction
        self.recent_limit = 50
        self.question_lock = threading.Lock()
        self.duplicate_threshold = 80  # Fuzzy match threshold (0-100)
        self.duplicate_cooldown = 20  # Seconds before allowing duplicates

        # Customer name personalization limits
        self.name_use_count = 0
        self.max_name_uses = 5  # Maximum name uses per conversation
        self.last_response_used_name = False
        self.name_usage_lock = threading.Lock()

        # Response caching and bundling
        self.bundle_index = 0
        self.response_cache = OrderedDict()  # LRU cache
        self.cache_lock = threading.Lock()
        self.cache_limit = 50

        # Intent classification and semantic caching
        self.intent_classifier = IntentClassifier()
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # pragma: no cover
            print(f"Semantic embedder disabled: {exc}")
            self.embedder = None
        # Response logging buffer
        self.responses_buffer: List[Dict[str, Any]] = []
        self.response_lock = threading.Lock()

    # ----------------------------------------------------------------- Public API

    def add_question(self, question_payload):
        """
        Add a question payload to the processing queue.

        The payload can be a string, list of strings, or dict containing
        questions and metadata like ASR confidence scores.

        Args:
            question_payload: Question text(s) and optional metadata
        """
        self.question_q.put(question_payload)

    def reset_name_usage(self):
        """
        Reset customer name usage counters.

        Allows name personalization to restart after conversation resets.
        Thread-safe operation.
        """
        with self.name_usage_lock:
            self.name_use_count = 0
            self.last_response_used_name = False

    def note_manual_name_usage(self, used_name: bool):
        """
        Record manual customer name usage for tracking limits.

        Args:
            used_name (bool): Whether a name was used in the response
        """
        with self.name_usage_lock:
            if used_name:
                if self.name_use_count < self.max_name_uses:
                    self.name_use_count += 1
                self.last_response_used_name = True
            else:
                self.last_response_used_name = False

    def stop(self):
        """
        Stop the processing thread and clean up resources.

        Signals the processing loop to exit and waits for thread completion.
        Non-blocking with timeout for graceful shutdown.
        """
        self.stop_event.set()
        self.question_q.put(None)  # Sentinel to wake thread
        self.thread.join(timeout=1.0)

    def reset_for_call(self):
        """
        Reset internal state for a new call session.

        Clears caches, context, and reporting buffers so each call
        generates isolated analytics.
        """
        self._ensure_response_logging_state()
        self._ensure_processing_tracking()

        with self.context_lock:
            self.context.clear()

        with self.cache_lock:
            self.response_cache.clear()

        with self.question_lock:
            self.recent_questions.clear()
            self.question_order.clear()

        with self.response_lock:
            self.responses_buffer.clear()

        with self.name_usage_lock:
            self.name_use_count = 0
            self.last_response_used_name = False

        self.bundle_index = 0
        self._drain_queue(self.question_q)

    def wait_until_idle(self, timeout: float = 3.0) -> bool:
        """
        Wait for the processing loop to finish outstanding questions.

        Args:
            timeout (float): Maximum seconds to wait

        Returns:
            bool: True if idle before timeout, False otherwise.
        """
        self._ensure_processing_tracking()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.stop_event.is_set():
                return True
            if self.question_q.empty():
                with self.processing_lock:
                    if self._active_processing == 0:
                        return True
            time.sleep(0.05)
        return False

    def _drain_queue(self, queue_obj: Queue):
        """Remove all items from an internal queue without blocking."""
        while True:
            try:
                queue_obj.get_nowait()
            except Empty:
                break

    def _ensure_response_logging_state(self):
        """
        Ensure response logging buffers/locks exist.

        Guards against partially-initialized instances when older objects
        are reused across reloads.
        """
        if not hasattr(self, "responses_buffer"):
            self.responses_buffer = []
        if not hasattr(self, "response_lock"):
            self.response_lock = threading.Lock()

    def _ensure_processing_tracking(self):
        """
        Ensure processing tracking primitives exist.
        """
        if not hasattr(self, "processing_lock"):
            self.processing_lock = threading.Lock()
        if not hasattr(self, "_active_processing"):
            self._active_processing = 0

    # --------------------------------------------------------------- Main Processing

    def _process_loop(self):
        """
        Main question processing loop running in dedicated thread.

        Continuously processes questions from the queue, handling extraction,
        deduplication, matching, and response generation. Runs until stopped.
        """
        self._ensure_processing_tracking()
        while not self.stop_event.is_set():
            try:
                # Get next question payload with timeout
                payload = self.question_q.get(timeout=0.1)
            except Empty:
                continue

            if payload is None:  # Sentinel value for shutdown
                break

            with self.processing_lock:
                self._active_processing += 1
            try:
                # Extract block_id if provided (for question editing)
                existing_block_id = None
                if isinstance(payload, dict):
                    existing_block_id = payload.get("block_id")

                # Extract and normalize questions
                questions, meta = self._extract_questions(payload)
                asr_conf = meta.get("asr_conf", 0.75)

                # Remove duplicates within this batch and against history
                deduped = []
                for question in questions:
                    if not self._is_duplicate_question(question, deduped):
                        deduped.append(question)
                questions = deduped
                if not questions:
                    continue

                # Process each question through matcher and classifier
                qa_bundle = []
                flattened_answers = []
                score_rows = []

                for raw_question in questions:
                    # Find matching Q&A pairs
                    matches = self.matcher.match(raw_question)
                    intent, intent_conf = self.intent_classifier.classify(raw_question)
                    match_conf = matches[0]["score"] / 100.0 if matches else 0.4

                    if matches:
                        top_match = matches[0]
                        canonical_question = top_match["question"]
                        answers = [match["answer"] for match in matches[:1]]
                        qa_bundle.append({
                            "question": canonical_question,
                            "answers": answers,
                            "score": top_match.get("score", 0),
                            "intent": intent,
                        })
                        flattened_answers.extend(answers)
                    else:
                        # No matches found
                        qa_bundle.append({
                            "question": raw_question,
                            "answers": [],
                            "score": 0,
                            "intent": intent,
                        })

                    # Calculate combined confidence score
                    combined = self._combined_confidence(asr_conf, intent_conf, match_conf)
                    score_rows.append({
                        "question": raw_question,
                        "intent": intent,
                        "combined": combined
                    })

                if not qa_bundle:
                    print("ResponseManager: no matching entries for questions -> skipping response.")
                    continue

                # Generate unique bundle ID for tracking, or use existing for edits
                bundle_id = existing_block_id if existing_block_id else self._bundle_id(qa_bundle)

                # Report average confidence if callback provided
                if self.confidence_callback and score_rows:
                    avg_conf = float(np.mean([row["combined"] for row in score_rows]))
                    try:
                        self.confidence_callback(avg_conf)
                    except Exception:
                        pass  # Ignore callback errors

                # Get current conversation context
                context_snapshot = self._get_context()

                if self.use_llm:
                    # Emit draft response and start LLM generation in background
                    self._emit_response(
                        qa_bundle,
                        None,
                        is_draft=True,
                        bundle_id=bundle_id,
                        replace=bool(existing_block_id),  # Replace if editing existing
                    )
                    threading.Thread(
                        target=self._llm_response,
                        args=(qa_bundle, flattened_answers, bundle_id, context_snapshot),
                        daemon=True,
                    ).start()
                else:
                    # Use fallback response without LLM
                    fallback = self._compose_fallback(qa_bundle)
                    customer_name = self._get_customer_name_if_allowed()
                    used_name = bool(customer_name and fallback)
                    if used_name:
                        fallback = f"{customer_name}, {fallback}"
                    self._emit_response(qa_bundle, fallback, bundle_id=bundle_id, replace=bool(existing_block_id))
                    self._record_name_usage(used_name)

                # Remember questions for duplicate detection (skip for edits to avoid marking as duplicate)
                if not existing_block_id:
                    self._remember_questions([item["question"] for item in qa_bundle])
            finally:
                with self.processing_lock:
                    self._active_processing = max(0, self._active_processing - 1)

    # ------------------------------------------------------------- LLM Response Generation

    def _llm_response(self, qa_bundle, answers, bundle_id=None, context=None):
        """
        Generate LLM response for a question bundle.

        Handles caching, streaming, and fallback logic for LLM responses.
        Runs in background thread to avoid blocking the main processing loop.

        Args:
            qa_bundle: List of question/answer dictionaries
            answers: Flattened list of answer texts
            bundle_id: Unique identifier for this response bundle
            context: Recent conversation context
        """
        customer_name = self._get_customer_name_if_allowed()

        # Build cache key including questions, answers, context, and name
        cache_key = self._build_cache_key(qa_bundle, context, customer_name)
        cached = self._get_cached_response(cache_key)

        # Semantic cache check using embeddings
        if not cached and self.embedder:
            question_text = " ".join(item["question"] for item in qa_bundle)
            q_emb = self.embedder.encode(question_text, normalize_embeddings=True)

            # Check similarity against cached embeddings
            for cached_q, (emb, resp) in self.cache_embeddings.items():
                sim = util.cos_sim(q_emb, emb)[0][0]
                if sim > 0.9:  # High similarity threshold
                    cached = resp
                    print(f"Semantic cache hit: {sim:.2f}")
                    break

        # Streaming callback for real-time response updates
        stream_buffer = []
        def on_token(token):
            stream_buffer.append(token)
            self._emit_response(
                qa_bundle,
                "".join(stream_buffer),
                bundle_id=bundle_id,
                replace=True,
                message_type="response",
            )

        if cached:
            # Use cached response
            response = cached
            success = True
        else:
            # Generate new response with LLM
            response, success = self.llm.generate_response(
                qa_bundle,
                context,
                customer_name=customer_name,
                stream_callback=on_token,
            )

            # Cache successful responses
            if success and response:
                self._store_cached_response(cache_key, response)

                # Store embedding for semantic caching
                if self.embedder:
                    q_emb = self.embedder.encode(question_text, normalize_embeddings=True)
                    self.cache_embeddings[question_text] = (q_emb, response)
                    # Limit semantic cache size
                    while len(self.cache_embeddings) > 20:
                        self.cache_embeddings.popitem(last=False)

        if success and response:
            # Emit final response
            self._emit_response(
                qa_bundle,
                response,
                bundle_id=bundle_id,
                replace=bool(bundle_id),
            )

            # Log the response
            self._log_response(qa_bundle, response, success, end_time - start_time)

        else:
            # Fallback to knowledge base answers
            fallback = " ".join(answers) if answers else "No matching response found."
            if customer_name and fallback:
                fallback = f"{customer_name}, {fallback}"
            self._emit_response(
                qa_bundle,
                fallback,
                bundle_id=bundle_id,
                replace=bool(bundle_id),
            )

            # Log the fallback response
            self._log_response(qa_bundle, fallback, False, end_time - start_time)

        # Track name usage
        self._record_name_usage(bool(customer_name))

        # Update conversation context
        question_summary = " ".join(item["question"] for item in qa_bundle)
        self._update_context(question_summary)

    def _log_response(self, qa_bundle: List[Dict], response_text: str, success: bool, latency: float):
        """
        Log a generated response for reporting.

        Args:
            qa_bundle: Question/answer bundle
            response_text: Generated response text
            success: Whether LLM generation was successful
            latency: Response generation time in seconds
        """
        timestamp = datetime.now().isoformat()
        question_text = " ".join(item.get("question", "") for item in qa_bundle)

        self._ensure_response_logging_state()
        with self.response_lock:
            self.responses_buffer.append({
                "timestamp": timestamp,
                "question": question_text,
                "answer": response_text,
                "llm_latency": latency,
                "success": success
            })

    def save_logs(self, call_dir: str):
        """
        Save response logs to file.

        Args:
            call_dir (str): Directory to save response logs
        """
        import os

        responses_file = os.path.join(call_dir, "responses.json")
        self._ensure_response_logging_state()
        with self.response_lock:
            try:
                with open(responses_file, 'w', encoding='utf-8') as f:
                    json.dump(self.responses_buffer, f, indent=2, ensure_ascii=False)
            except Exception as e:
                exception_logger.log_exception(e, "response_manager", "Failed to save responses")

    # ------------------------------------------------------------- Question Processing

    def _extract_questions(self, payload) -> tuple[list[str], dict]:
        """
        Extract and normalize questions from various payload formats.

        Handles string, list, and dict payloads containing question text
        and optional metadata like ASR confidence scores.

        Args:
            payload: Question payload (str, list, or dict)

        Returns:
            tuple[list[str], dict]: (normalized_questions, metadata)
        """
        asr_conf = 0.75  # Default confidence

        if isinstance(payload, dict):
            asr_conf = float(payload.get("asr_conf", asr_conf))
            raw = payload.get("questions") or payload.get("text") or payload.get("question")
            if isinstance(raw, (list, tuple)):
                raw_questions = [str(part).strip() for part in raw if part]
            elif raw:
                raw_questions = [str(raw).strip()]
            else:
                raw_questions = []
        elif isinstance(payload, (list, tuple)):
            raw_questions = [str(part).strip() for part in payload if part]
        else:
            raw_questions = [str(payload).strip()]

        # Normalize and filter questions
        normalized = self._normalize_questions(raw_questions)
        return normalized, {"asr_conf": asr_conf}

    def _normalize_questions(self, raw_questions: list[str]) -> list[str]:
        """
        Normalize a list of raw question strings.

        Splits compound questions, filters incomplete questions, and
        ensures proper formatting.

        Args:
            raw_questions: List of raw question strings

        Returns:
            list[str]: Normalized, complete questions
        """
        normalized = []
        for item in raw_questions:
            normalized.extend(self._split_questions(item))
        # Filter out empty and incomplete questions
        normalized = [q for q in normalized if q]
        return [q for q in normalized if self._is_complete_question(q)]

    def _split_questions(self, text: str) -> list[str]:
        """
        Split compound questions connected by conjunctions.

        Handles questions like "How much does it cost and when is installation?"
        by splitting on conjunctions and question marks.

        Args:
            text: Input text that may contain multiple questions

        Returns:
            list[str]: Individual question segments
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # Split on question marks first
        parts = [part.strip() for part in text.split("?") if part.strip()]
        if "?" in text and parts:
            return parts

        # Split on conjunctions
        segments = []
        start = 0
        for match in self.conjunction_pattern.finditer(text):
            segment = text[start: match.start()].strip(" ,")
            if self._looks_like_question(segment):
                segments.append(segment)
            start = match.end()

        # Add remaining text
        tail = text[start:].strip(" ,")
        if tail:
            segments.append(tail)

        return segments if segments else [text]

    def _looks_like_question(self, text: str) -> bool:
        """
        Determine if text appears to be a question.

        Uses keyword detection, intent phrases, and punctuation analysis.

        Args:
            text: Text to analyze

        Returns:
            bool: True if text appears to be a question
        """
        if not text:
            return False

        stripped = text.strip()
        lowered = stripped.lower().lstrip()
        lowered = self._remove_leading_fillers(lowered)

        # Explicit question mark
        if stripped.endswith("?"):
            return True

        # Intent phrases (e.g., "I'm curious about...")
        for phrase in self.intent_phrases:
            if lowered.startswith(phrase):
                return True

        # Question keywords at start
        words = lowered.split()
        if not words:
            return False
        return words[0] in self.leading_keywords

    def _is_complete_question(self, text: str) -> bool:
        """
        Check if a question is complete and meaningful.

        Requires minimum length and question-like characteristics.

        Args:
            text: Question text to validate

        Returns:
            bool: True if question is complete
        """
        stripped = text.strip()
        if not stripped:
            return False

        if stripped.endswith("?"):
            return True

        words = stripped.split()
        # Require at least 3 words and question-like structure
        return len(words) >= 3 and self._looks_like_question(stripped)

    # -------------------------------------------------------- Duplicate Detection

    def _is_duplicate_question(self, question: str, pending_batch: list = None) -> bool:
        """
        Check if a question is a duplicate of recent questions.

        Compares against both pending batch and historical questions
        using fuzzy matching and cooldown periods.

        Args:
            question: Question to check for duplication
            pending_batch: Other questions in current batch

        Returns:
            bool: True if question should be considered duplicate
        """
        normalized = self._normalize_for_dedupe(question)
        if not normalized:
            return True

        # Check against pending batch
        if pending_batch:
            normalized_batch = [
                self._normalize_for_dedupe(item) for item in pending_batch if item
            ]
            for asked in normalized_batch:
                if asked and self._questions_similar(normalized, asked):
                    return True

        # Check against historical questions
        now = time.time()
        with self.question_lock:
            self._prune_old_questions(now)
            for asked, timestamp in self.recent_questions.items():
                if now - timestamp > self.duplicate_cooldown:
                    continue
                if self._questions_similar(normalized, asked):
                    return True

        return False

    def _remember_questions(self, questions: list[str]):
        """
        Store questions for future duplicate detection.

        Args:
            questions: List of question strings to remember
        """
        now = time.time()
        for question in questions:
            normalized = self._normalize_for_dedupe(question)
            if not normalized:
                continue

            with self.question_lock:
                self.recent_questions[normalized] = now
                self.question_order.append((normalized, now))

                # Enforce size limits
                if len(self.question_order) > self.recent_limit:
                    self._prune_old_questions(now)

    def _normalize_for_dedupe(self, text: str) -> str:
        """
        Normalize text for duplicate comparison.

        Removes punctuation, converts to lowercase, and normalizes whitespace.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text for comparison
        """
        if not text:
            return ""
        # Remove non-alphanumeric characters and normalize whitespace
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _questions_similar(self, a: str, b: str) -> bool:
        """
        Determine if two normalized questions are similar.

        Uses fuzzy string matching with multiple algorithms and thresholds.

        Args:
            a, b: Normalized question strings to compare

        Returns:
            bool: True if questions are considered similar
        """
        if not a or not b:
            return False
        if a == b:
            return True

        # Fuzzy ratio matching
        ratio = max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b))
        if ratio >= self.duplicate_threshold:
            return True

        # Prefix matching for longer questions
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 10 and longer.startswith(shorter):
            return True

        # Word overlap matching
        return self._prefix_word_overlap(a, b) >= 3

    def _prefix_word_overlap(self, a: str, b: str) -> int:
        """
        Count overlapping words at the beginning of questions.

        Args:
            a, b: Question strings to compare

        Returns:
            int: Number of overlapping prefix words
        """
        aw = a.split()
        bw = b.split()
        count = 0
        for aword, bword in zip(aw, bw):
            if aword != bword:
                break
            count += 1
        return count

    # -------------------------------------------------------------- Context Management

    def _get_context(self) -> list[str]:
        """
        Get recent conversation context.

        Returns:
            list[str]: Recent questions for context
        """
        with self.context_lock:
            return list(self.context[-self.context_limit:])

    def _update_context(self, question: str):
        """
        Add question to conversation context.

        Args:
            question: Question to add to context
        """
        if not question:
            return

        with self.context_lock:
            self.context.append(question.strip())
            if len(self.context) > self.context_limit:
                self.context = self.context[-self.context_limit:]

    # -------------------------------------------------------------- Customer Names

    def _get_customer_name_if_allowed(self) -> str:
        """
        Get customer name if usage limits allow.

        Respects maximum usage count and prevents consecutive usage.

        Returns:
            str or None: Customer name if allowed, None otherwise
        """
        if not callable(self.customer_name_provider):
            return None

        with self.name_usage_lock:
            if self.name_use_count >= self.max_name_uses or self.last_response_used_name:
                return None

            name = self.customer_name_provider()
            if not name:
                return None

            cleaned = name.strip()
            if not cleaned:
                return None

            # Record usage
            self.name_use_count += 1
            self.last_response_used_name = True
            return cleaned

    def _record_name_usage(self, used_name: bool):
        """
        Record whether name was used in response.

        Args:
            used_name: Whether customer name was included
        """
        with self.name_usage_lock:
            if not used_name:
                self.last_response_used_name = False

    # -------------------------------------------------------------- Response Emission

    def _format_questions(self, qa_bundle: list[dict]) -> str:
        """
        Format questions for display in responses.

        Args:
            qa_bundle: List of question dictionaries

        Returns:
            str: Formatted question list
        """
        questions = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        return "\n".join(f"- {q}" for q in questions if q)

    def _emit_response(
        self,
        qa_bundle: list[dict],
        response_text: str,
        is_draft: bool = False,
        bundle_id: str = None,
        replace: bool = False,
        message_type: str = "response",
    ):
        """
        Emit a response payload to the callback.

        Args:
            qa_bundle: Question/answer bundle
            response_text: Generated response text
            is_draft: Whether this is a draft/preview response
            bundle_id: Unique bundle identifier
            replace: Whether to replace previous response with same ID
            message_type: Type of message ("response", etc.)
        """
        questions_text = self._format_questions(qa_bundle)

        if is_draft:
            loading = "(Preparing response...)"
            message = f"Answering:\n{questions_text or '- ...'}\n\n{loading}"
        else:
            if questions_text and message_type == "response":
                message = f"Question(s):\n{questions_text}\n\nResponse:\n{response_text}"
            else:
                message = response_text

        payload = {
            "text": message,
            "block_id": bundle_id,
            "replace": replace,
            "is_draft": is_draft,
            "message_type": message_type,
            "timestamp": time.time(),  # Add timestamp for ordering
            "questions": [item.get("question", "") for item in qa_bundle if item.get("question")],
            "response_text": response_text,
        }

        self.response_callback(payload)

    # ---------------------------------------------------------- Confidence Calculation

    def _combined_confidence(self, asr_conf: float, intent_conf: float, match_conf: float) -> float:
        """
        Calculate combined confidence score from multiple sources.

        Weights: 50% ASR, 30% intent classification, 20% knowledge base match

        Args:
            asr_conf: Speech recognition confidence (0.0-1.0)
            intent_conf: Intent classification confidence (0.0-1.0)
            match_conf: Knowledge base match confidence (0.0-1.0)

        Returns:
            float: Combined confidence score (0.0-1.0)
        """
        combined = (0.5 * asr_conf) + (0.3 * intent_conf) + (0.2 * match_conf)
        return float(max(0.0, min(1.0, combined)))

    # --------------------------------------------------------- Cache Management

    def _compose_fallback(self, qa_bundle: list[dict]) -> str:
        """
        Compose fallback response from knowledge base answers.

        Args:
            qa_bundle: Question bundle with answers

        Returns:
            str: Concatenated answer text
        """
        snippets = []
        for item in qa_bundle:
            if item["answers"]:
                snippets.append(item["answers"][0])
        return " ".join(snippets).strip()

    def _bundle_id(self, qa_bundle: list[dict]) -> str:
        """
        Generate unique bundle identifier for response tracking.

        Args:
            qa_bundle: Question bundle

        Returns:
            str: Unique bundle ID
        """
        questions = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        if not questions:
            return None

        self.bundle_index += 1
        joined = "|".join(questions)
        return f"{self.bundle_index}:{joined}"

    def _build_cache_key(self, qa_bundle: list[dict], context: list, customer_name: str):
        """
        Build cache key from question bundle and context.

        Args:
            qa_bundle: Question/answer bundle
            context: Conversation context
            customer_name: Customer name (if used)

        Returns:
            tuple: Immutable cache key
        """
        questions = tuple(sorted(item.get("question", "").strip().lower() for item in qa_bundle))
        answers = tuple(
            (item.get("question", "").strip().lower(), (item.get("answers") or [""])[0].strip().lower())
            for item in qa_bundle
        )
        context_key = tuple((context or [])[:self.context_limit])
        name_key = (customer_name or "").strip().lower()
        return (questions, answers, context_key, name_key)

    def _get_cached_response(self, cache_key):
        """
        Retrieve cached response if available.

        Args:
            cache_key: Cache key tuple

        Returns:
            str or None: Cached response text
        """
        if not cache_key:
            return None

        with self.cache_lock:
            response = self.response_cache.get(cache_key)
            if response:
                # Move to end (most recently used)
                self.response_cache.move_to_end(cache_key)
            return response

    def _store_cached_response(self, cache_key, response: str):
        """
        Store response in cache with LRU eviction.

        Args:
            cache_key: Cache key tuple
            response: Response text to cache
        """
        if not cache_key or not response:
            return

        with self.cache_lock:
            self.response_cache[cache_key] = response
            self.response_cache.move_to_end(cache_key)

            # Evict oldest entries if over limit
            while len(self.response_cache) > self.cache_limit:
                self.response_cache.popitem(last=False)

    def _prune_old_questions(self, now: float = None):
        """
        Remove expired questions from duplicate tracking.

        Args:
            now: Current timestamp (default: time.time())
        """
        if now is None:
            now = time.time()

        while self.question_order:
            normalized, timestamp = self.question_order[0]
            # Keep if within cooldown and under size limit
            if (now - timestamp) <= self.duplicate_cooldown and len(self.question_order) <= self.recent_limit:
                break

            # Remove expired entry
            self.question_order.popleft()
            if self.recent_questions.get(normalized) == timestamp:
                del self.recent_questions[normalized]

    def _remove_leading_fillers(self, lowered: str) -> str:
        """
        Remove leading filler words from text.

        Args:
            lowered: Lowercase text to clean

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
    def __init__(
        self,
        matcher,
        llm,
        response_callback,
        use_llm=True,
        customer_name_provider=None,
        confidence_callback=None,
    ):
        self.matcher = matcher
        self.llm = llm
        self.response_callback = response_callback
        self.confidence_callback = confidence_callback
        self.use_llm = use_llm
        self.customer_name_provider = customer_name_provider

        self.question_q = Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

        self.context = []
        self.context_lock = threading.Lock()
        self.context_limit = 2

        self.conjunction_pattern = re.compile(r"\b(and|also|what about|tell me about)\b", re.IGNORECASE)
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

        self.recent_questions = {}
        self.question_order = deque()
        self.recent_limit = 50
        self.question_lock = threading.Lock()
        self.duplicate_threshold = 80
        self.duplicate_cooldown = 20

        self.name_use_count = 0
        self.max_name_uses = 5
        self.last_response_used_name = False
        self.name_usage_lock = threading.Lock()

        self.bundle_index = 0
        self.response_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.cache_limit = 50

        self.intent_classifier = IntentClassifier()
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # pragma: no cover
            print(f"Semantic embedder disabled: {exc}")
            self.embedder = None
        self.cache_embeddings = OrderedDict()

    # ----------------------------------------------------------------- public

    def add_question(self, question_payload):
        self.question_q.put(question_payload)

    def reset_name_usage(self):
        with self.name_usage_lock:
            self.name_use_count = 0
            self.last_response_used_name = False

    def note_manual_name_usage(self, used_name):
        with self.name_usage_lock:
            if used_name:
                if self.name_use_count < self.max_name_uses:
                    self.name_use_count += 1
                self.last_response_used_name = True
            else:
                self.last_response_used_name = False

    def stop(self):
        self.stop_event.set()
        self.question_q.put(None)
        self.thread.join(timeout=1.0)

    # --------------------------------------------------------------- main loop

    def _process_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = self.question_q.get(timeout=0.1)
            except Empty:
                continue

            if payload is None:
                break

            questions, meta = self._extract_questions(payload)
            asr_conf = meta.get("asr_conf", 0.75)

            deduped = []
            for question in questions:
                if not self._is_duplicate_question(question, deduped):
                    deduped.append(question)
            questions = deduped
            if not questions:
                continue

            qa_bundle = []
            flattened_answers = []
            score_rows = []

            for raw_question in questions:
                matches = self.matcher.match(raw_question)
                intent, intent_conf = self.intent_classifier.classify(raw_question)
                match_conf = matches[0]["score"] / 100.0 if matches else 0.4
                if matches:
                    top_match = matches[0]
                    canonical_question = top_match["question"]
                    answers = [match["answer"] for match in matches[:1]]
                    qa_bundle.append(
                        {
                            "question": canonical_question,
                            "answers": answers,
                            "score": top_match.get("score", 0),
                            "intent": intent,
                        }
                    )
                    flattened_answers.extend(answers)
                else:
                    qa_bundle.append(
                        {
                            "question": raw_question,
                            "answers": [],
                            "score": 0,
                            "intent": intent,
                        }
                    )
                combined = self._combined_confidence(asr_conf, intent_conf, match_conf)
                score_rows.append({"question": raw_question, "intent": intent, "combined": combined})

            if not qa_bundle:
                print("ResponseManager: no matching entries for questions -> skipping response.")
                continue

            bundle_id = self._bundle_id(qa_bundle)
            if self.confidence_callback and score_rows:
                avg_conf = float(np.mean([row["combined"] for row in score_rows]))
                try:
                    self.confidence_callback(avg_conf)
                except Exception:
                    pass

            context_snapshot = self._get_context()

            if self.use_llm:
                self._emit_response(
                    qa_bundle,
                    None,
                    is_draft=True,
                    bundle_id=bundle_id,
                    replace=False,
                )
                threading.Thread(
                    target=self._llm_response,
                    args=(qa_bundle, flattened_answers, bundle_id, context_snapshot),
                    daemon=True,
                ).start()
            else:
                fallback = self._compose_fallback(qa_bundle)
                customer_name = self._get_customer_name_if_allowed()
                used_name = bool(customer_name and fallback)
                if used_name:
                    fallback = f"{customer_name}, {fallback}"
                self._emit_response(qa_bundle, fallback, bundle_id=bundle_id, replace=False)
                self._record_name_usage(used_name)

            self._remember_questions([item["question"] for item in qa_bundle])

    # ------------------------------------------------------------- LLM & cache

    def _llm_response(self, qa_bundle, answers, bundle_id=None, context=None):
        customer_name = self._get_customer_name_if_allowed()
        cache_key = self._build_cache_key(qa_bundle, context, customer_name)
        cached = self._get_cached_response(cache_key)

        # Semantic cache check
        if not cached and self.embedder:
            question_text = " ".join(item["question"] for item in qa_bundle)
            q_emb = self.embedder.encode(question_text, normalize_embeddings=True)
            for cached_q, (emb, resp) in self.cache_embeddings.items():
                sim = util.cos_sim(q_emb, emb)[0][0]
                if sim > 0.9:
                    cached = resp
                    print(f"Semantic cache hit: {sim:.2f}")
                    break

        stream_buffer = []

        def on_token(token):
            stream_buffer.append(token)
            self._emit_response(
                qa_bundle,
                "".join(stream_buffer),
                bundle_id=bundle_id,
                replace=True,
                message_type="response",
            )

        if cached:
            response = cached
            success = True
        else:
            response, success = self.llm.generate_response(
                qa_bundle,
                context,
                customer_name=customer_name,
                stream_callback=on_token,
            )
            if success and response:
                self._store_cached_response(cache_key, response)
                # Store embedding
                if self.embedder:
                    q_emb = self.embedder.encode(question_text, normalize_embeddings=True)
                    self.cache_embeddings[question_text] = (q_emb, response)
                    while len(self.cache_embeddings) > 20:
                        self.cache_embeddings.popitem(last=False)

        if success and response:
            self._emit_response(
                qa_bundle,
                response,
                bundle_id=bundle_id,
                replace=bool(bundle_id),
            )
        else:
            fallback = " ".join(answers) if answers else "No matching response found."
            if customer_name and fallback:
                fallback = f"{customer_name}, {fallback}"
            self._emit_response(
                qa_bundle,
                fallback,
                bundle_id=bundle_id,
                replace=bool(bundle_id),
            )
        self._record_name_usage(bool(customer_name))
        question_summary = " ".join(item["question"] for item in qa_bundle)
        self._update_context(question_summary)

    # ------------------------------------------------------------- utilities

    def _extract_questions(self, payload):
        asr_conf = 0.75
        if isinstance(payload, dict):
            asr_conf = float(payload.get("asr_conf", asr_conf))
            raw = payload.get("questions") or payload.get("text") or payload.get("question")
            if isinstance(raw, (list, tuple)):
                raw_questions = [str(part).strip() for part in raw if part]
            elif raw:
                raw_questions = [str(raw).strip()]
            else:
                raw_questions = []
        elif isinstance(payload, (list, tuple)):
            raw_questions = [str(part).strip() for part in payload if part]
        else:
            raw_questions = [str(payload).strip()]
        normalized = self._normalize_questions(raw_questions)
        return normalized, {"asr_conf": asr_conf}

    def _normalize_questions(self, raw_questions):
        normalized = []
        for item in raw_questions:
            normalized.extend(self._split_questions(item))
        normalized = [q for q in normalized if q]
        return [q for q in normalized if self._is_complete_question(q)]

    def _split_questions(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split("?") if part.strip()]
        if "?" in text and parts:
            return parts
        segments = []
        start = 0
        for match in self.conjunction_pattern.finditer(text):
            segment = text[start : match.start()].strip(" ,")
            if self._looks_like_question(segment):
                segments.append(segment)
            start = match.end()
        tail = text[start:].strip(" ,")
        if tail:
            segments.append(tail)
        return segments if segments else [text]

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

    def _is_complete_question(self, text):
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        words = stripped.split()
        return len(words) >= 3 and self._looks_like_question(stripped)

    # -------------------------------------------------------- dedupe tracking

    def _is_duplicate_question(self, question, pending_batch=None):
        normalized = self._normalize_for_dedupe(question)
        if not normalized:
            return True
        if pending_batch:
            normalized_batch = [
                self._normalize_for_dedupe(item) for item in pending_batch if item
            ]
            for asked in normalized_batch:
                if asked and self._questions_similar(normalized, asked):
                    return True

        now = time.time()
        with self.question_lock:
            self._prune_old_questions(now)
            for asked, timestamp in self.recent_questions.items():
                if now - timestamp > self.duplicate_cooldown:
                    continue
                if self._questions_similar(normalized, asked):
                    return True
        return False

    def _remember_questions(self, questions):
        now = time.time()
        for question in questions:
            normalized = self._normalize_for_dedupe(question)
            if not normalized:
                continue
            with self.question_lock:
                self.recent_questions[normalized] = now
                self.question_order.append((normalized, now))
                if len(self.question_order) > self.recent_limit:
                    self._prune_old_questions(now)

    def _normalize_for_dedupe(self, text):
        if not text:
            return ""
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _questions_similar(self, a, b):
        if not a or not b:
            return False
        if a == b:
            return True
        ratio = max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b))
        if ratio >= self.duplicate_threshold:
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 10 and longer.startswith(shorter):
            return True
        return self._prefix_word_overlap(a, b) >= 3

    def _prefix_word_overlap(self, a, b):
        aw = a.split()
        bw = b.split()
        count = 0
        for aword, bword in zip(aw, bw):
            if aword != bword:
                break
            count += 1
        return count

    # -------------------------------------------------------------- context

    def _get_context(self):
        with self.context_lock:
            return list(self.context[-self.context_limit :])

    def _update_context(self, question):
        if not question:
            return
        with self.context_lock:
            self.context.append(question.strip())
            if len(self.context) > self.context_limit:
                self.context = self.context[-self.context_limit :]

    # -------------------------------------------------------------- names

    def _get_customer_name_if_allowed(self):
        if not callable(self.customer_name_provider):
            return None
        with self.name_usage_lock:
            if self.name_use_count >= self.max_name_uses or self.last_response_used_name:
                return None
            name = self.customer_name_provider()
            if not name:
                return None
            cleaned = name.strip()
            if not cleaned:
                return None
            self.name_use_count += 1
            self.last_response_used_name = True
            return cleaned

    def _record_name_usage(self, used_name):
        with self.name_usage_lock:
            if not used_name:
                self.last_response_used_name = False

    # -------------------------------------------------------------- emission

    def _format_questions(self, qa_bundle):
        questions = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        return "\n".join(f"- {q}" for q in questions if q)

    def _emit_response(
        self,
        qa_bundle,
        response_text,
        is_draft=False,
        bundle_id=None,
        replace=False,
        message_type="response",
    ):
        questions_text = self._format_questions(qa_bundle)
        if is_draft:
            loading = "(Preparing response...)"
            message = f"Answering:\n{questions_text or '- ...'}\n\n{loading}"
        else:
            if questions_text and message_type == "response":
                message = f"Question(s):\n{questions_text}\n\nResponse:\n{response_text}"
            else:
                message = response_text
        payload = {
            "text": message,
            "block_id": bundle_id,
            "replace": replace,
            "is_draft": is_draft,
            "message_type": message_type,
            "timestamp": time.time(),  # Add timestamp for ordering
            "questions": [item.get("question", "") for item in qa_bundle if item.get("question")],
            "response_text": response_text,
        }
        self.response_callback(payload)

    # ---------------------------------------------------------- confidence log

    def _combined_confidence(self, asr_conf, intent_conf, match_conf):
        combined = (0.5 * asr_conf) + (0.3 * intent_conf) + (0.2 * match_conf)
        return float(max(0.0, min(1.0, combined)))

    # --------------------------------------------------------- cache helpers

    def _compose_fallback(self, qa_bundle):
        snippets = []
        for item in qa_bundle:
            if item["answers"]:
                snippets.append(item["answers"][0])
        return " ".join(snippets).strip()

    def _bundle_id(self, qa_bundle):
        questions = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        if not questions:
            return None
        self.bundle_index += 1
        joined = "|".join(questions)
        return f"{self.bundle_index}:{joined}"

    def _build_cache_key(self, qa_bundle, context, customer_name):
        questions = tuple(sorted(item.get("question", "").strip().lower() for item in qa_bundle))
        answers = tuple(
            (item.get("question", "").strip().lower(), (item.get("answers") or [""])[0].strip().lower())
            for item in qa_bundle
        )
        context_key = tuple((context or [])[: self.context_limit])
        name_key = (customer_name or "").strip().lower()
        return (questions, answers, context_key, name_key)

    def _get_cached_response(self, cache_key):
        if not cache_key:
            return None
        with self.cache_lock:
            response = self.response_cache.get(cache_key)
            if response:
                self.response_cache.move_to_end(cache_key)
            return response

    def _store_cached_response(self, cache_key, response):
        if not cache_key or not response:
            return
        with self.cache_lock:
            self.response_cache[cache_key] = response
            self.response_cache.move_to_end(cache_key)
            while len(self.response_cache) > self.cache_limit:
                self.response_cache.popitem(last=False)

    def _prune_old_questions(self, now=None):
        if now is None:
            now = time.time()
        while self.question_order:
            normalized, timestamp = self.question_order[0]
            if (now - timestamp) <= self.duplicate_cooldown and len(self.question_order) <= self.recent_limit:
                break
            self.question_order.popleft()
            if self.recent_questions.get(normalized) == timestamp:
                del self.recent_questions[normalized]

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
