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
    """Main orchestrator for question -> response pipeline."""

    def __init__(
        self,
        matcher,
        llm,
        response_callback,
        use_llm: bool = True,
        customer_name_provider=None,
        confidence_callback=None,
    ):
        # External components
        self.matcher = matcher
        self.llm = llm
        self.response_callback = response_callback
        self.confidence_callback = confidence_callback
        self.use_llm = use_llm
        self.customer_name_provider = customer_name_provider

        # Threading / queue
        self.question_q: Queue = Queue()
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        self._active_processing = 0
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

        # Context
        self.context: list[str] = []
        self.context_lock = threading.Lock()
        self.context_limit = 2

        # Parsing helpers
        self.conjunction_pattern = re.compile(r"\b(and|also|what about|tell me about)\b", re.IGNORECASE)
        self.question_keywords = {
            "how","what","why","when","where","can","do","does","is","are","will","would","should","could","who","which"
        }
        self.leading_keywords = set(self.question_keywords) | {"curious", "wondering"}
        self.intent_phrases = (
            "i'm curious","im curious","i am curious","i'm wondering","im wondering","i am wondering",
            "i also want to know","i also want to","i also need to know","i also need to","i also wonder"
        )
        self.leading_fillers = ("and", "so", "well", "also", "but", "then", "plus")

        # Duplicate tracking
        self.recent_questions: dict[str, float] = {}
        self.question_order: deque = deque()
        self.recent_limit = 50
        self.question_lock = threading.Lock()
        self.duplicate_threshold = 80
        self.duplicate_cooldown = 20

        # Name usage
        self.name_use_count = 0
        self.max_name_uses = 5
        self.last_response_used_name = False
        self.name_usage_lock = threading.Lock()

        # Caching
        self.bundle_index = 0
        self.response_cache: OrderedDict = OrderedDict()
        self.cache_lock = threading.Lock()
        self.cache_limit = 50

        # Embeddings / semantic cache
        self.intent_classifier = IntentClassifier()
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # pragma: no cover
            print(f"Semantic embedder disabled: {exc}")
            self.embedder = None
        self.cache_embeddings: OrderedDict[str, tuple[np.ndarray, str]] = OrderedDict()

        # Response logging
        self.responses_buffer: List[Dict[str, Any]] = []
        self.response_lock = threading.Lock()

    # ------------------------------ public API ------------------------------
    def add_question(self, question_payload):
        self.question_q.put(question_payload)

    def stop(self):
        self.stop_event.set()
        self.question_q.put(None)
        self.thread.join(timeout=1.0)

    def reset_for_call(self):
        with self.context_lock:
            self.context.clear()
        with self.cache_lock:
            self.response_cache.clear()
            self.cache_embeddings.clear()
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

    def save_logs(self, call_dir: str):
        import os
        path = os.path.join(call_dir, "responses.json")
        with self.response_lock:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(self.responses_buffer, fh, indent=2, ensure_ascii=False)
            except Exception as e:  # pragma: no cover
                exception_logger.log_exception(e, "response_manager", "Failed to save responses")

    # --------------------------- main processing loop -----------------------
    def _process_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = self.question_q.get(timeout=0.1)
            except Empty:
                continue
            if payload is None:
                break
            with self.processing_lock:
                self._active_processing += 1
            try:
                existing_block_id = payload.get("block_id") if isinstance(payload, dict) else None
                questions, meta = self._extract_questions(payload)
                asr_conf = meta.get("asr_conf", 0.75)
                deduped = []
                for q in questions:
                    if not self._is_duplicate_question(q, deduped):
                        deduped.append(q)
                questions = deduped
                if not questions:
                    continue
                qa_bundle, flattened_answers, score_rows = self._build_qa_bundle(questions, asr_conf)
                if not qa_bundle:
                    continue
                bundle_id = existing_block_id if existing_block_id else self._bundle_id(qa_bundle)
                if self.confidence_callback and score_rows:
                    avg_conf = float(np.mean([row["combined"] for row in score_rows]))
                    try:
                        self.confidence_callback(avg_conf)
                    except Exception:  # pragma: no cover
                        pass
                context_snapshot = self._get_context()
                if self.use_llm:
                    self._emit_response(qa_bundle, None, is_draft=True, bundle_id=bundle_id, replace=bool(existing_block_id))
                    threading.Thread(
                        target=self._llm_response,
                        args=(qa_bundle, flattened_answers, bundle_id, context_snapshot, existing_block_id is not None),
                        daemon=True,
                    ).start()
                else:
                    fallback = self._compose_fallback(qa_bundle)
                    customer_name = self._get_customer_name_if_allowed()
                    used_name = bool(customer_name and fallback)
                    if used_name:
                        fallback = f"{customer_name}, {fallback}"
                    self._emit_response(qa_bundle, fallback, bundle_id=bundle_id, replace=bool(existing_block_id))
                    self._record_name_usage(used_name)
                if not existing_block_id:
                    self._remember_questions([item["question"] for item in qa_bundle])
            finally:
                with self.processing_lock:
                    self._active_processing = max(0, self._active_processing - 1)

    # ------------------------- building QA bundle ---------------------------
    def _build_qa_bundle(self, questions: list[str], asr_conf: float):
        qa_bundle = []
        flattened_answers = []
        score_rows = []
        for raw_question in questions:
            matches = self.matcher.match(raw_question)
            intent, intent_conf = self.intent_classifier.classify(raw_question)
            match_conf = matches[0]["score"] / 100.0 if matches else 0.4
            if matches:
                top = matches[0]
                canonical = top["question"]
                answers = [m["answer"] for m in matches[:1]]
                qa_bundle.append({"question": canonical, "answers": answers, "score": top.get("score", 0), "intent": intent})
                flattened_answers.extend(answers)
            else:
                qa_bundle.append({"question": raw_question, "answers": [], "score": 0, "intent": intent})
            combined = self._combined_confidence(asr_conf, intent_conf, match_conf)
            score_rows.append({"question": raw_question, "intent": intent, "combined": combined})
        return qa_bundle, flattened_answers, score_rows

    # -------------------------- LLM response path ---------------------------
    def _llm_response(self, qa_bundle, answers, bundle_id=None, context=None, replace: bool = False):
        start_time = time.time()
        customer_name = self._get_customer_name_if_allowed()
        cache_key = self._build_cache_key(qa_bundle, context, customer_name)
        cached = self._get_cached_response(cache_key)
        question_text = " ".join(item["question"] for item in qa_bundle)
        if not cached and self.embedder:
            q_emb = self.embedder.encode(question_text, normalize_embeddings=True)
            for cached_q, (emb, resp) in self.cache_embeddings.items():
                sim = util.cos_sim(q_emb, emb)[0][0]
                if sim > 0.9:
                    cached = resp
                    break
        stream_buffer: list[str] = []

        def on_token(tok: str):
            stream_buffer.append(tok)
            self._emit_response(qa_bundle, "".join(stream_buffer), bundle_id=bundle_id, replace=True, message_type="response")

        if cached:
            response = cached
            success = True
        else:
            response, success = self.llm.generate_response(qa_bundle, context, customer_name=customer_name, stream_callback=on_token)
            if success and response:
                self._store_cached_response(cache_key, response)
                if self.embedder:
                    q_emb = self.embedder.encode(question_text, normalize_embeddings=True)
                    self.cache_embeddings[question_text] = (q_emb, response)
                    while len(self.cache_embeddings) > 20:
                        self.cache_embeddings.popitem(last=False)
        end_time = time.time()
        latency = end_time - start_time
        if success and response:
            self._emit_response(qa_bundle, response, bundle_id=bundle_id, replace=True)
            self._log_response(qa_bundle, response, True, latency)
        else:
            fallback = " ".join(answers) if answers else "No matching response found."
            if customer_name and fallback:
                fallback = f"{customer_name}, {fallback}"
            self._emit_response(qa_bundle, fallback, bundle_id=bundle_id, replace=True)
            self._log_response(qa_bundle, fallback, False, latency)
        self._record_name_usage(bool(customer_name))
        question_summary = " ".join(item["question"] for item in qa_bundle)
        self._update_context(question_summary)

    def _log_response(self, qa_bundle: List[Dict], response_text: str, success: bool, latency: float):
        timestamp = datetime.utcnow().isoformat()
        question_text = " ".join(item.get("question", "") for item in qa_bundle)
        with self.response_lock:
            self.responses_buffer.append({
                "timestamp": timestamp,
                "question": question_text,
                "answer": response_text,
                "llm_latency": latency,
                "success": success,
            })

    # ----------------------- extraction & normalization ---------------------
    def _extract_questions(self, payload) -> tuple[list[str], dict]:
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

    def _normalize_questions(self, raw_questions: list[str]) -> list[str]:
        out: list[str] = []
        for item in raw_questions:
            out.extend(self._split_questions(item))
        out = [q for q in out if q]
        return [q for q in out if self._is_complete_question(q)]

    def _split_questions(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split("?") if p.strip()]
        if "?" in text and parts:
            return parts
        segments = []
        start = 0
        for match in self.conjunction_pattern.finditer(text):
            segment = text[start:match.start()].strip(" ,")
            if self._looks_like_question(segment):
                segments.append(segment)
            start = match.end()
        tail = text[start:].strip(" ,")
        if tail:
            segments.append(tail)
        return segments if segments else [text]

    def _looks_like_question(self, text: str) -> bool:
        if not text:
            return False
        stripped = text.strip()
        lowered = self._remove_leading_fillers(stripped.lower().lstrip())
        if stripped.endswith("?"):
            return True
        for phrase in self.intent_phrases:
            if lowered.startswith(phrase):
                return True
        words = lowered.split()
        return bool(words) and words[0] in self.leading_keywords

    def _is_complete_question(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        words = stripped.split()
        return len(words) >= 3 and self._looks_like_question(stripped)

    # --------------------------- duplicate detection ------------------------
    def _is_duplicate_question(self, question: str, pending_batch: list[str] | None = None) -> bool:
        normalized = self._normalize_for_dedupe(question)
        if not normalized:
            return True
        if pending_batch:
            for asked in [self._normalize_for_dedupe(x) for x in pending_batch if x]:
                if asked and self._questions_similar(normalized, asked):
                    return True
        now = time.time()
        with self.question_lock:
            self._prune_old_questions(now)
            for asked, ts in self.recent_questions.items():
                if now - ts > self.duplicate_cooldown:
                    continue
                if self._questions_similar(normalized, asked):
                    return True
        return False

    def _remember_questions(self, questions: list[str]):
        now = time.time()
        for q in questions:
            normalized = self._normalize_for_dedupe(q)
            if not normalized:
                continue
            with self.question_lock:
                self.recent_questions[normalized] = now
                self.question_order.append((normalized, now))
                if len(self.question_order) > self.recent_limit:
                    self._prune_old_questions(now)

    def _normalize_for_dedupe(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _questions_similar(self, a: str, b: str) -> bool:
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

    def _prefix_word_overlap(self, a: str, b: str) -> int:
        aw = a.split(); bw = b.split(); count = 0
        for aword, bword in zip(aw, bw):
            if aword != bword:
                break
            count += 1
        return count

    def _prune_old_questions(self, now: float):
        while self.question_order:
            normalized, ts = self.question_order[0]
            if (now - ts) <= self.duplicate_cooldown and len(self.question_order) <= self.recent_limit:
                break
            self.question_order.popleft()
            if self.recent_questions.get(normalized) == ts:
                del self.recent_questions[normalized]

    # ------------------------------ context ----------------------------------
    def _get_context(self) -> list[str]:
        with self.context_lock:
            return list(self.context[-self.context_limit:])

    def _update_context(self, question: str):
        if not question:
            return
        with self.context_lock:
            self.context.append(question.strip())
            if len(self.context) > self.context_limit:
                self.context = self.context[-self.context_limit:]

    # ------------------------------ names ------------------------------------
    def _get_customer_name_if_allowed(self) -> str | None:
        if not callable(self.customer_name_provider):
            return None
        with self.name_usage_lock:
            if self.name_use_count >= self.max_name_uses or self.last_response_used_name:
                return None
            name = (self.customer_name_provider() or "").strip()
            if not name:
                return None
            self.name_use_count += 1
            self.last_response_used_name = True
            return name

    def _record_name_usage(self, used_name: bool):
        with self.name_usage_lock:
            if not used_name:
                self.last_response_used_name = False

    # ------------------------------ emission ---------------------------------
    def _format_questions(self, qa_bundle: list[dict]) -> str:
        qs = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        return "\n".join(f"- {q}" for q in qs if q)

    def _emit_response(self, qa_bundle: list[dict], response_text: str, is_draft: bool = False, bundle_id: str | None = None, replace: bool = False, message_type: str = "response"):
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
            "timestamp": time.time(),
            "questions": [item.get("question", "") for item in qa_bundle if item.get("question")],
            "response_text": response_text,
        }
        self.response_callback(payload)

    # ------------------------------ confidence -------------------------------
    def _combined_confidence(self, asr_conf: float, intent_conf: float, match_conf: float) -> float:
        return float(max(0.0, min(1.0, (0.5 * asr_conf) + (0.3 * intent_conf) + (0.2 * match_conf))))

    # ------------------------------ cache helpers ----------------------------
    def _compose_fallback(self, qa_bundle: list[dict]) -> str:
        snippets = []
        for item in qa_bundle:
            if item.get("answers"):
                snippets.append(item["answers"][0])
        return " ".join(snippets).strip()

    def _bundle_id(self, qa_bundle: list[dict]) -> str | None:
        questions = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        if not questions:
            return None
        self.bundle_index += 1
        joined = "|".join(questions)
        return f"{self.bundle_index}:{joined}"

    def _build_cache_key(self, qa_bundle: list[dict], context: list[str] | None, customer_name: str | None):
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
            resp = self.response_cache.get(cache_key)
            if resp:
                self.response_cache.move_to_end(cache_key)
            return resp

    def _store_cached_response(self, cache_key, response: str):
        if not cache_key or not response:
            return
        with self.cache_lock:
            self.response_cache[cache_key] = response
            self.response_cache.move_to_end(cache_key)
            while len(self.response_cache) > self.cache_limit:
                self.response_cache.popitem(last=False)

    # ------------------------------ misc helpers ----------------------------
    def _remove_leading_fillers(self, lowered: str) -> str:
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

    def _drain_queue(self, queue_obj: Queue):
        while True:
            try:
                queue_obj.get_nowait()
            except Empty:
                break
