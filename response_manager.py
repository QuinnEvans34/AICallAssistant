import re
import threading
import time
from collections import OrderedDict, deque
from queue import Empty, Queue

from rapidfuzz import fuzz


class ResponseManager:
    def __init__(self, matcher, llm, response_callback, use_llm=True, customer_name_provider=None):
        self.matcher = matcher
        self.llm = llm
        self.response_callback = response_callback
        self.use_llm = use_llm
        self.customer_name_provider = customer_name_provider
        self.question_q = Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        self.context = []
        self.context_lock = threading.Lock()
        self.context_limit = 1  # Reduced for speed
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
        self.duplicate_cooldown = 20  # seconds before same question allowed again
        self.name_use_count = 0
        self.max_name_uses = 5
        self.last_response_used_name = False
        self.name_usage_lock = threading.Lock()
        self.bundle_index = 0
        self.response_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.cache_limit = 50

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

    def _process_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = self.question_q.get(timeout=0.1)
            except Empty:
                continue

            if payload is None:
                break

            questions = self._normalize_questions(payload)
            deduped = []
            for question in questions:
                if not self._is_duplicate_question(question, deduped):
                    deduped.append(question)
            questions = deduped
            if not questions:
                continue

            qa_bundle = []
            flattened_answers = []
            for raw_question in questions:
                matches = self.matcher.match(raw_question)
                if matches:
                    top_match = matches[0]
                    canonical_question = top_match["question"]
                    truncated = [match["answer"] for match in matches[:1]]  # Limit to 1 answer per question for speed
                    qa_bundle.append(
                        {
                            "question": canonical_question,
                            "answers": truncated,
                            "score": top_match.get("score", 0),
                        }
                    )
                    flattened_answers.extend(truncated)

            if not qa_bundle:
                print("ResponseManager: no matching entries for questions -> skipping response.")
                continue

            bundle_id = self._bundle_id(qa_bundle)
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

    def _llm_response(self, qa_bundle, answers, bundle_id=None, context=None):
        customer_name = self._get_customer_name_if_allowed()
        cache_key = self._build_cache_key(qa_bundle, context, customer_name)
        cached = self._get_cached_response(cache_key)

        if cached:
            response = cached
            success = True
        else:
            response, success = self.llm.generate_response(
                qa_bundle, context, customer_name=customer_name
            )
            if success and response:
                self._store_cached_response(cache_key, response)

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

    def _normalize_questions(self, payload):
        if isinstance(payload, (list, tuple)):
            raw_questions = [part.strip() for part in payload if part]
        else:
            raw_questions = [str(payload).strip()]

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
        return len(words) >= 4 and self._looks_like_question(stripped)

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

    def _format_questions(self, qa_bundle):
        questions = [item.get("question", "").strip() for item in qa_bundle if item.get("question")]
        return "; ".join(questions)

    def _emit_response(self, qa_bundle, response_text, is_draft=False, bundle_id=None, replace=False):
        questions_text = self._format_questions(qa_bundle)
        label = "Question"
        if is_draft:
            loading = "(Preparing response...)"
            message = f"{label}: {questions_text or '...'}\n\n{loading}"
        else:
            if questions_text:
                message = f"{label}: {questions_text}\n\nResponse:\n{response_text}"
            else:
                message = response_text
        if bundle_id:
            payload = {
                "text": message,
                "block_id": bundle_id,
                "replace": replace,
                "is_draft": is_draft,
            }
            self.response_callback(payload)
        else:
            self.response_callback(message)

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
                # move to end to mark as recently used
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
            # Only remove if this timestamp is still current
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
