from __future__ import annotations

import datetime
from typing import Callable

from question_detector import QuestionDetector


class QuestionManager:
    """
    Wraps the legacy QuestionDetector and funnels detected questions to the
    response pipeline along with WebSocket broadcasts.
    """

    def __init__(
        self,
        on_question_event: Callable[[dict], None],
        on_question_payload: Callable[[dict], None],
    ):
        self.on_question_event = on_question_event
        self.on_question_payload = on_question_payload
        self.detector = QuestionDetector(self._handle_questions)

    def add_text(self, text: str):
        self.detector.add_text(text)

    def reset_for_call(self):
        self.detector.reset_for_call()

    def flush_pending(self):
        self.detector.flush_pending()

    def wait_until_idle(self):
        self.detector.wait_until_idle()

    def stop(self):
        self.detector.stop()

    def save_logs(self, call_dir: str):
        self.detector.save_logs(call_dir)

    def question_count(self) -> int:
        with self.detector.buffer_lock:
            return len(self.detector.detected_questions_buffer)

    def backpressure_count(self) -> int:
        return self.detector.backpressure_count

    # ------------------------------------------------------------------ callbacks
    def _handle_questions(self, payload):
        """
        Receive payloads from QuestionDetector and forward them.
        """
        questions = payload.get("questions") if isinstance(payload, dict) else payload
        if not questions:
            return

        if isinstance(questions, str):
            questions = [questions]

        timestamp = datetime.datetime.utcnow().isoformat()
        for question in questions:
            event = {"Timestamp": timestamp, "Text": question}
            self.on_question_event(event)

        self.on_question_payload(payload)
