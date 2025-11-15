from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from call_summary_generator import CallSummaryGenerator
from call_transcript_recorder import TranscriptRecorder
from evaluation_report_builder import EvaluationReportBuilder
from exception_logger import exception_logger
from llm import LLMManager
from matcher import Matcher

# Support both package and script execution for internal modules
try:
    from .heartbeat_manager import HeartbeatManager  # type: ignore
    from .question_manager import QuestionManager  # type: ignore
    from .response_manager import ResponseManager  # type: ignore
    from .transcript_stream import TranscriptStream  # type: ignore
except Exception:  # pragma: no cover
    from heartbeat_manager import HeartbeatManager  # type: ignore
    from question_manager import QuestionManager  # type: ignore
    from response_manager import ResponseManager  # type: ignore
    from transcript_stream import TranscriptStream  # type: ignore


class CallManager:
    """
    Coordinates ASR, question detection, and response generation lifecycles.
    """

    def __init__(
        self,
        transcript_stream: TranscriptStream,
        heartbeat: HeartbeatManager,
        broadcast_transcript: Callable[[str], None],
        broadcast_question: Callable[[dict], None],
        broadcast_suggestion: Callable[[dict], None],
        broadcast_status: Callable[[dict], None],
        logs_root: Optional[str] = None,
        transcript_recorder: Optional[TranscriptRecorder] = None,
        evaluation_builder: Optional[EvaluationReportBuilder] = None,
        summary_generator: Optional[CallSummaryGenerator] = None,
        matcher: Optional[Matcher] = None,
        llm_manager: Optional[LLMManager] = None,
    ):
        self.transcript_stream = transcript_stream
        self.heartbeat = heartbeat
        self.broadcast_transcript = broadcast_transcript
        self.broadcast_question = broadcast_question
        self.broadcast_suggestion = broadcast_suggestion
        self.broadcast_status = broadcast_status

        self.logs_root = logs_root or os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(self.logs_root, exist_ok=True)

        self.matcher = matcher or Matcher(self._load_qa_data())
        self.llm = llm_manager or LLMManager(self._load_persona())
        self.transcript_recorder = transcript_recorder or TranscriptRecorder()
        self.evaluation_builder = evaluation_builder or EvaluationReportBuilder()
        self.summary_generator = summary_generator or CallSummaryGenerator(self.llm)

        self.question_manager = QuestionManager(
            on_question_event=self._handle_question_event,
            on_question_payload=self._forward_question_payload,
        )
        self.response_manager = ResponseManager(
            matcher=self.matcher,
            llm=self.llm,
            response_callback=self._handle_response_payload,
            use_llm=True,
            customer_name_provider=self.get_caller_name,
            confidence_callback=self._handle_confidence,
        )

        self.transcript_stream.on_transcript = self._handle_transcript
        self.transcript_stream.on_status = self._handle_status_update
        self.transcript_stream.transcript_recorder = self.transcript_recorder

        self.live_mode = False
        self.current_call_id: Optional[str] = None
        self.call_dir: Optional[str] = None
        self.caller_name = ""
        self.transcript_buffer: list[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start_call(self) -> str:
        with self._lock:
            if self.live_mode:
                return self.current_call_id or ""

            call_id = datetime.utcnow().strftime("call_%Y%m%d_%H%M%S")
            call_dir = os.path.join(self.logs_root, call_id)
            os.makedirs(call_dir, exist_ok=True)

            self.current_call_id = call_id
            self.call_dir = call_dir

            error_log_path = os.path.join(call_dir, "errors.log")
            exception_logger.set_log_file(error_log_path)

            self.transcript_recorder.start(call_dir)
            self.evaluation_builder.start_call(call_dir)
            self.question_manager.reset_for_call()
            self.response_manager.reset_for_call()
            self.transcript_stream.reset_for_call()
            self.transcript_stream.start()
            self.transcript_buffer.clear()
            self.live_mode = True
            self.heartbeat.update(processing=True)

            return call_id

    def end_call(self):
        with self._lock:
            if not self.live_mode:
                return

            self.live_mode = False
            self.transcript_stream.stop()
            self.question_manager.flush_pending()
            self.question_manager.wait_until_idle()
            self.response_manager.wait_until_idle()

            call_dir = self.call_dir
            if call_dir:
                self.transcript_recorder.save()
                self.question_manager.save_logs(call_dir)
                self.response_manager.save_logs(call_dir)
                self.evaluation_builder.end_call()
                self.evaluation_builder.set_transcript_stats(
                    self.transcript_recorder.get_transcript_count(),
                    self.transcript_recorder.get_word_count(),
                )
                self.evaluation_builder.set_question_stats(self.question_manager.question_count())
                self.evaluation_builder.set_response_stats(
                    len(getattr(self.response_manager, "responses_buffer", [])),
                    [entry.get("llm_latency", 0) for entry in getattr(self.response_manager, "responses_buffer", [])],
                )
                self.evaluation_builder.metrics["detector_backpressure"] = self.question_manager.backpressure_count()
                self.evaluation_builder.save_report()
                self.summary_generator.generate_summary(call_dir)

            self.heartbeat.update(processing=False)
            self.current_call_id = None
            self.call_dir = None

    # ------------------------------------------------------------------ helpers
    def get_caller_name(self) -> str:
        return self.caller_name

    def set_caller_name(self, name: str):
        self.caller_name = (name or "").strip()

    def get_transcript_text(self) -> str:
        with self._lock:
            return "\n".join(self.transcript_buffer)

    # ------------------------------------------------------------------ callbacks
    def _handle_transcript(self, text: str):
        if not text:
            return
        with self._lock:
            self.transcript_buffer.append(text)
        self.broadcast_transcript(text)
        self.question_manager.add_text(text)

    def _handle_question_event(self, event: dict):
        self.broadcast_question(event)

    def _forward_question_payload(self, payload: dict):
        self.response_manager.add_question(payload)

    def _handle_response_payload(self, payload: dict):
        if payload.get("is_draft"):
            return

        questions = payload.get("questions") or []
        response_text = payload.get("response_text") or payload.get("text") or ""
        if not questions or not response_text:
            return

        event = {
            "Question": questions[0],
            "Items": [response_text.strip()],
        }
        self.broadcast_suggestion(event)

    def _handle_status_update(self, status: dict):
        self.heartbeat.update(**status)
        snapshot = self.heartbeat.snapshot()
        self.broadcast_status(
            {
                "AsrLatencyMs": int(round(snapshot.get("asr_latency_ms", 0.0))),
                "QueueDepth": snapshot.get("queue_depth", 0),
                "Errors": snapshot.get("errors", 0),
                "Processing": snapshot.get("processing", False),
            }
        )

    def _handle_confidence(self, value: float):
        self.broadcast_status(
            {
                "AsrLatencyMs": int(round(self.heartbeat.snapshot().get("asr_latency_ms", 0.0))),
                "QueueDepth": 0,
                "Errors": 0,
                "Processing": self.live_mode,
            }
        )

    # ------------------------------------------------------------------ config loaders
    def _load_qa_data(self):
        qa_path = os.path.join(os.getcwd(), "qa_script.json")
        with open(qa_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_persona(self) -> str:
        persona_path = os.path.join(os.getcwd(), "persona.txt")
        with open(persona_path, "r", encoding="utf-8") as fh:
            return fh.read()
