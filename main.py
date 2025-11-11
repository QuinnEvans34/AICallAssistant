import json
import os

from ui_flet import FletUI
from stream_asr import StreamASR
from question_detector import QuestionDetector
from response_manager import ResponseManager
from matcher import Matcher
from llm import LLMManager


class CallAssistApp:
    def __init__(self):
        print("App initializing")
        # Load Q&A data
        bundle_dir = os.getcwd()
        qa_file = os.path.join(bundle_dir, "qa_script.json")
        with open(qa_file, "r") as f:
            self.qa_data = json.load(f)

        # Load persona
        persona_file = os.path.join(bundle_dir, "persona.txt")
        with open(persona_file, "r", encoding="utf-8") as f:
            self.persona = f.read()

        # Initialize components
        self.matcher = Matcher(self.qa_data)
        self.llm = LLMManager(self.persona)
        self.caller_name = ""
        self.ui = FletUI(self.toggle_live, self.manual_input, self.on_caller_name_change)
        self.question_detector = QuestionDetector(self.on_question_detected)
        self.response_manager = ResponseManager(
            self.matcher,
            self.llm,
            self.ui.update_response,
            use_llm=True,
            customer_name_provider=self.get_caller_name,
        )
        self._prompt_for_caller_name()
        self.asr = StreamASR(self.on_partial_transcript)
        self._reset_transcript_tracking()

        # State
        self.live_mode = False
        print("App initialized")

    def toggle_live(self):
        print("Toggle live called")
        try:
            if self.live_mode:
                self.asr.stop_listening()
                self.live_mode = False
                self.ui.update_status("Off")
                self._reset_transcript_tracking()
            else:
                self.response_manager.reset_name_usage()
                self._reset_transcript_tracking()
                self.asr.start_listening()
                self.live_mode = True
                self.ui.update_status("Live")
                self._send_greeting()
        except Exception as e:
            print(f"Error in toggle_live: {e}")

    def on_partial_transcript(self, text):
        if not text:
            return
        novel_text = self._extract_novel_text(text)
        if not novel_text:
            return
        print(f"[ASR] {novel_text}")
        self.question_detector.add_text(novel_text)

    def on_question_detected(self, questions):
        self.response_manager.add_question(questions)

    def manual_input(self, question):
        self.response_manager.add_question(question)

    def on_caller_name_change(self, name):
        cleaned = (name or "").strip()
        self.caller_name = cleaned
        self.ui.set_caller_name(cleaned)
        if hasattr(self, "response_manager"):
            self.response_manager.reset_name_usage()

    def get_caller_name(self):
        return self.caller_name

    def _extract_novel_text(self, text):
        text = " ".join(text.strip().split())
        if not text:
            return ""

        tokens = self._tokenize(text)
        if not tokens:
            return ""

        if not self._last_tokens:
            self._last_tokens = tokens
            self._last_partial = text
            return text

        overlap = self._compute_overlap(self._last_tokens, tokens)
        if overlap == len(tokens):
            self._last_tokens = tokens
            self._last_partial = text
            return ""

        novel_tokens = tokens[overlap:]
        if not novel_tokens:
            return ""

        novel_text = " ".join(novel_tokens).strip()

        self._last_tokens = tokens
        self._last_partial = text
        return novel_text

    @staticmethod
    def _tokenize(text):
        return [tok for tok in text.split() if tok]

    @staticmethod
    def _compute_overlap(previous_tokens, current_tokens):
        max_overlap = min(len(previous_tokens), len(current_tokens))
        for size in range(max_overlap, 0, -1):
            prev_slice = [tok.lower() for tok in previous_tokens[-size:]]
            curr_slice = [tok.lower() for tok in current_tokens[:size]]
            if prev_slice == curr_slice:
                return size
        return 0

    def _reset_transcript_tracking(self):
        self._last_tokens = []
        self._last_partial = ""

    def run(self):
        import flet as ft
        ft.app(self.ui.run)

    def _prompt_for_caller_name(self):
        # For Flet, the name is entered in the UI
        pass

    def _send_greeting(self):
        name = self.get_caller_name() or "there"
        script = f"Hello {name}, this is John calling from ES Solar. Is now a good time for a call?"
        self.ui.update_response({'response': script, 'question': '', 'replace': False})
        if self.response_manager:
            self.response_manager.note_manual_name_usage(bool(self.get_caller_name()))

    def on_close(self):
        self.asr.stop_listening()
        self.question_detector.stop()
        self.response_manager.stop()


if __name__ == "__main__":
    app = CallAssistApp()
    app.run()
