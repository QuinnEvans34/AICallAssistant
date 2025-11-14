"""
CallAssist - Real-Time Speech Response System

A comprehensive desktop application that provides real-time speech-to-text transcription,
question detection, and intelligent response generation for customer service calls.

The system integrates multiple AI components:
- Automatic Speech Recognition (ASR) using Faster-Whisper
- Question detection and intent classification
- Fuzzy matching against a Q&A knowledge base
- Large Language Model (LLM) response generation with personality
- Modern GUI with real-time updates and manual controls

Architecture:
- Modular design with separate components for each functionality
- Thread-safe communication between components
- Real-time audio streaming and processing
- Offline operation with local AI models

Author: Quinn Evans
"""

import json
import os

from ui_flet import FletUI
from stream_asr import StreamASR
from question_detector import QuestionDetector
from response_manager import ResponseManager
from matcher import Matcher
from llm import LLMManager


class CallAssistApp:
    """
    Main application orchestrator for CallAssist.

    This class initializes and coordinates all system components:
    - User interface (Flet-based GUI)
    - Speech recognition (ASR streaming)
    - Question detection and processing
    - Response generation and management
    - Q&A matching and LLM integration

    The app operates in two modes:
    - Live mode: Real-time audio processing from microphone
    - Manual mode: Text input for testing and manual questions

    Attributes:
        qa_data (dict): Loaded Q&A knowledge base from JSON file
        persona (str): Personality and communication guidelines for LLM
        matcher (Matcher): Fuzzy matching engine for Q&A lookup
        llm (LLMManager): Large language model interface for response generation
        caller_name (str): Current caller's name for personalization
        ui (FletUI): Graphical user interface component
        question_detector (QuestionDetector): Real-time question identification
        response_manager (ResponseManager): Orchestrates matching and LLM responses
        asr (StreamASR): Audio streaming and speech recognition
        live_mode (bool): Current operational mode (live audio vs manual)
    """

    def __init__(self):
        """
        Initialize the CallAssist application.

        Loads configuration files, initializes all components, and sets up
        the processing pipeline. Components are connected through callbacks
        to enable real-time communication.

        Raises:
            FileNotFoundError: If required configuration files are missing
            json.JSONDecodeError: If qa_script.json is malformed
        """
        print("App initializing")

        # Load Q&A knowledge base - contains all scripted responses
        bundle_dir = os.getcwd()
        qa_file = os.path.join(bundle_dir, "qa_script.json")
        with open(qa_file, "r") as f:
            self.qa_data = json.load(f)

        # Load personality guidelines for LLM responses
        persona_file = os.path.join(bundle_dir, "persona.txt")
        with open(persona_file, "r", encoding="utf-8") as f:
            self.persona = f.read()

        # Initialize core AI components
        self.matcher = Matcher(self.qa_data)  # Fuzzy matching for Q&A lookup
        self.llm = LLMManager(self.persona)   # LLM interface for response generation
        self.caller_name = ""  # Current caller's name for personalization

        # Initialize UI with callback functions for user interactions
        self.ui = FletUI(
            toggle_callback=self.toggle_live,           # Start/stop live audio
            manual_callback=self.manual_input,          # Manual text input
            name_callback=self.on_caller_name_change,   # Caller name updates
            question_edit_callback=self.on_question_edit # Question editing
        )

        # Initialize question detection with callback for detected questions
        self.question_detector = QuestionDetector(self.on_question_detected)

        # Initialize response management system
        self.response_manager = ResponseManager(
            matcher=self.matcher,                       # Q&A matching engine
            llm=self.llm,                              # LLM response generator
            response_callback=self.ui.update_response,  # UI update function
            use_llm=True,                              # Enable LLM responses
            customer_name_provider=self.get_caller_name, # Dynamic name provider
            confidence_callback=self.ui.update_confidence # Confidence visualization
        )

        # Prompt for caller name setup (handled by UI)
        self._prompt_for_caller_name()

        # Initialize audio streaming with callbacks
        self.asr = StreamASR(
            transcript_callback=self.on_partial_transcript,  # Partial transcript updates
            speaker_callback=self.ui.update_speaker_status   # Speaker detection updates
        )

        # Reset transcript tracking state
        self._reset_transcript_tracking()

        # Operational state
        self.live_mode = False  # Whether live audio processing is active

        print("App initialized")

    def toggle_live(self):
        """
        Toggle between live audio processing and idle modes.

        When activating live mode:
        - Starts audio streaming from microphone
        - Resets response manager state
        - Sends initial greeting
        - Updates UI status

        When deactivating live mode:
        - Stops audio streaming
        - Updates UI status
        - Resets transcript tracking

        This method is called by the UI toggle button.
        """
        print("Toggle live called")
        try:
            if self.live_mode:
                # Deactivate live mode
                self.asr.stop_listening()
                self.live_mode = False
                self.ui.update_status("Off")
                self._reset_transcript_tracking()
            else:
                # Activate live mode
                self.response_manager.reset_name_usage()  # Reset personalization state
                self._reset_transcript_tracking()         # Clear transcript history
                self.asr.start_listening()                # Start audio capture
                self.live_mode = True
                self.ui.update_status("Live")
                self._send_greeting()                     # Send initial greeting
        except Exception as e:
            print(f"Error in toggle_live: {e}")

    def on_partial_transcript(self, text):
        """
        Handle partial transcript updates from ASR.

        Processes incoming speech text, extracts novel content (avoids duplicates),
        and feeds it to the question detector for real-time analysis.

        Args:
            text (str): Partial transcript text from speech recognition
        """
        if not text:
            return

        # Extract only the new content not seen before
        novel_text = self._extract_novel_text(text)
        if not novel_text:
            return

        print(f"[ASR] {novel_text}")
        # Feed novel text to question detector for analysis
        self.question_detector.add_text(novel_text)

    def on_question_detected(self, questions):
        """
        Handle detected questions from the question detector.

        Routes questions to the response manager for matching against
        the Q&A database and LLM processing.

        Args:
            questions: Detected question(s) - can be string or list
        """
        self.response_manager.add_question(questions)

    def manual_input(self, question):
        """
        Handle manual text input from the user interface.

        Allows users to manually submit questions for processing,
        useful for testing or when ASR fails.

        Args:
            question (str): Manually entered question text
        """
        self.response_manager.add_question(question)

    def on_question_edit(self, edited_question, block_id=None):
        """
        Handle edited questions from the user interface.

        Treats the edited question as an update to an existing question,
        regenerating the response for the same conversation block.

        Args:
            edited_question (str): The edited question text
            block_id (str, optional): ID of the response block to update
        """
        if block_id:
            # For edited questions, we want to update the existing response block
            # Create a payload that includes the block_id for replacement
            payload = {
                "question": edited_question,
                "block_id": block_id,
                "replace": True
            }
            self.response_manager.add_question(payload)
        else:
            # Fallback to treating as new question
            self.response_manager.add_question(edited_question)

    def on_caller_name_change(self, name):
        """
        Handle caller name updates from the UI.

        Updates the stored caller name and notifies the response manager
        to reset personalization state when the name changes.

        Args:
            name (str): New caller name from UI input
        """
        cleaned = (name or "").strip()
        self.caller_name = cleaned
        self.ui.set_caller_name(cleaned)
        if hasattr(self, "response_manager"):
            self.response_manager.reset_name_usage()

    def get_caller_name(self):
        """
        Get the current caller name for personalization.

        Returns:
            str: Current caller name, empty string if not set
        """
        return self.caller_name

    def _extract_novel_text(self, text):
        """
        Extract novel text content not seen in previous transcripts.

        Uses token-based overlap detection to avoid processing duplicate
        content from overlapping audio chunks.

        Args:
            text (str): New transcript text to process

        Returns:
            str: Novel text content, empty if no new content
        """
        text = " ".join(text.strip().split())  # Normalize whitespace
        if not text:
            return ""

        tokens = self._tokenize(text)
        if not tokens:
            return ""

        if not self._last_tokens:
            # First text received
            self._last_tokens = tokens
            self._last_partial = text
            return text

        # Find overlap with previous text
        overlap = self._compute_overlap(self._last_tokens, tokens)
        if overlap == len(tokens):
            # Complete overlap - no new content
            self._last_tokens = tokens
            self._last_partial = text
            return ""

        # Extract novel tokens
        novel_tokens = tokens[overlap:]
        if not novel_tokens:
            return ""

        novel_text = " ".join(novel_tokens).strip()

        # Update tracking state
        self._last_tokens = tokens
        self._last_partial = text
        return novel_text

    @staticmethod
    def _tokenize(text):
        """
        Tokenize text into words for overlap computation.

        Args:
            text (str): Text to tokenize

        Returns:
            list: List of word tokens
        """
        return [tok for tok in text.split() if tok]

    @staticmethod
    def _compute_overlap(previous_tokens, current_tokens):
        """
        Compute the maximum overlapping token sequence between two token lists.

        Used to detect duplicate content in streaming transcripts.

        Args:
            previous_tokens (list): Tokens from previous transcript
            current_tokens (list): Tokens from current transcript

        Returns:
            int: Length of maximum overlapping sequence
        """
        max_overlap = min(len(previous_tokens), len(current_tokens))
        for size in range(max_overlap, 0, -1):
            prev_slice = [tok.lower() for tok in previous_tokens[-size:]]
            curr_slice = [tok.lower() for tok in current_tokens[:size]]
            if prev_slice == curr_slice:
                return size
        return 0

    def _reset_transcript_tracking(self):
        """
        Reset transcript tracking state.

        Clears the token history used for duplicate detection.
        Called when starting a new conversation or switching modes.
        """
        self._last_tokens = []
        self._last_partial = ""

    def run(self):
        """
        Start the application main loop.

        Launches the Flet UI framework and begins the application.
        This method blocks until the application is closed.
        """
        import flet as ft
        ft.app(self.ui.run)

    def _prompt_for_caller_name(self):
        """
        Prompt for caller name setup.

        In the Flet UI implementation, the name is entered directly
        in the interface, so this method is a no-op.
        """
        # For Flet, the name is entered in the UI
        pass

    def _send_greeting(self):
        """
        Send initial greeting message to start the conversation.

        Uses the caller's name if available, otherwise uses a generic greeting.
        This greeting is sent through the normal response pipeline.
        """
        name = self.get_caller_name() or "there"
        script = f"Hello {name}, this is John calling from ES Solar. Is now a good time for a call?"
        self.ui.update_response({'response': script, 'question': '', 'replace': False})
        if self.response_manager:
            self.response_manager.note_manual_name_usage(bool(self.get_caller_name()))

    def on_close(self):
        """
        Handle application shutdown.

        Stops all background processes and cleans up resources.
        Called when the application window is closed.
        """
        self.asr.stop_listening()
        self.question_detector.stop()
        self.response_manager.stop()


# Application entry point
if __name__ == "__main__":
    # Create and run the application
    app = CallAssistApp()
    app.run()
