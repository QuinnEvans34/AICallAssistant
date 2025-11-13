"""
Flet-based User Interface for CallAssist

Provides a modern, responsive graphical interface for the CallAssist application
using the Flet framework. Handles real-time display of conversation messages,
user controls, and visual feedback for the speech recognition system.

Key Features:
- Real-time message display with chronological ordering
- Live audio status indicators (speaker detection, confidence levels)
- Manual question input and editing capabilities
- Thread-safe UI updates for background processing
- Scrollable conversation history
- Question editing with inline text fields

Architecture:
- Event-driven design with callback functions
- Thread-safe communication with background ASR/LLM processes
- Message storage with timestamps for proper ordering
- State management for UI interactions (editing, scrolling)

Dependencies:
- flet: Modern Python UI framework
- asyncio: For async UI operations
- threading: For thread-safe UI updates
- time: For message timestamping

Author: Quinn Evans
"""

import asyncio
import threading
import time

import flet as ft


class FletUI:
    """
    Main user interface class for CallAssist using Flet framework.

    Manages the graphical interface, handles user interactions, and provides
    real-time visual feedback for the speech recognition and response system.

    The UI consists of:
    - Status indicators (live/off, speaker detection, confidence)
    - Caller name input field
    - Scrollable conversation display area
    - Manual question input controls
    - Question editing capabilities

    Attributes:
        toggle_callback (callable): Function to toggle live audio processing
        manual_callback (callable): Function to submit manual questions
        name_callback (callable): Function to update caller name
        question_edit_callback (callable): Function to handle question edits
        page (ft.Page): Flet page object for UI management
        messages (list): Chronologically ordered message storage
        response_blocks (dict): Mapping of block IDs to UI controls
        editing_block (ft.Container): Currently edited message container
    """

    def __init__(self, toggle_callback, manual_callback, name_callback, question_edit_callback=None):
        """
        Initialize the Flet UI with callback functions.

        Sets up the UI component with all necessary callback functions for
        communication with the main application logic.

        Args:
            toggle_callback (callable): Called when user toggles live mode
            manual_callback (callable): Called when user submits manual question
            name_callback (callable): Called when caller name is updated
            question_edit_callback (callable, optional): Called when question is edited
        """
        # Callback functions for external communication
        self.toggle_callback = toggle_callback
        self.manual_callback = manual_callback
        self.name_callback = name_callback
        self.question_edit_callback = question_edit_callback

        # UI component references
        self.page = None
        self.status_text = None
        self.toggle_button = None
        self.name_entry = None
        self.response_list = None
        self.manual_button = None
        self.name_save_button = None
        self.speaker_indicator = None
        self.confidence_bar = None

        # State management
        self.editing_block = None  # Track which message block is being edited
        self._ui_thread_id = None  # Thread ID for thread-safe operations
        self.response_blocks = {}  # Map block IDs to UI controls for updates
        self.messages = []  # Store messages with timestamps for chronological ordering

    # ------------------------------------------------------------------- UI Setup

    def run(self, page: ft.Page):
        """
        Initialize and display the main UI layout.

        Sets up the Flet page with all UI components arranged in a responsive
        layout. Configures window properties, theme, and event handlers.

        Args:
            page (ft.Page): Flet page object provided by the framework
        """
        self.page = page
        self._ui_thread_id = threading.get_ident()

        # Configure page properties
        page.title = "CallAssist Streaming"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window_width = 960
        page.window_height = 720
        page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        page.padding = 20

        # Create status and control components
        self.status_text = ft.Text("Status: Off", size=18, weight=ft.FontWeight.BOLD)
        self.toggle_button = ft.FilledButton("Start Live", on_click=lambda _: self.toggle_callback())

        # Speaker detection indicator (red = customer speaking, gray = waiting)
        self.speaker_indicator = ft.Container(width=20, height=20, bgcolor=ft.Colors.GREY, border_radius=10)

        # Confidence level progress bar
        self.confidence_bar = ft.ProgressBar(value=0, width=200)

        # Caller name input field with save button
        self.name_entry = ft.TextField(label="Caller Name", expand=1, on_submit=self._on_name_submit)
        self.name_save_button = ft.IconButton(icon="check_circle_outline", tooltip="Save name", on_click=self._on_name_submit)

        # Scrollable conversation display area
        self.response_list = ft.ListView(expand=1, spacing=8)
        response_container = ft.Container(
            expand=1,
            bgcolor=ft.Colors.BLUE_50,
            padding=15,
            border_radius=8,
            content=self.response_list,
        )

        # Manual question input controls
        self.manual_entry = ft.TextField(label="Manual Question", expand=1, multiline=True, min_lines=1, max_lines=3, on_submit=self._manual_submit)
        self.manual_button = ft.ElevatedButton("Submit", on_click=self._manual_submit)

        # Layout arrangement
        header = ft.Row(
            controls=[self.status_text, self.toggle_button, self.speaker_indicator, self.confidence_bar],
            wrap=True,
            spacing=12,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        name_row = ft.Row(
            controls=[self.name_entry, self.name_save_button],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )

        manual_row = ft.Row(
            controls=[self.manual_entry, self.manual_button],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Main layout structure
        page.add(
            ft.Column(
                spacing=16,
                expand=True,
                controls=[
                    header,                    # Status and controls
                    name_row,                  # Caller name input
                    ft.Text("Responses:", size=16, weight=ft.FontWeight.BOLD),  # Section header
                    response_container,        # Conversation display
                    manual_row,               # Manual input controls
                ],
            )
        )

    # ------------------------------------------------------------------- Event Handlers

    def _on_name_submit(self, _):
        """
        Handle caller name submission from UI input.

        Extracts the name value and calls the name callback function.
        Triggered by Enter key or save button click.

        Args:
            _ : Event object (unused)
        """
        value = (self.name_entry.value or "").strip()
        self.name_callback(value)

    def set_caller_name(self, name):
        """
        Update the caller name display in the UI.

        Sets the text field value and refreshes the display.

        Args:
            name (str): New caller name to display
        """
        self.name_entry.value = name or ""
        self.page.update()

    def _manual_submit(self, _=None):
        """
        Handle manual question submission.

        Extracts the question text, calls the manual callback,
        and clears the input field.

        Args:
            _ : Event object (unused, defaults to None)
        """
        question = (self.manual_entry.value or "").strip()
        if question:
            self.manual_callback(question)
            self.manual_entry.value = ""
            self.page.update()

    # ------------------------------------------------------------------- Status Updates

    def update_status(self, status):
        """
        Update the live/off status display and button text.

        Thread-safe method that updates the status text and toggle button
        based on the current operational mode.

        Args:
            status (str): Current status ("Live" or "Off")
        """
        def _update():
            self.status_text.value = f"Status: {status}"
            self.toggle_button.text = "Stop Live" if status == "Live" else "Start Live"
            self.page.update()

        self._thread_safe_call(_update)

    def update_speaker_status(self, state):
        """
        Update the speaker detection indicator.

        Changes the color of the speaker indicator based on who is speaking:
        - Red: Customer is speaking
        - Gray: Waiting/listening

        Args:
            state (str): Speaker state ("customer" or "waiting")
        """
        def _update():
            color = ft.Colors.RED if state == "customer" else ft.Colors.GREY
            self.speaker_indicator.bgcolor = color
            self.page.update()

        self._thread_safe_call(_update)

    def update_confidence(self, confidence):
        """
        Update the confidence level progress bar.

        Displays the system's confidence in current processing results.
        Value is clamped between 0.0 and 1.0.

        Args:
            confidence (float): Confidence level (0.0 to 1.0)
        """
        confidence = max(0.0, min(1.0, float(confidence)))

        def _update():
            self.confidence_bar.value = confidence
            self.page.update()

        self._thread_safe_call(_update)

    def update_response(self, payload):
        """
        Update the conversation display with new or modified messages.

        Handles adding new messages, updating existing ones, and maintaining
        chronological order. Supports different message types and draft states.

        Args:
            payload (dict): Message data containing:
                - text (str): Message content
                - block_id (str, optional): Unique identifier for updates
                - replace (bool): Whether to replace existing message
                - message_type (str): Type of message ("response", "clarification")
                - is_draft (bool): Whether message is in draft state
                - timestamp (float): Message timestamp for ordering
        """
        if not isinstance(payload, dict):
            payload = {"text": str(payload) if payload is not None else ""}

        text = payload.get("text", "")
        if not text:
            return

        block_id = payload.get("block_id")
        replace = payload.get("replace", False)
        message_type = payload.get("message_type", "response")
        is_draft = payload.get("is_draft", False)
        timestamp = payload.get("timestamp", time.time())

        def _update():
            # Store message with metadata for chronological ordering
            message_data = {
                "text": text,
                "block_id": block_id,
                "replace": replace,
                "message_type": message_type,
                "is_draft": is_draft,
                "timestamp": timestamp
            }

            if replace and block_id:
                # Update existing message in storage
                for i, msg in enumerate(self.messages):
                    if msg["block_id"] == block_id:
                        self.messages[i] = message_data
                        break
            else:
                # Add new message to storage
                self.messages.append(message_data)

            # Sort messages chronologically by timestamp
            self.messages.sort(key=lambda x: x["timestamp"])

            # Rebuild the UI controls in correct order
            self.response_list.controls.clear()
            self.response_blocks.clear()

            for msg in self.messages:
                control = self._build_message_control(
                    msg["text"],
                    msg["message_type"],
                    msg["is_draft"]
                )
                if msg["block_id"]:
                    self.response_blocks[msg["block_id"]] = control
                self.response_list.controls.append(control)

            self.response_list.update()
            self.page.update()

        self._thread_safe_call(_update)

    def update_response_partial(self, text, block_id):
        """
        Update a streaming response with partial text.

        Used for real-time display of LLM responses as they are generated.
        Creates a payload for partial updates with current timestamp.

        Args:
            text (str): Partial response text
            block_id (str): Identifier of the message block to update
        """
        if not block_id:
            return
        payload = {
            "text": text,
            "block_id": block_id,
            "replace": True,
            "message_type": "response",
            "timestamp": time.time(),
        }
        self.update_response(payload)

    # ------------------------------------------------------------------- Message Building

    def _build_message_control(self, text, message_type, is_draft):
        """
        Create a UI control for displaying a message.

        Builds appropriate visual styling and interactive elements based on
        message type and state. Handles special cases like editable questions.

        Args:
            text (str): Message text content
            message_type (str): Type of message ("response", "clarification")
            is_draft (bool): Whether message is in draft/preliminary state

        Returns:
            ft.Container: Styled container with message content
        """
        # Determine visual styling based on message type and state
        if message_type == "clarification":
            color = ft.Colors.BLUE_GREY_700
            italic = True
            bgcolor = ft.Colors.GREY_200
        elif is_draft:
            color = ft.Colors.GREY_700
            italic = True
            bgcolor = ft.Colors.GREY_100
        else:
            color = ft.Colors.BLACK
            italic = False
            bgcolor = ft.Colors.WHITE

        # Special handling for editable question messages
        if text.startswith("Question") and message_type == "response" and not is_draft:
            # Create editable question with edit button
            question_text = text
            txt = ft.Text(question_text, selectable=True, size=16, color=color, italic=italic)
            edit_btn = ft.IconButton(
                icon=ft.Icons.EDIT,
                tooltip="Edit question",
                on_click=lambda e: self._start_edit_question(e.control.parent, question_text)
            )
            content = ft.Row([txt, edit_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        else:
            # Standard text display
            content = ft.Text(text, selectable=True, size=16, color=color, italic=italic)

        return ft.Container(content=content, bgcolor=bgcolor, padding=ft.padding.all(10), border_radius=8)

    def _start_edit_question(self, container, original_text):
        """
        Initiate inline editing of a question message.

        Switches the message display to an editable text field with save/cancel controls.
        Only one message can be edited at a time.

        Args:
            container (ft.Container): The message container to edit
            original_text (str): Original question text
        """
        if self.editing_block:
            return  # Already editing another message

        self.editing_block = container

        # Extract the question content (remove prefixes)
        if original_text.startswith("Question(s): "):
            question_part = original_text[len("Question(s): "):]
        elif original_text.startswith("Question: "):
            question_part = original_text[len("Question: "):]
        else:
            question_part = original_text

        # Create editing controls
        edit_field = ft.TextField(value=question_part, expand=True, multiline=True, min_lines=1, max_lines=3)
        save_btn = ft.IconButton(
            icon=ft.Icons.CHECK,
            tooltip="Save",
            on_click=lambda e: self._save_edit_question(container, edit_field.value, original_text)
        )
        cancel_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            tooltip="Cancel",
            on_click=lambda e: self._cancel_edit_question(container, original_text)
        )

        container.content = ft.Row([edit_field, save_btn, cancel_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        self.page.update()

    def _save_edit_question(self, container, new_question, original_text):
        """
        Save the edited question and update the display.

        Validates the new question text, updates the message, and notifies
        the callback function for further processing.

        Args:
            container (ft.Container): Message container being edited
            new_question (str): New question text from user input
            original_text (str): Original question text for formatting reference
        """
        if not new_question.strip():
            self._cancel_edit_question(container, original_text)
            return

        self.editing_block = None

        # Reconstruct the full question text with proper formatting
        if original_text.startswith("Question(s): "):
            updated_text = f"Question(s): {new_question.strip()}"
        elif original_text.startswith("Question: "):
            updated_text = f"Question: {new_question.strip()}"
        else:
            updated_text = new_question.strip()

        # Restore display controls with edit button
        txt = ft.Text(updated_text, selectable=True, size=16, color=ft.Colors.BLACK, italic=False)
        edit_btn = ft.IconButton(
            icon=ft.Icons.EDIT,
            tooltip="Edit question",
            on_click=lambda e: self._start_edit_question(container, updated_text)
        )
        container.content = ft.Row([txt, edit_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        # Notify external callback for processing
        if self.question_edit_callback:
            self.question_edit_callback(new_question.strip())

        self.page.update()

    def _cancel_edit_question(self, container, original_text):
        """
        Cancel question editing and restore original display.

        Reverts the message container to its original state without changes.

        Args:
            container (ft.Container): Message container being edited
            original_text (str): Original question text to restore
        """
        self.editing_block = None

        # Restore original display with edit button
        txt = ft.Text(original_text, selectable=True, size=16, color=ft.Colors.BLACK, italic=False)
        edit_btn = ft.IconButton(
            icon=ft.Icons.EDIT,
            tooltip="Edit question",
            on_click=lambda e: self._start_edit_question(container, original_text)
        )
        container.content = ft.Row([txt, edit_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        self.page.update()

    # ------------------------------------------------------------------- Thread Safety

    def _thread_safe_call(self, func):
        """
        Execute a UI update function in a thread-safe manner.

        Ensures UI updates are performed on the correct thread to avoid
        Flet framework violations. Uses Flet's built-in thread communication.

        Args:
            func (callable): Function to execute safely
        """
        if threading.get_ident() == self._ui_thread_id:
            func()
            return
        try:
            self.page.call_from_thread(func)
        except AttributeError:
            asyncio.run(self._async_wrapper(func))

    async def typing_stream(self, text, control, delay=0.02):
        """
        Animate text appearance with typing effect.

        Displays text character by character with a delay, creating a
        "typing" animation effect. Used for live transcription display.

        Args:
            text (str): Text to animate
            control: UI control to update with text
            delay (float): Delay between characters in seconds
        """
        displayed = ""
        for ch in text:
            displayed += ch
            control.value = displayed
            self.page.update()
            await asyncio.sleep(delay)

    async def _async_wrapper(self, func):
        """
        Wrapper for async execution of UI functions.

        Provides compatibility for async UI operations when needed.

        Args:
            func (callable): Function to execute asynchronously
        """
        func()
