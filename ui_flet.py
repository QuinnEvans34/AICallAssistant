import flet as ft
import threading


class FletUI:
    def __init__(self, toggle_callback, manual_callback, name_callback):
        self.toggle_callback = toggle_callback
        self.manual_callback = manual_callback
        self.name_callback = name_callback
        self.page = None
        self.status_text = None
        self.toggle_button = None
        self.name_entry = None
        self.response_list = None
        self.manual_button = None
        self.name_save_button = None
        self._ui_thread_id = None
        self.response_blocks = {}
        self.response_blocks = {}

    def run(self, page: ft.Page):
        self.page = page
        self._ui_thread_id = threading.get_ident()
        page.title = "CallAssist Streaming"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window_width = 960
        page.window_height = 700
        page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        page.padding = 20
        page.scroll = ft.ScrollMode.AUTO

        self.status_text = ft.Text("Status: Off", size=18, weight=ft.FontWeight.BOLD)
        self.toggle_button = ft.FilledButton("Start Live", on_click=lambda e: self.toggle_callback())

        self.name_entry = ft.TextField(
            label="Caller Name",
            expand=1,
            on_submit=self._on_name_submit,
        )
        self.name_save_button = ft.IconButton(
            icon="check_circle_outline",
            tooltip="Save caller name",
            on_click=self._on_name_submit,
        )
        self.set_caller_name("")

        self.response_list = ft.Column(scroll=ft.ScrollMode.AUTO)
        response_container = ft.Container(
            expand=1,
            bgcolor=ft.Colors.BLUE_50,
            padding=15,
            border_radius=8,
            content=self.response_list,
        )

        self.manual_entry = ft.TextField(
            label="Manual Question",
            expand=1,
            multiline=True,
            min_lines=1,
            max_lines=3,
            on_submit=self._manual_submit,
        )
        self.manual_button = ft.ElevatedButton("Submit", on_click=self._manual_submit)

        header = ft.Row(
            controls=[self.status_text, self.toggle_button],
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

        layout = ft.Column(
            spacing=16,
            expand=True,
            controls=[
                header,
                name_row,
                ft.Text("Responses:", size=16, weight=ft.FontWeight.BOLD),
                response_container,
                manual_row,
            ],
        )

        page.add(layout)

    def _on_name_submit(self, e):
        if not self.name_entry:
            return
        value = (self.name_entry.value or "").strip()
        self.name_callback(value)

    def set_caller_name(self, name):
        if not self.name_entry:
            return
        self.name_entry.value = name or ""
        if self.page:
            self.page.update()

    def _manual_submit(self, e=None):
        if not self.manual_entry:
            return
        question = (self.manual_entry.value or "").strip()
        if question:
            self.manual_callback(question)
            self.manual_entry.value = ""
            self.page.update()

    def update_status(self, status):
        if not self.status_text:
            return

        def _update():
            self.status_text.value = f"Status: {status}"
            self.toggle_button.text = "Stop Live" if status == "Live" else "Start Live"
            self.page.update()

        self._thread_safe_call(_update)

    def update_response(self, payload):
        if not self.response_list:
            return
        if not isinstance(payload, dict):
            payload = {"response": str(payload) if payload is not None else ""}

        raw_response = payload.get("response") or payload.get("text", "")
        text = raw_response
        if not text:
            return

        block_id = payload.get("block_id")

        def _update():
            if block_id:
                if block_id in self.response_blocks:
                    self.response_blocks[block_id].value = text
                else:
                    control = ft.Text(text, selectable=True, size=16, color=ft.Colors.BLACK)
                    self.response_blocks[block_id] = control
                    self.response_list.controls.append(control)
            else:
                self.response_list.controls.append(
                    ft.Text(text, selectable=True, size=16, color=ft.Colors.BLACK)
                )
            self.response_list.update()
            self.page.update()

        self._thread_safe_call(_update)

    def _thread_safe_call(self, func):
        func()

    def _call_from_thread(self, func):
        func()
