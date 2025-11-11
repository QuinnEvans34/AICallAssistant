import threading
import tkinter as tk
from tkinter import ttk


class StreamUI:
    def __init__(self, toggle_callback, manual_callback, name_callback):
        self.root = tk.Tk()
        self.root.title("CallAssist Streaming")
        self.root.geometry("620x420")

        self.toggle_callback = toggle_callback
        self.manual_callback = manual_callback
        self.name_callback = name_callback

        # Status label
        self.status_label = ttk.Label(self.root, text="Status: Off", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Caller name input
        name_frame = ttk.Frame(self.root)
        name_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(name_frame, text="Caller Name:").pack(side=tk.LEFT)
        self.caller_name_var = tk.StringVar()
        self.caller_name_entry = ttk.Entry(name_frame, textvariable=self.caller_name_var, width=25)
        self.caller_name_entry.pack(side=tk.LEFT, padx=5)
        self.caller_name_entry.bind("<Return>", self._save_caller_name)
        self.name_button = ttk.Button(name_frame, text="Save", command=self._save_caller_name)
        self.name_button.pack(side=tk.LEFT)

        # Toggle button
        self.toggle_button = ttk.Button(self.root, text="Start Live", command=self._toggle)
        self.toggle_button.pack(pady=10)

        # Response display
        ttk.Label(self.root, text="Responses:").pack()
        response_frame = ttk.Frame(self.root)
        response_frame.pack(fill=tk.X, padx=10, pady=5)
        self.response_text = tk.Text(response_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(response_frame, orient=tk.VERTICAL, command=self.response_text.yview)
        self.response_text.config(yscrollcommand=scrollbar.set)
        self.response_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Manual input
        ttk.Label(self.root, text="Manual Question:").pack()
        self.manual_entry = ttk.Entry(self.root)
        self.manual_entry.pack(fill=tk.X, padx=10, pady=5)
        self.manual_entry.bind("<Return>", self._manual_submit)

        self.manual_button = ttk.Button(self.root, text="Submit", command=self._manual_submit)
        self.manual_button.pack(pady=5)

        self.manual_entry.focus()
        self.root.focus_force()
        self.response_blocks = {}

    def _toggle(self):
        self.toggle_callback()

    def _manual_submit(self, event=None):
        question = self.manual_entry.get().strip()
        if question:
            self.manual_callback(question)
            self.manual_entry.delete(0, tk.END)

    def update_status(self, status):
        def _update():
            self.status_label.config(text=f"Status: {status}")
            self.toggle_button.config(text="Stop Live" if status == "Live" else "Start Live")

        self._thread_safe_call(_update)

    def update_response(self, response):
        def _update():
            if isinstance(response, dict):
                formatted = (response.get("text") or "").strip()
                block_id = response.get("block_id")
                replace = response.get("replace", False)
            else:
                formatted = (response or "").strip()
                block_id = None
                replace = False

            if not formatted:
                return

            lower = formatted.lower()
            if (
                "\n" not in formatted
                and not lower.startswith("response:")
                and not lower.startswith("questions:")
                and not lower.startswith("question:")
            ):
                formatted = f"Response: {formatted}"

            if block_id and replace:
                self._delete_block(block_id)

            block_text = formatted + "\n" + ("-" * 60) + "\n"
            self.response_text.insert(tk.END, block_text)
            self.response_text.see(tk.END)

            if block_id:
                end_index = self.response_text.index(tk.END)
                start_index = self.response_text.index(f"{end_index}-{len(block_text)}c")
                tag = f"block_{block_id}"
                self.response_text.tag_add(tag, start_index, end_index)
                self.response_blocks[block_id] = tag

        self._thread_safe_call(_update)

    def _delete_block(self, block_id):
        tag = self.response_blocks.get(block_id)
        if not tag:
            return
        ranges = self.response_text.tag_ranges(tag)
        if ranges:
            self.response_text.delete(ranges[0], ranges[1])
        self.response_text.tag_delete(tag)
        del self.response_blocks[block_id]

    def _save_caller_name(self, event=None):
        if self.name_callback:
            self.name_callback(self.caller_name_var.get().strip())

    def set_caller_name(self, name):
        self.caller_name_var.set(name or "")

    def _thread_safe_call(self, func):
        if threading.current_thread() is threading.main_thread():
            func()
        else:
            self.root.after(0, func)

    def run(self):
        self.root.mainloop()
