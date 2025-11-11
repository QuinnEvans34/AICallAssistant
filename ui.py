import tkinter as tk
from tkinter import ttk


class CallAssistUI:
    def __init__(self, toggle_callback, manual_callback):
        self.root = tk.Tk()
        self.root.title("CallAssist")
        self.root.geometry("600x400")

        self.toggle_callback = toggle_callback
        self.manual_callback = manual_callback

        # Status label
        self.status_label = ttk.Label(self.root, text="Status: Off", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Toggle button
        self.toggle_button = ttk.Button(self.root, text="Start Live", command=self._toggle)
        self.toggle_button.pack(pady=10)

        # Transcript display
        ttk.Label(self.root, text="Transcript:").pack()
        self.transcript_text = tk.Text(self.root, height=5, wrap=tk.WORD)
        self.transcript_text.pack(fill=tk.X, padx=10, pady=5)

        # Response display
        ttk.Label(self.root, text="Response:").pack()
        self.response_text = tk.Text(self.root, height=5, wrap=tk.WORD)
        self.response_text.pack(fill=tk.X, padx=10, pady=5)

        # Manual input
        ttk.Label(self.root, text="Manual Question:").pack()
        self.manual_entry = ttk.Entry(self.root)
        self.manual_entry.pack(fill=tk.X, padx=10, pady=5)
        self.manual_entry.bind("<Return>", self._manual_submit)

        self.manual_button = ttk.Button(self.root, text="Submit", command=self._manual_submit)
        self.manual_button.pack(pady=5)

        self.manual_entry.focus()
        self.root.focus_force()

    def _toggle(self):
        self.toggle_callback()

    def _manual_submit(self, event=None):
        question = self.manual_entry.get().strip()
        if question:
            self.manual_callback(question)
            self.manual_entry.delete(0, tk.END)

    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")
        self.toggle_button.config(text="Stop Live" if status == "Live" else "Start Live")

    def update_transcript(self, text):
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(tk.END, text)

    def update_response(self, text):
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, text)

    def prompt_manual_input(self):
        self.update_response("Didn't catch that. Type your question manually.")

    def run(self):
        self.root.mainloop()
