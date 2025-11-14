"""
Call Summary Generator

Generates human-readable summaries of completed calls using LLM analysis.
Creates both JSON data and HTML reports for call review and analysis.

Features:
- LLM-powered call summarization
- HTML report generation with browser display
- Question and answer extraction
- Transcript integration
- Minimal, readable styling

Author: Quinn Evans
"""

import json
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from llm import LLMManager


class CallSummaryGenerator:
    """
    Generates comprehensive summaries of completed calls.

    Uses LLM to create brief summaries of customer interest and compiles
    all call data into human-readable HTML reports.

    Attributes:
        llm (LLMManager): LLM interface for summary generation
        call_dir (str): Directory containing call data files
    """

    def __init__(self, llm: LLMManager):
        """
        Initialize the summary generator.

        Args:
            llm (LLMManager): LLM manager instance for summary generation
        """
        self.llm = llm
        self.call_dir = None

    def generate_summary(self, call_dir: str):
        """
        Generate complete call summary from call data files.

        Reads transcript, questions, and responses, generates LLM summary,
        and creates both JSON and HTML outputs.

        Args:
            call_dir (str): Directory containing call data files
        """
        self.call_dir = call_dir

        # Load call data
        transcript = self._load_transcript()
        questions = self._load_questions()
        responses = self._load_responses()

        if not transcript:
            print("Warning: No transcript data found")
            return

        # Generate LLM summary
        summary_text = self._generate_llm_summary(transcript)

        # Create summary data
        summary_data = {
            "summary_text": summary_text,
            "questions": questions,
            "answers": [r.get("answer", "") for r in responses],
            "transcript_file": "transcript.json",
            "generated_at": datetime.now().isoformat()
        }

        # Save JSON summary
        self._save_json_summary(summary_data)

        # Generate and save HTML report
        html_content = self._generate_html_report(summary_data, transcript, questions, responses)
        self._save_html_report(html_content)

        # Open HTML report in browser
        self._open_html_report()

    def _load_transcript(self) -> List[Dict[str, Any]]:
        """
        Load transcript data from transcript.json.

        Returns:
            List[Dict[str, Any]]: List of transcript segments
        """
        transcript_file = os.path.join(self.call_dir, "transcript.json")
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading transcript: {e}")
            return []

    def _load_questions(self) -> List[str]:
        """
        Load detected questions from detected_questions.json.

        Returns:
            List[str]: List of question texts
        """
        questions_file = os.path.join(self.call_dir, "detected_questions.json")
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [q.get("question", "") for q in data]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading questions: {e}")
            return []

    def _load_responses(self) -> List[Dict[str, Any]]:
        """
        Load responses from responses.json.

        Returns:
            List[Dict[str, Any]]: List of response data
        """
        responses_file = os.path.join(self.call_dir, "responses.json")
        try:
            with open(responses_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading responses: {e}")
            return []

    def _generate_llm_summary(self, transcript: List[Dict[str, Any]]) -> str:
        """
        Generate a brief LLM summary of the call.

        Args:
            transcript (List[Dict[str, Any]]): Transcript segments

        Returns:
            str: Brief summary text
        """
        if not transcript:
            return "No transcript available for summarization."

        # Combine transcript into text
        transcript_text = " ".join([segment.get("text", "") for segment in transcript])

        if not transcript_text.strip():
            return "Transcript contains no readable text."

        # Create summary prompt
        prompt = f"""Provide a brief 2-3 sentence summary of the customer's overall interest in solar based only on this transcript. Do not invent specifics or make assumptions beyond what's stated.

Transcript:
{transcript_text}

Summary:"""

        try:
            # Use LLM to generate summary
            summary, success = self.llm.generate_response(
                [{"question": "Summarize customer interest in solar", "answers": []}],
                [],
                prompt_override=prompt
            )
            return summary if success else "Unable to generate summary."
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return "Summary generation failed."

    def _save_json_summary(self, summary_data: Dict[str, Any]):
        """
        Save summary data to summary.json.

        Args:
            summary_data (Dict[str, Any]): Summary data to save
        """
        summary_file = os.path.join(self.call_dir, "summary.json")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving JSON summary: {e}")

    def _generate_html_report(self, summary_data: Dict[str, Any],
                            transcript: List[Dict[str, Any]],
                            questions: List[str],
                            responses: List[Dict[str, Any]]) -> str:
        """
        Generate HTML report content.

        Args:
            summary_data (Dict[str, Any]): Summary information
            transcript (List[Dict[str, Any]]): Transcript segments
            questions (List[str]): Detected questions
            responses (List[Dict[str, Any]]): Generated responses

        Returns:
            str: Complete HTML content
        """
        summary_text = summary_data.get("summary_text", "No summary available")

        # Build questions list
        questions_html = ""
        for i, question in enumerate(questions, 1):
            questions_html += f"<li>{question}</li>\n"

        # Build answers list
        answers_html = ""
        for i, response in enumerate(responses, 1):
            answer = response.get("answer", "")
            latency = response.get("llm_latency", 0)
            answers_html += f"<li>{answer} <small>(generated in {latency:.1f}s)</small></li>\n"

        # Build transcript
        transcript_html = ""
        for segment in transcript:
            timestamp = segment.get("timestamp", "")
            text = segment.get("text", "")
            transcript_html += f"<div class='transcript-segment'><strong>{timestamp}:</strong> {text}</div>\n"

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Summary - CallAssist</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 1.1em;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
        }}
        .transcript-segment {{
            background: #f8f9fa;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            border-left: 3px solid #bdc3c7;
        }}
        small {{
            color: #7f8c8d;
        }}
        .stats {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Call Summary - CallAssist</h1>

        <div class="summary">
            <h2>Customer Interest Summary</h2>
            <p>{summary_text}</p>
        </div>

        <div class="stats">
            <h2>Call Statistics</h2>
            <p><strong>Questions Detected:</strong> {len(questions)}</p>
            <p><strong>Responses Generated:</strong> {len(responses)}</p>
            <p><strong>Transcript Segments:</strong> {len(transcript)}</p>
        </div>

        <h2>Questions Asked</h2>
        <ul>
            {questions_html}
        </ul>

        <h2>AI Responses</h2>
        <ul>
            {answers_html}
        </ul>

        <h2>Full Transcript</h2>
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
            {transcript_html}
        </div>
    </div>
</body>
</html>"""

        return html

    def _save_html_report(self, html_content: str):
        """
        Save HTML report to summary.html.

        Args:
            html_content (str): Complete HTML content
        """
        html_file = os.path.join(self.call_dir, "summary.html")
        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
            print(f"Error saving HTML report: {e}")

    def _open_html_report(self):
        """
        Open the HTML report in the default web browser.
        """
        html_file = os.path.join(self.call_dir, "summary.html")
        if os.path.exists(html_file):
            try:
                webbrowser.open(f"file://{os.path.abspath(html_file)}")
            except Exception as e:
                print(f"Error opening HTML report in browser: {e}")