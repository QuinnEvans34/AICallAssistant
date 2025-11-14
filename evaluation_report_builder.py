"""
Call Evaluation Report Builder

Generates comprehensive performance statistics and evaluation metrics
for completed calls. Analyzes transcript quality, response times, and
system performance.

Features:
- Performance metrics calculation
- Statistical analysis of call data
- JSON report generation
- Thread-safe data collection

Author: Quinn Evans
"""

import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class EvaluationReportBuilder:
    """
    Builds comprehensive evaluation reports for call analysis.

    Collects performance metrics throughout a call and generates
    detailed statistics about ASR quality, response times, and system behavior.

    Attributes:
        call_dir (str): Directory for saving the report
        start_time (float): Call start timestamp
        end_time (float): Call end timestamp
        metrics (Dict): Collected performance metrics
    """

    def __init__(self):
        """Initialize the evaluation report builder."""
        self.call_dir = None
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def start_call(self, call_dir: str):
        """
        Initialize evaluation tracking for a new call.

        Args:
            call_dir (str): Directory path for this call's logs
        """
        self.call_dir = call_dir
        self.start_time = time.time()
        self.end_time = None
        self.metrics = {
            'transcript_segments': 0,
            'words_transcribed': 0,
            'questions_detected': 0,
            'responses_generated': 0,
            'llm_latencies': [],
            'clarification_events': 0,
            'asr_dropouts': 0,
            'detector_backpressure': 0
        }

    def end_call(self):
        """
        Mark the call as ended and calculate final metrics.

        Sets the end time and prepares for report generation.
        """
        self.end_time = time.time()

    def set_transcript_stats(self, segment_count: int, word_count: int):
        """
        Update transcript statistics.

        Args:
            segment_count (int): Number of ASR segments
            word_count (int): Total words transcribed
        """
        self.metrics['transcript_segments'] = segment_count
        self.metrics['words_transcribed'] = word_count

    def set_question_stats(self, question_count: int):
        """
        Update question detection statistics.

        Args:
            question_count (int): Number of questions detected
        """
        self.metrics['questions_detected'] = question_count

    def set_response_stats(self, response_count: int, latencies: List[float]):
        """
        Update response generation statistics.

        Args:
            response_count (int): Number of responses generated
            latencies (List[float]): List of LLM response latencies in seconds
        """
        self.metrics['responses_generated'] = response_count
        self.metrics['llm_latencies'] = latencies

    def increment_clarifications(self):
        """Increment the clarification events counter."""
        self.metrics['clarification_events'] += 1

    def increment_asr_dropouts(self):
        """Increment the ASR dropout counter."""
        self.metrics['asr_dropouts'] += 1

    def increment_backpressure(self):
        """Increment the detector backpressure counter."""
        self.metrics['detector_backpressure'] += 1

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate the complete evaluation report.

        Calculates all statistics and returns the evaluation data.

        Returns:
            Dict[str, Any]: Complete evaluation report data
        """
        if not self.start_time or not self.end_time:
            return {}

        call_duration = self.end_time - self.start_time
        latencies = self.metrics['llm_latencies']

        report = {
            "total_transcript_segments": self.metrics['transcript_segments'],
            "total_words_transcribed": self.metrics['words_transcribed'],
            "total_questions_detected": self.metrics['questions_detected'],
            "total_responses_generated": self.metrics['responses_generated'],
            "avg_llm_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
            "max_llm_latency_seconds": max(latencies) if latencies else 0.0,
            "clarification_events": self.metrics['clarification_events'],
            "asr_dropouts_detected": self.metrics['asr_dropouts'],
            "detector_backpressure_events": self.metrics['detector_backpressure'],
            "call_duration_seconds": call_duration,
            "call_start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "call_end_time": datetime.fromtimestamp(self.end_time).isoformat()
        }

        return report

    def save_report(self):
        """
        Save the evaluation report to evaluation_report.json.

        Generates and writes the complete evaluation report to the call directory.
        """
        if not self.call_dir:
            return

        report = self.generate_report()
        if not report:
            return

        report_file = os.path.join(self.call_dir, "evaluation_report.json")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving evaluation report: {e}")

    def get_call_duration(self) -> float:
        """
        Get the total call duration in seconds.

        Returns:
            float: Call duration, or 0.0 if call not ended
        """
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time