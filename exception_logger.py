"""
Exception Logger for CallAssist

Provides centralized error logging and exception handling for all
CallAssist components. Logs errors with timestamps, modules, and stack traces.

Features:
- Thread-safe error logging
- Stack trace capture
- Module-specific error categorization
- Automatic log file management
- Exception context preservation

Author: Quinn Evans
"""

import os
import sys
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class ExceptionLogger:
    """
    Centralized exception logging for CallAssist system.

    Captures and logs exceptions from all system components with
    timestamps, module names, and full stack traces.

    Attributes:
        log_file (str): Path to the current log file
        lock (threading.Lock): Thread-safe file access
    """

    def __init__(self):
        """Initialize the exception logger."""
        self.log_file = None
        self.lock = threading.Lock()

    def set_log_file(self, log_file_path: str):
        """
        Set the log file path for error logging.

        Args:
            log_file_path (str): Full path to the error log file
        """
        self.log_file = log_file_path

        # Ensure directory exists
        log_dir = os.path.dirname(log_file_path)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def log_exception(self, exception: Exception, module: str = "unknown",
                     context: Optional[str] = None):
        """
        Log an exception with full details.

        Args:
            exception (Exception): The exception that occurred
            module (str): Name of the module where exception occurred
            context (str, optional): Additional context information
        """
        if not self.log_file:
            print(f"Exception in {module}: {exception}")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = str(exception)

        # Format the log entry
        log_entry = f"\n[{timestamp}] [{module.upper()}] {error_message}\n"

        if context:
            log_entry += f"Context: {context}\n"

        # Add stack trace
        log_entry += "Stack Trace:\n"
        log_entry += "".join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))
        log_entry += "\n" + "="*80 + "\n"

        # Write to file thread-safely
        with self.lock:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            except Exception as write_error:
                print(f"Failed to write to error log: {write_error}")
                print(f"Original error: {log_entry}")

    def log_error(self, error_message: str, module: str = "unknown",
                  context: Optional[str] = None):
        """
        Log a custom error message without an exception object.

        Args:
            error_message (str): The error message to log
            module (str): Name of the module where error occurred
            context (str, optional): Additional context information
        """
        if not self.log_file:
            print(f"Error in {module}: {error_message}")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"[{timestamp}] [{module.upper()}] {error_message}"
        if context:
            log_entry += f" | Context: {context}"
        log_entry += "\n"

        with self.lock:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            except Exception as write_error:
                print(f"Failed to write to error log: {write_error}")
                print(f"Original error: {log_entry}")


# Global exception logger instance
exception_logger = ExceptionLogger()


def log_exception(exc: Exception, module: str = "unknown", context: Optional[str] = None):
    """
    Convenience function to log exceptions using the global logger.

    Args:
        exc (Exception): Exception to log
        module (str): Module name
        context (str, optional): Additional context
    """
    exception_logger.log_exception(exc, module, context)


def log_error(message: str, module: str = "unknown", context: Optional[str] = None):
    """
    Convenience function to log error messages using the global logger.

    Args:
        message (str): Error message
        module (str): Module name
        context (str, optional): Additional context
    """
    exception_logger.log_error(message, module, context)