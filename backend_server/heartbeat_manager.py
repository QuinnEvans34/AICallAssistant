from __future__ import annotations

import threading
import time
from typing import Dict


class HeartbeatManager:
    """
    Tracks rolling ASR/status metrics for WebSocket broadcasting and log parity.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._status = {
            "asr_latency_ms": 0.0,
            "queue_depth": 0,
            "errors": 0,
            "processing": False,
            "updated_at": time.time(),
        }

    def update(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)
            self._status["updated_at"] = time.time()

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._status)
