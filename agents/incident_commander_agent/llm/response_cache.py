"""
File: llm/response_cache.py
Purpose: In-memory TTL cache for LLM responses.
Dependencies: Standard library only.
Performance: O(1) get/put, <1ms cache hit.

Caches LLM-generated responses keyed on a SHA-256 hash of
the input, avoiding redundant LLM calls for identical input.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

from agents.incident_commander_agent.telemetry import get_logger

logger = get_logger("incident_commander_agent.llm.response_cache")


class ResponseCache:
    """In-memory TTL cache for LLM responses.

    Args:
        ttl_seconds: Time-to-live for cache entries.
        max_entries: Maximum cache size.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._store: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if exists and not expired.

        Args:
            key: Cache key (SHA-256 hash).

        Returns:
            Cached response dict, or None.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            stored_at, result = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._store[key]
                return None
            return result

    def put(self, key: str, result: Dict[str, Any]) -> None:
        """Store a response in the cache.

        Args:
            key: Cache key (SHA-256 hash).
            result: Response dict to cache.
        """
        with self._lock:
            if (
                len(self._store) >= self._max_entries
                and key not in self._store
            ):
                oldest_key = min(
                    self._store, key=lambda k: self._store[k][0]
                )
                del self._store[oldest_key]
            self._store[key] = (time.monotonic(), result)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._store)

    @staticmethod
    def compute_key(signals: Dict[str, Any]) -> str:
        """Compute a deterministic cache key from signals.

        Args:
            signals: Input signals dict.

        Returns:
            SHA-256 hex digest.
        """
        payload = json.dumps(signals, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
