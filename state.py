"""Durable state store: an atomic JSON-blob facade over a KeyValueSink.

Hermes-free. The whole plugin state lives in one blob under a single sink key
(``<prefix>.state``) so read-modify-write cycles are atomic under the lock and
partial-key visibility never leaks between components. A sink that raises is
treated as a corrupt/unavailable store: reads fail open to defaults and the
error is recorded in ``last_error``.

The hermes ``PluginState`` (profile-scoped, file-locked, ``get(key, default)``
/ ``set(key, value)``) satisfies the sink protocol directly.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional


class Store:
    def __init__(self, sink, prefix: str = "oom_breaker"):
        self._sink = sink
        self._key = f"{prefix}.state"
        self._lock = threading.RLock()
        self.last_error: str = ""

    def _read(self) -> Dict[str, Any]:
        try:
            blob = self._sink.get(self._key, None)
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"read: {exc!r}"
            return {}
        if not isinstance(blob, dict):
            self.last_error = f"read: blob is {type(blob).__name__}, expected object"
            return {}
        return blob

    def _write(self, blob: Dict[str, Any]) -> None:
        try:
            self._sink.set(self._key, blob)
        except Exception as exc:
            self.last_error = f"write: {exc!r}"

    def get(self, key: str, default: Any = None) -> Any:
        """Read one key from the blob; fails open to *default* on store errors."""
        with self._lock:
            return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Atomically set one key in the blob without dropping other keys."""
        with self._lock:
            blob = self._read()
            blob[key] = value
            self._write(blob)

    def update(self, transform: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Atomic read-modify-write of the whole blob; returns the new blob."""
        with self._lock:
            blob = self._read()
            new_blob = transform(blob)
            if not isinstance(new_blob, dict):
                self.last_error = f"update: transform returned {type(new_blob).__name__}"
                return self._read()
            self._write(new_blob)
            return new_blob

    def reset(self) -> None:
        with self._lock:
            self._write({})
