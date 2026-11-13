"""In-flight ledger and exact fit-gate arithmetic.

The ledger is the plugin's admission accounting: every gate-admitted brain
request is tracked from admission until its response usage (or error) arrives.
``in_flight`` is the sum of the seeded ``prompt + max_tokens`` upper bounds, so
the gate's arithmetic is exact by construction — no estimation, no margin.

Hermes-free: the wall clock is injected; the hermes adapter feeds it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from urllib.parse import urlsplit


@dataclass
class _Entry:
    request_id: str
    session_id: str
    target: str
    needed: int
    admitted_at: float
    streaming: bool = False


def target_of(base_url: Optional[str], model: Optional[str],
              targets: Dict[str, "object"]) -> Optional[str]:
    """First target whose ``base_urls`` AND ``model`` match the call.

    URL match is on scheme + host + port (path ignored). Missing/invalid
    base_url → None (the gate fails open: untracked calls pass through).
    ``model`` is a real match key: a target with ``model="x"`` matches only
    requests carrying that exact model; a target with ``model=""`` (absent)
    matches any model, including requests that carry no model at all
    (``None``/empty are normalized to ``""``, so a model-keyed target never
    matches a model-less request).
    """
    if not base_url:
        return None
    try:
        p = urlsplit(str(base_url))
    except ValueError:
        return None
    scheme = (p.scheme or "").lower()
    host = p.hostname or ""
    port = p.port
    model = model or ""
    for name, t in targets.items():
        if t.model not in ("", model):
            continue
        for u in t.base_urls:
            try:
                q = urlsplit(u)
            except ValueError:
                continue
            if (q.scheme or "").lower() != scheme:
                continue
            if (q.hostname or "") != host:
                continue
            if (q.port or None) != (port or None):
                continue
            return name
    return None


class Ledger:
    """Per-target in-flight accounting with exact admit/hold/release rules."""

    def __init__(self, now: Callable[[], float], hold_escape: float = 300.0):
        self._now = now
        self._hold_escape = hold_escape
        self._lock = threading.RLock()
        self._entries: Dict[str, _Entry] = {}
        self._held_since: Dict[str, float] = {}
        self._idle_marker: Optional[float] = None
        # observability: (request_id, seeded_need, generated) of last release
        self.last_release: Optional[tuple] = None

    # -- internals -----------------------------------------------------
    def _refresh_idle(self) -> None:
        total = sum(e.needed for e in self._entries.values())
        if total == 0 and self._idle_marker is None:
            self._idle_marker = self._now()
        if total > 0:
            self._idle_marker = None

    def _add(self, request_id: str, session_id: str, target: str, needed: int) -> _Entry:
        entry = _Entry(request_id, session_id, target, needed, self._now())
        self._entries[request_id] = entry
        self._refresh_idle()
        return entry

    # -- queries -------------------------------------------------------
    def in_flight(self, target: str) -> int:
        with self._lock:
            return sum(e.needed for e in self._entries.values() if e.target == target)

    def in_flight_total(self) -> int:
        with self._lock:
            return sum(e.needed for e in self._entries.values())

    def idle_since(self, now: Optional[float] = None) -> Optional[float]:
        """Timestamp when in-flight went to zero (engine idle), or None if busy."""
        with self._lock:
            self._refresh_idle()
            if self._idle_marker is None and not self._entries:
                self._idle_marker = self._now() if now is None else now
            return self._idle_marker

    def held_since(self, request_id: str) -> Optional[float]:
        with self._lock:
            return self._held_since.get(request_id)

    # -- transitions ---------------------------------------------------
    def admit(self, request_id: str, session_id: str, target: str, needed: int,
              L: int, throttle: bool, engine_idle_since: Optional[float] = None) -> tuple:
        """Admit iff it fits — or immediately when the engine is idle.

        Load-gate (v0.2.1): a hold engages ONLY when there is another request
        in flight. When the engine is idle (total in-flight 0) there is no
        concurrent KV pressure, so the request is admitted regardless of the
        (possibly depressed) learned limit L. THROTTLE still forces serial
        admission (nothing is admitted while another request is in flight).

        Returns ``(admitted, reason)`` with reason in
        ``"admit" | "hold" | "throttle" | "already-tracked"``.
        ``engine_idle_since`` is advisory (logging); the ledger derives
        idleness itself.
        """
        with self._lock:
            if request_id in self._entries:
                return (True, "already-tracked")
            if throttle and self.in_flight_total() > 0:
                return (False, "throttle")
            if self.in_flight_total() == 0:
                # Engine idle: no concurrent OOM risk → admit regardless of L.
                self._add(request_id, session_id, target, needed)
                return (True, "admit")
            if self.in_flight(target) + needed > L:
                return (False, "hold")
            self._add(request_id, session_id, target, needed)
            return (True, "admit")

    def held(self, request_id: str, session_id: str, target: str, needed: int,
             L: int, throttle: bool = False) -> tuple:
        """Re-check a held request; releases when the engine is idle, when it
        fits, or on the 300s hold-escape.

        Load-gate (v0.2.1): the moment the engine goes idle (no other request
        in flight) the held request is released immediately — it no longer
        waits out the 300s escape. That release is reported as
        ``"hold_escape"`` so the adapter's ``on_hold_escape_admit`` L-raise
        still fires (the learned limit recovers exactly as before). The 300s
        idle-escape is kept below as the defensive L-raise backstop.

        THROTTLE (mirroring ``admit``) holds regardless of fit: serial
        admission is THROTTLE's safety property — a held request must not be
        admitted while another request is in flight, even when it fits.

        Returns ``(admitted, reason)`` with reason ``"admit" | "hold" |
        "throttle" | "hold_escape" | "already-tracked"``.
        """
        with self._lock:
            if request_id in self._entries:
                return (True, "already-tracked")
            if throttle and self.in_flight_total() > 0:
                # Record the held-since marker (first re-check) so the gate's
                # max_hold deadlock guard stays effective through the
                # THROTTLE window.
                self._held_since.setdefault(request_id, self._now())
                return (False, "throttle")
            # Fit first: a held request that now fits the learned limit is a
            # normal "admit" (preserves the fit→"admit" distinction the
            # throttle test relies on).
            if self.in_flight(target) + needed <= L:
                self._add(request_id, session_id, target, needed)
                self._held_since.pop(request_id, None)
                return (True, "admit")
            # Load-gate: the engine is idle (no other request in flight), so
            # release now — no 300s wait. A solo request that no longer fits
            # the (possibly depressed) learned limit cannot deadlock. Reported
            # as "hold_escape" so the L-raise backstop still fires.
            if self.in_flight_total() == 0:
                self._add(request_id, session_id, target, needed)
                self._held_since.pop(request_id, None)
                return (True, "hold_escape")
            idle = self.idle_since()
            if idle is not None and self._now() - idle >= self._hold_escape:
                # Defensive backstop: in_flight_total()==0 already released
                # above, so this 300s path is only reached if that check is
                # ever bypassed; kept so the L-raise recovery never regresses.
                self._add(request_id, session_id, target, needed)
                self._held_since.pop(request_id, None)
                return (True, "hold_escape")
            self._held_since.setdefault(request_id, self._now())
            return (False, "hold")

    def force_admit(self, request_id: str, session_id: str, target: str,
                    needed: int) -> tuple:
        """Track *request_id* without the capacity check (the gate's
        max_hold escape, log_only mode). A forced admit is TRACKED, so
        other requests see the load. Dedupes like ``admit``: returns
        ``(True, "already-tracked")`` for a live id; otherwise the entry
        is added and ``(True, "force")`` is returned.
        """
        with self._lock:
            if request_id in self._entries:
                return (True, "already-tracked")
            self._add(request_id, session_id, target, needed)
            self._held_since.pop(request_id, None)
            return (True, "force")

    def release(self, request_id: str, generated: Optional[int] = None) -> bool:
        """Drop a ledger entry (idempotent). ``generated`` is recorded for
        observability; in-flight accounting for the request drops to zero.
        Returns True when an entry was actually dropped."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
            self._held_since.pop(request_id, None)
            if entry is not None and generated is not None:
                self.last_release = (request_id, entry.needed, generated)
            self._refresh_idle()
            return entry is not None

    def release_session(self, session_id: str) -> int:
        """Drop all of a session's tracked entries (release-on-error
        fallback). Hermes is sequential per session, so at most one entry
        is live in practice. Returns the number released."""
        with self._lock:
            doomed = [rid for rid, e in self._entries.items()
                      if e.session_id == session_id]
            for rid in doomed:
                self._entries.pop(rid, None)
                self._held_since.pop(rid, None)
            if doomed:
                self._refresh_idle()
            return len(doomed)

    def mark_streaming(self, request_id: str) -> None:
        """Flag an entry as streaming: its upper bound stays until the
        terminal usage arrives (the hermes adapter releases via
        ``post_api_request``)."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                entry.streaming = True

    def entry(self, request_id: str) -> Optional[_Entry]:
        with self._lock:
            return self._entries.get(request_id)
