"""Reactive breaker FSM: NORMAL -> RECOVERY -> THROTTLE.

Reactive only: nothing happens until a brain 5xx arrives. State persists in
the durable Store, so a hermes restart mid-RECOVERY or mid-THROTTLE resumes
with the 15-min escalation window and the 30-min serial deadline intact.
Hermes-free (clock injected).
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional, Tuple


class Breaker:
    def __init__(self, cfg, store, learner, now: Callable[[], float]):
        self._cfg = cfg
        self._store = store
        self._learner = learner
        self._now = now
        self._lock = threading.RLock()

    # -- storage ---------------------------------------------------------
    def _blob(self) -> dict:
        return self._store.get("breaker", {}) or {}

    def _write(self, d: dict) -> None:
        def tf(blob):
            blob["breaker"] = d
            return blob
        self._store.update(tf)

    # -- state ------------------------------------------------------------
    def state(self) -> str:
        """Current state; THROTTLE auto-expires to normal (persisted)."""
        with self._lock:
            b = self._blob()
            if b.get("state") == "throttle" and self._now() >= b.get("throttle_until", 0):
                b["state"] = "normal"
                b["queue"] = []
                b["active"] = None
                self._write(b)
            return b.get("state", "normal")

    def throttle_active(self, now: float) -> bool:
        with self._lock:
            b = self._blob()
            return b.get("state") == "throttle" and now < b.get("throttle_until", 0)

    def holds_big(self, session_id: str, context_tokens: int) -> bool:
        """RECOVERY: hold the session's retried requests while its compaction
        is pending (queued) or in flight (active)."""
        with self._lock:
            if self.state() != "recovery" or context_tokens <= self._cfg.compact_above:
                return False
            b = self._blob()
            return session_id in b.get("queue", []) or b.get("active") == session_id

    # -- events -----------------------------------------------------------
    def hit(self, target: str, in_flight: int, session_ids: List[str],
            status_code: int, now: float) -> Tuple[str, bool]:
        """Register a hit. Returns ``(state, deduped)`` — the resulting state
        and whether this hit was deduped as a burst duplicate.

        Burst dedupe: 5xx inside ``hit_dedupe`` of the previous hit = the same
        hit -> ``(current state, True)``, nothing is updated. Escalation: a
        *new* hit inside ``escalate_window`` of the previous hit -> THROTTLE
        (``"throttle", False``; checked against the previous hit, before this
        one updates the learner's timestamp). Otherwise -> ``"recovery"``.
        """
        with self._lock:
            b = self._blob()
            last = b.get("last_hit")
            if last is not None and now - last < self._cfg.hit_dedupe:
                return b.get("state", "normal"), True

            prior_hit = self._learner.last_hit_ts()
            escalate = (
                prior_hit is not None
                and (now - prior_hit) < self._cfg.escalate_window
            )
            self._learner.on_hit(target, in_flight)

            if escalate:
                b.update({
                    "state": "throttle",
                    "throttle_until": now + self._cfg.throttle_duration,
                    "queue": [],
                    "active": None,
                })
            else:
                seen = set()
                queue = [s for s in session_ids if not (s in seen or seen.add(s))]
                b.update({"state": "recovery", "queue": queue, "active": None})
            b["last_hit"] = now
            self._write(b)
            return b["state"], False

    # -- compaction sequencing ---------------------------------------------
    def start_compaction(self, session_id: str) -> bool:
        """1-at-a-time: begin compacting *session_id* (must be queued, none active)."""
        with self._lock:
            if self.state() != "recovery":
                return False
            b = self._blob()
            if b.get("active") is not None or session_id not in b.get("queue", []):
                return False
            b["active"] = session_id
            self._write(b)
            return True

    def compaction_done(self, session_id: str, status: str) -> List[str]:
        """Compaction finished (or failed). Returns the remaining queue.

        ``unserviceable`` (compaction request 400'd: session context exceeds
        max-model-len) -> THROTTLE + flag; the session is never re-queued.
        Queue empty -> back to normal.
        """
        with self._lock:
            b = self._blob()
            if b.get("state") != "recovery" or b.get("active") != session_id:
                return list(b.get("queue", []))
            b["active"] = None
            if status == "unserviceable":
                b.setdefault("unserviceable", {})[session_id] = self._now()
                b["state"] = "throttle"
                b["throttle_until"] = self._now() + self._cfg.throttle_duration
                b["queue"] = []
                self._write(b)
                return []
            queue = [s for s in b.get("queue", []) if s != session_id]
            b["queue"] = queue
            if not queue:
                b["state"] = "normal"
                b["throttle_until"] = None
            self._write(b)
            return queue

    def unserviceable_ids(self) -> List[str]:
        with self._lock:
            return list(self._blob().get("unserviceable", {}).keys())

    # -- accessors (Task 7) --------------------------------------------------
    def pending_queue(self) -> List[str]:
        """The recovery queue from the blob (empty list when not in
        recovery or absent)."""
        with self._lock:
            if self.state() != "recovery":
                return []
            return list(self._blob().get("queue", []))

    def active_session(self) -> Optional[str]:
        """The session id currently being compacted (None otherwise)."""
        with self._lock:
            if self.state() != "recovery":
                return None
            return self._blob().get("active")
