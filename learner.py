"""Adaptive limit L: seed, lower on 5xx, raise on hold-escape, reset, reprobe.

Every 5xx is a data point; no calibration window. L is keyed per target and
persisted through the durable Store so a hermes restart keeps the learned
limits. Hermes-free (clock + store injected).

Wiring in v0.1.0: the downward path is ``on_hit`` (breaker, fired on every
tracked 5xx); the ONLY automatic upward path is ``on_hold_escape_admit``
(fired when the gate admits a held request after a long idle escape, capped
at the target's kv budget). ``on_episode_peak`` and ``maybe_reprobe`` remain
part of the API but have no production call sites in v0.1.0 (the ``reprobe``
settings knob was removed; manual ``oom_reset`` is the other reset path).
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

HISTORY_CAP = 50
REPROBE_CLEAN_SECONDS = 86_400


class Learner:
    def __init__(self, now: Callable[[], float], store,
                 race_reserve: int = 5000, escalate_window: float = 900.0):
        self._now = now
        self._store = store
        self._eps = race_reserve
        self._window = escalate_window
        self._lock = threading.RLock()

    # -- storage helpers ------------------------------------------------
    def _meta(self, target: str) -> dict:
        meta = self._store.get("meta", {}).get(target)
        if meta is None:
            raise KeyError(f"target {target!r} not seeded")
        return meta

    def _record(self, target: str, old: int, new: int, why: str) -> None:
        def tf(d):
            h = d.setdefault("history", [])
            h.append([self._now(), target, old, new, why])
            del h[:-HISTORY_CAP]
            return d
        self._store.update(tf)

    # -- API -------------------------------------------------------------
    def seed(self, target: str, t) -> int:
        """L := min(max_model_len, kv_budget); persist target metadata."""
        with self._lock:
            L = min(t.max_model_len, t.kv_budget)

            def tf(d):
                d.setdefault("L", {})[target] = L
                d.setdefault("meta", {})[target] = {
                    "max_model_len": t.max_model_len, "kv_budget": t.kv_budget,
                }
                return d
            self._store.update(tf)
            return L

    def current(self, target: str, defaults: Optional[Dict[str, object]] = None) -> int:
        with self._lock:
            L = self._store.get("L", {}).get(target)
            if L is None and defaults and target in defaults:
                return self.seed(target, defaults[target])
            if L is None:
                raise KeyError(f"target {target!r} not seeded")
            return int(L)

    def on_hit(self, target: str, in_flight_at_hit: int) -> int:
        """Any brain 5xx: L := min(L, max(1, in_flight - eps)). Exact ledger number."""
        with self._lock:
            old = self.current(target, {})
            new = max(1, min(old, in_flight_at_hit - self._eps))
            ts = self._now()

            def tf(d):
                d.setdefault("L", {})[target] = new
                d["last_hit_ts"] = ts
                return d
            self._store.update(tf)
            self._record(target, old, new, "hit")
            return new

    def on_episode_peak(self, target: str, peak: int) -> int:
        """Clean busy episode peaking above L raises L (cap: kv budget).
        API surface only in v0.1.0: no production call site (episode peaks
        are not observed automatically); hold-escape admission is the wired
        upward path."""
        return self._raise(target, peak, "episode")

    def on_hold_escape_admit(self, target: str, peak: int) -> int:
        """Hold-escape firing is the upward-exploration path: same raise rule."""
        return self._raise(target, peak, "hold_escape")

    def _raise(self, target: str, peak: int, why: str) -> int:
        with self._lock:
            old = self.current(target, {})
            budget = self._meta(target)["kv_budget"]
            if not (old < peak <= budget):
                return old
            new = int(peak)

            def tf(d):
                d.setdefault("L", {})[target] = new
                return d
            self._store.update(tf)
            self._record(target, old, new, why)
            return new

    def reset(self, target: str) -> int:
        """Back to the target's seed; clears the escalation window."""
        with self._lock:
            meta = self._meta(target)
            new = min(meta["max_model_len"], meta["kv_budget"])
            old = self.current(target, {})

            def tf(d):
                d.setdefault("L", {})[target] = new
                d["last_hit_ts"] = None
                return d
            self._store.update(tf)
            if old != new:
                self._record(target, old, new, "reset")
            return new

    def within_escalation_window(self, now: float) -> bool:
        """True when *now* is inside the escalate window after the last hit."""
        with self._lock:
            last = self._store.get("last_hit_ts")
            return last is not None and (now - last) < self._window

    def last_hit_ts(self) -> Optional[float]:
        return self._store.get("last_hit_ts")

    def maybe_reprobe(self, target: str, now: float, reprobe: bool = False) -> Optional[int]:
        """Optional: after 24 h clean, raise L by eps for one episode (once per
        hit). API surface only in v0.1.0: the ``reprobe`` settings knob was
        removed, so nothing in the shipped wiring calls this."""
        if not reprobe:
            return None
        with self._lock:
            last = self._store.get("last_hit_ts")
            if last is None or (now - last) < REPROBE_CLEAN_SECONDS:
                return None
            if self._store.get("last_reprobe", {}).get(target) == last:
                return None
            old = self.current(target, {})
            budget = self._meta(target)["kv_budget"]
            new = min(old + self._eps, budget)

            def tf(d):
                d.setdefault("L", {})[target] = new
                d.setdefault("last_reprobe", {})[target] = last
                return d
            self._store.update(tf)
            self._record(target, old, new, "reprobe")
            return new

    def history(self) -> list:
        with self._lock:
            return list(self._store.get("history", []))
