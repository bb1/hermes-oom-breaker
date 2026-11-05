"""Engine-metrics probe: background poller for engine liveness + cache usage.

Hermes-free: the HTTP fetcher and the clock are injected so the module is
unit-testable without hermes; the hermes adapter injects a ``urllib`` fetcher
(bounded by a 5 s timeout) later.

Misconfig-safe: a probe that never succeeded is NOT "engine down" — only an
engine that succeeded before and then went silent past 3 * poll_interval is,
and only when the probe is authoritative.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from .config import Target

# The cache-usage gauge the engine exposes on its metrics endpoint.
_CACHE_USAGE_GAUGE = "vllm:gpu_cache_usage_perc"
# Silent for more than this many poll intervals after ever succeeding -> down.
_DOWN_FACTOR = 3.0


def parse_gauge(body: str, name: str) -> Optional[float]:
    """Value of metric *name* in a Prometheus text exposition, or None.

    Matches ``name{labels} value`` and bare ``name value``; comment lines
    (``# HELP``/``# TYPE``) are skipped; the first matching series wins.
    """
    labeled = name + "{"
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(labeled):
            payload = line[line.rfind("}") + 1:]
        elif line.startswith(name) and len(line) > len(name) and line[len(name)] in " \t":
            payload = line[len(name):]
        else:
            continue
        for token in payload.split():
            try:
                return float(token)
            except ValueError:
                continue
    return None


class _TargetState:
    """Per-URL liveness + usage snapshot (only touched under the probe lock)."""

    __slots__ = ("ever_ok", "last_ok_at", "cache_usage")

    def __init__(self) -> None:
        self.ever_ok = False
        self.last_ok_at: Optional[float] = None
        self.cache_usage: Optional[float] = None


class MetricsProbe:
    """Polls each target's metrics URL; tracks liveness and cache usage.

    ``fetch`` returns the body of ``GET <url>``; ``now`` is the clock.
    All mutable state sits behind a single RLock; the poll thread is a daemon
    and waits on a threading.Event, so ``stop()`` interrupts it within one
    poll interval.
    """

    def __init__(self, fetch: Callable[[str], str], now: Callable[[], float],
                 authoritative: bool, poll_interval: float = 10.0):
        self._fetch = fetch
        self._now = now
        self._authoritative = authoritative
        self._poll_interval = poll_interval
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._urls: List[str] = []
        self._state: Dict[str, _TargetState] = {}

    # -- state -------------------------------------------------------------
    def tick(self, url: str, value: Optional[float]) -> None:
        """Record one successful poll of *url*; clamps usage into 0..1."""
        with self._lock:
            st = self._state.setdefault(url, _TargetState())
            st.ever_ok = True
            st.last_ok_at = self._now()
            if value is not None:
                st.cache_usage = min(1.0, max(0.0, float(value)))

    def engine_down(self, url: str) -> bool:
        """True only when the engine was ever ok, then went silent for more
        than 3 * poll_interval — and only when the probe is authoritative."""
        with self._lock:
            if not self._authoritative:
                return False
            st = self._state.get(url)
            if st is None or not st.ever_ok or st.last_ok_at is None:
                return False
            return (self._now() - st.last_ok_at) > _DOWN_FACTOR * self._poll_interval

    def cache_usage(self, url: str) -> Optional[float]:
        """Last seen cache-usage fraction (0..1), or None if never reported."""
        with self._lock:
            st = self._state.get(url)
            return st.cache_usage if st is not None else None

    # -- lifecycle -----------------------------------------------------------
    def start(self, targets: Dict[str, Target]) -> None:
        """Begin polling the targets' metrics URLs (idempotent)."""
        with self._lock:
            if self._thread is not None:
                return
            seen = set()
            urls = []
            for t in targets.values():
                if t.metrics_url and t.metrics_url not in seen:
                    seen.add(t.metrics_url)
                    urls.append(t.metrics_url)
            self._urls = urls
            self._state = {}
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="oom-breaker-metrics", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the poll thread and wait for it to exit (idempotent)."""
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join()

    # -- poll thread ----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                urls = list(self._urls)
            for url in urls:
                if self._stop.is_set():
                    break
                try:
                    body = self._fetch(url)  # never under the lock: may block
                except Exception:
                    continue  # failed fetch = non-success, never "down" alone
                self.tick(url, parse_gauge(body, _CACHE_USAGE_GAUGE))
            self._stop.wait(self._poll_interval)
