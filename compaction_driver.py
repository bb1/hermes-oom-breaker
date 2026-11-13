"""1-at-a-time session compaction through the gateway's own
``session.mutate(compress)`` control-socket path.

The reactive breaker (``adapter.py``) enqueues the offending session when a
5xx lands; this module drives exactly one compaction at a time on a dedicated
daemon worker. A finished attempt is never auto-retried: the queue is size 1,
and a busy/erroring session stays where the breaker put it, bounded by the
escalation window (next 5xx -> THROTTLE).

Driver vocabulary (returned to ``on_finished``):

- ``"compressed"``        committed compress (result carries messages/summary)
- ``"nothing_to_do"``     nothing to compress ('noop'/'nothing'/
                          'nothing_to_compress')
- ``"lock_skipped"``      gateway is busy with the session ('session_busy')
- ``"unserviceable"``     the session cannot be compressed at all
                          ('model_resolution_failed' / 'unsupported_compress_options' /
                          'invalid_params') — breaker THROTTLEs + flags it
- ``"error:<msg>"``       anything else, including 'permission_denied'
                          (surfaced as ``"error:permission_denied"``) and
                          exceptions ('error:<ExcType>')

All hermes imports live inside function bodies, so importing this module
stays hermes-free (doctor must pass in a hermes-free environment; the tests
import it without hermes installed).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, List, Optional

logger = logging.getLogger("oom_breaker")

# GatewayClientError messages that mean "this session cannot be compressed
# at all" (the breaker THROTTLEs + flags it; it is never re-queued).
_UNSERVICEABLE = {"model_resolution_failed", "unsupported_compress_options",
                  "invalid_params"}

# GatewayClientError messages that mean "the session fit; there was nothing
# to compress".
_NOTHING = {"nothing_to_compress"}


class CompactionDriver:
    """Base/protocol for compaction drivers.

    ``on_finished`` is an attribute: ``Callable[[str, str], None]`` or None.
    ``compact`` enqueues 1-at-a-time and returns False when the queue is full.
    """

    on_finished: Optional[Any] = None   # Callable[[str, str], None] or None

    def start(self) -> None:
        """Idempotent: bring the driver up (a real driver starts its worker)."""

    def stop(self) -> None:
        """Idempotent: shut the driver down; must never hang."""

    def compact(self, session_id: str, home: Optional[str] = None) -> bool:
        raise NotImplementedError

    @property
    def in_flight(self) -> bool:
        """True while an attempt is in flight (implementation-defined)."""
        return False


class FakeDriver(CompactionDriver):
    """Test double, no threads: ``compact`` records the attempt but does NOT
    call ``on_finished`` inline — the test controls timing via ``finish``
    (which invokes ``on_finished`` with the explicit, scripted, or default
    ("compressed") status)."""

    def __init__(self) -> None:
        self.queued: List[str] = []
        self.attempts: dict = {}
        self._statuses: dict = {}
        self._inflight: dict = {}

    def script(self, session_id: str, status: str) -> None:
        """Pre-set the terminal status of a session's attempts."""
        self._statuses[session_id] = status

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def compact(self, session_id: str, home: Optional[str] = None) -> bool:
        self.queued.append(session_id)
        self.attempts[session_id] = self.attempts.get(session_id, 0) + 1
        self._inflight[session_id] = self._statuses.get(session_id,
                                                        "compressed")
        return True

    @property
    def in_flight(self) -> bool:
        return bool(self._inflight)

    def finish(self, session_id: str, status: Optional[str] = None) -> None:
        """Complete the attempt: fire ``on_finished`` with the explicit,
        scripted, or default status."""
        if status is None:
            status = self._inflight.pop(session_id, "compressed")
        else:
            self._inflight.pop(session_id, None)
        if self.on_finished is not None:
            self.on_finished(session_id, status)


class GatewayDriver(CompactionDriver):
    """One **daemon** worker thread + ``queue.Queue(maxsize=1)``: strictly
    1-at-a-time. The worker never dies: any attempt exception becomes
    ``"error:<ExcType>"`` and any ``on_finished`` error is logged.

    ``home`` is the profile home of the session being compacted, captured at
    enqueue time on the hook thread (the per-turn HERMES_HOME override
    ContextVar does not propagate into the worker thread); when None, it is
    resolved at attempt time via ``hermes_constants.get_hermes_home()``.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gen = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Idempotent: spawn the daemon worker (one per generation).

        Restart is supported after any stop; after an incomplete (timed-out)
        stop the old worker self-terminates on its next pickup (its
        generation token no longer matches the current generation).
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            self._gen += 1
            gen = self._gen
            thread = threading.Thread(target=self._work, args=(gen,),
                                      name="oom-breaker-compact", daemon=True)
            thread.start()
            self._thread = thread

    def stop(self, timeout: float = 5.0) -> None:
        """Idempotent: shut the driver down; bounded by ~2×``timeout``
        (a blocking sentinel put and a thread join, each capped at
        ``timeout`` — worst case when the worker is mid-attempt).

        Bumps the generation token: a stale worker that survives an
        incomplete (timed-out) stop never processes a later item — it
        self-terminates on its next pickup.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._gen += 1
            thread = self._thread
            self._thread = None
        # Deliver the stop sentinel even when the single slot is briefly
        # full: a bounded blocking put (the worker drains the slot promptly);
        # the daemon thread can never wedge us past the deadline.
        try:
            self._queue.put(_STOP, block=True, timeout=timeout)
        except queue.Full:
            logger.warning("oom-breaker compaction: stop sentinel not "
                           "delivered within %ss; daemon worker exits on "
                           "process shutdown", timeout)
        if thread is not None:
            thread.join(timeout=timeout)

    # -- driver interface ---------------------------------------------------
    def compact(self, session_id: str, home: Optional[str] = None) -> bool:
        try:
            self._queue.put_nowait((session_id, home))
            return True
        except queue.Full:
            logger.warning("oom-breaker compaction: queue full; %r not queued "
                           "(1-at-a-time; it stays queued in the breaker and "
                           "the next completion signal resumes)", session_id)
            return False

    @property
    def in_flight(self) -> bool:
        """True only while an attempt is **queued** (between enqueue and the
        worker's pickup) — not while the attempt itself is running."""
        return not self._queue.empty()

    # -- worker ---------------------------------------------------------------
    def _work(self, gen: int) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            # Generation token: a worker that survived an incomplete stop is
            # stale — it self-terminates on this pickup (the item is simply
            # dropped; a subsequent operator action can re-enqueue it).
            if gen != self._gen:
                logger.warning("oom-breaker compaction: stale worker (gen "
                               "%d) dropped %r on pickup and exiting; "
                               "re-enqueue via a new compact if still "
                               "needed", gen, item[0])
                return
            session_id, home = item
            try:
                status = self._attempt(session_id, home)
            except Exception as exc:
                # Belt and braces over _rpc_attempt's own mapping: the thread
                # must never die silently.
                status = f"error:{type(exc).__name__}"
            callback = self.on_finished
            if callback is not None:
                try:
                    callback(session_id, status)
                except Exception:
                    logger.warning("oom-breaker compaction: on_finished "
                                   "callback failed for %r", session_id,
                                   exc_info=True)

    def _attempt(self, session_id: str, home: Optional[str]) -> str:
        # A private event loop per attempt (hermes' own ``_run_async``
        # worker-loop pattern): the agent turn runs on worker threads, not
        # the gateway loop, so there is no gateway loop to run on — and none
        # is wanted.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._rpc_attempt(session_id, home))
        finally:
            loop.close()

    async def _rpc_attempt(self, session_id: str, home: Optional[str]) -> str:
        if home is None:
            home = _resolve_home()
        if home is None:
            return "error:no-home"
        try:
            from hermes_cli.gateway_runtime import discover_gateway_endpoint
            # ``timeout`` is keyword-only in the current hermes
            # ``discover_gateway_endpoint(profile_home, *, timeout=2.0)``;
            # passing it positionally raises TypeError and every compaction
            # then fails (see the 2026-11-13 family RECOVERY wedge).
            info = await asyncio.to_thread(discover_gateway_endpoint,
                                           home, timeout=2.0)
            endpoint = getattr(info, "endpoint", None)
        except Exception as exc:
            return _err(exc)
        if not endpoint:
            return "error:no-endpoint"
        try:
            from hermes_cli.gateway_client import (connect_gateway,
                                                   GatewayClientError)
            from hermes_cli.gateway_mutations import (PreparedMutations,
                                                      compress_payload)
        except Exception as exc:
            return _err(exc)
        try:
            # subject=None: the socket identity — relabeled subjects must be
            # structured auth:v1 login subjects, which we never have.
            async with connect_gateway(endpoint=endpoint,
                                       subject=None) as client:
                result = await PreparedMutations().apply(
                    client, session_id, "compress", compress_payload(""),
                    confirm=None)
        except GatewayClientError as exc:
            return _map_refusal(str(exc))
        except Exception as exc:
            return _err(exc)
        return _map_result(result)


def _resolve_home() -> Optional[str]:
    """The hermes home of this process (lazy import; any failure -> None,
    so the attempt reports ``error:no-home`` rather than dying)."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return None


# -- status mapping (the driver's vocabulary) -------------------------------

def _err(exc: Exception) -> str:
    return f"error:{str(exc) or type(exc).__name__}"


def _map_refusal(message: str) -> str:
    """A definitive GatewayClientError refusal -> driver status."""
    if message == "session_busy":
        return "lock_skipped"
    if message in _UNSERVICEABLE:
        return "unserviceable"
    if message == "permission_denied":
        return "error:permission_denied"   # surface; never retry
    if message in _NOTHING:
        return "nothing_to_do"
    # revision_conflict / stale_generation / unknown_execution / anything else
    return f"error:{message}"


def _map_result(result: Any) -> str:
    """A committed compress returns the prepared params + messages + summary
    (no ``status`` key); a no-op reports a status of 'noop'/'nothing'."""
    if not isinstance(result, dict):
        return "error:unexpected-result"
    status = result.get("status")
    if status in ("noop", "nothing"):
        return "nothing_to_do"
    if status is None and ("messages" in result or "summary" in result):
        return "compressed"
    return f"error:unexpected-result-status:{status!r}"


_STOP = object()
