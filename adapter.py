"""Gate adapter: the llm_execution middleware half of hermes-oom-breaker.

``GateController`` is the in-process facade the hermes ``register(ctx)``
wiring (Task 8) talks to. This module imports nothing from hermes: payloads
arrive as plain dicts/kwargs, the clock and sleep are injected, and the
middleware contract is fail-open — the callback never raises out of its own
gate logic (hermes' contract: a raise before ``next_call`` makes hermes skip
the wrapper and proceed ungated; availability is preferred over strictness).
The two User-Agent attribution resolvers import hermes lazily inside their
bodies, so the module import itself stays hermes-free (doctor must stay
green in a hermes-free environment).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from urllib.request import urlopen

from .breaker import Breaker
from .config import Config, Target
from .ledger import Ledger, target_of
from .learner import Learner
from .metrics import MetricsProbe
from .notify import Notifier

if TYPE_CHECKING:
    from .compaction_driver import CompactionDriver

logger = logging.getLogger("oom_breaker")


def _default_fetch(url: str) -> str:
    """Default MetricsProbe fetcher: stdlib urllib, bounded by a 5 s timeout.

    The probe's poll thread tolerates fetch exceptions (non-success, never
    "down"); Task 8's wiring starts the thread — never from a constructor.
    """
    with urlopen(url, timeout=5.0) as resp:
        return resp.read().decode("utf-8", "replace")


# -- llm_request middleware (User-Agent attribution) -------------------------
# Optional observability, off by default (Config.user_agent): stamps the
# outgoing request's User-Agent so the engine's router dashboard can
# attribute big requests to a hermes profile (and cron job). Not on the
# gate path — a failure here must never change the request.

USER_AGENT_PREFIX = "Hermes"


def _default_profile_fn() -> Optional[str]:
    """Default profile resolver: the profile the current turn runs FOR
    (per-turn HERMES_HOME override in the multiplexed single-gateway
    process). Imports hermes lazily so this module's import stays
    hermes-free; any failure — import or call — yields None, so the label
    falls back to the bare prefix."""
    try:
        from hermes_cli.profiles import current_profile_name
        return current_profile_name()
    except Exception:
        return None


def _default_cron_fn() -> Optional[str]:
    """Default cron resolver: the ``job_name`` of the cron execution the
    scheduler bound to this turn's ContextVar (propagated into the agent
    thread via ``contextvars.copy_context()``). Lazy hermes import, same
    contract: any failure — import, call, or missing attribute — yields
    None, so no job suffix is appended."""
    try:
        from cron.execution_identity import current_cron_execution
        execution = current_cron_execution()
        return None if execution is None else execution.job_name
    except Exception:
        return None


def _current_home() -> Optional[str]:
    """The hermes home of this process, captured at enqueue time on the hook
    thread. The per-turn HERMES_HOME override (a ContextVar) does not
    propagate into the driver's worker thread, so the profile home of the
    session being compacted is pinned when the attempt is queued. Lazy hermes
    import, same contract as the UA resolvers: any failure -> None (the
    driver resolves the home at attempt time)."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return None


def estimate_tokens(request: dict) -> int:
    """Default estimator: total message-content chars // 4.

    Ruling R2: the hermes wiring (Task 8) swaps ``GateController.estimator``
    for hermes' ``estimate_request_context_tokens`` when importable, so the
    ledger's accounting agrees with hermes' own.
    """
    total = 0
    messages = request.get("messages") if isinstance(request, dict) else None
    for m in messages or []:
        if isinstance(m, dict):
            total += len(m.get("content") or "") // 4
    return total


def _completion_tokens(usage: Any) -> Optional[int]:
    """``usage.completion_tokens`` duck-typed (attribute or dict key);
    tolerate absence → None."""
    if usage is None:
        return None
    value = getattr(usage, "completion_tokens", None)
    if value is None and isinstance(usage, dict):
        value = usage.get("completion_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


class GateController:
    """The fit-gate facade: one per gateway process, wired by Task 8.

    ``now``/``sleep`` are injected (tests drive a virtual clock). The probe
    is built here but NOT started: ``load_targets()`` seeds the learner only
    (Ruling D3), and Task 8's ``register`` calls ``probe.start``.
    """

    def __init__(self, cfg: Config, store,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 notifier: Optional[Notifier] = None):
        self.cfg = cfg
        self.store = store
        self.now = now
        self.sleep = sleep
        # Debug posts (off unless cfg.debug): injected in tests; the
        # production notifier reads the scope's home .env on each post.
        self._notifier = notifier if notifier is not None else Notifier(cfg, now=now)
        self.ledger = Ledger(now, hold_escape=cfg.hold_escape)
        self.learner = Learner(now, store, race_reserve=cfg.race_reserve,
                               escalate_window=cfg.escalate_window)
        self.breaker = Breaker(cfg, store, self.learner, now)
        self.probe = MetricsProbe(_default_fetch, now, cfg.probe_authoritative)
        self.estimator: Callable[[dict], int] = estimate_tokens
        self.profile_fn: Callable[[], Optional[str]] = _default_profile_fn
        self.cron_fn: Callable[[], Optional[str]] = _default_cron_fn
        self.compaction_driver = None              # Task 7 installs the driver
        self.api_error_hook_registered = False     # Task 7's hook is canonical
        self.last_driver_status: Optional[str] = None  # Task 7's oom_status
        # Debug-post meta for driver-thread completions: (stashed home,
        # attribution label) per dispatched session — the driver's worker
        # thread has no per-turn context, so both are captured at dispatch.
        self._dispatch_meta: Dict[str, Tuple[Optional[str], str]] = {}
        self._meta_lock = threading.Lock()
        # Aliased so target resolution works before load_targets() runs.
        self.targets: Dict[str, Target] = cfg.targets

    # -- setup --------------------------------------------------------------
    def load_targets(self) -> None:
        """Seed the learner from cfg.targets (Ruling D3: learner only — the
        probe thread is started by Task 8's wiring, never here).

        Seeding goes through the learner's lazy path so a restart seeds only
        targets that have no learned limit yet — a learned L persists.
        """
        self.targets = self.cfg.targets
        for name, t in self.cfg.targets.items():
            self.learner.current(name, {name: t})

    # -- llm_request middleware -----------------------------------------------
    def ua_llm_request(self, request: dict, **context: Any) -> Optional[dict]:
        """The ``llm_request`` callback (Task 8 registers it; it is always
        registered and a runtime no-op while ``Config.user_agent`` is off).
        Stamps the outgoing request's User-Agent through ``extra_headers``
        — the OpenAI SDK merges per-request ``extra_headers`` on top of the
        client defaults, overriding the wire UA — so the router dashboard
        can attribute big requests.

        Never raises: observability must never break a request. Disabled or
        any internal failure returns None (hermes leaves the request
        unchanged). The input dict is never mutated: the result carries a
        fresh ``request`` mapping and a fresh ``extra_headers`` dict.
        """
        try:
            if not self.cfg.user_agent:
                return None
            extra_headers = request.get("extra_headers")
            headers = dict(extra_headers) if isinstance(extra_headers, dict) else {}
            headers["User-Agent"] = self.build_user_agent()
            return {"request": {**request, "extra_headers": headers}}
        except Exception:
            logger.warning("oom-breaker ua_llm_request: internal error; "
                           "leaving the request unchanged", exc_info=True)
            return None

    def build_user_agent(self) -> str:
        """The stamped label: ``Hermes/<profile>``, or the bare ``Hermes``
        when no profile resolves, with the cron job name appended
        (``Hermes/<profile>:<job>``) for a cron execution. May raise — the
        injected resolvers are untrusted here; the caller swallows it and
        leaves the request unchanged."""
        label = USER_AGENT_PREFIX
        profile = self.profile_fn()
        if profile:
            label += f"/{profile}"
        job = self.cron_fn()
        if job:
            label += f":{job}"
        return label

    # -- debug posts (opt-in one-line notices) -------------------------------
    def _attribution(self) -> str:
        """The debug-post label: "" (no profile resolved), "<profile>", or
        "<profile>:<job>" for a cron execution. Reuses the UA resolvers but
        is stricter than ``build_user_agent``: a job without a profile yields
        no label (not ``":job"``), and resolver failures are swallowed here
        (the post proceeds unlabeled rather than lost)."""
        try:
            profile = self.profile_fn()
        except Exception:
            profile = None
        try:
            job = self.cron_fn()
        except Exception:
            job = None
        if not profile:
            return ""
        label = str(profile)
        if job:
            label += f":{job}"
        return label

    def _debug_post(self, event: str, line: str, *,
                    home: Optional[str] = None,
                    label: Optional[str] = None) -> None:
        """One opt-in debug notice: "[oom-breaker] (label) line", the
        parenthetical omitted when the label is empty. The notifier never
        raises (off unless cfg.debug; rate-limited per event type), so this
        is called directly — no extra wrapper."""
        text = (f"[oom-breaker] ({label}) {line}" if label
                else f"[oom-breaker] {line}")
        self._notifier.post(event, text, home=home)

    # -- llm_execution middleware -------------------------------------------
    def llm_execution(self, request: dict, next_call, **context: Any) -> Any:
        """The middleware callback. Never raises out of its own gate logic.

        Phase 1 (gate: target/estimation/admission/hold) runs fail-open —
        any internal error logs loud and the request passes through
        untracked. Phase 2 (the call itself): an exception releases the
        ledger entry and re-raises (hermes' retry re-enters the wrapper).
        Phase 3 (settle): streaming keeps the upper bound; non-streaming
        success releases with the generated tokens.
        """
        plan = None
        try:
            plan = self._plan(request, context)
        except Exception:
            logger.warning(
                "oom-breaker gate: internal error for request %r; failing "
                "open (untracked passthrough)", context.get("api_request_id"),
                exc_info=True)
        if plan is None:
            # Untracked provider, reshaped context, or a gate error: the
            # request proceeds without the gate.
            return next_call(request)
        _target, request_id, _session_id, _needed = plan
        try:
            result = next_call(request)
        except Exception:
            self.ledger.release(request_id)
            raise
        try:
            self._settle(result, request_id)
        except Exception:
            logger.warning(
                "oom-breaker gate: post-call settle failed for %r; entry "
                "stays tracked until the post_api_request hook releases it",
                request_id, exc_info=True)
        return result

    def _plan(self, request: dict, context: dict) -> Optional[tuple]:
        """Pre-call gate: resolve the target, size the request, admit — or
        hold until admitted. Returns None when the call must pass through
        untracked (untracked provider or reshaped context). May raise: the
        caller fails open."""
        target = target_of(context.get("base_url"), context.get("model"),
                           self.targets)
        request_id = context.get("api_request_id")
        session_id = context.get("session_id")
        if target is None or not request_id or not session_id:
            # Untracked provider, or a context field hermes renamed/dropped:
            # plain passthrough — no ledger, no error.
            return None
        needed = self.estimator(request) + int(request.get("max_tokens") or 0)
        limit = self.learner.current(target, self.targets)
        admitted, _why = self.ledger.admit(
            request_id, session_id, target, needed, limit,
            self.breaker.throttle_active(self.now()), self.ledger.idle_since())
        if not admitted:
            if self.cfg.log_only:
                # Never hold in log_only mode — but keep the request TRACKED
                # (tracked force-admit) so in-flight stays exact for others.
                logger.warning(
                    "oom-breaker log_only: would hold %r (session %s) on "
                    "target %s; proceeding tracked",
                    request_id, session_id, target)
                self.ledger.force_admit(request_id, session_id, target, needed)
            else:
                self._hold(target, request_id, session_id, needed, limit)
        return target, request_id, session_id, needed

    def _hold(self, target: str, request_id: str, session_id: str,
              needed: int, limit: int) -> None:
        """Hold until the request fits (or an escape fires).

        Sleeps go through the injected sleep; ledger/breaker locks are held
        only inside their own method calls, never across a sleep.
        """
        while True:
            if self.breaker.holds_big(session_id, needed):
                # RECOVERY: this session's compaction is pending or in
                # flight — wait for its queue to drain, rechecking at the
                # cadence.
                self.sleep(self.cfg.recheck_cadence)
                continue
            now = self.now()
            held_at = self.ledger.held_since(request_id)
            if held_at is not None and now - held_at > self.cfg.max_hold:
                # Deadlock guard: admit anyway, tracked, so other requests
                # see the load.
                _ok, why = self.ledger.force_admit(request_id, session_id,
                                                   target, needed)
                logger.warning(
                    "oom-breaker: request %r held %.0fs on target %s past "
                    "max_hold; admitting anyway (reason=%s)",
                    request_id, now - held_at, target, why)
                self._debug_post(
                    "delayed",
                    f"request held {now - held_at:.0f}s on '{target}' "
                    "(max_hold); admitted anyway",
                    home=_current_home(), label=self._attribution())
                return
            admitted, reason = self.ledger.held(
                request_id, session_id, target, needed, limit,
                throttle=self.breaker.throttle_active(self.now()))
            if admitted:
                if reason == "hold_escape":
                    # Upward exploration (Ruling D5): the engine sat idle
                    # long enough that firing is a new-peak data point; the
                    # learner caps at the target's kv budget. Best-effort:
                    # the admission already happened — a hook error must not
                    # fail the request (or leak its tracked entry).
                    try:
                        self.learner.on_hold_escape_admit(target, needed)
                    except Exception:
                        logger.warning(
                            "oom-breaker: hold-escape learner update failed "
                            "for %r (target %s)", request_id, target,
                            exc_info=True)
                    line = (f"request held {now - held_at:.0f}s on "
                            f"'{target}' (hold_escape); admitted"
                            if held_at is not None
                            else f"request admitted on '{target}' "
                                 "(hold_escape)")
                    self._debug_post("delayed", line, home=_current_home(),
                                     label=self._attribution())
                return
            self.sleep(self.cfg.recheck_cadence)

    def _settle(self, result: Any, request_id: str) -> None:
        """Post-success: a streaming result (an iterator without ``.usage``)
        keeps its upper bound — the terminal usage releases it via
        ``post_api_request``; a non-streaming result releases now, with the
        generated tokens when extractable."""
        if hasattr(result, "__iter__") and not hasattr(result, "usage"):
            self.ledger.mark_streaming(request_id)
            return
        self.ledger.release(
            request_id, generated=_completion_tokens(getattr(result, "usage", None)))

    # -- post_api_request observer -------------------------------------------
    def post_api_request(self, **payload: Any) -> None:
        """Observer callback — success-intake. On terminal usage: release
        with the generated tokens. On a 5xx from a tracked target: release,
        and — only when the canonical ``api_request_error`` hook (Task 7) is
        NOT registered, defensive fallback — record the hit, with the
        in-flight total taken BEFORE releasing this entry. In production
        hermes fires ``api_request_error`` on failures (which does the
        release-on-error AND the hit) and ``post_api_request`` on success,
        so this 5xx branch is reachable only in the Task 6 hook
        configuration. 4xx is never a hit. Never raises."""
        try:
            self._post_api_request(payload)
        except Exception:
            logger.warning("oom-breaker post_api_request: internal error",
                           exc_info=True)

    def _post_api_request(self, payload: dict) -> None:
        request_id = payload.get("api_request_id")
        usage = payload.get("usage")
        if usage is not None:
            if request_id is not None:
                self.ledger.release(request_id,
                                    generated=_completion_tokens(usage))
            return
        status = payload.get("status_code")
        if (not isinstance(status, int) or isinstance(status, bool)
                or not 500 <= status < 600):
            return
        target = target_of(payload.get("base_url"), payload.get("model"),
                           self.targets)
        if target is None:
            return  # untracked: nothing was admitted, nothing to record
        in_flight = self.ledger.in_flight_total()   # BEFORE releasing r
        if request_id is not None:
            self.ledger.release(request_id)
        if not self.api_error_hook_registered:
            session_id = payload.get("session_id") or ""
            _state, _deduped = self.breaker.hit(
                target, in_flight, [session_id], status, self.now())

    # -- breaker hooks (Task 7: the reactive half) ---------------------------
    def install_breaker_hooks(self, driver: "CompactionDriver") -> None:
        """Wire the reactive half: the canonical 5xx hook is registered by
        Task 8's ``register`` as ``api_request_error``; this binds the
        driver — its completion callback (the PRIMARY compaction completion
        signal) and its worker start.

        ``api_error_hook_registered`` flips the Task 6 ``post_api_request``
        5xx fallback OFF: without it, the fallback would record the same 5xx
        alongside the canonical hook (double hit -> spurious THROTTLE).
        """
        self.api_error_hook_registered = True
        self.compaction_driver = driver
        driver.on_finished = self._on_compaction_finished
        driver.start()

    def _resolve_target(self, base_url: Optional[str],
                        model: Optional[str]) -> Optional[str]:
        """Target resolution shared by the gate and the breaker hooks: the
        base_url (scheme + host + port) is the match key; unknown/None ->
        None (the call is untracked)."""
        return target_of(base_url, model, self.targets)

    def _release_error_entry(self, request_id: Optional[str],
                             session_id: str) -> None:
        """Release a failed request's ledger entry (best-effort): by the
        gate's request id first (the same id ``_plan`` tracked), falling back
        to the session — hermes is sequential per session, so at most one
        live entry per session in practice."""
        if request_id is not None and self.ledger.release(request_id):
            return
        if session_id:
            self.ledger.release_session(session_id)

    # -- api_request_error (the canonical 5xx path) --------------------------
    def api_request_error(self, **payload: Any) -> None:
        """Observer callback — the canonical 5xx path (hermes fires it on
        every API failure path). On ANY failure (5xx, 4xx, non-HTTP) the
        dead request's ledger entry is released — the canonical
        release-on-error path: a dead stream must not keep over-counting
        in-flight until a gateway restart. Only ``500 <= status_code < 600``
        on a tracked target is additionally a breaker hit (L learns down,
        recovery/THROTTLE); 4xx and non-HTTP errors (status None) never are.
        On recovery, the queue head is dispatched to the driver 1-at-a-time.
        Never raises (observer contract)."""
        try:
            self._api_request_error(payload)
        except Exception:
            logger.warning("oom-breaker api_request_error: internal error",
                           exc_info=True)

    def _api_request_error(self, payload: dict) -> None:
        target = self._resolve_target(payload.get("base_url"),
                                      payload.get("model"))
        if target is None:
            return  # untracked: nothing gated, nothing to release or react
        status = payload.get("status_code")
        request_id = payload.get("api_request_id")
        session_id = str(payload.get("session_id") or "")
        # The in-flight total is captured BEFORE the release: a 5xx hit must
        # see the true load at the moment of the crash, including this
        # request's own upper bound.
        in_flight = self.ledger.in_flight_total()
        self._release_error_entry(request_id, session_id)
        if not (isinstance(status, int) and not isinstance(status, bool)
                and 500 <= status < 600):
            # Non-5xx failure (4xx, connection error, timeout): not an OOM
            # data point — no L change, no recovery. The entry above is
            # already released.
            return
        new_state, deduped = self.breaker.hit(
            target, in_flight, [session_id], status, self.now())
        line = (f"{status} on '{target}' (in-flight {in_flight:,} tok) "
                f"-> {new_state}")
        if new_state == "throttle":
            logger.warning(
                "oom-breaker: 5xx %s on target %s inside the escalate "
                "window: THROTTLE for %.0fs (serial admission; oom_status "
                "surfaces it)", status, target, self.cfg.throttle_duration)
        elif new_state == "recovery":
            queue = self.breaker.pending_queue()
            if queue:
                self._dispatch_compaction(queue[0])
                line += f", compacting {queue[0]}"
        if not deduped:
            # A burst (deduped 5xx) is one post: only the hit that actually
            # moved the state posts.
            self._debug_post("5xx", line, home=_current_home(),
                             label=self._attribution())

    def _pop_dispatch_meta(self, session_id: str) -> Tuple[Optional[str], str]:
        """The stashed (home, label) for a driver-dispatched compaction, or
        the post-time fallback (home re-resolved, label re-attributed) when
        the compression completed outside the driver's dispatch. Pop = the
        first completer owns the post; a second signal finds nothing."""
        with self._meta_lock:
            home, label = self._dispatch_meta.pop(session_id, (None, None))
        if home is None:
            home = _current_home()
        if label is None:
            label = self._attribution()
        return home, label

    def _discard_dispatch_meta(self, session_id: str) -> None:
        """Drop a session's stashed (home, label) without posting: its
        compaction completed terminally without a committed compress
        (nothing_to_do / unserviceable), so the entry is stale."""
        with self._meta_lock:
            self._dispatch_meta.pop(session_id, None)

    # -- compaction dispatch (1-at-a-time) -----------------------------------
    def _dispatch_compaction(self, session_id: str) -> None:
        """Mark a queued session active and enqueue it in the driver.
        Any failure logs loud and leaves the session where the breaker has
        it (recovery): the next completion signal, or the escalation window
        on the next 5xx, bounds the stall."""
        if not self.breaker.start_compaction(session_id):
            logger.warning("oom-breaker: could not start compaction of %r "
                           "(active compaction in flight or state changed)",
                           session_id)
            return
        driver = self.compaction_driver
        if driver is None:
            logger.warning("oom-breaker: recovery but no compaction driver "
                           "installed; session %r stays queued", session_id)
            return
        home = _current_home()
        with self._meta_lock:
            self._dispatch_meta[session_id] = (home, self._attribution())
        if not driver.compact(session_id, home=home):
            logger.warning("oom-breaker: compaction queue full; session %r "
                           "stays queued until the next completion signal",
                           session_id)

    # -- post_compaction (the fallback completion signal) ---------------------
    def post_compaction(self, **payload: Any) -> None:
        """Observer callback: the gateway fires it after a committed
        compress. The driver's ``on_finished`` is the PRIMARY signal; this
        covers a lost driver signal (and a compress done outside the
        driver). Only a recovery match — ``session_id == active`` or
        ``old_session_id == active`` (a compressed segment gets a new
        session id; the gateway passes both) — completes the compaction and
        chains the next queued session. Never filter on kind/reason.
        Never raises."""
        try:
            self._post_compaction(payload)
        except Exception:
            logger.warning("oom-breaker post_compaction: internal error",
                           exc_info=True)

    def _post_compaction(self, payload: dict) -> None:
        if self.breaker.state() != "recovery":
            return
        active = self.breaker.active_session()
        if active is None:
            return
        sid = payload.get("session_id")
        if not (sid == active
                or str(payload.get("old_session_id") or "") == active):
            return
        # Complete under the ACTIVE session id: the breaker only accepts a
        # completion for the session it is compacting (a segment's new id
        # differs from it). When sid == active the two forms coincide.
        remaining = self.breaker.compaction_done(active, "compressed")
        # Post like the driver's signal: meta was stashed under the ACTIVE
        # id at dispatch; a compress done outside the driver (no meta — e.g.
        # after a restart) falls back to post-time resolution.
        home, label = self._pop_dispatch_meta(active)
        self._debug_post("compacted", self._compaction_line(active, remaining),
                         home=home, label=label)
        if remaining and self.breaker.state() == "recovery":
            self._dispatch_compaction(remaining[0])

    # -- driver completion (the PRIMARY completion signal) --------------------
    def _on_compaction_finished(self, session_id: str, status: str) -> None:
        """Driver callback (runs on the driver's worker thread). Never
        raises: a callback error must not kill the driver thread."""
        try:
            self._compaction_finished(session_id, status)
        except Exception:
            logger.warning("oom-breaker compaction finished: internal error",
                           exc_info=True)

    def _compaction_finished(self, session_id: str, status: str) -> None:
        self.last_driver_status = status
        if status in ("compressed", "nothing_to_do"):
            # Only the completer that actually completes it posts: a late
            # duplicate signal finds the session already done and stays
            # silent (pop semantics, with the fallback for undriven ones).
            completing = self.breaker.active_session() == session_id
            remaining = self.breaker.compaction_done(session_id, status)
            logger.info("oom-breaker: compaction of %r -> %r; queue now %r",
                        session_id, status, remaining)
            if completing:
                if status == "compressed":
                    # "nothing_to_do" is a no-op: it never posts, but the
                    # session is done — discard the stashed meta either way.
                    home, label = self._pop_dispatch_meta(session_id)
                    self._debug_post("compacted",
                                     self._compaction_line(session_id, remaining),
                                     home=home, label=label)
                else:
                    self._discard_dispatch_meta(session_id)
            if remaining and self.breaker.state() == "recovery":
                self._dispatch_compaction(remaining[0])
        elif status == "unserviceable":
            self.breaker.compaction_done(session_id, "unserviceable")
            self._discard_dispatch_meta(session_id)
            logger.warning(
                "oom-breaker: compaction of %r refused (unserviceable: the "
                "session context exceeds the model's limit): THROTTLE + "
                "operator flag; the session is never re-queued", session_id)
        else:
            # lock_skipped / error:<msg>: the attempt produced no committed
            # result; the session STAYS active (no auto-retry) — bounded by
            # the escalation window: the next 5xx -> THROTTLE.
            logger.warning(
                "oom-breaker: compaction of %r ended %r; staying in RECOVERY "
                "(the escalation window bounds it; oom_status surfaces it)",
                session_id, status)

    def _compaction_line(self, session_id: str, remaining: List[str]) -> str:
        """The one-line completion notice shared by both completion signals."""
        if remaining:
            return f"session {session_id} compacted; queue now " \
                   f"{len(remaining)} pending"
        return f"session {session_id} compacted; queue now empty"

    # -- operator surface ------------------------------------------------------
    def oom_status(self) -> dict:
        """Operator/tool surface: breaker state, per-target L and in-flight,
        the last hit, the throttle deadline, unserviceable flags, and the
        compaction position. Synchronous; safe from any thread."""
        state = self.breaker.state()      # may auto-expire an expired THROTTLE
        blob = self.store.get("breaker", {}) or {}
        return {
            "state": state,
            "L": {name: self.learner.current(name, self.targets)
                  for name in self.targets},
            "in_flight": {name: self.ledger.in_flight(name)
                          for name in self.targets},
            "last_hit": blob.get("last_hit"),
            "throttle_until": blob.get("throttle_until"),
            "unserviceable": self.breaker.unserviceable_ids(),
            "compaction": {
                "active": self.breaker.active_session(),
                "queue": self.breaker.pending_queue(),
                "last_driver_status": self.last_driver_status,
            },
        }

    def oom_reset(self, target: str) -> dict:
        """Operator/tool surface: restore a target's learned L to its seed
        (clears the escalation window). Unknown target -> raises ValueError
        (the original exception stays visible via the exc_info warning
        log)."""
        try:
            new_l = self.learner.reset(target)
        except Exception:
            logger.warning("oom-breaker oom_reset: unknown target %r",
                           target, exc_info=True)
            raise ValueError(f"unknown target: {target!r}") from None
        logger.info("oom-breaker oom_reset: target %r L back to %d",
                    target, new_l)
        return {"reset": target, "l": new_l}
