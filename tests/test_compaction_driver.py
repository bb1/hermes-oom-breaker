"""Task 7 fix round: M1 — direct tests for the pure status-mapping
functions, and M3 — generation-token restart behavior of GatewayDriver.

M1 covers ``_map_result``/``_map_refusal``/``_err`` directly (no driver
instance, no threads, no hermes mocks). The tests pin the *current*
implementation — GREEN by construction against the Task 7 code, and the
refusal split is validated ground truth (verified against hermes_cli
``PreparedMutations.apply`` on the deployment host, 2026-11-04): the 9
refusal strings surface ONLY as ``GatewayClientError`` exceptions — the
``_map_refusal`` path — and never as result-dict statuses; a result dict
carries only status 'nothing'/'cancelled'/'confirmation_required' (the
last loops into an exception with confirm=None) or the committed shape
(messages/summary, no status key). The original §B.4 table conflated the
two paths; the DEFENSIVE rows below pin ``_map_result``'s fallback for
dict statuses that cannot occur.

M3 drives ``GatewayDriver`` with a stubbed ``_attempt`` (a gate holds the
worker inside the attempt) to prove that a restart after an incomplete
(timed-out) stop cannot double-drain the queue.
"""
import threading
import time

import pytest

import compaction_driver as cd
from compaction_driver import GatewayDriver


# -- M1: _map_result ---------------------------------------------------------
#
# Table rows that match the implementation: committed shapes (messages
# and/or summary, no status key), the disclosed superset row (explicit
# status: None + messages/summary), the 'noop'/'nothing' no-ops, and
# non-dict results.
#
# DEFENSIVE rows: refusal strings NEVER arrive as result-dict statuses —
# verified against hermes_cli PreparedMutations.apply (deployment host,
# 2026-11-04), they surface ONLY as GatewayClientError exceptions, which
# _map_refusal maps to the §B.4 vocabulary. The original §B.4 table
# conflated the two paths; these rows pin _map_result's
# error:unexpected-result-status:<repr> fallback for dict statuses that
# cannot occur. Pinned as-implemented (values unchanged by the fix round).

def _unexpected_result_status(status):
    return f"error:unexpected-result-status:{status!r}"


@pytest.mark.parametrize("result, expected", [
    # committed: carries messages and/or summary, no status key
    ({"messages": ["m"]}, "compressed"),
    ({"summary": "s"}, "compressed"),
    ({"messages": ["m"], "summary": "s"}, "compressed"),
    # the disclosed superset row: explicit status: None + messages/summary
    ({"status": None, "messages": ["m"]}, "compressed"),
    ({"status": None, "summary": "s"}, "compressed"),
    # no-ops
    ({"status": "noop"}, "nothing_to_do"),
    ({"status": "nothing"}, "nothing_to_do"),
    # DEFENSIVE: refusal string — _map_refusal path only (see header)
    ({"status": "nothing_to_compress"},
     _unexpected_result_status("nothing_to_compress")),
    # DEFENSIVE: refusal string — _map_refusal path only (see header)
    ({"status": "session_busy"},
     _unexpected_result_status("session_busy")),
    # DEFENSIVE: unserviceable refusal strings — _map_refusal path only
    ({"status": "model_resolution_failed"},
     _unexpected_result_status("model_resolution_failed")),
    ({"status": "unsupported_compress_options"},
     _unexpected_result_status("unsupported_compress_options")),
    ({"status": "invalid_params"},
     _unexpected_result_status("invalid_params")),
    # DEFENSIVE: refusal string — _map_refusal path only (see header)
    ({"status": "permission_denied"},
     _unexpected_result_status("permission_denied")),
    # DEFENSIVE: other refusal strings — _map_refusal path only (see header)
    ({"status": "revision_conflict"},
     _unexpected_result_status("revision_conflict")),
    # a status key overrides the committed shape
    ({"messages": ["m"], "status": "nothing_to_compress"},
     _unexpected_result_status("nothing_to_compress")),
    # no status and nothing committed: not a committed shape
    ({}, _unexpected_result_status(None)),
    ({"status": None}, _unexpected_result_status(None)),
    # non-dict results (matches §B.4; verified behavior, pinned exactly)
    (None, "error:unexpected-result"),
    ("committed", "error:unexpected-result"),
    (["m"], "error:unexpected-result"),
])
def test_map_result_pins_current_behavior(result, expected):
    assert cd._map_result(result) == expected


# -- M1: _map_refusal ----------------------------------------------------------
#
# The 'session_busy' -> 'lock_skipped' logic lives here (not in _attempt):
# GatewayClientError messages are mapped directly, so it is fully testable.

@pytest.mark.parametrize("message, expected", [
    ("session_busy", "lock_skipped"),
    # the unserviceable set (breaker THROTTLEs + flags; never re-queued)
    ("model_resolution_failed", "unserviceable"),
    ("unsupported_compress_options", "unserviceable"),
    ("invalid_params", "unserviceable"),
    # surfaced, never retried (== error:<msg> for this message)
    ("permission_denied", "error:permission_denied"),
    # the nothing-to-compress refusal
    ("nothing_to_compress", "nothing_to_do"),
    # anything else -> error:<msg>
    ("revision_conflict", "error:revision_conflict"),
    ("stale_generation", "error:stale_generation"),
    ("unknown_execution", "error:unknown_execution"),
    # empty message degrades to a bare 'error:' prefix
    ("", "error:"),
])
def test_map_refusal_pins_current_behavior(message, expected):
    assert cd._map_refusal(message) == expected


# -- M1: _err -------------------------------------------------------------------

def test_err_pins_current_behavior():
    # a non-empty str(exc) is the message
    assert cd._err(RuntimeError("boom")) == "error:boom"
    # an empty str(exc) degrades to the exception type name
    assert cd._err(RuntimeError()) == "error:RuntimeError"


# -- M3: generation token — restart after an incomplete stop -------------------
#
# Workers are named "oom-breaker-compact" by start(); counting them is the
# only way to observe which generations survived (stop() clears
# driver._thread, so the restart's worker must be found via enumerate()).

def _worker_threads():
    return [t for t in threading.enumerate()
            if t.name == "oom-breaker-compact"]


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for driver event")
        time.sleep(0.01)


def test_restart_after_incomplete_stop_new_worker_processes_once():
    """M3, spec ordering: the stop join times out with the worker hung
    mid-attempt (incomplete stop). Once the old worker drains the stop
    sentinel and exits, a restart + new enqueue is processed by the NEW
    worker exactly once — 1-at-a-time survives the incomplete-stop restart.

    (The gate is released before the restart so the old worker can drain
    the sentinel the incomplete stop left in the queue; otherwise the slot
    stays full and the new enqueue cannot land.)
    """
    gate = threading.Event()      # holds the worker inside the stub attempt
    events = []
    d = GatewayDriver()
    d.on_finished = lambda sid, status: events.append((sid, status))

    def attempt(sid, home):
        gate.wait()
        return "compressed"
    d._attempt = attempt          # bypass the RPC seam (as in the Task 7 tests)

    d.start()
    assert d.compact("s1") is True
    _wait(lambda: not d.in_flight)        # picked up; now blocked in the stub

    d.stop(timeout=0.05)                  # incomplete: the join times out
    assert len(_worker_threads()) == 1     # the old worker is still alive

    gate.set()                             # the old worker finishes s1 ...
    _wait(lambda: len(events) >= 1)
    assert events[0] == ("s1", "compressed")
    _wait(lambda: len(_worker_threads()) == 0)   # ... then drains the sentinel

    d.start()                              # restart: a new worker spawns
    _wait(lambda: len(_worker_threads()) == 1)
    assert d.compact("s2") is True
    _wait(lambda: len(events) >= 2)
    assert events == [("s1", "compressed"),
                      ("s2", "compressed")]  # s2 finished exactly once

    d.stop()
    _wait(lambda: len(_worker_threads()) == 0)


def test_stale_worker_drops_and_exits_after_incomplete_stop():
    """M3, token property: the worker that survived the timed-out stop is
    stale — on its next pickup it self-terminates BEFORE _attempt: the item
    is dropped (no attempt, no on_finished) and the worker exits, so a
    restart cannot double-drain the queue.

    Note: the stop sentinel left behind by the incomplete stop is consumed
    by the NEW worker, which therefore exits too; the dropped item must be
    re-enqueued by a later operator action (see the fix-round report).
    """
    gate = threading.Event()
    events = []
    attempts = []
    d = GatewayDriver()
    d.on_finished = lambda sid, status: events.append((sid, status))

    def attempt(sid, home):
        attempts.append(sid)
        gate.wait()
        return "compressed"
    d._attempt = attempt

    d.start()
    assert d.compact("s1") is True
    _wait(lambda: not d.in_flight)        # old worker picked it up, blocked

    d.stop(timeout=0.05)                  # incomplete: old worker alive
    assert len(_worker_threads()) == 1
    d.start()                             # the new worker spawns ...
    _wait(lambda: d._thread is not None
          and not d._thread.is_alive())   # ... and exits on the stop
    # sentinel the incomplete stop left in the queue (see docstring)
    assert len(_worker_threads()) == 1     # ... while the old worker lives

    gate.set()
    _wait(lambda: len(events) >= 1)
    assert events[0] == ("s1", "compressed")

    assert d.compact("s2") is True         # the slot is free
    _wait(lambda: len(_worker_threads()) == 0)   # the stale worker dropped s2
    # The stale worker self-terminated on pickup: s2 was never attempted
    # and never finished.
    assert events == [("s1", "compressed")]
    assert attempts == ["s1"]
    assert d.in_flight is False            # the dropped item is gone
    d.stop()
