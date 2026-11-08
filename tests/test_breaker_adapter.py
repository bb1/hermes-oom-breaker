"""Task 7: the breaker's hook half on GateController.

Covers the 11 binding behaviors: canonical 5xx hits via ``api_request_error``,
driver completion as the primary completion signal, ``post_compaction`` as
the fallback signal (both session-id and old_session_id matches), the
1-at-a-time sequence, escalation, the never-raise observer contract, the
Task 6 defensive-fallback flag, and the oom_status/oom_reset operator
surface.

The driver is the ``FakeDriver`` double (no threads): tests control when a
compaction finishes via ``driver.finish``. Payload dicts use hermes'
verified hook field names (Task 7 brief, section A).
"""
import pytest
import time
from types import SimpleNamespace

from config import Config, Target
from state import Store
from test_state import DictSink
from adapter import GateController
from compaction_driver import FakeDriver

# Virtual clock, driven by the injected now= in make().
NOW = [0.0]

# Verified post_api_request / api_request_error field names (brief A.1/A.2).
CTX = {"session_id": "s1", "model": "brain-model", "provider": "openai",
       "base_url": "http://brain.test:9999/v1", "api_request_id": "r1",
       "api_mode": "chat", "task_id": "t1", "turn_id": "u1", "platform": "cli",
       "api_call_count": 1}


def make():
    cfg = Config()
    store = Store(DictSink())
    cfg.targets = {"brain": Target("brain", 245760, 254723,
                                   ["http://brain.test:9999"], "")}
    g = GateController(cfg, store, now=lambda: NOW[0])
    g.load_targets()          # Task 8's register() does this before hooks fire
    driver = FakeDriver()
    g.install_breaker_hooks(driver)
    return g, driver


def fire(g, status=500, session_id="s1", **extra):
    """Fire one api_request_error payload at the controller."""
    g.api_request_error(status_code=status, session_id=session_id,
                        model="brain-model", provider="openai",
                        base_url="http://brain.test:9999/v1", retry_count=0,
                        **extra)


def test_5xx_hit_triggers_recovery_and_compaction_sequence():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    assert g.breaker.state() == "recovery"
    assert driver.queued[0] == "s1"         # compaction dispatched 1-at-a-time
    driver.finish("s1", "compressed")       # driver completion is primary
    assert g.breaker.state() == "normal"
    # The gateway's post_compaction still arrives — idempotent no-op.
    g.post_compaction(session_id="s1", old_session_id="", kind="summarize",
                      reason="manual")
    assert g.breaker.state() == "normal"


def test_unserviceable_throttles_no_retry():
    NOW[0] = 100.0
    g, driver = make()
    driver.script("s1", "unserviceable")
    fire(g)
    driver.finish("s1")                     # uses the scripted status
    assert g.breaker.state() == "throttle"
    assert driver.attempts["s1"] == 1       # never re-queued
    assert "s1" in g.breaker.unserviceable_ids()


def test_4xx_and_nonhttp_are_not_hits():
    NOW[0] = 100.0
    g, driver = make()
    fire(g, status=400)
    g.api_request_error(status_code=None, session_id="s1", model="brain-model",
                        base_url="http://brain.test:9999/v1", provider="openai",
                        retry_count=0)      # non-HTTP error (e.g. timeout)
    assert g.breaker.state() == "normal"
    assert driver.queued == []


def test_post_compaction_completes_when_driver_signal_lost():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    assert g.breaker.active_session() == "s1"
    # No driver.finish — the driver signal was lost; the gateway hook still
    # fires after the committed compress.
    g.post_compaction(session_id="s1", old_session_id="", kind="summarize",
                      reason="manual")
    assert g.breaker.state() == "normal"


def test_post_compaction_segment_id_via_old_session():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    # A compressed segment gets a new session id; the gateway passes both.
    g.post_compaction(session_id="s1-seg", old_session_id="s1",
                      kind="summarize", reason="manual")
    assert g.breaker.state() == "normal"


def test_post_compaction_unrelated_session_noop():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    g.post_compaction(session_id="zzz", old_session_id="")
    assert g.breaker.state() == "recovery"
    assert g.breaker.active_session() == "s1"


def test_driver_error_keeps_recovery():
    NOW[0] = 100.0
    g, driver = make()
    driver.script("s1", "error:connect")
    fire(g)
    driver.finish("s1")
    assert g.breaker.state() == "recovery"  # no auto-retry; the window bounds
    assert g.breaker.active_session() == "s1"


def test_second_hit_inside_window_escalates():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    assert g.breaker.state() == "recovery"
    # 60s later: past hit_dedupe (10s), inside escalate_window (900s).
    NOW[0] = 160.0
    fire(g, session_id="s2", api_request_id="r2")
    assert g.breaker.state() == "throttle"


def test_oom_status_shape():
    NOW[0] = 100.0
    g, driver = make()
    s = g.oom_status()
    assert s["state"] == "normal"
    assert s["L"] == {"brain": 245760}
    assert s["in_flight"] == {"brain": 0}
    assert s["last_hit"] is None
    assert s["throttle_until"] is None
    assert s["unserviceable"] == []
    assert s["compaction"] == {"active": None, "queue": [],
                               "last_driver_status": None}


def test_oom_reset():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)                                   # hit lowered L (in-flight 0)
    assert g.learner.current("brain", g.targets) < 245760
    assert g.oom_reset("brain") == {"reset": "brain", "l": 245760}
    assert g.learner.current("brain", g.targets) == 245760


def test_oom_reset_unknown_target():
    NOW[0] = 100.0
    g, driver = make()
    with pytest.raises(ValueError):
        g.oom_reset("nope")


def test_streaming_error_releases_ledger_entry():
    # Review Focus #2 (the canonical path, host-verified payload): a stream
    # that dies after first delta must not leak its ledger entry. The real
    # api_request_error payload carries api_request_id/session_id/model/
    # base_url and status_code (None for non-HTTP failures); post_api_request
    # is success-intake only in production, so release-on-error lives HERE.
    import itertools
    NOW[0] = 100.0
    g, driver = make()
    stream = (c for c in itertools.chain(
        ["a", "b"], [SimpleNamespace(usage=SimpleNamespace(completion_tokens=7))]))
    g.llm_execution({"messages": [{"content": "x" * 4000}], "max_tokens": 40000},
                    lambda r: stream, **{**CTX, "api_request_id": "r9"})
    assert g.ledger.in_flight("brain") > 0    # streaming: upper bound held
    g.api_request_error(status_code=500, api_request_id="r9", session_id="s1",
                        model="brain-model", base_url="http://brain.test:9999/v1",
                        provider="openai", retry_count=0)
    assert g.ledger.in_flight("brain") == 0   # canonical path releases the entry
    assert g.breaker.state() == "recovery"    # and the 5xx is a hit
    assert g.learner.current("brain", g.targets) < 245760  # L learned down


def test_nonhttp_error_releases_without_hit():
    # Connection error / timeout: status_code is None (host-verified). The
    # dead stream's entry must still be released, but a non-HTTP failure is
    # NOT an OOM data point: no L change, no recovery.
    NOW[0] = 100.0
    g, driver = make()
    g.llm_execution({"messages": [{"content": "x" * 4000}], "max_tokens": 40000},
                    lambda r: iter(["chunk"]), **{**CTX, "api_request_id": "rx"})
    assert g.ledger.in_flight("brain") > 0
    g.api_request_error(status_code=None, api_request_id="rx", session_id="s1",
                        model="brain-model", base_url="http://brain.test:9999/v1",
                        provider="openai", error={"type": "connection",
                                                  "message": "refused"})
    assert g.ledger.in_flight("brain") == 0    # the entry was released
    assert g.breaker.state() == "normal"       # but no hit
    assert g.learner.current("brain", g.targets) == 245760  # L untouched
    assert driver.queued == []


def test_error_release_falls_back_to_session():
    # The hook's api_request_id can differ from the gate's (a hermes retry or
    # a renamed field): the session-scoped fallback still releases the live
    # entry.
    NOW[0] = 100.0
    g, driver = make()
    g.llm_execution({"messages": [{"content": "x" * 4000}], "max_tokens": 40000},
                    lambda r: iter(["chunk"]), **{**CTX, "api_request_id": "gate-id"})
    assert g.ledger.in_flight("brain") > 0
    fire(g, api_request_id="other-id")        # mismatched request id
    assert g.ledger.in_flight("brain") == 0    # released via the session
    assert g.breaker.state() == "recovery"


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for driver event")
        time.sleep(0.01)


def test_gateway_driver_one_at_a_time_and_crash_safe():
    """Crash-safety constraints: 1-at-a-time queue, start() idempotent, a
    worker exception never kills the thread (status degrades to
    error:<ExcType>), a bad on_finished never kills it either, and stop()
    is bounded (~2×timeout worst case)."""
    from compaction_driver import GatewayDriver
    events = []
    d = GatewayDriver()
    d.on_finished = lambda sid, status: events.append((sid, status))
    # Queue is size 1: the second enqueue is rejected while the first is
    # queued (worker not started — deterministic full).
    assert d.compact("s1") is True
    assert d.compact("s2") is False
    assert d.in_flight is True
    d._attempt = lambda sid, home: "compressed"    # bypass the RPC seam
    d.start()
    d.start()                                      # idempotent
    _wait(lambda: len(events) >= 1)
    assert events[0] == ("s1", "compressed")
    # A worker exception degrades to error:<ExcType>; the thread survives.
    def boom(sid, home):
        raise RuntimeError("kaput")
    d._attempt = boom
    d.compact("s3")
    _wait(lambda: len(events) >= 2)
    assert events[1] == ("s3", "error:RuntimeError")
    # The worker still processes after the crash.
    d._attempt = lambda sid, home: "nothing_to_do"
    d.compact("s4")
    _wait(lambda: len(events) >= 3)
    assert events[2] == ("s4", "nothing_to_do")
    # A raising on_finished is logged, never escapes the thread.
    d.on_finished = lambda sid, status: 1 / 0
    d.compact("s5")
    _wait(lambda: d.in_flight is False)
    d.on_finished = lambda sid, status: events.append((sid, status))
    d._attempt = lambda sid, home: "compressed"
    d.compact("s6")
    _wait(lambda: len(events) >= 4)
    assert events[3] == ("s6", "compressed")
    d.stop()
    d.stop()                                      # idempotent
    assert d._thread is None


def test_flag_prevents_fallback_double_hit():
    NOW[0] = 100.0
    g, driver = make()
    assert g.api_error_hook_registered is True   # install_breaker_hooks set it
    # A tracked streaming request so the synthetic 5xx has a real entry to
    # release.
    g.llm_execution({"messages": [{"content": "x" * 4000}], "max_tokens": 40000},
                    lambda r: iter(["chunk"]), **CTX)
    assert g.ledger.in_flight("brain") > 0
    NOW[0] = 105.0
    fire(g)                       # the canonical 5xx hook
    assert g.breaker.state() == "recovery"
    ts = g.learner.last_hit_ts()
    assert ts == 105.0
    # The real post_api_request payload has no status_code; this synthetic
    # one pins the Task 6 defensive-fallback contract: release the ledger
    # entry, but — because the canonical hook is registered — NO second
    # breaker hit. Without the flag this lands 15s after the first hit:
    # past hit_dedupe, inside the escalate window -> a spurious THROTTLE.
    NOW[0] = 120.0
    g.post_api_request(status_code=500, api_request_id="r1", session_id="s1",
                       model="brain-model", base_url="http://brain.test:9999/v1")
    assert g.ledger.in_flight("brain") == 0     # the entry was released
    assert g.breaker.state() == "recovery"      # not escalated to throttle
    assert g.learner.last_hit_ts() == ts        # no second hit
