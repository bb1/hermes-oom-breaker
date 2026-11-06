"""The three debug post sites on GateController (opt-in, cfg.debug).

Site A: the canonical 5xx hook (one post per burst; recovery names the
session being dispatched, throttle does not). Site B: compaction-completed
posts ("compacted") — only for a committed "compressed" (never
"nothing_to_do"), via the driver's PRIMARY signal and the post_compaction
FALLBACK signal; the home/label are stashed at dispatch time (the driver's
worker thread has no per-turn context). Site C: the gate's two escape
admissions ("delayed": max_hold deadlock guard, hold_escape) — a normal fit
admission stays silent. Plus the contracts: the defensive 5xx fallback in
post_api_request never posts, a raising notifier never propagates (observer
contract), and double completion signals post exactly once.

Harness: a minimal copy of test_breaker_adapter.py's make()/fire(), with the
shared DictSink double (the one cross-test import, as test_breaker_adapter.py
does) and a recording FakeNotifier injected through the controller's
notifier= parameter.
"""
import pytest
from types import SimpleNamespace

import adapter
from adapter import GateController
from compaction_driver import FakeDriver
from config import Config, Target
from notify import Notifier
from state import Store
from test_state import DictSink

# Virtual clock, advanced by the injected now=/sleep= (the pattern from
# test_adapter_gate.py's held-request test).
NOW = [0.0]

# Verified hook field names (same as test_breaker_adapter.py).
CTX = {"session_id": "s1", "model": "brain-model", "provider": "openai",
       "base_url": "http://brain.test:9999/v1", "api_request_id": "r1",
       "api_mode": "chat", "task_id": "t1", "turn_id": "u1", "platform": "cli",
       "api_call_count": 1}


class FakeNotifier:
    """Recording notifier double: remembers (event, text, home) tuples;
    optionally raises, to prove the observer contract survives it."""

    def __init__(self, raise_exc: bool = False):
        self.posts = []
        self.raise_exc = raise_exc

    def post(self, event, text, *, home=None):
        if self.raise_exc:
            raise RuntimeError("notifier exploded")
        self.posts.append((event, text, home))


def make(cfg=None, notifier=None, sleep=None):
    """A GateController with the standard brain target and a FakeDriver,
    like test_breaker_adapter.make() — plus an injected FakeNotifier (or a
    caller-supplied cfg/notifier/sleep)."""
    if cfg is None:
        cfg = Config()
    store = Store(DictSink())
    cfg.targets = {"brain": Target("brain", 245760, 254723,
                                   ["http://brain.test:9999"], "")}
    if sleep is None:
        sleep = lambda s: NOW.__setitem__(0, NOW[0] + s)  # virtual: never blocks
    g = GateController(cfg, store, now=lambda: NOW[0], sleep=sleep,
                       notifier=notifier if notifier is not None
                       else FakeNotifier())
    g.load_targets()
    driver = FakeDriver()
    g.install_breaker_hooks(driver)
    return g, driver


def fire(g, status=500, session_id="s1", **extra):
    """Fire one api_request_error payload at the controller."""
    g.api_request_error(status_code=status, session_id=session_id,
                        model="brain-model", provider="openai",
                        base_url="http://brain.test:9999/v1", retry_count=0,
                        **extra)


# -- Site A: the canonical 5xx -----------------------------------------------

def test_5xx_posts_once_with_event_state_and_target():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    assert len(g._notifier.posts) == 1
    event, text, home = g._notifier.posts[0]
    assert event == "5xx"
    assert text.startswith("[oom-breaker] ")
    assert "500" in text
    assert "'brain'" in text
    assert "-> recovery" in text
    assert home is None      # _current_home() is None in a hermes-free test


@pytest.mark.parametrize("profile,job,want_prefix", [
    ("prof", None, "[oom-breaker] (prof) 500"),
    ("prof", "job", "[oom-breaker] (prof:job) 500"),
    (None, None, "[oom-breaker] 500"),       # no label at all
])
def test_5xx_label_follows_profile_and_cron(profile, job, want_prefix):
    NOW[0] = 100.0
    g, driver = make()
    g.profile_fn = lambda: profile
    g.cron_fn = lambda: job
    fire(g)
    assert g._notifier.posts[0][1].startswith(want_prefix)


def test_second_5xx_in_dedupe_window_adds_no_post():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    NOW[0] = 105.0     # inside hit_dedupe (10s): a burst duplicate
    fire(g)
    assert [p[0] for p in g._notifier.posts] == ["5xx"]


def test_5xx_recovery_suffix_names_compacting_session():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    assert g._notifier.posts[0][1].endswith(", compacting s1")
    assert driver.queued == ["s1"]


def test_5xx_throttle_has_no_compacting_suffix():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    NOW[0] = 160.0     # past hit_dedupe (10s), inside escalate_window (900s)
    fire(g, session_id="s2", api_request_id="r2")
    event, text, _home = g._notifier.posts[1]
    assert event == "5xx"
    assert "-> throttle" in text
    assert "compacting" not in text


# -- the defensive fallback (must stay silent) -------------------------------

def test_post_api_request_5xx_fallback_posts_nothing():
    NOW[0] = 100.0
    g, driver = make()
    g.api_error_hook_registered = False     # the Task 6 hook configuration
    g.post_api_request(status_code=500, api_request_id="r1", session_id="s1",
                       model="brain-model", base_url="http://brain.test:9999/v1")
    assert g._notifier.posts == []
    assert g.breaker.state() == "recovery"   # the hit was recorded, no post


# -- Site B: compaction-completed (the stashed meta) ---------------------------

def test_compacted_post_carries_stashed_home_and_label(monkeypatch):
    NOW[0] = 100.0
    g, driver = make()
    monkeypatch.setattr(adapter, "_current_home", lambda: "/h/prof")
    g.profile_fn = lambda: "prof"
    fire(g)                                   # dispatch stashes home + label
    g.profile_fn = lambda: None               # gone by completion time
    monkeypatch.setattr(adapter, "_current_home", lambda: None)
    driver.finish("s1", "compressed")
    event, text, home = g._notifier.posts[-1]
    assert event == "compacted"
    assert home == "/h/prof"                  # stashed, not re-resolved
    assert "(prof)" in text                   # stashed, not re-resolved
    assert "session s1 compacted; queue now empty" in text


@pytest.mark.parametrize("status", ["nothing_to_do", "lock_skipped",
                                    "error:boom"])
def test_driver_non_compressed_statuses_post_nothing(status):
    NOW[0] = 100.0
    g, driver = make()
    driver.finish("s1", status)
    assert g._notifier.posts == []


def test_double_completion_posts_exactly_once():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    driver.finish("s1", "compressed")         # primary signal completes + posts
    g.post_compaction(session_id="s1", old_session_id="", kind="summarize",
                      reason="manual")        # fallback: idempotent no-op
    assert [p[0] for p in g._notifier.posts] == ["5xx", "compacted"]


def test_fallback_first_then_stale_driver_finish_posts_once():
    NOW[0] = 100.0
    g, driver = make()
    fire(g)
    g.post_compaction(session_id="s1", old_session_id="")    # fallback first
    driver.finish("s1", "compressed")       # stale driver signal: silent
    assert [p[0] for p in g._notifier.posts] == ["5xx", "compacted"]


def test_terminal_completions_discard_stashed_meta():
    """A dispatched session that completes terminally without a committed
    compress ('nothing_to_do', 'unserviceable') discards its stashed
    (home, label): no post, no stale entry. 'lock_skipped'/'error:*' keep
    it — the session stays active and a later completion may still need
    the stashed context."""
    for status, keep in [("nothing_to_do", False), ("unserviceable", False),
                         ("lock_skipped", True), ("error:boom", True)]:
        NOW[0] = 100.0
        g, driver = make()
        fire(g)                        # dispatch stashes the meta for s1
        assert "s1" in g._dispatch_meta
        driver.finish("s1", status)
        assert ("s1" in g._dispatch_meta) is keep
        assert [p[0] for p in g._notifier.posts] == ["5xx"]


# -- Site C: the hold-loop escape admissions -----------------------------------

def test_max_hold_escape_admission_posts_delayed():
    NOW[0] = 100.0
    cfg = Config(hold_escape=10_000.0)       # hold_escape must not preempt
    g, driver = make(cfg=cfg)                # sleep= advances the virtual clock
    out = g.llm_execution(
        {"messages": [{"content": "x" * 1_000_000}], "max_tokens": 40_000},
        lambda r: SimpleNamespace(usage=None),
        **{**CTX, "api_request_id": "rheld"})
    assert out is not None
    assert [p[0] for p in g._notifier.posts] == ["delayed"]
    event, text, home = g._notifier.posts[0]
    assert event == "delayed"
    assert "held " in text
    assert "(max_hold)" in text
    assert "admitted anyway" in text
    assert NOW[0] >= 100.0 + 600.0           # held past cfg.max_hold


def test_hold_escape_admission_posts_delayed():
    NOW[0] = 100.0
    g, driver = make()                       # default hold_escape=300
    g.llm_execution(
        {"messages": [{"content": "x" * 1_000_000}], "max_tokens": 40_000},
        lambda r: SimpleNamespace(usage=None),
        **{**CTX, "api_request_id": "rheld"})
    assert [p[0] for p in g._notifier.posts] == ["delayed"]
    event, text, home = g._notifier.posts[0]
    assert event == "delayed"
    assert "held 300s" in text
    assert "(hold_escape)" in text
    assert "admitted" in text
    assert NOW[0] >= 100.0 + 300.0


def test_normal_fit_admission_posts_nothing():
    NOW[0] = 100.0
    g, driver = make()
    g.llm_execution({"messages": [{"content": "x" * 4000}], "max_tokens": 4000},
                    lambda r: SimpleNamespace(usage=None),
                    **{**CTX, "api_request_id": "rfit"})
    assert g._notifier.posts == []
    assert g.ledger.in_flight("brain") == 0  # admitted, settled, silent


# -- the notifier contracts -----------------------------------------------------

def test_debug_off_spawner_never_called():
    NOW[0] = 100.0
    cfg = Config(debug=False)
    spawned, sent = [], []
    n = Notifier(cfg, now=lambda: NOW[0], home_fn=lambda: "/h",
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok",
                                       "TELEGRAM_HOME_CHANNEL": "111"},
                 sender=lambda token, chat, thread, text: sent.append(text) or True,
                 spawner=lambda fn: spawned.append(fn))
    g, driver = make(cfg=cfg, notifier=n)
    fire(g)
    assert spawned == []                     # the post never reached the spawner
    assert sent == []
    assert g.breaker.state() == "recovery"   # gate logic ran; only the post skipped


def test_raising_notifier_never_propagates():
    NOW[0] = 100.0
    g, driver = make(notifier=FakeNotifier(raise_exc=True))
    fire(g)                                  # observer contract: returns normally
    assert g.breaker.state() == "recovery"   # the hit landed despite the failure
