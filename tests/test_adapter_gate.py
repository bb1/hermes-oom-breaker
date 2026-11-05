import time
from types import SimpleNamespace

from config import Config, Target
from state import Store
from test_state import DictSink
from adapter import GateController


def make_gate(**kw):
    cfg = Config(); store = Store(DictSink())
    cfg.targets = {"brain": Target("brain", 245760, 254723, ["http://brain.test:9999"], "")}
    return GateController(cfg, store, **kw)

CTX = {"session_id": "s1", "model": "brain-model", "provider": "openai",
       "base_url": "http://brain.test:9999/v1", "api_request_id": "r1", "api_mode": "chat",
       "task_id": "t1", "turn_id": "u1", "platform": "cli", "api_call_count": 1}

def test_untracked_provider_passes_through():
    g = make_gate(); calls = []
    out = g.llm_execution({"messages": [], "max_tokens": 10, "model": "other"},
                          lambda r: calls.append(r) or "ok",
                          **{**CTX, "base_url": "http://elsewhere:9"})
    assert out == "ok" and calls and g.ledger.in_flight_total() == 0

def test_fit_admits_and_releases_with_usage():
    g = make_gate(); calls = []
    resp = SimpleNamespace(usage=SimpleNamespace(completion_tokens=42))
    out = g.llm_execution({"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 40000},
                          lambda r: calls.append(r) or resp, **CTX)
    assert out is resp and len(calls) == 1
    assert g.ledger.in_flight_total() == 0     # released with generated=42

def test_second_big_request_held_until_first_completes():
    import threading
    t2 = [0.0]   # virtual clock, advanced by the injected sleep
    g = make_gate(now=lambda: t2[0], sleep=lambda s: t2.__setitem__(0, t2[0] + s))
    release_r1 = threading.Event()
    def slow_first(r):
        release_r1.wait(timeout=30)
        return SimpleNamespace(usage=SimpleNamespace(completion_tokens=1))
    th = threading.Thread(target=lambda: g.llm_execution(
        {"messages": [{"content": "x" * 700000}], "max_tokens": 40000},
        slow_first, **{**CTX, "api_request_id": "r1"}))
    th.start()
    while g.ledger.in_flight("brain") == 0:      # wait for r1 admission
        time.sleep(0.01)
    out = None
    def run_r2():
        nonlocal out
        out = g.llm_execution({"messages": [{"content": "x" * 700000}], "max_tokens": 40000},
                              lambda r: "second", **{**CTX, "api_request_id": "r2"})
    start = t2[0]                                # before r2: its virtual race must not predate start
    th2 = threading.Thread(target=run_r2)
    th2.start()
    while t2[0] - start < 0.6:                   # virtual: let r2 hold ≥ one recheck
        time.sleep(0.005)
    release_r1.set()
    th.join(timeout=30); th2.join(timeout=30)
    assert out == "second"
    assert t2[0] - start >= 0.5                  # r2 held ≥ one recheck before firing

def test_missing_context_fields_fail_open_passthrough():
    g = make_gate(); calls = []
    ctx = {k: v for k, v in CTX.items() if k != "base_url"}   # hermes update dropped the field
    out = g.llm_execution({"messages": [], "max_tokens": 10},
                          lambda r: calls.append(r) or "ok", **ctx)
    assert out == "ok" and calls and g.ledger.in_flight_total() == 0

def test_error_releases_entry_and_reraises():
    import pytest
    g = make_gate()
    def boom(r): raise RuntimeError("500")
    with pytest.raises(RuntimeError):
        g.llm_execution({"messages": [], "max_tokens": 1000}, boom, **CTX)
    assert g.ledger.in_flight_total() == 0

def test_internal_error_fails_open():
    g = make_gate()
    g.estimator = lambda request: 1 / 0         # gate logic crashes
    calls = []
    out = g.llm_execution({"messages": [], "max_tokens": 10}, lambda r: calls.append(r) or "ok", **CTX)
    assert out == "ok" and calls            # admitted anyway, request not lost
