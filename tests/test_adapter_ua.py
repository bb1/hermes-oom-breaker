from config import Config
from state import Store
from test_state import DictSink
from adapter import GateController


def make():
    cfg = Config()
    return cfg, GateController(cfg, Store(DictSink()))


def test_ua_disabled_is_noop():
    cfg, g = make()
    assert g.ua_llm_request({"messages": []}) is None


def test_ua_profile_label_and_cron_suffix():
    cfg, g = make(); cfg.user_agent = True
    g.profile_fn = lambda: "alpha"; g.cron_fn = lambda: None
    out = g.ua_llm_request({"messages": []})
    assert out == {"request": {"messages": [], "extra_headers": {"User-Agent": "Hermes/alpha"}}}
    g.cron_fn = lambda: "nightly_brief"
    assert g.ua_llm_request({"messages": []})["request"]["extra_headers"]["User-Agent"] == \
        "Hermes/alpha:nightly_brief"


def test_ua_no_profile_falls_back_to_base():
    cfg, g = make(); cfg.user_agent = True
    g.profile_fn = lambda: None
    out = g.ua_llm_request({"messages": []})
    assert out["request"]["extra_headers"]["User-Agent"] == "Hermes"


def test_ua_preserves_existing_extra_headers_and_input():
    cfg, g = make(); cfg.user_agent = True
    g.profile_fn = lambda: "alpha"
    req = {"messages": [], "extra_headers": {"X-Foo": "bar"}}
    out = g.ua_llm_request(req)
    assert out["request"]["extra_headers"] == {"X-Foo": "bar", "User-Agent": "Hermes/alpha"}
    assert req["extra_headers"] == {"X-Foo": "bar"}     # input not mutated


def test_ua_resolution_failure_never_changes_request():
    cfg, g = make(); cfg.user_agent = True
    g.profile_fn = lambda: 1 / 0            # resolver crashes
    assert g.ua_llm_request({"messages": []}) is None
