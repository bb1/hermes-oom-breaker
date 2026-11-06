import logging

import pytest

from config import Config
from notify import Notifier, _load_env


def mk_cfg(**over) -> Config:
    base = {"debug": True, "debug_min_interval": 60.0}
    base.update(over)
    return Config(**base)


class FakeSender:
    """A recording sender (injected as ``sender``): returns a fixed result
    or raises, and remembers every call as (token, chat, thread, text)."""

    def __init__(self, result: bool = True, raise_exc: bool = False):
        self.result = result
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, token, chat, thread, text):
        self.calls.append((token, chat, thread, text))
        if self.raise_exc:
            raise RuntimeError("sender exploded")
        return self.result


def test_debug_off_is_total_noop():
    """debug off: post() returns before touching the clock, the spawner,
    or the sender — even when a home is given."""
    NOW = [0.0]
    clock_calls = []
    spawned = []
    sender = FakeSender()

    def now():
        clock_calls.append(1)
        return NOW[0]

    n = Notifier(mk_cfg(debug=False), now=now,
                 home_fn=lambda: "/h",
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok",
                                       "TELEGRAM_HOME_CHANNEL": "111"},
                 sender=sender,
                 spawner=lambda fn: spawned.append(fn))
    n.post("5xx", "one line", home="/h")
    assert clock_calls == []
    assert spawned == []
    assert sender.calls == []


def test_env_home_channel_posted():
    """home given + env with token and home channel -> one send to that
    channel (thread None)."""
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok",
                                       "TELEGRAM_HOME_CHANNEL": "111"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line", home="/h")
    assert sender.calls == [("tok", "111", None, "one line")]


def test_debug_channel_wins_over_env():
    """A set debug_channel pins chat+thread regardless of the env (the
    token still comes from the profile's .env)."""
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(debug_channel="telegram:123:45"), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line", home="/h")
    assert sender.calls == [("tok", "123", "45", "one line")]


def test_debug_channel_trailing_colon_has_no_thread():
    """``telegram:123:`` parses as chat 123 with NO thread (an empty thread
    normalizes to None, not an empty string)."""
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(debug_channel="telegram:123:"), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line", home="/h")
    assert sender.calls == [("tok", "123", None, "one line")]


@pytest.mark.parametrize("channel", ["telegram:", "slack:1", "123"])
def test_invalid_debug_channel_never_sends(channel):
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(debug_channel=channel), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line", home="/h")
    assert sender.calls == []


def test_missing_token_never_sends():
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_HOME_CHANNEL": "111"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line", home="/h")
    assert sender.calls == []


def test_missing_home_channel_never_sends():
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line", home="/h")
    assert sender.calls == []


def test_no_home_never_sends(caplog):
    """home=None and home_fn() -> None: dropped, with a loud warning that
    names the event."""
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0], home_fn=lambda: None,
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok",
                                       "TELEGRAM_HOME_CHANNEL": "111"},
                 sender=sender, spawner=lambda fn: fn())
    with caplog.at_level(logging.WARNING, logger="oom_breaker"):
        n.post("5xx", "one line")
    assert sender.calls == []
    assert any("5xx" in r.message for r in caplog.records
               if r.levelno == logging.WARNING)


def test_home_falls_back_to_home_fn():
    """home=None but home_fn() returns a home: that home's .env is used."""
    NOW = [0.0]
    seen = []

    def env_loader(h):
        seen.append(h)
        return {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_HOME_CHANNEL": "222"}

    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0], home_fn=lambda: "/profile/home",
                 env_loader=env_loader, sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "one line")
    assert seen == ["/profile/home"]
    assert sender.calls == [("tok", "222", None, "one line")]


def test_rate_limit_drops_within_interval():
    """Per-event rate limit: t=0 sends, t=59 is a silent drop, t=60 sends
    again (exactly min_interval is allowed)."""
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok",
                                       "TELEGRAM_HOME_CHANNEL": "111"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "a", home="/h")
    NOW[0] = 59.0
    n.post("5xx", "b", home="/h")
    NOW[0] = 60.0
    n.post("5xx", "c", home="/h")
    assert [c[3] for c in sender.calls] == ["a", "c"]


def test_rate_limit_is_per_event():
    """Buckets are per event type: a different event is not rate-limited by
    another event's post."""
    NOW = [0.0]
    sender = FakeSender()
    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=lambda h: {"TELEGRAM_BOT_TOKEN": "tok",
                                       "TELEGRAM_HOME_CHANNEL": "111"},
                 sender=sender, spawner=lambda fn: fn())
    n.post("5xx", "a", home="/h")
    NOW[0] = 5.0
    n.post("compacted", "b", home="/h")
    assert [c[3] for c in sender.calls] == ["a", "b"]


def test_sender_false_or_raising_never_raises_out_of_post(caplog):
    """A sender that returns False or raises is contained inside the
    spawned work: post() returns normally either way (loud warning; the
    raising one carries exc_info)."""
    NOW = [0.0]
    env = {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_HOME_CHANNEL": "111"}
    s_false, s_raise = FakeSender(result=False), FakeSender(raise_exc=True)
    with caplog.at_level(logging.WARNING, logger="oom_breaker"):
        for s in (s_false, s_raise):
            n = Notifier(mk_cfg(), now=lambda: NOW[0],
                         env_loader=lambda h: dict(env),
                         sender=s, spawner=lambda fn: fn())
            n.post("5xx", "one line", home="/h")   # must not raise
    assert s_false.calls == [("tok", "111", None, "one line")]
    assert s_raise.calls == [("tok", "111", None, "one line")]
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not delivered" in m for m in msgs)
    assert any(r.exc_info for r in caplog.records
               if r.levelno == logging.WARNING)


def test_raising_env_loader_never_raises_out_of_post(caplog):
    """A failure in any resolution step (here: the env loader) is contained
    by the spawned work's catch-all: post() returns normally with a loud
    warning; nothing is sent."""
    NOW = [0.0]
    sender = FakeSender()

    def boom(h):
        raise OSError("env unreadable")

    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=boom, sender=sender, spawner=lambda fn: fn())
    with caplog.at_level(logging.WARNING, logger="oom_breaker"):
        n.post("5xx", "one line", home="/h")   # must not raise
    assert sender.calls == []
    assert any(r.exc_info for r in caplog.records
               if r.levelno == logging.WARNING)


def test_real_work_defers_into_the_spawned_callable():
    """post() itself only rate-limits and hands off: no env read and no send
    happen until the spawner actually runs the work callable (no I/O on the
    caller's thread)."""
    NOW = [0.0]
    work = []
    env_reads = []
    sender = FakeSender()

    def env_loader(h):
        env_reads.append(h)
        return {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_HOME_CHANNEL": "111"}

    n = Notifier(mk_cfg(), now=lambda: NOW[0],
                 env_loader=env_loader, sender=sender,
                 spawner=lambda fn: work.append(fn))
    n.post("5xx", "one line", home="/h")
    assert len(work) == 1
    assert env_reads == [] and sender.calls == []
    work[0]()
    assert env_reads == ["/h"]
    assert sender.calls == [("tok", "111", None, "one line")]


def test_load_env_parses_plain_quoted_comment_blank(tmp_path):
    (tmp_path / ".env").write_text(
        "# bot settings\n"
        "\n"
        "TELEGRAM_BOT_TOKEN=plain-token\n"
        "TELEGRAM_HOME_CHANNEL='111'\n",
        encoding="utf-8")
    assert _load_env(str(tmp_path)) == {
        "TELEGRAM_BOT_TOKEN": "plain-token",
        "TELEGRAM_HOME_CHANNEL": "111"}


def test_load_env_missing_file_returns_empty(tmp_path):
    assert _load_env(str(tmp_path / "nope")) == {}
