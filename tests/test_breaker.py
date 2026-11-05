from breaker import Breaker
from config import Config, Target
from learner import Learner
from state import Store
from test_state import DictSink


def make(t, **kw):
    cfg = Config()
    store = Store(DictSink())
    lr = Learner(lambda: t[0], store)
    lr.seed("a", Target("a", 245760, 254723, ["http://a.test:9999"], ""))
    return Breaker(cfg, store, lr, lambda: t[0])


def test_first_hit_starts_recovery():
    t = [0.0]
    b = make(t)
    assert b.hit("a", 250000, ["s1", "s2"], 500, t[0]) == "recovery"


def test_burst_within_dedupe_is_one_hit():
    t = [0.0]
    b = make(t)
    b.hit("a", 250000, ["s1"], 500, t[0])
    assert b.hit("a", 250000, ["s2"], 502, t[0] + 5) == "recovery"  # same hit
    assert b.state() == "recovery"


def test_second_hit_within_window_throttles_flat_30min():
    t = [0.0]
    b = make(t)
    b.hit("a", 250000, ["s1"], 500, t[0])
    t[0] = 600  # 10 min later — inside the 15-min escalation window
    assert b.hit("a", 200000, ["s3"], 500, t[0]) == "throttle"
    assert b.throttle_active(600 + 1799) and not b.throttle_active(600 + 1801)


def test_third_hit_after_window_is_plain_recovery():
    t = [0.0]
    b = make(t)
    b.hit("a", 250000, ["s1"], 500, t[0])
    t[0] = 901  # outside the 15-min window
    assert b.hit("a", 200000, ["s3"], 500, t[0]) == "recovery"


def test_compaction_strictly_one_at_a_time():
    t = [0.0]
    b = make(t)
    b.hit("a", 250000, ["s1", "s2", "s3"], 500, t[0])
    assert b.start_compaction("s1") is True
    assert b.start_compaction("s2") is False          # s1 still active
    assert b.compaction_done("s1", "compressed") == ["s2", "s3"]
    assert b.start_compaction("s2") is True


def test_unserviceable_400_goes_throttle_no_retry():
    t = [0.0]
    b = make(t)
    b.hit("a", 250000, ["s1", "s2"], 500, t[0])
    b.start_compaction("s1")
    assert b.compaction_done("s1", "unserviceable") == []
    assert b.state() == "throttle"
    assert "s1" in b.unserviceable_ids()
    assert b.start_compaction("s1") is False          # never re-queued


def test_recovery_state_survives_restart():
    t = [0.0]
    cfg = Config()
    store = Store(DictSink())
    lr = Learner(lambda: t[0], store)
    lr.seed("a", Target("a", 245760, 254723, ["http://a.test:9999"], ""))
    b1 = Breaker(cfg, store, lr, lambda: t[0])
    b1.hit("a", 250000, ["s1"], 500, t[0])
    b2 = Breaker(cfg, store, lr, lambda: t[0])        # fresh process, same store
    assert b2.state() == "recovery"
    assert b2.start_compaction("s1") is True


def test_holds_big_only_in_recovery_for_big_contexts():
    t = [0.0]
    b = make(t)
    assert b.holds_big("s1", 60000) is False          # normal: nothing held
    b.hit("a", 250000, ["s1", "s2"], 500, t[0])
    assert b.holds_big("s1", 60000) is True           # recovery + above compact_above
    assert b.holds_big("s1", 40000) is False          # small: just retries


def test_pending_queue_and_active_session_track_recovery():
    t = [0.0]
    b = make(t)
    assert b.pending_queue() == []                   # normal: nothing pending
    assert b.active_session() is None
    b.hit("a", 250000, ["s1", "s2", "s3"], 500, t[0])
    assert b.pending_queue() == ["s1", "s2", "s3"]
    assert b.active_session() is None
    assert b.start_compaction("s1") is True
    assert b.active_session() == "s1"
    assert b.compaction_done("s1", "compressed") == ["s2", "s3"]
    assert b.active_session() is None
    assert b.pending_queue() == ["s2", "s3"]
    assert b.start_compaction("s2") is True
    assert b.compaction_done("s2", "compressed") == ["s3"]
    assert b.start_compaction("s3") is True
    assert b.compaction_done("s3", "nothing_to_do") == []
    assert b.state() == "normal"
    assert b.pending_queue() == []
    assert b.active_session() is None


def test_pending_queue_and_active_session_empty_outside_recovery():
    t = [0.0]
    b = make(t)
    b.hit("a", 250000, ["s1"], 500, t[0])
    t[0] = 600                                       # inside the escalate window
    assert b.hit("a", 200000, ["s3"], 500, t[0]) == "throttle"
    assert b.pending_queue() == []
    assert b.active_session() is None
