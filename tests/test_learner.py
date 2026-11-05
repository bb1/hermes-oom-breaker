from config import Target
from learner import Learner
from state import Store
from test_state import DictSink


def make():
    t = [0.0]
    store = Store(DictSink())
    lr = Learner(lambda: t[0], store)
    lr.seed("a", Target("a", 245760, 254723, ["http://a.test:1"], ""))
    return t, lr


def test_seed_is_maxlen_capped_at_budget():
    t, lr = make()
    assert lr.current("a", {}) == 245760
    t2 = [0.0]
    lr2 = Learner(lambda: t2[0], Store(DictSink()))
    lr2.seed("b", Target("b", 150000, 211855, ["http://b.test:1"], ""))
    assert lr2.current("b", {}) == 150000


def test_seed_capped_at_budget_when_maxlen_exceeds():
    t = [0.0]
    lr = Learner(lambda: t[0], Store(DictSink()))
    lr.seed("c", Target("c", 300000, 200000, ["http://c.test:1"], ""))
    assert lr.current("c", {}) == 200000


def test_hit_lowers_below_observed_minus_epsilon():
    t, lr = make()
    lr.on_hit("a", 250000)           # observed in-flight at hit (workspace edge, just past L)
    assert lr.current("a", {}) == 250000 - 5000
    assert lr.within_escalation_window(1.0)


def test_hit_never_raises_l():
    t, lr = make()
    lr.on_hit("a", 999999)           # min rule: a hit only tightens
    assert lr.current("a", {}) == 245760


def test_clean_episode_peak_raises():
    t, lr = make()
    assert lr.on_episode_peak("a", 300000) == 245760   # above budget cap -> ignored
    assert lr.on_episode_peak("a", 254000) == 254000    # at/below cap -> raised


def test_hold_escape_admit_raises_same_rule():
    t, lr = make()
    assert lr.on_hold_escape_admit("a", 250000) == 250000
    assert lr.on_hold_escape_admit("a", 1000) == 250000  # below L -> no change


def test_reset_restores_seed():
    t, lr = make()
    lr.on_hit("a", 100000)
    assert lr.current("a", {}) == 95000
    assert lr.reset("a") == 245760
    assert not lr.within_escalation_window(t[0] + 1)


def test_reprobe_off_by_default_and_on_fires_once():
    t, lr = make()
    lr.on_hit("a", 100000)            # L: 245760 -> 95000
    t[0] = 90_000                     # > 24 h clean
    assert lr.maybe_reprobe("a", t[0], reprobe=False) is None
    assert lr.maybe_reprobe("a", t[0], reprobe=True) == 100000  # L + eps, back to observed
    assert lr.maybe_reprobe("a", t[0] + 10, reprobe=True) is None  # once per hit
    assert lr.current("a", {}) == 100000
