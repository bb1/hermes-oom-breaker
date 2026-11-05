from config import Target
from ledger import Ledger, target_of


def make_ledger(L=254723):
    t = [0.0]
    lg = Ledger(lambda: t[0])
    targets = {"brain": Target("brain", 245760, L, ["http://brain.test:9999"], "")}
    return t, lg, targets


def test_solo_maxlen_admitted_on_idle():
    t, lg, targets = make_ledger()
    ok, why = lg.admit("r1", "s1", "brain", 245760, 254723, False, None)
    assert ok and lg.in_flight("brain") == 245760


def test_two_big_sessions_second_holds_then_fires():
    # The 326k replay: two sessions whose needs sum past the budget.
    t, lg, targets = make_ledger()
    ok1, _ = lg.admit("r1", "s1", "brain", 226859, 254723, False, None)
    ok2, why2 = lg.admit("r2", "s2", "brain", 226859, 254723, False, None)
    assert ok1 and not ok2 and why2 == "hold"
    lg.release("r1")
    ok3, _ = lg.held("r2", "s2", "brain", 226859, 254723)
    assert ok3 and lg.in_flight("brain") == 226859


def test_throttle_serializes():
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    ok, why = lg.admit("r2", "s2", "brain", 1000, 254723, True, None)
    assert not ok and why == "throttle"
    lg.release("r1")
    ok, _ = lg.held("r2", "s2", "brain", 1000, 254723)
    assert ok


def test_held_stays_held_during_throttle_even_when_it_fits():
    # THROTTLE's safety property: serial admission. A held request must not
    # be admitted while another request is in flight, even when it fits —
    # fit-alone would bypass the 1-at-a-time guarantee THROTTLE exists for.
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 100000, 254723, False, None)
    ok, why = lg.held("r2", "s2", "brain", 100000, 254723, throttle=True)
    assert not ok and why == "throttle"
    assert lg.in_flight("brain") == 100000   # r2 NOT admitted despite fit
    assert lg.held_since("r2") == 0.0       # marker set for the max_hold guard
    t[0] += 1.5
    ok, why = lg.held("r2", "s2", "brain", 100000, 254723, throttle=True)
    assert not ok and why == "throttle"     # still held while r1 in flight
    lg.release("r1")
    ok, why = lg.held("r2", "s2", "brain", 100000, 254723, throttle=True)
    assert ok and why == "admit"            # engine drained: fits as usual
    assert lg.in_flight("brain") == 100000


def test_two_keyed_targets_same_router_url():
    # Rollout shape: one router URL, two model-keyed targets. Admission is
    # accounted per target: each request competes only under its own
    # target's budget.
    t = [0.0]
    lg = Ledger(lambda: t[0])
    targets = {
        "dflash": Target("dflash", 245760, 254723, ["http://router.test:9871"], "", model="model-a"),
        "batch": Target("batch", 150000, 160000, ["http://router.test:9871"], "", model="model-b"),
    }
    assert target_of("http://router.test:9871/v1", "model-a", targets) == "dflash"
    assert target_of("http://router.test:9871/v1", "model-b", targets) == "batch"
    assert target_of("http://router.test:9871/v1", "model-x", targets) is None
    ok1, _ = lg.admit("r1", "s1", "dflash", 200000, 254723, False, None)
    ok2, _ = lg.admit("r2", "s2", "batch", 100000, 160000, False, None)
    assert ok1 and ok2
    assert lg.in_flight("dflash") == 200000 and lg.in_flight("batch") == 100000
    ok3, why3 = lg.admit("r3", "s3", "batch", 40000, 160000, False, None)
    assert ok3 and why3 == "admit"     # batch headroom independent of dflash
    ok4, why4 = lg.admit("r4", "s4", "batch", 30000, 160000, False, None)
    assert not ok4 and why4 == "hold"  # 100k + 40k + 30k > 160k
    for rid in ("r1", "r2", "r3"):
        lg.release(rid)
    assert lg.in_flight_total() == 0


def test_hold_escape_when_idle():
    # Deadlock guard: L was tightened below a solo request's need (e.g. after
    # a 5xx lowered L). Engine is idle, request still doesn't fit -> after
    # hold_escape it fires anyway (solo-long sessions can never deadlock).
    t, lg, targets = make_ledger(L=500)
    ok, why = lg.held("r1", "s1", "brain", 900, 500)
    assert not ok and why == "hold"
    t[0] += 301                # engine idle the whole time, > hold_escape
    ok, why = lg.held("r1", "s1", "brain", 900, 500)
    assert ok and why == "hold_escape"


def test_hold_escape_requires_idle_engine():
    t, lg, targets = make_ledger(L=1000)
    lg.admit("r1", "s1", "brain", 900, 1000, False, None)
    t[0] += 301                # long wait, but r1 still in flight
    ok, why = lg.held("r2", "s2", "brain", 900, 1000)
    assert not ok              # engine not idle -> no escape


def test_retry_reentry_no_double_count():
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    ok, why = lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    assert ok and why == "already-tracked" and lg.in_flight("brain") == 1000


def test_release_idempotent_and_generated_adjusts_down():
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    lg.release("r1", generated=100)
    assert lg.in_flight("brain") == 0
    lg.release("r1")           # idempotent


def test_force_admit_bypasses_capacity_and_dedupes():
    # The gate's max_hold / log_only escapes track the request (tracked
    # force-admit), so in-flight accounting stays exact for other requests.
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 254723, 254723, False, None)
    ok, why = lg.force_admit("r2", "s2", "brain", 254723)  # would not fit
    assert ok and why == "force" and lg.in_flight("brain") == 2 * 254723
    ok, why = lg.force_admit("r1", "s1", "brain", 254723)  # live id: dedupe
    assert ok and why == "already-tracked" and lg.in_flight("brain") == 2 * 254723


def test_target_of_matches_scheme_host():
    targets = {"brain": Target("brain", 245760, 254723,
                               ["http://brain.test:9999", "https://alt.test:8443"], "")}
    assert target_of("http://brain.test:9999/v1", "m", targets) == "brain"
    assert target_of("http://brain.test:9999", "m", targets) == "brain"
    assert target_of("https://alt.test:8443/v1/chat/completions", "m", targets) == "brain"
    assert target_of("http://other.test:1234/v1", "m", targets) is None
    assert target_of("", "m", targets) is None
    assert target_of(None, "m", targets) is None


def test_target_of_model_keyed_target_matches_same_model():
    targets = {"alpha": Target("alpha", 100, 120, ["http://router.test:9999"], "", model="alpha"),
               "any": Target("any", 100, 120, ["http://router.test:9999"], "")}
    assert target_of("http://router.test:9999/v1", "alpha", targets) == "alpha"


def test_target_of_model_mismatch_falls_to_later_catchall():
    targets = {"alpha": Target("alpha", 100, 120, ["http://router.test:9999"], "", model="alpha"),
               "any": Target("any", 100, 120, ["http://router.test:9999"], "")}
    assert target_of("http://router.test:9999/v1", "beta", targets) == "any"
    # no catch-all present -> mismatch resolves to None
    assert target_of("http://router.test:9999/v1", "beta",
                     {"alpha": targets["alpha"]}) is None


def test_target_of_model_keyed_target_ignores_modelless_requests():
    targets = {"alpha": Target("alpha", 100, 120, ["http://router.test:9999"], "", model="alpha")}
    assert target_of("http://router.test:9999/v1", None, targets) is None
    assert target_of("http://router.test:9999/v1", "", targets) is None


def test_target_of_catchall_matches_any_model_incl_none():
    targets = {"any": Target("any", 100, 120, ["http://router.test:9999"], "")}
    for m in ("alpha", "beta", "", None):
        assert target_of("http://router.test:9999/v1", m, targets) == "any"


def test_target_of_first_match_wins_keyed_before_catchall():
    targets = {"alpha": Target("alpha", 100, 120, ["http://router.test:9999"], "", model="alpha"),
               "any": Target("any", 100, 120, ["http://router.test:9999"], "")}
    assert target_of("http://router.test:9999/v1", "alpha", targets) == "alpha"
    assert target_of("http://router.test:9999/v1", "beta", targets) == "any"
