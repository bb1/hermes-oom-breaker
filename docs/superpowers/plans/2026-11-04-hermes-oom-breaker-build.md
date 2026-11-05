# hermes-oom-breaker v0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-layer OOM protection plugin (exact fit-gate, reactive compaction breaker, on-the-fly learned limit L) as a standalone, hermes-free-core plugin that `hermes plugins doctor` accepts.

**Architecture:** Pure-logic modules (config, state, ledger, learner, breaker, metrics) with zero hermes imports — unit-testable anywhere with plain pytest. A thin adapter layer (`adapter.py`, `compaction_driver.py`) is the only code that touches hermes surfaces: the `llm_execution` execution middleware (the gate), observer hooks (breaker trigger, compaction signal, usage updates), two plugin tools, and the gateway's manual-compaction path (breaker's compaction driver). One gateway process serves all profiles → module-level singleton state, idempotent `register(ctx)`.

**Tech Stack:** Python 3.14 (hermes toolchain), stdlib only (no new dependencies), pytest for unit tests, `hermes plugins doctor` on the hermes host for integration.

**Spec:** `specs/DESIGN.md` (the agreed design; this plan implements it).

## Global Constraints

- Core-logic files (`config.py`, `state.py`, `ledger.py`, `learner.py`, `breaker.py`, `metrics.py`) must import nothing from hermes. Only `adapter.py`, `compaction_driver.py`, `__init__.py` may.
- The `llm_execution` middleware callback must never raise out of itself: hermes' contract is raise-before-`next_call` → callback skipped, request proceeds *without the gate*. All adapter logic is wrapped; internal error → admit + loud warning (availability over strictness).
- No deployment data in this repo: no hostnames, IPs, model names, profile names, or absolute paths in code, tests, comments, or docs. Targets (maxlen, KV budget, metrics URL, base_url filter) come from plugin settings at deploy time (operator's private `config.yaml` under `plugins.entries.hermes-oom-breaker.settings`).
- `hermes plugins doctor` must stay green after every task that touches `plugin.yaml`, `__init__.py`, or registration.
- Unit tests run locally with `python3 -m pytest tests/ -q` (no hermes present locally). Commands marked `[host]` run on the hermes host via ssh.
- Every module that holds mutable state is protected by a single `threading.RLock`; held-request wait loops must release the lock while sleeping.
- Plan and code contain no deployment identifiers: no hostnames, IP addresses, engine ports, model names, or user home paths. The hermes host appears only as `~` under the operator's ssh alias; `[host]` commands are written for that alias (e.g. `ssh <hermes-host> '…'`).

### Verified seams (surveyed 2026-11-04 against the hermes-agent checkout on the hermes host)

| seam | finding |
|---|---|
| `llm_execution` middleware | `agent/turn_api_call.py::perform_api_call` is the **per-attempt** call of the turn's retry loop; it wraps the provider call in `run_llm_execution_middleware(api_kwargs, _perform_api_call, original_request=…, task_id, turn_id, api_request_id, session_id, platform, model, provider, base_url, api_mode, api_call_count, middleware_trace)`. **Every retry re-enters the wrapper.** Callback contract: receives `request` (full provider kwargs incl. `messages`, `max_tokens`) + all context above + `next_call`; `next_call` is single-use per frame; raising before `next_call` → callback skipped (fail-open), chain continues; downstream errors re-raise original (`hermes_cli/middleware.py::_run_execution_chain`). |
| Auxiliary calls | `agent/auxiliary_client.py` builds its own OpenAI client; `agent/conversation_compression.py` never calls `run_llm_execution_middleware`. Auxiliary/compression calls **do not flow through the gate** → the breaker's compaction driver runs them 1-at-a-time by construction. |
| Breaker trigger | `api_request_error` observer hook fires per attempt with `status_code`, `session_id`, `model`, `provider`, `base_url`, `retry_count`, `retryable`, sanitized `request` (`agent/api_request_hooks.py`). |
| Usage update | `post_api_request` observer carries `usage`, `api_duration`, `api_request_id` — used to update/release ledger entries for streaming calls (the wrapper sees the stream iterator return, not the terminal usage). |
| Compaction driver | Gateway `/compress` path: `gateway/slash_commands_session.py::_run_manual_compression(source, session_entry, history, request)` → `compress_now(tmp_agent, msgs, request, system_message="", skip_without_window=True, task_id=session_id)` (`agent/conversation_compression_manual.py`; manual path has no cooldown gate; returns status `compressed` / `lock_skipped` / `nothing_to_do` / `nothing_to_do`-family). Async mutation entry: `gateway/session_mutations.py::mutate_session(authority, actor, ref, params)`. |
| Plugin ctx | `ctx.register_middleware(kind, cb)`; `ctx.register_hook(hook_name, cb)` (callback invoked as `callback(**payload)`); `ctx.register_tool(name, toolset, schema, handler, check_fn=None, requires_env=None, is_async=False, description="", emoji="", override=False)`; `ctx.get_config(key, default)` (reads `plugins.entries.<plugin_id>.settings.<key>`); `ctx.state` → `PluginState` (atomic JSON kv, profile-scoped, `get(key, default)` / `set(key, value)`, file-locked, quota-bounded). |
| Topology | One `gateway run` process serves all profiles (systemd user service); separate dashboard process. → module-level singleton is valid. All profile homes + root route to the brain → plugin must be enabled per profile home. |
| doctor | `HERMES_HOME=<home> <checkout>/.hermes/bin/hermes plugins doctor <plugin-dir>` → skeleton green ("runtime discovery, manifest parsing, import, and registration passed"). |

### Targets (deploy-time config — NOT in this repo)

Settings shape (operator's `config.yaml`): `settings.targets` = `{ <name>: {max_model_len: int, kv_budget: int, base_urls: [str], metrics_url: str} }`; `settings.user_agent` = bool (default false) enabling the User-Agent attribution feature (Task 6b). Seeds for the two current engine profiles (from their boot logs): interactive profile maxlen 245,760 / budget 254,723; batch profile maxlen 150,000 / budget 211,855. L seed = target's `max_model_len`; cap = target's `kv_budget`.

## Review Focus

1. **Hermes update changes middleware context kwargs** (rename/drop a field) → gate must fail open: unknown/missing `base_url`/`model` → skip gating for that call + one loud warning, never a broken turn; tests pin "missing field → pass-through".
2. **Stream that dies after first delta** → ledger entry must be released on `api_request_error` (matched by `api_request_id`/session), not leak; a leaked entry permanently under-admits. Test pins release-on-error.
3. **`PluginState` is profile-scoped** → if a second hermes process with a different profile home ever talks to the same engine, per-process ledgers undercount. Singleton must detect a second `register(ctx)` in a different process (pid in state) and log a loud warning (degrade, don't crash).
4. **Metrics probe misconfigured vs engine dead** → probe that never succeeded must not hold all traffic (misconfig ≠ death); only a probe that succeeded before and then stopped is treated as engine-down. Test pins both.
5. **Compaction 400 fallback loop** (session context > maxlen) → breaker must not retry compaction forever: one 400 → THROTTLE + operator flag, session marked unserviceable. Test pins no-retry.
6. **User-Agent attribution (Task 6b) must be fail-soft**: it is observability, not protection — any resolver failure or unexpected payload shape leaves the request untouched; no exception may ever escape the `llm_request` callback. Test pins "resolver raises → no change".

## File Structure

```
__init__.py            register(ctx): singleton bootstrap, wiring (Task 8)
config.py              knobs + defaults + settings merge (Task 1)
state.py               Store: atomic durable kv over an injected sink (Task 1)
ledger.py              Ledger: in-flight accounting, admit/hold/release (Task 2)
learner.py             Learner: per-target L rules (Task 3)
breaker.py             Breaker FSM: NORMAL/RECOVERY/THROTTLE (Task 4)
metrics.py             MetricsProbe: /metrics poll, liveness (Task 5)
adapter.py             hermes surfaces: middleware cbs, hook cbs, tools (Tasks 6, 6b, 7)
compaction_driver.py   compact_session(session_id) interface + gateway impl (Task 7)
plugin.yaml            manifest (bumped in Task 8)
tests/                 pytest, hermes-free
```

---

### Task 1: Config + durable state store

**Files:**
- Create: `config.py`, `state.py`, `tests/test_config.py`, `tests/test_state.py`
- Modify: `tests/` (new dir)

**Interfaces:**
- Produces: `load_config(get_setting: Callable[[str, Any], Any]) -> Config` — `Config` dataclass with fields `race_reserve: int = 5000`, `recheck_cadence: float = 1.5`, `hold_escape: float = 300.0`, `hit_dedupe: float = 10.0`, `compact_above: int = 50_000`, `escalate_window: float = 900.0`, `throttle_duration: float = 1800.0`, `max_hold: float = 600.0`, `log_only: bool = False`, `probe_authoritative: bool = False`, `reprobe: bool = False`, `targets: dict[str, Target]`; `Target` dataclass with `name: str`, `max_model_len: int`, `kv_budget: int`, `base_urls: list[str]`, `metrics_url: str = ""`. Unknown/invalid settings values → keep default + collect into `Config.warnings: list[str]` (never raise).
- Produces: `Store` — `Store(sink: KeyValueSink, prefix: str = "oom_breaker")` where `KeyValueSink` has `get(key, default)` / `set(key, value)` (the hermes `PluginState` satisfies it; tests use a dict-backed fake). Methods: `get(key, default)`, `set(key, value)`, `update(transform: Callable[[dict], dict]) -> dict` (atomic read-modify-write of the whole state blob under one lock), `reset()`. A corrupt read (sink raises) → return defaults, record `self.last_error: str`, continue (fail open).

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
def test_defaults():
    cfg = load_config(lambda k, d=None: d)
    assert cfg.race_reserve == 5000 and cfg.recheck_cadence == 1.5
    assert cfg.hold_escape == 300.0 and cfg.hit_dedupe == 10.0
    assert cfg.compact_above == 50_000 and cfg.escalate_window == 900.0
    assert cfg.throttle_duration == 1800.0 and cfg.max_hold == 600.0
    assert not cfg.log_only and not cfg.probe_authoritative and not cfg.reprobe
    assert cfg.targets == {} and cfg.warnings == []

def test_settings_override_and_targets():
    def get(k, d=None):
        vals = {"race_reserve": 999,
                "targets": {"a": {"max_model_len": 100, "kv_budget": 120,
                                  "base_urls": ["http://x:1"], "metrics_url": "http://x:2/metrics"}}}
        return vals.get(k, d)
    cfg = load_config(get)
    assert cfg.race_reserve == 999
    t = cfg.targets["a"]
    assert t.max_model_len == 100 and t.kv_budget == 120 and t.metrics_url == "http://x:2/metrics"

def test_invalid_values_keep_default_and_warn():
    cfg = load_config(lambda k, d=None: "garbage" if k == "race_reserve" else d)
    assert cfg.race_reserve == 5000
    assert any("race_reserve" in w for w in cfg.warnings)
```

`tests/test_state.py` (dict-backed fake sink; corrupt = sink raises on get):
```python
class DictSink:
    def __init__(self): self.d = {}
    def get(self, k, d=None): return self.d.get(k, d)
    def set(self, k, v): self.d[k] = v

def test_set_get_roundtrip():
    s = Store(DictSink())
    s.set("L", {"a": 245760}); assert s.get("L") == {"a": 245760}

def test_update_transforms_atomically():
    s = Store(DictSink())
    s.update(lambda d: {**d, "hits": 1})
    assert s.update(lambda d: {**d, "hits": d.get("hits", 0) + 1})["hits"] == 2

def test_corrupt_read_fails_open():
    class Boom:
        def get(self, k, d=None): raise OSError("corrupt")
        def set(self, k, v): pass
    s = Store(Boom())
    assert s.get("x", 5) == 5 and s.last_error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/ -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'` (or collection errors on the two new test files).

- [ ] **Step 3: Implement `config.py` and `state.py`**

`config.py`: `load_config` reads each knob via `get_setting(k, default)` with type/coercion checks (`int`/`float`/`bool`/`dict`); `targets` parsed into `Target` objects (missing required field → drop that target + warning). All parse failures warn, never raise. `state.py`: `Store` with one `RLock`; `update` = read whole blob (key `"state"`) → `transform(blob)` → `set` under the same lock hold; every sink error caught → defaults + `last_error`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (6/6).

- [ ] **Step 5: Commit**

```bash
git add config.py state.py tests/
git commit -m "feat: config knobs + atomic durable state store"
```

---

### Task 2: Ledger + fit-gate arithmetic

**Files:**
- Create: `ledger.py`, `tests/test_ledger.py`

**Interfaces:**
- Consumes: `Config` (Task 1) for `compact_above` (used by breaker later, not here).
- Produces: `Ledger(now: Callable[[], float])` — injectable clock.
  - `target_of(base_url: str, model: str, targets: dict[str, Target]) -> Optional[str]` — first target whose `base_urls` matches (exact match on the URL's scheme://host[:port] prefix) — module-level pure function.
  - `admit(request_id: str, session_id: str, target: str, needed: int, L: int, throttle: bool, engine_idle_since: Optional[float]) -> tuple[bool, str]` — returns `(admitted, reason)`. Rules: unknown target → `(True, "untracked")`; `in_flight(target) + needed <= L` and not (`throttle` and `in_flight_all > 0`) → admit, record entry `(request_id, session_id, target, needed, admitted_at)`; else `(False, "hold"|"throttle")`. Re-admit of a live `request_id` → no double-count (`(True, "already-tracked")` or hold with the same single entry).
  - `release(request_id: str, generated: Optional[int] = None) -> None` — drop the entry (idempotent).
  - `in_flight(target: str) -> int` / `in_flight_total() -> int` — Σ of entry `needed` (upper bound; `generated` adjusts down when known).
  - `held(request_id: str, session_id: str, target: str, needed: int, L: int) -> tuple[bool, str]` — recheck loop body: re-run admission for an already-known held request; if not admitted and `engine_idle_since is not None` and `now() - engine_idle_since >= hold_escape` → admit anyway with reason `"hold_escape"`.
  - `idle_since(now: float) -> Optional[float]` — tracks when `in_flight_total() == 0` began (reset on any admit).
- Produces: module constant `EPSILON_KEY = "race_reserve"` (the learner applies it; ledger just admits against the L it's given).

- [ ] **Step 1: Write the failing tests** (injectable `now`; the 326k replay is the anchor scenario — two sessions whose needs sum past L: the first admits, the second holds, fires when the first releases)
```python
def make_ledger(L=254723):
    t = [0.0]
    lg = Ledger(lambda: t[0])
    targets = {"brain": Target("brain", 245760, 254723, ["http://brain.test:9999"], "")}
    return t, lg, targets

def test_solo_maxlen_admitted_on_idle():
    t, lg, targets = make_ledger()
    ok, why = lg.admit("r1", "s1", "brain", 245760, 254723, False, None)
    assert ok and lg.in_flight("brain") == 245760

def test_two_big_sessions_second_holds_then_fires():
    t, lg, targets = make_ledger()
    ok1, _ = lg.admit("r1", "s1", "brain", 226859, 254723, False, None)  # 172.5k+73.3k-ish
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

def test_hold_escape_when_idle():
    t, lg, targets = make_ledger(L=1000)
    lg.admit("r1", "s1", "brain", 900, 1000, False, None)
    ok, why = lg.held("r2", "s2", "brain", 900, 1000)
    assert not ok
    t[0] += 301  # engine idle 5 min 1 s
    ok, why = lg.held("r2", "s2", "brain", 900, 1000)
    assert ok and why == "hold_escape"

def test_retry_reentry_no_double_count():
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    ok, why = lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    assert ok and why == "already-tracked" and lg.in_flight("brain") == 1000

def test_release_idempotent_and_generated_adjusts_down():
    t, lg, targets = make_ledger()
    lg.admit("r1", "s1", "brain", 1000, 254723, False, None)
    lg.release("r1", generated=100); assert lg.in_flight("brain") == 0
    lg.release("r1")  # no error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ledger.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ledger'`.

- [ ] **Step 3: Implement `ledger.py`**

Entries: `dict[request_id, Entry]` under the ledger's `RLock`. `needed` seeded as the prompt+max_tokens upper bound; `release(request_id, generated)` — when `generated` is known the entry's accounting drops to `prompt + generated` before removal (matters only for observability/log lines; in-flight after release is 0). `idle_since`: timestamp of the most recent transition to total==0, computed in `held`/`admit` from the current total and a stored marker. `target_of` compares `urllib.parse.urlsplit(base_url)` scheme+netloc against each target's `base_urls` (same normalization).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ledger.py -q`
Expected: PASS (6/6); then full `python3 -m pytest tests/ -q` PASS.

- [ ] **Step 5: Commit**

```bash
git add ledger.py tests/test_ledger.py
git commit -m "feat: in-flight ledger with exact fit-gate arithmetic"
```

---

### Task 3: Adaptive limit L (learner)

**Files:**
- Create: `learner.py`, `tests/test_learner.py`

**Interfaces:**
- Consumes: `Target` (Task 1), `Config.race_reserve` (Task 1).
- Produces: `Learner(now: Callable[[], float])`:
  - `seed(target: str, t: Target) -> int` — `L = t.max_model_len`, capped at `t.kv_budget`; persists to the injected `Store` (Task 1) under key `"L"`.
  - `current(target: str, defaults: dict[str, Target]) -> int`
  - `on_hit(target: str, in_flight_at_hit: int) -> int` — `L := min(L, max(0, in_flight_at_hit - race_reserve))`, floored at 1; records `last_hit_ts` + appends to an L-history ring (cap 50) in the store.
  - `on_episode_peak(target: str, peak: int) -> int` — raise only when `peak > L` and `peak <= kv_budget`: `L := peak`.
  - `on_hold_escape_admit(target: str, peak: int) -> int` — same raise rule as `on_episode_peak` (upward-exploration path).
  - `reset(target: str) -> int` — back to the target's seed; clears `last_hit_ts`.
  - `within_escalation_window(now: float) -> bool` — `now - last_hit_ts < escalate_window` (read from store).
  - `maybe_reprobe(target: str, now: float) -> Optional[int]` — only when `Config.reprobe` is True: if `now - last_hit > 86_400` and no reprobe has fired since that hit → `L += race_reserve` exactly once (returns the new L, else None). `seed()` persists the target's `max_model_len`/`kv_budget` into the store; all other methods read them back (no `defaults` needed after seeding).

- [ ] **Step 1: Write the failing tests** (store-backed via the Task 1 `Store` + `DictSink` from `tests/test_state.py` — import it; clock injected)
```python
def make():
    from state import Store
    from test_state import DictSink
    t = [0.0]; store = Store(DictSink())
    lr = Learner(lambda: t[0], store)
    lr.seed("a", Target("a", 245760, 254723, ["http://x:1"], ""))
    return t, lr

def test_seed_is_maxlen_capped_at_budget():
    t, lr = make(); assert lr.current("a", {}) == 245760
    t2, lr2 = make_budget(); assert lr2.current("b", {}) == 150000

def test_hit_lowers_below_observed_minus_epsilon():
    t, lr = make()
    lr.on_hit("a", 326101)  # observed in-flight at hit
    assert lr.current("a", {}) == 326101 - 5000
    t[0] += 1
    assert lr.within_escalation_window(t[0])

def test_clean_episode_peak_raises():
    t, lr = make()
    lr.on_episode_peak("a", 300000)          # above cap → ignored
    assert lr.current("a", {}) == 245760
    lr.on_episode_peak("a", 254000)
    assert lr.current("a", {}) == 254000     # ≤ budget → raised

def test_hold_escape_admit_raises_same_rule():
    t, lr = make()
    lr.on_hold_escape_admit("a", 250000)
    assert lr.current("a", {}) == 250000

def test_reset_restores_seed():
    t, lr = make()
    lr.on_hit("a", 100000)
    assert lr.reset("a") == 245760
    assert not lr.within_escalation_window(t[0] + 1)

def test_reprobe_off_by_default_and_on_fires_once():
    t, lr = make()
    lr.on_hit("a", 100000)
    t[0] = 90_000
    assert lr.maybe_reprobe("a", t[0], reprobe=False) is None
    assert lr.maybe_reprobe("a", t[0], reprobe=True) == 95000
    assert lr.maybe_reprobe("a", t[0] + 10, reprobe=True) is None  # once per hit
```
(`make_budget` seeds a target `"b"` with maxlen 150,000 / budget 211,855 — the batch-profile shape, no names.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_learner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'learner'`.

- [ ] **Step 3: Implement `learner.py`**

All L reads/writes through `Store.update` (atomic). L-history ring: list of `(ts, target, old, new, why)` capped at 50 entries. `on_hit` uses the **exact** `in_flight_at_hit` passed in (the adapter passes the ledger's number — no re-estimation).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_learner.py -q` then full suite.
Expected: PASS (5/5 + all).

- [ ] **Step 5: Commit**

```bash
git add learner.py tests/test_learner.py
git commit -m "feat: adaptive limit L — seed/lower/raise/reset rules"
```

---

### Task 4: Breaker FSM

**Files:**
- Create: `breaker.py`, `tests/test_breaker.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `Store` (Task 1), `Learner` (Task 3).
- Produces: `Breaker(cfg: Config, store: Store, learner: Learner, now: Callable[[], float])`:
  - `state() -> str` — `"normal" | "recovery" | "throttle"` (persisted in store; survives restarts — reload on construction).
  - `hit(target: str, in_flight: int, session_ids: list[str], status_code: int, now: float) -> str` — registers a hit. Returns the resulting state. Rules: `now - last_hit < hit_dedupe` → same hit, state unchanged (dedupe); else `learner.on_hit`; if `learner.within_escalation_window(now)` (a prior hit inside `escalate_window`) → `"throttle"` with `throttle_until = now + throttle_duration`; else `"recovery"` with `compaction_queue = [sid for sid in session_ids]` (deduped, order preserved); `last_hit_ts = now`.
  - `start_compaction(session_id: str) -> bool` — RECOVERY only; pops the session from the queue head *only if* `active_compaction is None`; sets `active_compaction = session_id` (1-at-a-time lock).
  - `compaction_done(session_id: str, status: str) -> list[str]` — clears `active_compaction`; on `status == "unserviceable"` (the 400 case) → `"throttle"` + `unserviceable[session_id] = now` (never re-queued); when the queue empties → `"normal"`. Returns sessions still queued.
  - `throttle_active(now: float) -> bool` — `state == "throttle"` and `now < throttle_until` (auto-expires → `"normal"`).
  - `holds_big(session_id: str, context_tokens: int) -> bool` — RECOVERY: `context_tokens > compact_above` → hold (gate keeps it parked until the queue drains).
  - `unserviceable_ids() -> list[str]`.

- [ ] **Step 1: Write the failing tests** (fresh `Breaker` per test with shared `Store`/`Learner`; clock = `t[0]`)
```python
def make(t, **kw):
    cfg = Config(); store = Store(DictSink()); lr = Learner(lambda: t[0], store)
    lr.seed("a", Target("a", 245760, 254723, ["http://x:1"], ""))
    return Breaker(cfg, store, lr, lambda: t[0])

def test_first_hit_starts_recovery():
    t = [0.0]; b = make(t)
    assert b.hit("a", 326101, ["s1", "s2"], 500, t[0]) == "recovery"

def test_burst_within_dedupe_is_one_hit():
    t = [0.0]; b = make(t)
    b.hit("a", 326101, ["s1"], 500, t[0])
    assert b.hit("a", 326101, ["s2"], 502, t[0] + 5) == "recovery"  # still one hit
    assert b.state() == "recovery"

def test_second_hit_within_window_throttles_flat_30min():
    t = [0.0]; b = make(t)
    b.hit("a", 326101, ["s1"], 500, t[0])
    t[0] = 600  # 10 min later — inside 15-min window
    assert b.hit("a", 200000, ["s3"], 500, t[0]) == "throttle"
    assert b.throttle_active(600 + 1799) and not b.throttle_active(600 + 1801)

def test_third_hit_after_window_is_plain_recovery():
    t = [0.0]; b = make(t)
    b.hit("a", 326101, ["s1"], 500, t[0])
    t[0] = 901
    assert b.hit("a", 200000, ["s3"], 500, t[0]) == "recovery"

def test_compaction_strictly_one_at_a_time():
    t = [0.0]; b = make(t)
    b.hit("a", 326101, ["s1", "s2", "s3"], 500, t[0])
    assert b.start_compaction("s1") is True
    assert b.start_compaction("s2") is False          # s1 still active
    assert b.compaction_done("s1", "compressed") == ["s2", "s3"]
    assert b.start_compaction("s2") is True

def test_unserviceable_400_goes_throttle_no_retry():
    t = [0.0]; b = make(t)
    b.hit("a", 326101, ["s1", "s2"], 500, t[0])
    b.start_compaction("s1")
    assert b.compaction_done("s1", "unserviceable") == []
    assert b.state() == "throttle"
    assert "s1" in b.unserviceable_ids()
    assert b.start_compaction("s1") is False          # never re-queued

def test_recovery_state_survives_restart():
    t = [0.0]; cfg = Config(); store = Store(DictSink())
    lr = Learner(lambda: t[0], store); lr.seed("a", Target("a", 245760, 254723, ["http://x:1"], ""))
    b1 = Breaker(cfg, store, lr, lambda: t[0])
    b1.hit("a", 326101, ["s1"], 500, t[0])
    b2 = Breaker(cfg, store, lr, lambda: t[0])       # fresh process, same store
    assert b2.state() == "recovery" and b2.start_compaction("s1") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_breaker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'breaker'`.

- [ ] **Step 3: Implement `breaker.py`**

One `RLock`; all transitions persist via `Store.update` (state, `last_hit_ts`, `throttle_until`, `compaction_queue`, `active_compaction`, `unserviceable`). Construction reloads persisted state (restart safety). `hit` dedupe compares against `last_hit_ts`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_breaker.py -q` then full suite.
Expected: PASS (7/7 + all).

- [ ] **Step 5: Commit**

```bash
git add breaker.py tests/test_breaker.py
git commit -m "feat: reactive breaker FSM — dedupe, escalation, throttle, 1-at-a-time"
```

---

### Task 5: Engine metrics probe

**Files:**
- Create: `metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Target.metrics_url` (Task 1), `Config.probe_authoritative` (Task 1).
- Produces: `MetricsProbe(fetch: Callable[[str], str], now: Callable[[], float], authoritative: bool, poll_interval: float = 10.0)` — `fetch` returns the body of `GET <metrics_url>` (adapter injects `urllib.request`; tests inject a fake).
  - `tick(url: str, value: Optional[float]) -> None` — fed by the poll thread; tracks `last_ok_at`, `ever_ok`, `cache_usage` (0..1).
  - `engine_down(url: str) -> bool` — True only when `ever_ok(url) and now - last_ok_at > 3*poll_interval` **and** `authoritative` is True. A probe that never succeeded → False (misconfig ≠ death).
  - `start(targets: dict[str, Target])` / `stop()` — background daemon thread (uses `poll_interval`); per-target last-success map; thread must be interruptible by `stop()` within one interval (uses a `threading.Event` wait, no busy sleep).
- [ ] **Step 1: Write the failing tests**
```python
def test_never_ok_means_not_down():
    n = [0.0]; p = MetricsProbe(lambda u: "boom", lambda: n[0], authoritative=True)
    p.tick("http://x/metrics", 0.5) if False else None  # no ticks → never ok
    assert p.engine_down("http://x/metrics") is False

def test_ok_then_dead_and_authoritative_means_down():
    n = [100.0]; p = MetricsProbe(lambda u: "", lambda: n[0], authoritative=True, poll_interval=10.0)
    p.tick("http://x/metrics", 0.4); n[0] = 140.0
    assert p.engine_down("http://x/metrics") is True

def test_dead_but_not_authoritative_means_not_down():
    n = [100.0]; p = MetricsProbe(lambda u: "", lambda: n[0], authoritative=False, poll_interval=10.0)
    p.tick("http://x/metrics", 0.4); n[0] = 140.0
    assert p.engine_down("http://x/metrics") is False

def test_parses_gauge_line():
    body = "# HELP vllm:gpu_cache_usage_perc ...\nvllm:gpu_cache_usage_perc{api_server=\"0\"} 0.888\n"
    assert parse_gauge(body, "vllm:gpu_cache_usage_perc") == 0.888
    assert parse_gauge("nothing", "vllm:gpu_cache_usage_perc") is None
```
- [ ] **Step 2: Run tests to verify they fail** — `python3 -m pytest tests/test_metrics.py -q` → FAIL (no `metrics` module).
- [ ] **Step 3: Implement `metrics.py`** — `parse_gauge` scans lines `name{labels} value`; thread loop: per target `fetch(url)` with a 5 s timeout via the injected fetcher, `parse_gauge`, `tick`; on fetch exception → skip (counted as non-success); `stop()` sets the event; thread joins on `stop()` (idempotent).
- [ ] **Step 4: Run tests to verify they pass** — `python3 -m pytest tests/test_metrics.py -q` → PASS (4/4 + full suite).
- [ ] **Step 5: Commit** — `git add metrics.py tests/test_metrics.py && git commit -m "feat: engine metrics probe — liveness + cross-check, misconfig-safe"`

---

### Task 6: Gate adapter (llm_execution middleware)

**Files:**
- Create: `adapter.py` (gate half), `tests/test_adapter_gate.py`

**Interfaces:**
- Consumes: `Ledger` (Task 2), `Learner` (Task 3), `Breaker` (Task 4), `MetricsProbe` (Task 5), `Config` (Task 1), `Store` (Task 1).
- Produces: `GateController` — the in-process singleton facade the `register(ctx)` wiring talks to (no hermes imports; hermes payload dicts are passed in as plain dicts). Exposes `ledger`, `learner`, `breaker`, `probe`, `estimator`, `compaction_driver` attributes (tests and Task 7 use them).
  - `__init__(cfg: Config, store: Store, now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep)` — builds ledger/learner/breaker/probe (`now`/`sleep` injected for tests); `load_targets()` seeds learner + probe from `cfg.targets`.
  - `llm_execution(self, request: dict, next_call, **context) -> Any` — the middleware callback (must never raise; outer `try/except` → `return next_call(request)` on internal error after a loud `logging.warning`). Flow:
    1. `target = target_of(context.get("base_url"), context.get("model"), self.targets)`; None → `next_call(request)` (untracked provider passes through).
    2. `needed = prompt_tokens + max_tokens`: `prompt_tokens = estimate(request)` — injected estimator (default: sum of `len(m.get("content", ""))` char-count `// 4` over messages — the hermes adapter overrides with `estimate_request_context_tokens`; see Ruling R2); `max_tokens = int(request.get("max_tokens") or 0)`.
    3. Admission via `ledger.admit(api_request_id, session_id, target, needed, learner.current(target, targets), breaker.throttle_active(now), ledger.idle_since)`.
    4. Hold loop: while not admitted: if `breaker.holds_big(session_id, needed)` and breaker in recovery → wait for `breaker` queue drain (recheck `recheck_cadence`); elif `now - held_since > max_hold` → admit anyway (log loud); else sleep `recheck_cadence` (lock released while sleeping); `log_only` mode → never hold, only log would-be holds.
    5. `result = next_call(request)`; on success (non-streaming result object with `.usage`): `ledger.release(request_id, generated=usage.completion_tokens)`; streaming (result is an iterator/generator): keep upper bound, defer release to `post_api_request` hook (Task 7) — mark entry as `"streaming"`.
    6. On exception from `next_call`: `ledger.release(request_id)` (error path), re-raise (let hermes' retry re-enter the wrapper).
  - `post_api_request(self, **payload) -> None` — observer callback: on `usage` present → `ledger.release(api_request_id, generated=usage.completion_tokens)`; on error status → `ledger.release(api_request_id)` + `breaker.hit(...)` with `in_flight = ledger.in_flight_total()` **before** releasing this entry (order matters: hit sees the true in-flight total) — but dedupe: the `api_request_error` hook (Task 7) is the canonical trigger; `post_api_request` only releases + records, and calls `breaker.hit` **only** when the `api_request_error` hook is not registered (defensive fallback, guarded by a flag).
- Produces: module function `estimate_tokens(request: dict) -> int` (the default estimator; overridable by the hermes wiring).

- [ ] **Step 1: Write the failing tests** (fake `next_call` capturing calls; fake context dict with the verified fields; clock via `GateController(cfg, store, now=...)` constructor injection)
```python
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
    g = make_gate(now=lambda: t2[0], sleep=lambda s: t2[0] += s)
    release_r1 = threading.Event()
    def slow_first(r):
        release_r1.wait(timeout=10)
        return SimpleNamespace(usage=SimpleNamespace(completion_tokens=1))
    th = threading.Thread(target=lambda: g.llm_execution(
        {"messages": [{"content": "x" * 700000}], "max_tokens": 40000},
        slow_first, **{**CTX, "api_request_id": "r1"}))
    th.start()
    while g.ledger.in_flight("brain") == 0:      # wait for r1 admission
        time.sleep(0.01)
    start = t2[0]
    out = g.llm_execution({"messages": [{"content": "x" * 700000}], "max_tokens": 40000},
                          lambda r: "second", **{**CTX, "api_request_id": "r2"})
    release_r1.set(); th.join()
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
```

- [ ] **Step 2: Run tests to verify they fail** — `python3 -m pytest tests/test_adapter_gate.py -q` → FAIL (no `adapter`).
- [ ] **Step 3: Implement the gate half of `adapter.py`** — `GateController` as specified. Hold loop releases its lock around `time.sleep` (configurable `sleep: Callable[[float], None] = time.sleep` for tests). Streaming detection: `hasattr(result, "__iter__") and not hasattr(result, "usage")`. Ruling R2: the default char/4 estimator keeps the module hermes-free; the hermes wiring (`__init__.py`, Task 8) swaps in hermes' `estimate_request_context_tokens(api_payload)` when importable (ledger parity with hermes itself).
- [ ] **Step 4: Run tests to verify they pass** — `python3 -m pytest tests/test_adapter_gate.py -q` → PASS (7/7 + full suite).
- [ ] **Step 5: Commit** — `git add adapter.py tests/test_adapter_gate.py && git commit -m "feat: fit-gate adapter — llm_execution middleware callback, fail-open"`

---

### Task 6b: User-Agent attribution (llm_request middleware) — optional, off by default

**Why:** the brain router's dashboard should show *which* hermes profile (and which cron job, for scheduler runs) issued each request — the big requests are the ones that matter. When enabled, the plugin stamps the outgoing request's `User-Agent` (otherwise the OpenAI SDK's default) with `Hermes/<profile>`, or `Hermes/<profile>:<cron job>` for an active cron execution. Off by default; **on in the rollout** (Task 10).

**Files:**
- Modify: `config.py` (new knob `user_agent: bool = False`), `adapter.py` (label builder + `ua_llm_request` callback)
- Create: `tests/test_adapter_ua.py`; extend `tests/test_config.py`

**Verified seams (surveyed on the hermes host, 2026-11-04):**
- `llm_request` request middleware: `agent/turn_api_request.py` applies `apply_llm_request_middleware(api_kwargs, task_id, turn_id, api_request_id, session_id, platform, model, provider, base_url, api_mode, api_call_count)` before every per-attempt provider call. The registered callback is invoked with `request=<copy of api_kwargs>` + those context kwargs; returning `{"request": {...}}` replaces the provider kwargs.
- Header mechanism: the OpenAI SDK's `_build_headers` merges per-request `extra_headers` **on top of** the client's `default_headers` (which carry the default `User-Agent`) → `extra_headers["User-Agent"]` overrides the wire UA.
- Profile identity: `hermes_cli.profiles.current_profile_name()` — the profile the current turn runs FOR (per-turn `HERMES_HOME` override in the multiplexed single-gateway process; no session-store access needed).
- Cron identity: `cron.execution_identity.current_cron_execution()` — public getter over the ContextVar the scheduler binds and propagates into the agent thread via `contextvars.copy_context()`; `.job_name` is the label.

**Interfaces:**
- `Config.user_agent: bool = False` — new knob in `_KNOWN_KNOBS` (bool).
- `GateController.build_user_agent(self) -> str` — `f"{USER_AGENT_PREFIX}/{profile}"` with `USER_AGENT_PREFIX = "Hermes"` (constant); no profile → `"Hermes"`; appends `f":{job}"` when `self.cron_fn()` returns a truthy job name.
- `GateController.ua_llm_request(self, request: dict, **context) -> Optional[dict]` — the `llm_request` callback. **Must never raise** (observability must never break a request): any internal failure → return `None` (no change). Flow: `cfg.user_agent` False → `None`; compute label; copy-on-write merge `extra_headers` (set `User-Agent`); return `{"request": {**request, "extra_headers": xh}}` — never mutate the input dict.
- `profile_fn` / `cron_fn` are injectable attributes (tests inject fakes). Defaults are hermes-bound lazily: small functions that import `hermes_cli.profiles` / `cron.execution_identity` inside the body and return `None` on any failure — `adapter.py`'s module import stays hermes-free (doctor must stay green in a hermes-free env).
- Task 8's `register(ctx)`: additionally `ctx.register_middleware("llm_request", controller.ua_llm_request)` (always registered; runtime no-op while disabled).

- [ ] **Step 1: Write the failing tests** — extend `tests/test_config.py` (`test_defaults` asserts `not cfg.user_agent`; `test_settings_override_and_targets` serves `"user_agent": True` and asserts it), and create `tests/test_adapter_ua.py`:
```python
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
```
- [ ] **Step 2: Verify RED** — `python3 -m pytest tests/test_adapter_ua.py tests/test_config.py -q` → FAIL (no `ua_llm_request` / `user_agent`).
- [ ] **Step 3: Implement** — the config knob; the adapter additions. The default resolver functions import hermes lazily inside the body, each wrapped in try/except → `None`.
- [ ] **Step 4: Verify GREEN** — focused tests + full `python3 -m pytest tests/ -q`.
- [ ] **Step 5: Commit** — `git add config.py tests/test_config.py adapter.py tests/test_adapter_ua.py && git commit -m "feat: optional User-Agent attribution (Hermes/<profile>[:<cron job>]) via llm_request middleware"`

---

### Task 7: Breaker adapter + compaction driver

**Files:**
- Create: `compaction_driver.py`, extend `adapter.py` (hook half), `tests/test_breaker_adapter.py`

**Interfaces:**
- Consumes: `GateController` (Task 6), `Breaker` (Task 4), `Store` (Task 1).
- Produces: `compaction_driver.py`: `CompactionDriver` protocol — `async compact(session_id: str) -> str` returning `"compressed" | "lock_skipped" | "unserviceable" | "nothing_to_do" | "error:<msg>"`. `GatewayDriver` implementation — **Step 1 pins the exact hermes entry** (see Step 1); `FakeDriver` for tests (returns scripted statuses).
- Produces (adapter.py extensions):
  - `api_request_error(self, **payload) -> None` — observer callback: `status = payload.get("status_code")`; if `500 <= status < 600` and target tracked: `breaker.hit(target, ledger.in_flight_total(), [payload["session_id"]], status, now)`; then `learner` already lowered L inside `hit`; if breaker entered THROTTLE → log loud. Never raise (observer contract).
  - `post_compaction(self, **payload) -> None` — observer callback (`session_id`, `kind`, `reason`): if `breaker.state() == "recovery"` and `session_id` matches `breaker.active_compaction` → `breaker.compaction_done(session_id, "compressed")`; on transition to normal → log + (optional) `learner.on_episode_peak` from the recorded peak.
  - `requeue_or_finish(...)` — after `compaction_done` with a non-empty remaining queue: `GateController` fires the next `driver.compact(session_id)` from a worker thread (not the hook thread — hooks must stay fast; bounded queue of 1, `queue.Full` → log, breaker stays recovery, next `post_compaction`/recheck resumes).
  - `oom_status(self) -> dict` / `oom_reset(self, target: str) -> dict` — tool handlers (synchronous): status returns `{state, L: {…}, in_flight: {…}, last_hit, throttle_until, unserviceable}`; reset calls `learner.reset(target)`.

- [ ] **Step 1: Pin the gateway compaction entry (verification step, hermes host)**

Run `[host]`:
```
ssh <hermes-host> 'cd ~/.hermes/hermes-agent && sed -n "660,700p" gateway/slash_commands_session.py && grep -n "def \|_slash_commands\|async_session_store" gateway/session_mutations.py | head -20'
```
Expected: the `/compress` handler calls `self._run_manual_compression(source, session_entry, history, request)`; `gateway/session_mutations.py::mutate_session` is the async mutation entry. **Decision recorded here (Ruling R3):** `GatewayDriver.compact(session_id)` = (a) obtain the gateway's session store + `source`/`session_entry` for `session_id` (verify the accessor exists on the gateway singleton — if the reachable path requires an operator `Principal`, construct the plugin's own principal per the gateway's own code), (b) call `_run_manual_compression`, (c) map `CompressResult.status` → driver statuses (`"compressed"` → `compressed`; `"lock_skipped"` → `lock_skipped`; anything where the compression request itself 400s — i.e. `error` mentioning 400/`max` context → `unserviceable`). If `_run_manual_compression` proves not reachable from plugin context without core wrapping: fallback = emit the session's `/compress` slash command through the gateway's command dispatch (same public path an operator uses); if that is also unreachable: STOP and surface to the operator (this is the one seam that has no generic fallback).

- [ ] **Step 2: Write the failing tests** (scripted `FakeDriver`; hook callbacks fed plain payload dicts with the verified field names)
```python
def make():
    cfg = Config(); store = Store(DictSink())
    cfg.targets = {"brain": Target("brain", 245760, 254723, ["http://brain.test:9999"], "")}
    g = GateController(cfg, store, now=lambda: NOW[0])
    driver = FakeDriver(); g.install_breaker_hooks(driver)
    return g, driver

def test_5xx_hit_triggers_recovery_and_compaction_sequence():
    NOW = [0.0]; g, driver = make(); NOW[0] = 100.0
    g.api_request_error(status_code=500, session_id="s1", model="brain-model",
                        provider="openai", base_url="http://brain.test:9999/v1", retry_count=0)
    assert g.breaker.state() == "recovery"
    assert driver.queued[0] == "s1"           # compaction dispatched 1-at-a-time
    driver.finish("s1", "compressed")         # post_compaction arrives
    g.post_compaction(session_id="s1", kind="manual", reason="oom_breaker")
    assert g.breaker.state() == "normal"

def test_error_400_unserviceable_throttles_no_retry():
    NOW = [0.0]; g, driver = make(); NOW[0] = 100.0
    driver.script("s1", "unserviceable")
    g.api_request_error(status_code=500, session_id="s1", model="brain-model",
                        base_url="http://brain.test:9999/v1", provider="openai", retry_count=0)
    driver.finish("s1", "unserviceable")
    g.post_compaction(session_id="s1", kind="manual", reason="oom_breaker")
    assert g.breaker.state() == "throttle"
    assert driver.attempts["s1"] == 1         # never re-queued

def test_400s_are_not_hits():
    NOW = [0.0]; g, driver = make()
    g.api_request_error(status_code=400, session_id="s1", model="m", base_url="http://other", retry_count=0)
    assert g.breaker.state() == "normal"      # 400 outside tracked target: no hit

def test_streaming_error_releases_ledger_entry():
    # Review Focus #2: a stream that dies after first delta must not leak its entry.
    import itertools
    NOW = [0.0]; g, driver = make(); NOW[0] = 100.0
    stream = (c for c in itertools.chain(["a", "b"], [SimpleNamespace(usage=SimpleNamespace(completion_tokens=7))]))
    g.llm_execution({"messages": [{"content": "x" * 4000}], "max_tokens": 40000},
                    lambda r: stream, **{**CTX, "api_request_id": "r9"})
    assert g.ledger.in_flight("brain") > 0    # streaming: upper bound held
    g.post_api_request(status_code=500, api_request_id="r9", session_id="s1",
                       model="brain-model", base_url="http://brain.test:9999/v1")
    assert g.ledger.in_flight("brain") == 0   # error path releases the entry

def test_oom_status_tool_shape():
    NOW = [0.0]; g, driver = make()
    s = g.oom_status()
    assert s["state"] == "normal" and "L" in s and "in_flight" in s
```
- [ ] **Step 3: Run tests to verify they fail** — `python3 -m pytest tests/test_breaker_adapter.py -q` → FAIL.
- [ ] **Step 4: Implement** — `GatewayDriver` per Step 1's ruling (hermes imports inside a function, lazy, so the module imports cleanly without hermes — keeps doctor green in hermes-free test envs); hook callbacks as specified; `install_breaker_hooks(driver)` wires `api_request_error` + `post_compaction` onto `GateController` and starts the single-worker compaction dispatcher. Tool handler registration happens in Task 8.
- [ ] **Step 5: Run tests to verify they pass** — full suite PASS.
- [ ] **Step 6: Commit** — `git add compaction_driver.py adapter.py tests/test_breaker_adapter.py && git commit -m "feat: breaker hooks + 1-at-a-time compaction driver (gateway /compress path)"`

---

### Task 8: register(ctx) assembly + manifest bump

**Files:**
- Modify: `__init__.py`, `plugin.yaml` (version → `0.1.0`)
- Create: `tests/test_register.py`

**Interfaces:**
- Consumes: `GateController` (Task 6-7), `ctx` API (verified: `register_middleware`, `register_hook`, `register_tool`, `get_config`, `state`).
- Produces: `register(ctx) -> None`:
  1. Singleton: module-level `_controller`; second `register` in the same process → idempotent no-op; a pid recorded in `Store` differs from `os.getpid()` → log a loud warning (cross-process detection — Review Focus #3) and still proceed (degrade, don't crash).
  2. `cfg = load_config(ctx.get_config)`; `store = Store(Shim(ctx.state))` where `Shim` adapts `PluginState.get/set` to the `KeyValueSink` interface.
  3. `controller.load_targets()`; if no targets configured → log loud "unconfigured: gate is a pass-through" (doctor-green, zero behavior change — the safe default).
  4. Swap the estimator for hermes' `estimate_request_context_tokens` when importable (R2).
  5. `ctx.register_middleware("llm_execution", controller.llm_execution)`; `ctx.register_middleware("llm_request", controller.ua_llm_request)` (Task 6b; runtime no-op while `user_agent` is false); `ctx.register_hook("api_request_error", controller.api_request_error)`; `ctx.register_hook("post_compaction", controller.post_compaction)`; `ctx.register_hook("post_api_request", controller.post_api_request)`.
  6. `ctx.register_tool("oom_status", "oom_breaker", schema, controller.oom_status, description="OOM-breaker status: per-target L, in-flight, breaker state")` and `ctx.register_tool("oom_reset", "oom_breaker", schema, controller.oom_reset, description="Reset a target's learned limit L to its seed")` — schemas: `oom_status` `{}` → object; `oom_reset` `{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}`.
  7. `controller.probe.start(cfg.targets)` (only for targets with `metrics_url`).
- `unregister`/`on_unload` (if the ctx exposes it — verify in Step 1): stop the probe thread.

- [ ] **Step 1: Verify the register-surface details (hermes host)** — `ssh <hermes-host> 'cd ~/.hermes/hermes-agent && grep -n "on_unload\|def register_hook" hermes_cli/plugins.py | head'` Expected: confirm whether the plugin lifecycle exposes an unload callback (record in ledger; if not, the probe thread is daemon and process-lifetime — acceptable).
- [ ] **Step 2: Write the failing test** `tests/test_register.py` — a `FakeCtx` implementing the five used methods (records registrations); assert: `register(FakeCtx())` registers 2 middlewares (`llm_execution`, `llm_request`), 3 hooks (`api_request_error`, `post_compaction`, `post_api_request`), 2 tools (`oom_status`, `oom_reset`); a second `register` call adds nothing (idempotent); a fresh process (fake pid in store) logs the cross-process warning but still registers; a fresh `GateController` over a store holding a lowered L preserves it (lazy `load_targets` — D3').
- [ ] **Step 3: Run → fail** (`python3 -m pytest tests/test_register.py -q` → FAIL).
- [ ] **Step 4: Implement** per the Produces block; bump `plugin.yaml` `version: 0.1.0` + description unchanged.
- [ ] **Step 5: Run full suite** — PASS.
- [ ] **Step 6: doctor green on the hermes host**
  Run `[host]`: `ssh <hermes-host> 'cd ~/code/hermes-oom-breaker && git pull -q && HERMES_HOME=~/.hermes ~/.hermes/hermes-agent/.hermes/bin/hermes plugins doctor ~/code/hermes-oom-breaker'`
  Expected: `OK: runtime discovery, manifest parsing, import, and registration passed` with `2 tool(s), 3 hook(s)` + 2 middlewares (unconfigured → pass-through; `user_agent` off → no-op; no behavior change yet).
- [ ] **Step 7: Commit** — `git add __init__.py plugin.yaml tests/test_register.py && git commit -m "feat: register(ctx) assembly — middleware, hooks, tools, probe; v0.1.0"` + `git push`

---

### Task 9: Integration on the hermes host (engine up, staging profile, no production traffic)

**Files:**
- Modify: none in-repo (verification task; results recorded in the ledger; a `tests/integration/` script may be added if the replays are worth keeping — decide from Step 2's friction).

**Preconditions (operator action, coordinated):** a staging hermes profile home whose brain target points at the engine (or a temporary `plugins.entries` override with `settings.targets`), the engine up, no production sessions on that profile.

- [ ] **Step 1: Enable on the staging profile + restart** `[host]` — add `plugins.entries.hermes-oom-breaker.enabled: true` + `settings.targets` (the engine's real seeds: maxlen 245,760 / budget 254,723; `metrics_url` = the engine's loopback metrics endpoint; `base_urls` = the brain router's base URL) to the staging profile's config; full gateway restart (hot-reload does not re-import plugin modules). Expected: gateway log shows plugin registration (2 tools, 3 hooks, middleware) + "targets loaded: 1".
- [ ] **Step 2: Solo near-maxlen replay** `[host]` — drive one session to ~200k context on the idle engine (or reuse a large real session transcript on the staging profile). Expected: admitted immediately (0 + needed ≤ L), full response, zero 5xx, gate log line shows admit + release with usage.
- [ ] **Step 3: Two-big concurrent replay** `[host]` — start session A (~170k, long generation), then session B (~150k) mid-flight. Expected: B **held** in the wrapper (gate log: `hold recheck …`), zero 5xx, B fires when A completes. (Day-1 proof — the two-big-sessions-past-budget scenario prevented.)
- [ ] **Step 4: Simulated 5xx → RECOVERY** `[host]` — with two big sessions in flight, stop the engine process (or drop its port for the request path — pick the least disruptive; a real engine restart is the real thing, do it in a quiet window). Expected: 5xx observed → one hit (deduped), L lowered below observed minus ε (state file), offending big session(s) compacted **1-at-a-time** via the gateway `/compress` path (watch gateway log for the manual-compression run), `post_compaction` drives the next, re-fired requests fit, state → normal.
- [ ] **Step 5: Second hit inside 15 min → THROTTLE** `[host]` — repeat Step 4's trigger within the window. Expected: THROTTLE 30 min flat: subsequent brain calls serialize (in_flight ≤ 1); non-brain providers unaffected (switch a staging session to an external provider mid-THROTTLE if available — escape hatch works).
- [ ] **Step 6: Restart mid-THROTTLE** `[host]` — restart the gateway during THROTTLE. Expected: re-loads state, stays serial until the persisted deadline (log line `throttle until …`), then normal.
- [ ] **Step 7: Record** — ledger lines with observed values (L before/after, hold durations, compaction counts); any deviation from Expected → systematic-debugging, fix, re-run the step.

---

### Task 10: Rollout + docs

**Files:**
- Modify: `README.md` (status → v0.1.0 shipped; install/enablement per profile home; config reference table; the `settings.targets` shape), `specs/DESIGN.md` (mark build-time checks resolved with one line each; note Rulings R1–R3).

- [ ] **Step 1: Per-profile enablement** `[host]` — operator applies the same `plugins.entries` block to all 9 homes (root + 8 profiles — the verified audit: all route to the brain), with `settings.user_agent: true` (attribution is ON in the rollout, per design decision); gateway restarts (staggered if production). Expected: each home's gateway log shows registration; `oom_status` tool returns sane per-target state; no behavior change for small sessions (gate admits instantly); the brain router's dashboard shows requests attributed as `Hermes/<profile>` (`Hermes/<profile>:<cron job>` for scheduler runs).
- [ ] **Step 2: Optional `log_only` soak** — if the operator wants one: enable with `settings.log_only: true` for 24 h, review would-be-hold log lines, flip off. (Default rollout: direct on — the gate is protective from minute zero by design.)
- [ ] **Step 3: Update README + DESIGN.md**, commit, push. `[host]` pull on the hermes host. Expected: docs match shipped behavior; rollback note accurate (remove `plugins.entries` per profile; engine untouched).
- [ ] **Step 4: Final verification** — full unit suite green locally; `hermes plugins doctor` green on the hermes host; `git log` clean (one commit per task). Ledger: all tasks complete.
