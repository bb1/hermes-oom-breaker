---
type: Plan
title: hermes-oom-breaker design (agreed, extension points verified vs official docs, ready to build)
description: "Full design for the hermes-side plugin that prevents vLLM EngineCore OOM crashes (500/502 bursts) under concurrent long-context sessions: an exact ledger-based fit-gate (always on, owned by hermes' llm_execution execution middleware), a reactive compaction breaker (on 5xx, 1-at-a-time, 15 min/30 min escalation), and an on-the-fly learned limit L (every 5xx is a data point; no calibration window). Extension points verified against the hermes-agent official docs + open-source checkout."
---

# `hermes-oom-breaker` — design (agreed; v0.1.0 shipped 2026-11-05)

> Build spec for this repo; the build followed the acceptance gates at the
> bottom. Design verified against the hermes-agent official docs
> (developer guide: plugins, middleware, observer-hooks, model-provider)
> and cross-checked against the open-source hermes-agent checkout.
>
> **Status (2026-11-05):** v0.1.0 implemented, unit-green (115 tests), and
> integration-tested on a live host (see the acceptance gates below). One
> integration finding: the engine's `/metrics` endpoint is auth-protected
> (401), so probe-based budget seeding is unavailable there and L-learning
> runs from 5xx data points only (the designed fallback).

## Problem

Hermes agent sessions (main session, delegation subagents, batch fan-out)
can run large concurrent contexts against a single vLLM engine. When the
sum of in-flight tokens exceeds the engine's KV-cache budget, the engine
OOM-crashes (`EngineCore encountered a fatal error`), in-flight requests die
as 500s, and the 1–3 min cold respawn serves 502s.

Two facts shape the design:

- A single long session (up to `max-model-len`) must keep working at all
  times — only the *combined* concurrent load is the failure mode.
- The fatal allocation is not necessarily the KV tensor itself: vLLM maps
  runtime workspace outside the KV tensor, so the engine can OOM before the
  KV gauge reads 100 %. The KV gauge alone is not a safe admission signal —
  an exact admission ledger is.

## Constraints (binding decisions)
1. **Reactive breaker only** — the breaker activates only after an error is
   encountered. No preemptive compaction.
2. **One long session must always work** — sometimes there is exactly one
   long-context session; this only becomes an issue with several of them.
3. **No percent fudge on the limit** — the state of the engine is known and
   the tokens of outgoing requests are calculable; the gate is exact
   arithmetic, not a margin.
4. **No dedicated calibration window** — learn on the fly; every 5xx error
   is a data point.
5. No KV-cache reduction, no `max-model-len` reduction — engine profile
   untouched.
6. Standalone plugin repo. Hermes policy (`plugins/AGENTS.md` + official
   docs): *plugins never touch core* — no wrapping core methods/modules at
   runtime (the catalog refuses it, "no core override"); use the documented
   surfaces only.

## Architecture — three layers
| layer | role | when |
|---|---|---|
| **1. Fit-gate** | admit or hold each outgoing brain request against the exact limit L | always on |
| **2. Breaker** | on 5xx: stop the retry storm, compact offending sessions 1-at-a-time, re-fire; escalation to hard 1-at-a-time | reactive (any brain 5xx) |
| **3. Adaptive L** | the limit L itself, learned from operation | on-the-fly, always |

### Layer 1 — exact fit-gate (`llm_execution` middleware, no estimation)
**The seam:** hermes' **execution middleware** (official docs:
`/docs/developer-guide/middleware`; checkout: `hermes_cli/middleware.py`,
`agent/turn_api_call.py`). `ctx.register_middleware("llm_execution", cb)`
wraps **the actual provider call**: the callback receives the effective provider
request (full kwargs — `messages`, `model`, `max_tokens`, `base_url`, provider
options) + runtime context (`session_id`, `task_id`, `turn_id`, `api_request_id`,
`provider`, `model`, `api_mode`) + `next_call`. Calling `next_call(request)`
continues the chain to the real provider call; returning its result (or a
translated one) completes it. Hermes' retry, streaming, interrupt, and observer
hooks are preserved — and because the wrap point is the per-attempt call
(`turn_api_call.py`), **every retry attempt re-enters the wrapper**. This is the
only documented surface that owns the request lifecycle: observer hooks are
read-only and fail-open (a timeout/raise skips the callback and the request
proceeds), `ProviderTransport` is format-conversion only, and a model-provider
profile is config + request-shaping — none of them can hold a request.

```
in_flight = Σ over gate-admitted, unfinished brain requests: (prompt_tokens + generated_so_far)
needed    = prompt_tokens + max_tokens   (of the new request, from the kwargs the wrapper sees)
admit     iff  in_flight + needed ≤ L(target model)   [and THROTTLE ⇒ only if in_flight == 0]
else        hold: the wrapper waits (re-check every 1–2 s) before calling next_call
```

- The hold is a wait *inside* the wrapper — no hook timeout applies (it's the
  execution path, not an observer). The only cap is hermes' provider request
  timeout (the wait counts against the attempt's duration): a hold that outlives
  it makes hermes abandon + retry the attempt; the retry re-enters the wrapper
  and is re-held. Documented cost: a held turn during RECOVERY may re-run (same
  as today's 500-retry behavior). Build-time check: `get_provider_request_timeout`
  (`hermes_cli/timeouts.py`) + openai client timeout per profile.
- Fail-open semantics: if the wrapper **raises** before `next_call`, hermes logs
  and runs the base call *without* the middleware — so gate logic must never
  raise; on internal error it admits + logs a loud warning (availability over
  strictness; the breaker covers the hole).
- Ledger entry seeded with the `prompt + max_tokens` upper bound at admission
  (conservative by construction), updated from the response usage when
  `next_call` returns (streaming: from chunk/terminal usage).
- **L is keyed by target model**: the brain may be routed to multiple engine
  profiles; each target has its own L. The wrapper sees each call's
  `provider`/`base_url`/`model` → only brain calls are gated; every other
  provider lane passes through untouched.
- **Provider/model switching is preserved** — the wrapper wraps call
  *execution*, it does no routing: `/model` switches, `providers:` blocks, the
  model picker, delegation routing, and engine-profile switches all keep
  working unchanged — a switched-to engine simply has its own L entry. THROTTLE
  serializes brain calls only; every other provider keeps full concurrency
  (switching *to* another provider is a valid mid-THROTTLE escape hatch).
- The engine's own `vllm:gpu_cache_usage_perc` (polled from its `/metrics`
  endpoint by the plugin on a background thread) = **cross-check + liveness
  only** (port dead = engine down = hold all). The gauge can lag during prefill
  bursts — on the deployment that motivated this design, the engine OOMed with
  the gauge well below the true admitted total; the ledger can't have that gap.
- `pre_api_request` / `post_api_request` observer hooks (official observer-hooks
  doc: `pre_api_request` carries `max_tokens`, `approx_input_tokens`,
  sanitized `request`; `post_api_request` carries `usage`, `api_duration`)
  corroborate the ledger from the hermes side.
- **Auxiliary calls** (compression, titling, MoA — `pre_auxiliary_call`
  observers) are also big brain requests: the breaker's 1-at-a-time compaction
  sequencing is their gate; whether they flow through the `llm_execution`
  wrapper is a build-time check.

**Day-1 guarantee (before any learning has happened):** L is seeded at the
target's `max-model-len` (L₀). A session up to maxlen fits an idle engine
(`0 + maxlen ≤ L₀`) and runs untouched; a second concurrent session whose sum
with the in-flight one exceeds L₀ is **held** in the wrapper, re-checking
every 1–2 s, and fires the instant the first completes. The crash scenario this
plugin exists to prevent — two big sessions summing past the KV budget — is
prevented deterministically on day one, with zero 5xx and zero compaction,
before a single data point is learned.

### Layer 1b — optional attribution (llm_request middleware, off by default)
The brain router's dashboard should show *which* hermes profile (and which
cron job, for scheduler runs) issued each request — the big requests are the
ones that matter. When enabled, the plugin stamps the outgoing request's
`User-Agent` (which is otherwise the OpenAI SDK's default) with
`Hermes/<profile>`, or `Hermes/<profile>:<cron job>` when a cron execution is
active. Profile comes from hermes' own per-turn resolver; the cron job from the
scheduler's execution-identity context. Observability only: any resolution
failure leaves the request untouched — this feature must never break a turn.
Off by default; **on in the rollout**.

### Layer 2 — reactive breaker (backstop)
Any 5xx the gate can't prevent (the workspace edge above, engine restarts, host
reboots, ledger gaps) is still handled — the breaker generates the learning
data points and keeps the system moving:

```
NORMAL ──5xx──▶ RECOVERY
  1. register one "hit" (dedupe: all 5xx within ~10 s of the first = one hit)
  2. the wrapper owns the call — failed requests are held (RECOVERY holds big
     ones until compaction completes); hermes' own retry of the originals
     re-enters the llm_execution wrapper (every attempt does) and waits —
     no need to cancel hermes' retry loop (the error hook is observational)
  3. force-compact (cooldown-bypassing `force` flag) the offending in-flight
     sessions (context > ~50k) — **1 at a time** (a big-session compaction is
     itself a big request); completion signalled by the `post_compaction` hook
  4. re-fire the compacted requests concurrently → back to NORMAL
     (compaction at target_ratio ~0.2 shrinks large sessions to tens of k
      tokens; the re-fired total lands well under budget)

RECOVERY or NORMAL ──5xx within 15 min of last hit──▶ THROTTLE
  hard 1-at-a-time (all brain calls) for 30 min → NORMAL
  (flat 30 min; during THROTTLE a single session can never OOM:
   1 × ≤ maxlen < KV budget, as long as maxlen ≤ budget)
```

- **400 fallback:** if a session's context exceeds `max-model-len`, the
  compaction request itself 400s → skip straight to THROTTLE and flag the
  operator — that session is structurally unserviceable on this engine (hermes
  believes a bigger window than the engine has).
- Only the offending (in-flight, big) sessions get compacted — not everything;
  `protect_last_n` / `protect_first_n` preserve working context.
- All breaker state persists (see below) — a hermes restart mid-recovery does not
  lose the 15-min window, the held requests, or the THROTTLE deadline.

### Layer 3 — adaptive limit L (learn on the fly)
No calibration window. L converges from real operation; every 5xx is a data point
(keyed per target model, as the gate is):

| event | rule |
|---|---|
| seed (first deploy) | `L₀ = max-model-len of that target` — the profile's own design point: vLLM's boot line "Maximum concurrency for N tokens per request: ~1.0x" means it fits exactly one max-length request |
| any brain 5xx | `L[target] := min(L[target], Y − ε)` where `Y` = in-flight at hit time from the **in-process ledger** (exact) and `ε = 5k` fixed race reserve |
| clean busy episode peaking above L | `L[target] := max(L[target], peak)` — upward learning is driven by real workloads, not clocks |
| request held while engine is **idle** for 5 min (`hold_escape`) | fire it anyway → success raises L to the observed peak; a 5xx lowers L *and* the breaker compacts the session, so it is still served (solo-long sessions can never deadlock; also the only upward-exploration path) |
| profile / model / vLLM-version change | `L[target] := L₀` for that target (manual reset command; auto-detection a build-time detail) |

- Cap: `L ≤ KV budget of that target` (read from the engine's boot log — beyond
  it vLLM 400s anyway).
- Self-healing false positives: an unrelated engine death (host reboot) at heavy
  load tightens L one notch; the next clean episode restores it. We deliberately
  do not try to distinguish OOM from reboot (unreliable from the hermes box).
- **Total cost of learning: ≤ 1 incident per target, ever** (until a profile
  change resets it). Residual slip-through: a concurrent total at the very top of
  the budget might OOM once (the runtime workspace lives outside both the KV
  tensor and the ledger); that single 5xx tightens L just below the observed
  point, exact from then on. Day-1 (L₀) already prevents the two-big-sessions
  scenario deterministically.

## Hermes extension points (verified — official docs + checkout)
The contract was verified against the official site:
`/docs/developer-guide/plugins` (authoring), `/observer-hooks` (telemetry
contract), `/middleware` (behavior-changing wrappers), `/model-provider-plugin`,
and cross-checked against the open-source hermes-agent checkout.

Authoring contract: standalone plugin repo → `~/.hermes/plugins/<name>/` with
`plugin.yaml` + `register(ctx)`; `hermes plugins doctor/validate` tests a plugin
in isolation; durable state via `plugin_data_dir()` / `plugin_db()` (profile-aware,
survives `plugins update/remove`); dependencies via `python_dependencies` /
`pyproject.toml`; **middleware only runs for enabled plugins** (per-profile
enablement).

Verified seams (official docs + checkout file refs):

| seam | where | role in oom-breaker |
|---|---|---|
| **`llm_execution` middleware** — `ctx.register_middleware("llm_execution", cb)`; `run_llm_execution_middleware(request, next_call, …)` | docs: Middleware (wraps the actual provider call; preserves retry/streaming/interrupt; per-attempt); checkout: `hermes_cli/middleware.py`, `agent/turn_api_call.py`, `hermes_cli/plugins.py` | **the gate + breaker serialization** — holds before `next_call`, admits, updates the ledger from the response |
| `api_request_error` observer hook — payload: `status_code`, `error{type,message}`, `retry_count`, `max_retries`, `retryable`, `session_id`, `model`, `provider`, `base_url`, sanitized `request` | docs: Observer Hooks (Request-Scoped API Hooks); checkout: `agent/api_request_hooks.py`, fired from `agent/turn_api_error.py` | **breaker trigger** — 500/502 per attempt; burst dedupe plugin-side (the wrapper also sees `next_call` exceptions) |
| `pre_api_request` / `post_api_request` observers — `pre` carries `session_id`, `api_request_id`, `model`, `provider`, `base_url`, `max_tokens`, `approx_input_tokens`, sanitized `request`; `post` carries `usage`, `api_duration`, `finish_reason` | docs: Observer Hooks; checkout: `agent/turn_api_request.py` | telemetry — ledger corroboration + brain-vs-external filter |
| `pre_auxiliary_call` / `post_auxiliary_call` observers (`aux_task`) | docs: Observer Hooks | watch the compression (auxiliary) requests |
| `post_compaction` hook — `session_id`, `old_session_id`, `kind`, `reason` | checkout: `agent/compaction_events.py` | breaker's "compaction done" signal (1-at-a-time driver) |
| `compress_context(agent, messages, system_message, *, force=True, …)` | checkout: `agent/conversation_compression.py` | breaker's forced per-session compaction (`force` clears the summary-failure cooldown) |
| `estimate_request_context_tokens(api_payload)` | checkout: `agent/chat_completion_helpers.py` | the token counter — same function hermes itself uses → ledger parity for free |
| `llm_request` / `tool_request` / `tool_execution` middleware, `ProviderTransport`, model-provider profiles, `pre_llm_call` | docs: Middleware / model-provider-plugin; checkout: `agent/transports/base.py`, `agent/turn_context.py` | **not used** — request-shaping/format-only/turn-context seams; recorded so a future builder doesn't re-derive this |

**Decisive finding:** the `llm_execution` middleware is the documented request-
lifecycle seam — the gate holds a request by simply not calling `next_call` yet.
No sidecar, no new provider, no base_url re-wiring, no SSE handling, no core
patches; the plugin is a single general plugin, enabled per profile.

## Deployment (per-profile — the key operational detail)
- The plugin must be enabled in **every profile that calls the brain**
  (`plugins.enabled` entry per profile, or `hermes -p <profile> plugins
  enable …`): middleware "only runs for enabled plugins" and plugin discovery is
  **per profile home**. A profile left without it bypasses the gate and the
  ledger → undercount → the original OOM. (Which homes actually route to the
  brain, incl. delegation lanes: per-profile audit at build time.)
- **Process topology check (build-time):** if the gateway is one process serving
  all profiles (expected), the ledger/L/breaker state is a process-global
  module-level singleton with idempotent `register(ctx)` (the established
  long-lived-hermes-plugin pattern — cross-thread SQLite
  `check_same_thread=False` + `RLock`). If hermes runs one process per profile,
  the gate stays the middleware but the ledger moves to the shared durable store
  (`plugin_data_dir()` + file locking / atomic CAS).
- Hot-reload caveat: control-socket `reload-plugins` re-registers handlers but
  does **not** re-import the plugin module → a full gateway restart is required
  after plugin code changes.

## State file
- Via `plugin_data_dir()` / `plugin_db()` (doc-verified durable-state API).
- Contents: per-target `L` + L-history ring, `last_hit_ts`, escalation state +
  `throttle_until_ts`, held/compacting requests, ledger snapshot.
- Atomic write on every change; reloaded at plugin start — a hermes restart
  mid-THROTTLE stays serial; mid-RECOVERY re-enters with the 15-min window intact.

## Config knobs (defaults, all overridable)
| knob | default | notes |
|---|---|---|
| `race_reserve` (ε) | 5,000 tokens | fixed constant, not a percentage |
| `recheck_cadence` | 1–2 s | held-request polling inside the wrapper |
| `hold_escape` | 5 min | idle-engine held request fires anyway |
| `hit_dedupe` | 10 s | burst → one hit |
| `compact_above` | 50k context | smaller sessions just retry |
| `escalate_window` | 15 min | 2nd hit inside → THROTTLE |
| `throttle_duration` | 30 min | flat |
| ~~`reprobe`~~ | *removed in v0.1.0* | was a silent no-op (never consumed); hold-escape admission is the only automatic upward path for L (manual `oom_reset` is the other) |
| `log_only` | off (rollout soak) | gate logs would-be-holds without enforcing |
| `user_agent` | off (on in rollout) | stamp outgoing requests with `User-Agent: Hermes/<profile>[:<cron job>]` for dashboard attribution |

## Build-time verification (all 8 resolved on-host during the build)
1. **Per-attempt re-entry** — ✅ resolved: hermes' retry loop re-invokes the
   `llm_execution` chain per attempt (verified against the host checkout's
   per-call module); a held request re-enters cleanly on each retry, and the
   release-on-error path covers every dead attempt (observed live: a
   mid-stream engine kill drove several failed attempts, each releasing its
   own ledger entry).
2. **Auxiliary calls** — ✅ resolved: auxiliary provider calls that flow
   through the wrapper are tracked; those that do not (e.g. internal
   compression passes) are covered by the breaker's 1-at-a-time compaction
   sequencing — acceptable by design, as written above.
3. **Middleware payload fields** — ✅ resolved: `max_tokens`, `session_id`,
   `provider`, `base_url`, `model` are present in the `llm_execution`
   context on the host (the gate sized the staged 172k-token request from
   them, exactly); the `api_request_error` payload carries the same ids plus
   `status_code` (None for non-HTTP failures) — both seam shapes verified
   against the running host.
4. **Session `agent` handle** for `compress_context` — ✅ resolved: the
   compaction driver does not need a plugin-context session handle; it drives
   the gateway's own `session.mutate(compress)` control path (host-verified
   seam), which is also where the fallback `post_compaction` signal comes from.
5. **Per-target budgets** — ✅ resolved: each target's `max_model_len` and
   `kv_budget` come from its engine boot log; L seeds at
   `min(max_model_len, kv_budget)` and is capped at the kv budget (staged
   state file shows the seeds).
6. **Per-profile audit** — ✅ resolved: every profile home that routes to the
   brain was audited on the host; the rollout enables the plugin per profile
   (see Rollout & rollback).
7. **Process topology** — ✅ resolved: the host runs ONE gateway process
   serving all profiles → process-global singleton with idempotent
   `register(ctx)` (as the design anticipated); plugin settings are
   process-scoped (root home config), plugin state is per profile
   (`<profile-home>/plugin-data/...`).
8. **`hermes plugins doctor` green** — ✅ resolved: doctor green on the host
   (hermes-free import of the plugin module confirmed — its hermes imports
   are lazy, inside resolver bodies); the plugin registers silently, and
   behavior + state file are the registration proof.

## Acceptance gates (status as of v0.1.0, 2026-11-05)
1. **Unit — ✅ complete:** 115 tests pass (gate arithmetic incl. the two-big
   replay, THROTTLE serialization incl. the hold re-check, release-on-error
   on the canonical path, FSM transitions (dedupe, escalation, THROTTLE
   expiry), learner updates (lower on 5xx, hold-escape raise, reset), state
   persist/reload, fail-open-on-raise wrapper).
2. **Integration (engine up, no production traffic) — ✅ complete with
   documented coverage split** (staged on a live host inside a pre-dawn
   maintenance window):
   a. two large concurrent sessions → second held in the wrapper, zero 5xx
      while held: the live replay queued the second session for the first's
      duration (zero 5xx); after the first completed, the shared session's
      combined context exceeded the model window and hermes' native
      compaction exhausted it before the provider call (`context_overflow`,
      session-level — no engine involvement). The engine-level two-big
      hold-and-fire (both in flight on the engine) is pinned by unit tests
      (ledger two-big scenario + gate); a live two-big replay was timeboxed
      behind the host's nightly render window.
   b. solo near-maxlen on a cold engine → ran to completion (the session's
      own native compaction kept it under the context window across 3 API
      calls; the gate admitted the ~172k request exactly); ✅
   c. engine death mid-in-flight → the live run exercised the seam's hard
      case: the failure arrived as a non-HTTP/connection error (the API
      server's graceful shutdown held the stream open; the router reported
      the dead engine as `ready` for minutes), and the pre-fix handler
      recorded nothing for it — the dead stream's ledger entry leaked until
      restart (permanent under-admission). v0.1.0's release-on-error fix
      (every failure releases; 5xx only adds the hit) covers both arrival
      shapes, pinned by unit tests. A live 5xx-arrival replay (L tightening
      observed in the state file) was timeboxed behind the render window.
   d. second simulated hit within 15 min → THROTTLE 30 min; restart hermes
      mid THROTTLE → stays serial until the persisted deadline: covered by
      unit tests (incl. the hold re-check staying serial during THROTTLE);
      a live replay was timeboxed (see c).
3. **Production soak — ⏳ in progress:** all brain-facing profiles enabled,
   `user_agent` stamping on; zero-5xx target under concurrent big sessions;
   L history + state file monitored.

## Rollout & rollback
1. Build the plugin; optionally one `log_only` soak to confirm gate decisions.
2. Per-profile rollout: add the `plugins.enabled` entry (root first, then the
   delegation lanes' homes), gateway restart per the hot-reload caveat. No
   provider/base_url/config-model changes anywhere.
3. **Rollback:** remove the `plugins.enabled` entry (per profile). The engine
   profile is untouched by this design — nothing to revert on the engine side.
4. Engine-side hard floor (`--max-num-seqs 1`, drop `--async-scheduling`) remains
   documented as an *optional* floor if a second client ever appears — **not**
   part of this plan.

## Follow-ups (separate threads)
- L auto-reset detection on profile/vLLM change (manual command ships first).
- Non-OOM engine deaths (host reboots, power loss) exercise the breaker's
  tighten-then-self-heal path — worth watching in the soak.
