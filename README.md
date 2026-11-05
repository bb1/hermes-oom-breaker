# hermes-oom-breaker

A [Hermes](https://hermes-agent.nousresearch.com) plugin that prevents GPU OOM
crashes (500/502 bursts) when concurrent long-context agent sessions overflow a
vLLM engine's KV-cache budget.

## Status

**v0.1.0 shipped (2026-11-05).** Implemented against the agreed design
([DESIGN.md](specs/DESIGN.md)) — exact fit-gate, reactive compaction breaker,
adaptive limit L, optional User-Agent attribution — and integration-tested on
a live host inside a pre-dawn maintenance window:

- a ~172k-token solo session was admitted by the fit-gate on a cold engine and
  ran to completion (hermes' own in-place compaction kept it under the context
  window across multiple API calls — "one long session always works" held);
- a second large session submitted while the first was in flight was queued
  for the first's duration with zero 5xx while held; after the first
  completed, the shared session's combined context (~356k tokens) exceeded the
  model window and hermes' native compaction exhausted it
  (`context_overflow`) before the provider call — a session-level limit, not
  an engine OOM. The engine-level two-big hold-and-fire is pinned by unit
  tests; a live two-big replay was timeboxed behind the host's nightly
  render window.
- the live run also surfaced the seam gap that the final pre-merge review
  caught: when the engine dies, the stream's failure arrives as a
  non-HTTP/connection error, and the pre-fix `api_request_error` handler
  recorded nothing for it (no release, no hit) — the dead stream's ledger
  entry leaked until a gateway restart (permanent under-admission).
  **v0.1.0 fixes that seam**: every failure (5xx *and* non-HTTP) releases the
  dead request's ledger entry; only 500–599 additionally count as a breaker
  hit (L learning + recovery). Pinned by unit tests.

Known limitations (v0.1.0):

- the engine's `/metrics` endpoint is auth-protected (401) on the target
  host, so probe-based budget reads are unavailable there and L-learning runs
  from 5xx data points only — the designed fallback;
- the brain router (llama-swap) can report a dead engine as `ready` for
  minutes after an EngineCore death (stale health) — engine-side behavior,
  not plugin behavior; the breaker's release-on-error seam handles it;
- THROTTLE (second-hit) escalation is unit-pinned; a live second-hit replay
  was timeboxed behind the host's nightly render window.

## Problem

Hermes agent sessions (main + delegated subagents) run 150k–175k-token
concurrent contexts against a single vLLM brain. Two such sessions exceed the
engine's KV-cache budget, the engine OOM-crashes (`EngineCore encountered a
fatal error`), in-flight requests die as 500s, and the 1–3 min cold respawn
serves 502s. A single long session must keep working at all times — only the
*combined* concurrent load is the failure mode.

## How it works

1. **Exact fit-gate (always on)** — an `llm_execution` execution-middleware
   wrapper around brain provider calls: admit iff `in-flight tokens + needed
   tokens ≤ L`, else hold (never reject) until KV frees. Hermes is the only
   brain client, so its own admission ledger is the exact in-flight accounting
   — no estimation, no percentage margin. Dead streams release their ledger
   entries on the `api_request_error` seam (5xx *and* non-HTTP failures).
2. **Reactive breaker (backstop)** — on any brain 5xx: register the hit
   (deduped), hold retries, force-compact the offending big sessions
   1-at-a-time, re-fire the compacted requests; a second hit
   within the escalate window → hard 1-at-a-time (THROTTLE) for the throttle
   duration. No preemptive compaction: the breaker activates only after an
   error.
3. **Adaptive limit L (on-the-fly learning)** — seeded at each target's
   `min(max_model_len, kv_budget)`; every 5xx lowers L below the observed
   in-flight total (the exact ledger number minus a fixed race reserve);
   hold-escape admission (an idle engine holding a request long enough) is the
   automatic upward path, capped at the target's KV budget. No calibration
   window — every 5xx is a data point. Expected learning cost: ≤ 1 incident
   per engine target, ever.

Provider/model switching is unaffected: the wrapper only gates calls targeting
a configured target (matched by base URL + model name); every other provider
keeps full concurrency, including during the breaker's throttle.

## Install

```sh
git clone https://github.com/bb1/hermes-oom-breaker ~/.hermes/plugins/hermes-oom-breaker
```

**Plugin settings are process-scoped**: the `plugins.entries` settings block
belongs in the gateway process's root home config. Each profile that may talk
to the brain enables the plugin in its own profile config. Plugin state is
stored per profile under `<profile-home>/plugin-data/`. A full gateway
restart is required (plugin handlers are not hot-reloaded).

Root home config (`~/.hermes/config.yaml`):

```yaml
plugins:
  entries:
    hermes-oom-breaker:
      settings:
        user_agent: true
        targets:
          <target-name>:
            model: <model name as served by the router>
            max_model_len: <engine --max-model-len>
            kv_budget: <engine KV budget from its boot log>
            base_urls:
              - http://<router-host>:<port>/v1
            # optional; only used when the engine's /metrics is reachable
            metrics_url: http://<engine-host>:<port>/metrics
```

Each brain-facing profile config (`~/.hermes/profiles/<profile>/config.yaml`):

```yaml
plugins:
  enabled:
    - hermes-oom-breaker
```

## Config

Settings live under `plugins.entries.hermes-oom-breaker.settings` (defaults in
brackets; see [DESIGN.md](specs/DESIGN.md) for the full design):

| key | default | meaning |
|---|---|---|
| `targets` | — | mapping of target name → `{model, max_model_len, kv_budget, base_urls[, metrics_url]}`; the gate only tracks calls whose base URL (scheme+host+port) and model match a target |
| `race_reserve` | 5000 | fixed token reserve ε subtracted from the observed in-flight total when learning L down (never a percentage) |
| `recheck_cadence` | 1.5 s | held-request re-check cadence inside the wrapper |
| `hold_escape` | 300 s | a held request fires anyway after this much idle engine time (deadlock guard + the automatic upward path for L) |
| `hit_dedupe` | 10 s | 5xx burst → one hit |
| `compact_above` | 50000 | only sessions above this context size are force-compacted (smaller ones just retry) |
| `escalate_window` | 900 s | second hit inside this window after the first → THROTTLE |
| `throttle_duration` | 1800 s | hard 1-at-a-time duration after escalation |
| `max_hold` | 600 s | deadlock guard: a held request is admitted anyway (tracked) after this |
| `log_only` | false | gate logs would-be-holds without enforcing (rollout soak mode) |
| `probe_authoritative` | false | trust the metrics probe over 5xx learning when available |
| `user_agent` | false | stamp outgoing brain requests with `User-Agent: Hermes/<profile>[:<cron job>]` for dashboard attribution |

Operator tools (registered with the agent): `oom_status` (state, per-target L
and in-flight, last hit, throttle deadline, compaction position) and
`oom_reset <target>` (restore a target's learned L to its seed).
