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

**v0.2.0 (2026-11-06).** Adds the opt-in debug notices: one-line Telegram
posts to the scope's home channel at the three moments that matter — a
registered 5xx hit on a tracked target (`5xx`), a committed session
compaction (`compacted`), and a request finally admitted after a long hold
(`delayed`: the `max_hold` deadlock guard or the `hold_escape` idle timeout).
Off by default; the direct-Bot-API transport (the fleet's "h3" pattern)
delivers them even while the LLM brain is down. See [Debug posts](#debug-posts).

Known limitations (v0.1.0):

- admission is **per scope**: the host-verified topology is one gateway
  process with the plugin module imported per profile scope, so each scope
  runs its own controller, ledger, and learned L. Concurrent big calls
  *within* one scope are jointly gated (the incident pattern: one profile's
  main session + its subagents); concurrent big calls *across different
  scopes* are each gated only against their own scope's load — a shared
  cross-scope ledger is a follow-up. A 5xx tightens the L of the scope that
  saw the hit;
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

Plugin **code is also per scope**: each scope loads the plugin from its own
home's `plugins/` directory. Clone (or copy) the runtime files — the top-level
`*.py` + `plugin.yaml` — into `~/.hermes/plugins/hermes-oom-breaker/` (root
scope) and into `~/.hermes/profiles/<profile>/plugins/hermes-oom-breaker/`
for each brain-facing profile:

```sh
git clone https://github.com/bb1/hermes-oom-breaker
cp hermes-oom-breaker/{*.py,plugin.yaml} ~/.hermes/plugins/hermes-oom-breaker/
# and likewise into each profile's plugins/ dir
```

**Plugin settings are resolved per scope** (host-verified: one gateway
process, but the plugin module is imported per profile scope — each scope
reads its *own* home config, with no inheritance from the root config).
Every scope that may talk to the brain therefore needs **both**, in its own
home config: the `plugins.entries.hermes-oom-breaker` settings block **and**
`hermes-oom-breaker` in its `plugins.enabled` list. Plugin state is stored
per scope under `<scope-home>/plugin-data/`. A full gateway restart is
required (plugin handlers are not hot-reloaded). An enabled scope whose
config lacks settings degrades to a zero-behavior pass-through and logs a
loud `unconfigured` warning at registration — grep the gateway log for it
after a rollout.

Each brain-facing home config (the root home `~/.hermes/config.yaml` for the
root scope; each profile home `~/.hermes/profiles/<profile>/config.yaml`):

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

Each brain-facing profile config (`~/.hermes/profiles/<profile>/config.yaml`)
carries the **same** `plugins.entries.hermes-oom-breaker` settings block (the
settings are not inherited from the root config) plus the enablement:

```yaml
plugins:
  entries:
    hermes-oom-breaker:
      settings:
        # identical block to the one above
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
| `debug` | false | master switch for the one-line debug posts (off = total no-op) |
| `debug_channel` | `""` | post destination: empty → the scope's own home channel (its home `.env`'s `TELEGRAM_HOME_CHANNEL`); else an explicit `telegram:<chat_id>[:<thread_id>]` |
| `debug_min_interval` | 60.0 s | per-event-type rate limit in seconds (a 5xx burst → one post per interval) |

Operator tools (registered with the agent): `oom_status` (state, per-target L
and in-flight, last hit, throttle deadline, compaction position) and
`oom_reset <target>` (restore a target's learned L to its seed).

### Debug posts

One-line Telegram posts for operator visibility, off by default. With
`debug: true`, one post per event type fires at:

- **`5xx`** — a registered 5xx hit on a tracked target. A burst is one post:
  the breaker's `hit_dedupe` window collapses it into one hit, and the
  per-event rate limit bounds everything after;
- **`compacted`** — a session compaction committed (the driver's completion
  signal, or the `post_compaction` fallback);
- **`delayed`** — a request finally admitted after a long hold: the
  `max_hold` deadlock guard, or the `hold_escape` idle timeout.

Message shape: `[oom-breaker] (label) <one line>` — the label is the
User-Agent attribution convention: the profile name, `profile:cron-job` when
the turn is a cron run — omitted when no profile resolves (a cron job alone
yields no label):

```
[oom-breaker] (infra:nightly-render) 503 on 'brain' (in-flight 172,340 tok) -> recovery, compacting s123
[oom-breaker] (infra) session s123 compacted; queue now empty
[oom-breaker] (infra) request held 612s on 'brain' (max_hold); admitted anyway
```

**Delivery** — the fleet's "h3" pattern: the plugin calls the Telegram Bot
API directly (stdlib `urllib`) with `TELEGRAM_BOT_TOKEN` +
`TELEGRAM_HOME_CHANNEL` from the scope's own home `.env` — no gateway, no
hermes runtime, so the posts still land while the LLM brain is down. The
token is never logged. A set `debug_channel` pins a specific chat/thread
instead (the token still comes from the `.env`); an invalid one warns and
drops the post.

**Rate limit** — one bucket per event type: at most one post per
`debug_min_interval` seconds (a bucket's first post is never rate-limited),
so a 5xx burst is one post per interval.

**Failure mode** — a post is a single attempt; every failure (no home,
missing token or channel, network, Bot API error) is a loud gateway-log
warning and a dropped post — never a raised error. The send runs on a daemon
thread (fire-and-forget) and the synchronous half of a post is a lock plus a
thread spawn, so no hook path ever blocks and the request path is never
broken.

**Enabling** — set `debug: true` in the scope's settings; no
`debug_channel` is needed when the scope's `.env` already carries
`TELEGRAM_BOT_TOKEN` / `TELEGRAM_HOME_CHANNEL`. Set `debug_channel` to pin a
specific chat/thread for scopes whose home channel shouldn't receive the
noise. As with every knob, invalid values warn and keep the default.
