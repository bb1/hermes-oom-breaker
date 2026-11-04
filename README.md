# hermes-oom-breaker

A [Hermes](https://hermes-agent.nousresearch.com) plugin that prevents GPU OOM
crashes (500/502 bursts) when concurrent long-context agent sessions overflow a
vLLM engine's KV-cache budget.

## Status

**Design complete (2026-11-04) — implementation pending.** v0.0.1 is a loadable
skeleton: it registers nothing and changes no behavior. The agreed design is
[DESIGN.md](specs/DESIGN.md); the build follows its acceptance gates.

## Problem

Hermes agent sessions (main + delegated subagents) run 150k–175k-token
concurrent contexts against a single vLLM brain. Two such sessions exceed the
engine's KV-cache budget, the engine OOM-crashes (`EngineCore encountered a
fatal error`), in-flight requests die as 500s, and the 1–3 min cold respawn
serves 502s. A single long session must keep working at all times — only the
*combined* concurrent load is the failure mode.

## How it works (agreed design)

1. **Exact fit-gate (always on)** — an `llm_execution` execution-middleware
   wrapper around brain provider calls: admit iff `in-flight tokens + needed
   tokens ≤ L`, else hold (never reject) until KV frees. Hermes is the only
   brain client, so its own admission ledger is the exact in-flight accounting —
   no estimation, no percentage margin.
2. **Reactive breaker (backstop)** — on any brain 5xx: register the hit
   (deduped), hold retries, force-compact the offending big sessions
   1-at-a-time, re-fire the compacted requests concurrently; a second hit
   within 15 min → hard 1-at-a-time for 30 min. No preemptive compaction: the
   breaker activates only after an error.
3. **Adaptive limit L (on-the-fly learning)** — seeded at each target's
   `max-model-len`; every 5xx lowers L below the observed in-flight total;
   clean episodes that exceed L raise it. No calibration window — every 5xx is
   a data point. Expected learning cost: ≤ 1 incident per engine target, ever.

Provider/model switching is unaffected: the wrapper only gates calls targeting
the local brain (L is keyed per target model); every other provider keeps full
concurrency, including during the breaker's 30-min throttle.

## Install (once implemented)

```sh
git clone https://github.com/bb1/hermes-oom-breaker ~/.hermes/plugins/hermes-oom-breaker
# per profile that talks to the brain:
hermes -p <profile> plugins enable hermes-oom-breaker
```

(plus the per-profile brain audit — specs/DESIGN.md, build-time check #6)

## Config

Knobs + defaults in [DESIGN.md](specs/DESIGN.md) (race reserve ε 5k · recheck 1–2 s ·
hold-escape 5 min · hit dedupe 10 s · compact-above 50k · escalate window
15 min · throttle 30 min), overridable under
`plugins.entries.hermes-oom-breaker.settings`.

## Development

Build-time verification checklist + acceptance gates in [DESIGN.md](specs/DESIGN.md).
Testing follows the existing plugin-repo pattern (`tests/` +
`hermes plugins doctor`).
