"""hermes-oom-breaker — prevents vLLM KV-cache OOM crashes (5xx) under
concurrent long-context hermes sessions.

v0.0.1 is a loadable skeleton (design complete, build pending — see
specs/DESIGN.md): registers nothing, changes no behavior.

Planned surfaces (specs/DESIGN.md — all documented hermes extension points):

- ``ctx.register_middleware("llm_execution", ...)`` — the exact fit-gate +
  breaker serialization; wraps the actual brain provider call and holds
  requests that would not fit in the KV budget
- ``ctx.register_hook("api_request_error", ...)`` — the reactive breaker
  trigger (500/502, per attempt)
- ``ctx.register_hook("post_compaction", ...)`` — compaction-completion
  signal driving the 1-at-a-time compaction sequence
- ``ctx.register_tool("oom_status", ...)`` / ``ctx.register_tool("oom_reset", ...)``
  — ops surface (inspect L / history, reset the learned limit)
"""


def register(ctx) -> None:
    """Plugin entry point. No-op until the build phase lands the design."""
    pass
