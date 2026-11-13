"""hermes-oom-breaker — v0.2.2: load-gated KV fit-gate (hold only when a
request is in flight) + reactive compaction breaker, preventing vLLM KV-cache
OOM crashes (5xx) under concurrent long-context hermes sessions.

The plugin entry point is ``register(ctx)``: it assembles one
``GateController`` per gateway process (module singleton, idempotent) and
wires hermes' documented surfaces —

- ``ctx.register_middleware("llm_execution", ...)`` — the exact KV fit-gate
  (admits a request that fits the learned limit L, or immediately when the
  engine is idle; holds only when another request is in flight and it is
  over the limit)
- ``ctx.register_middleware("llm_request", ...)`` — optional User-Agent
  attribution for router dashboards (runtime no-op while ``user_agent`` off)
- ``ctx.register_hook("api_request_error", ...)`` — the canonical 5xx hit
  (drives the 1-at-a-time reactive compaction breaker)
- ``ctx.register_hook("post_compaction", ...)`` — the compaction-completion
  fallback signal (the driver's ``on_finished`` is primary)
- ``ctx.register_hook("post_api_request", ...)`` — usage release + the
  defensive 5xx fallback (flag-guarded while the canonical hook is up)
- ``ctx.register_tool("oom_status" / "oom_reset", ...)`` — the operator
  surface (per-target L / in-flight / breaker state; reset L to its seed)
- ``ctx.on_unload(...)`` — stops the metrics probe thread and the compaction
  driver (both teardowns are owned here)

This module imports nothing from hermes at module level: the hermes estimator
swap and everything else hermes-specific happens lazily inside ``register()``
(try/except with a loud warning on failure), so the import stays hermes-free
(doctor must stay green in a hermes-free environment). Sibling project modules
are imported relatively, matching how hermes imports the plugin (as a package
whose ``__path__`` is the plugin directory).
"""
from __future__ import annotations

import json
import logging

# Relative sibling imports: hermes loads a directory plugin as a PACKAGE
# (hermes_plugins.<slug> with __path__ = the plugin dir — see
# hermes_cli/plugins_loader.py), so bare "import adapter" would not resolve
# there. The test harness mirrors this (tests/test_register.py).
from .adapter import GateController
from .compaction_driver import GatewayDriver
from .config import load_config
from .state import Store

logger = logging.getLogger("oom_breaker")

# One GateController per gateway process.
_controller = None

# Driver seam: the production default is the real (daemon-worker)
# GatewayDriver; tests swap in a threadless FakeDriver (test_register.py).
_driver_cls = GatewayDriver


class _StateShim:
    """Adapts hermes' ``PluginState`` to the ``KeyValueSink`` interface used
    by ``Store`` (state.py): ``get(key, default)`` / ``set(key, value)``.
    Currently 1:1 — kept as a single seam so a signature drift in
    PluginState is a one-line change here, not a Store/learner edit."""

    def __init__(self, state):
        self._state = state

    def get(self, key, default=None):
        return self._state.get(key, default)

    def set(self, key, value):
        return self._state.set(key, value)


def register(ctx) -> None:
    """Plugin entry point (hermes runs it under a per-plugin deadline):
    fast and non-blocking — no I/O beyond the small state write, only
    daemon threads. A second register in the same process is an idempotent
    no-op (nothing re-registered)."""
    global _controller
    if _controller is not None:
        # Loud signal (QA t1): a second register is a no-op — an operator who
        # never re-runs doctor must still see it in the gateway log.
        logger.warning(
            "oom-breaker: register() called again in this process: the "
            "controller is already assembled; re-register is a no-op")
        return

    # -- config + state ----------------------------------------------------
    cfg = load_config(ctx.get_config)
    for w in cfg.warnings:
        logger.warning("oom-breaker config: %s", w)
    store = Store(_StateShim(ctx.state))

    # -- controller + targets (D3' lazy seed: a learned L survives) --------
    controller = GateController(cfg, store)
    controller.load_targets()
    if not cfg.targets:
        logger.warning(
            "oom-breaker: unconfigured: no targets in settings; the gate is "
            "a pass-through (zero behavior change)")

    # -- driver + breaker hooks --------------------------------------------
    driver = _driver_cls()
    controller.install_breaker_hooks(driver)

    # -- estimator swap (R2): use hermes' own sizing when importable -------
    try:
        from agent.chat_completion_helpers import (
            estimate_request_context_tokens as _hermes_estimator)
        controller.estimator = _hermes_estimator
    except Exception:
        logger.warning(
            "oom-breaker: estimator swap failed: could not import hermes' "
            "estimate_request_context_tokens; keeping the default "
            "estimate_tokens (ledger sizing may disagree with hermes' own "
            "accounting)", exc_info=True)

    # -- the exact registration set ------------------------------------------
    ctx.register_middleware("llm_execution", controller.llm_execution)
    ctx.register_middleware("llm_request", controller.ua_llm_request)
    ctx.register_hook("api_request_error", controller.api_request_error)
    ctx.register_hook("post_compaction", controller.post_compaction)
    ctx.register_hook("post_api_request", controller.post_api_request)
    ctx.register_tool("oom_status", "oom_breaker", {}, _oom_status_tool,
                      description="OOM-breaker status: per-target L, "
                                  "in-flight, breaker state; the result "
                                  "is a JSON document string")
    ctx.register_tool("oom_reset", "oom_breaker",
                      {"type": "object",
                       "properties": {"target": {"type": "string"}},
                       "required": ["target"]},
                      _oom_reset_tool,
                      description="Reset a target's learned limit L to its "
                                  "seed; the result is a JSON document "
                                  "string")
    ctx.on_unload(_on_unload)

    # -- metrics probe: only for targets that have a metrics_url ------------
    if any(t.metrics_url for t in cfg.targets.values()):
        controller.probe.start(cfg.targets)

    _controller = controller


def _oom_status_tool(args):
    """``oom_status`` tool handler: the operator status as a JSON document
    string (hermes' tool registry accepts str only)."""
    return json.dumps(_controller.oom_status(), indent=2)


def _oom_reset_tool(args):
    """``oom_reset`` tool handler: reset a target's learned L to its seed;
    the result is a JSON document string (hermes' tool registry accepts
    str only). An unknown target raises ValueError — hermes' dispatch
    converts handler exceptions into the standard error envelope, so the
    wrapper never has two result shapes."""
    return json.dumps(_controller.oom_reset(args.get("target")), indent=2)


def _on_unload() -> None:
    """Plugin unload: Task 8 owns both teardowns — the metrics probe thread
    and the compaction driver. Never raises (observer contract)."""
    controller = _controller
    if controller is None:
        return
    try:
        controller.probe.stop()
    except Exception:
        logger.warning("oom-breaker: probe stop failed", exc_info=True)
    driver = controller.compaction_driver
    if driver is not None:
        try:
            driver.stop()
        except Exception:
            logger.warning("oom-breaker: driver stop failed", exc_info=True)
