"""Task 8: ``register(ctx)`` assembly.

The module under test is the repo-root package ``__init__.py``. It is loaded
per test as a fresh synthetic top-level module via
``importlib.util.spec_from_file_location`` (the package lives at the repo
root, so a plain package import is not available here). The key pins:

- importing/executing the module must SUCCEED in this hermes-free test env
  (no hermes imports at module level — binding 7);
- ``register(FakeCtx())`` wires the EXACT surface: 2 middlewares
  (``llm_execution``, ``llm_request``), 3 hooks (``api_request_error``,
  ``post_compaction``, ``post_api_request``), 2 tools (``oom_status``,
  ``oom_reset``), 1 ``on_unload`` callback.

``FakeCtx`` records every registration and stands in for hermes' plugin ctx
(``get_config``, ``state``, ``register_middleware``, ``register_hook``,
``register_tool``, ``on_unload``). The driver is swapped to a threadless
``FakeDriver`` through the module's driver seam, so no worker threads are
spawned except where a binding requires the real probe thread.
"""
import ast
import importlib
import importlib.util
import itertools
import json
import logging
import os
import pytest
import sys
import types

from adapter import GateController, estimate_tokens
from compaction_driver import FakeDriver
from config import Config, Target
from state import Store
from test_state import DictSink

_INIT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, "__init__.py"))
_REPO_ROOT = os.path.dirname(_INIT)
_COUNTER = itertools.count()

# hermes-internal top-level packages (never importable in this environment):
# a module-level import of any of them would break the hermes-free import
# contract (binding 7) — the lazy import list the plugin uses (estimator
# swap, UA resolvers, gateway client) all lives inside function bodies.
_HERMES_TOP_LEVEL = {"hermes", "hermes_cli", "hermes_constants", "agent", "cron"}

# Namespace parent mirroring hermes' loader (hermes_cli/plugins_loader.py:
# _load_directory_module creates the "hermes_plugins" namespace package once,
# then imports each directory plugin as "hermes_plugins.<slug>" with
# submodule_search_locations=[plugin_dir]).
_NS = "oom_breaker_test_ns"
if _NS not in sys.modules:
    _ns_pkg = types.ModuleType(_NS)
    _ns_pkg.__path__ = []
    _ns_pkg.__package__ = _NS
    sys.modules[_NS] = _ns_pkg


def fresh_module():
    """Exec ``__init__.py`` as a brand-new synthetic PACKAGE module, mirroring
    hermes' directory-plugin loader (relative sibling imports require the
    package context: ``__package__`` + ``__path__`` set to the plugin dir).

    The sibling submodules are pre-seeded from the bare top-level modules so
    cross-module identity (e.g. ``ctrl.estimator is estimate_tokens``) matches
    the rest of the suite's bare imports.

    The execution itself IS binding 7: it must succeed in this hermes-free
    environment (any top-level hermes import would raise ImportError here).
    """
    name = "%s.under_test_%d" % (_NS, next(_COUNTER))
    spec = importlib.util.spec_from_file_location(
        name, _INIT, submodule_search_locations=[_REPO_ROOT])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name
    mod.__path__ = [_REPO_ROOT]
    for _sub in ("adapter", "compaction_driver", "config", "state",
                 "ledger", "learner", "breaker", "metrics"):
        sys.modules["%s.%s" % (name, _sub)] = importlib.import_module(_sub)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


class FakeState:
    """In-memory stand-in for hermes' ``PluginState`` (``get``/``set``)."""

    def __init__(self):
        self.d = {}

    def get(self, key, default=None):
        return self.d.get(key, default)

    def set(self, key, value):
        self.d[key] = value


class FakeCtx:
    """Records every registration; ``settings`` backs ``get_config``."""

    def __init__(self, settings=None):
        self.settings = settings or {}
        self.middlewares = []
        self.hooks = []
        self.tools = []
        self.unloads = []
        self.state = FakeState()

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_middleware(self, kind, callback):
        self.middlewares.append((kind, callback))

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))

    def register_tool(self, name, toolset, schema, handler, check_fn=None,
                      requires_env=None, is_async=False, description=""):
        self.tools.append({"name": name, "toolset": toolset, "schema": schema,
                           "handler": handler, "description": description})

    def on_unload(self, callback):
        self.unloads.append(callback)


# A fake target (no deployment identifiers): fake host labels, fake sizes.
TARGET = {"max_model_len": 245760, "kv_budget": 254723,
          "base_urls": ["http://brain.test:9999"], "model": "brain-model"}


def register_with(mod, ctx, driver_cls=FakeDriver):
    """Point the module's driver seam at *driver_cls*, register, return the
    controller."""
    mod._driver_cls = driver_cls
    mod.register(ctx)
    return mod._controller


def test_registers_exact_surface_set(caplog):
    """Binding 1: exactly 2 middlewares, 3 hooks, 2 tools, 1 on_unload —
    and the registered callbacks are the controller's own bound methods."""
    mod = fresh_module()
    ctx = FakeCtx()
    ctrl = register_with(mod, ctx)

    assert [k for k, _ in ctx.middlewares] == ["llm_execution", "llm_request"]
    assert [n for n, _ in ctx.hooks] == ["api_request_error", "post_compaction",
                                         "post_api_request"]
    assert [t["name"] for t in ctx.tools] == ["oom_status", "oom_reset"]
    assert len(ctx.unloads) == 1

    # == (not is): bound-method objects are re-created per attribute access;
    # equality compares __self__ + __func__, which is what we are pinning.
    assert ctx.middlewares[0][1] == ctrl.llm_execution
    assert ctx.middlewares[1][1] == ctrl.ua_llm_request
    by_hook = dict(ctx.hooks)
    assert by_hook["api_request_error"] == ctrl.api_request_error
    assert by_hook["post_compaction"] == ctrl.post_compaction
    assert by_hook["post_api_request"] == ctrl.post_api_request

    # Both tools sit in the oom_breaker toolset with the pinned schemas.
    tools = {t["name"]: t for t in ctx.tools}
    for t in ctx.tools:
        assert t["toolset"] == "oom_breaker"
    assert tools["oom_status"]["schema"] == {}
    assert tools["oom_reset"]["schema"] == {"type": "object",
                                            "properties": {"target": {"type": "string"}},
                                            "required": ["target"]}
    assert "OOM-breaker status" in tools["oom_status"]["description"]
    assert "JSON" in tools["oom_status"]["description"]
    assert "Reset a target's learned limit L" in tools["oom_reset"]["description"]
    assert "JSON" in tools["oom_reset"]["description"]

    # Breaker hooks installed: canonical-hook flag set, driver wired.
    assert ctrl.api_error_hook_registered is True
    assert ctrl.compaction_driver is not None

    # hermes-free env: the estimator stays the project default, and the
    # failed swap is logged loudly (no silent fallback).
    assert ctrl.estimator is estimate_tokens
    assert any("estimate_request_context_tokens" in r.message
               for r in caplog.records if r.levelno == logging.WARNING)

    # The tool handlers reach the controller and return JSON document
    # strings (hermes' tool registry accepts str only — the wrappers
    # serialize the controller's dict shape); a bad target RAISES
    # ValueError, which hermes' dispatch converts into the standard
    # error envelope.
    status_doc = tools["oom_status"]["handler"]({})
    assert isinstance(status_doc, str)
    status = json.loads(status_doc)
    assert set(status) == {"state", "L", "in_flight", "last_hit",
                           "throttle_until", "unserviceable", "compaction"}
    assert status["state"] == "normal" and status["L"] == {}
    with pytest.raises(ValueError):
        tools["oom_reset"]["handler"]({"target": "nope"})


def test_second_register_is_idempotent(caplog):
    """Binding 2: a second register in the same process adds nothing."""
    mod = fresh_module()
    ctx = FakeCtx()
    register_with(mod, ctx)
    assert len(ctx.middlewares) == 2 and len(ctx.hooks) == 3
    assert len(ctx.tools) == 2 and len(ctx.unloads) == 1   # the first did work
    snap = (list(ctx.middlewares), list(ctx.hooks), list(ctx.tools),
            list(ctx.unloads), dict(ctx.state.d))
    with caplog.at_level(logging.WARNING, logger="oom_breaker"):
        mod.register(ctx)
    # Loud no-op signal (QA t1): the second register must be visible in logs.
    assert any("no-op" in r.message
               for r in caplog.records if r.levelno == logging.WARNING)
    assert (list(ctx.middlewares), list(ctx.hooks), list(ctx.tools),
            list(ctx.unloads), dict(ctx.state.d)) == snap


def test_stale_pid_field_in_store_is_inert(caplog):
    """Binding 3 (updated v0.2.1): a store blob carrying a foreign/legacy
    ``pid`` field registers cleanly with no stale-state warning; the field
    is inert metadata (not consumed, not rewritten, not deleted)."""
    mod = fresh_module()
    ctx = FakeCtx()
    ctx.state.set("oom_breaker.state", {"pid": 999_999_999})
    with caplog.at_level(logging.WARNING, logger="oom_breaker"):
        register_with(mod, ctx)
    # Registration completed fully.
    assert len(ctx.middlewares) == 2 and len(ctx.hooks) == 3
    assert len(ctx.tools) == 2 and len(ctx.unloads) == 1
    # No "stale state" warning — and no "degraded" claim either.
    assert not any("stale state" in r.message or "degraded" in r.message
                   for r in caplog.records
                   if r.levelno == logging.WARNING)
    # The legacy field is inert: untouched by the register.
    assert ctx.state.d["oom_breaker.state"]["pid"] == 999_999_999


def test_load_targets_preserves_lowered_learned_l():
    """Binding 4 (D3' lazy seed): a learned L — here LOWERED by a past 5xx —
    must survive a restart's load_targets(); the seed only fills targets
    that have no learned L yet."""
    sink = DictSink()
    store = Store(sink)
    store.set("L", {"brain": 100_000})
    store.set("meta", {"brain": {"max_model_len": 245760,
                                 "kv_budget": 254723}})
    cfg = Config()
    cfg.targets = {"brain": Target("brain", 245760, 254723,
                                   ["http://brain.test:9999"], "")}
    g = GateController(cfg, store)
    g.load_targets()
    assert g.learner.current("brain", g.targets) == 100_000

    # ...while a target with no learned L is still seeded.
    cfg.targets["batch"] = Target("batch", 100_000, 150_000,
                                  ["http://batch.test:9998"], "")
    g2 = GateController(cfg, Store(sink))
    g2.load_targets()
    assert g2.learner.current("batch", g2.targets) == 100_000
    assert g2.learner.current("brain", g2.targets) == 100_000   # still preserved


def test_unconfigured_registers_passthrough_and_warns(caplog):
    """Binding 5: no targets in settings → everything still registered
    (doctor-green), the loud unconfigured warning is logged, and no probe
    thread is started (nothing to poll)."""
    mod = fresh_module()
    ctx = FakeCtx()
    with caplog.at_level(logging.WARNING, logger="oom_breaker"):
        register_with(mod, ctx)
    assert len(ctx.middlewares) == 2 and len(ctx.hooks) == 3
    assert len(ctx.tools) == 2 and len(ctx.unloads) == 1
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("unconfigured" in m and "pass-through" in m for m in msgs)
    assert mod._controller.probe._thread is None


class _RecordingDriver(FakeDriver):
    """FakeDriver that records ``stop()`` calls."""

    def __init__(self):
        super().__init__()
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_on_unload_stops_driver_and_probe():
    """Binding 6: the on_unload callback stops the driver AND the probe
    thread (Task 8 owns both teardowns)."""
    mod = fresh_module()
    ctx = FakeCtx(settings={"targets": {
        "brain": {**TARGET, "metrics_url": "http://metrics.test:9999/metrics"}}})
    ctrl = register_with(mod, ctx, _RecordingDriver)
    driver = ctrl.compaction_driver
    assert isinstance(driver, _RecordingDriver)
    # The metrics_url target started the probe (daemon poll thread).
    assert ctrl.probe._thread is not None
    ctx.unloads[0]()
    assert driver.stop_calls >= 1
    assert ctrl.probe._thread is None


def test_import_contract_no_top_level_hermes_imports():
    """Binding 7: the module import must succeed in this hermes-free env
    (fresh_module's exec already proved it — the test would error here
    otherwise), and the static pin: no top-level import of a hermes-internal
    package (the lazy imports live inside function bodies)."""
    mod = fresh_module()
    assert callable(mod.register)
    tree = ast.parse(open(_INIT).read())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level.add(node.module.split(".")[0])
    assert not (top_level & _HERMES_TOP_LEVEL), \
        "top-level hermes imports: %r" % (top_level & _HERMES_TOP_LEVEL)
