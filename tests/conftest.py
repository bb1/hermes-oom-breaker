"""Pytest bootstrap: load the plugin (the repo-root package) the way hermes
loads it — as a PACKAGE.

hermes' directory-plugin loader (hermes_cli/plugins_loader.py,
``_load_directory_module``) imports a plugin's ``__init__.py`` as
``hermes_plugins.<slug>`` with ``__path__`` set to the plugin directory, so
every intra-plugin import is RELATIVE (the plugin dir is never put on
sys.path). The modules in this repo follow that contract; a bare
``import adapter`` would fail here, so this conftest mirrors the loader:

1. exec ``__init__.py`` once as the synthetic package ``oom_breaker_pkg``
   (with ``__path__`` = the repo root, like hermes'
   ``submodule_search_locations=[plugin_dir]``);
2. alias every submodule under its bare name in ``sys.modules`` so the
   existing test files keep their bare imports (``from adapter import
   GateController``) — resolving to the SAME module objects as the package's
   submodules, so cross-module identity assertions (``is``) hold across the
   suite.

Running under pytest therefore exercises the exact same module graph the
gateway will (relative imports inside the package), which is also why the
hermes-free import contract is proven by the suite: any top-level hermes
import would raise here.
"""
import importlib
import importlib.util
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
_PKG = "oom_breaker_pkg"

if _PKG not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PKG, os.path.join(_ROOT, "__init__.py"),
        submodule_search_locations=[_ROOT])
    pkg = importlib.util.module_from_spec(spec)
    pkg.__package__ = _PKG
    pkg.__path__ = [_ROOT]
    sys.modules[_PKG] = pkg
    spec.loader.exec_module(pkg)
    # Every submodule exists now: __init__ imports the four it needs and the
    # adapter's own (now relative) imports pull the rest in transitively.
    for _sub in ("adapter", "compaction_driver", "config", "state",
                 "ledger", "learner", "breaker", "metrics", "notify"):
        sys.modules[_sub] = sys.modules["%s.%s" % (_PKG, _sub)]
