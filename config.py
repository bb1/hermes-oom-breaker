"""Configuration knobs and target definitions for hermes-oom-breaker.

Hermes-free: settings arrive via an injected ``get_setting`` callable so the
module is unit-testable without hermes. Invalid values never raise — they fall
back to the default and are reported in ``Config.warnings``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class Target:
    """One gated engine target (an engine profile behind the brain router)."""

    name: str
    max_model_len: int
    kv_budget: int
    base_urls: List[str]
    metrics_url: str = ""
    model: str = ""


@dataclass
class Config:
    race_reserve: int = 5000
    recheck_cadence: float = 1.5
    hold_escape: float = 300.0
    hit_dedupe: float = 10.0
    compact_above: int = 50_000
    escalate_window: float = 900.0
    throttle_duration: float = 1800.0
    max_hold: float = 600.0
    log_only: bool = False
    probe_authoritative: bool = False
    user_agent: bool = False
    targets: Dict[str, Target] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


_KNOWN_KNOBS: Dict[str, type] = {
    "race_reserve": int,
    "recheck_cadence": float,
    "hold_escape": float,
    "hit_dedupe": float,
    "compact_above": int,
    "escalate_window": float,
    "throttle_duration": float,
    "max_hold": float,
    "log_only": bool,
    "probe_authoritative": bool,
    "user_agent": bool,
}


def _valid_number(value: Any, type_: type) -> bool:
    if isinstance(value, bool):
        return False
    if type_ is int:
        return isinstance(value, int)
    if type_ is float:
        return isinstance(value, (int, float))
    return False


def _coerce(value: Any, type_: type) -> Any:
    if type_ is float:
        return float(value)
    return value


def _parse_targets(raw: Any, warnings: List[str]) -> Dict[str, Target]:
    targets: Dict[str, Target] = {}
    if raw is None:
        return targets
    if not isinstance(raw, dict):
        warnings.append(f"targets: expected mapping, got {type(raw).__name__}; using none")
        return targets
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            warnings.append(f"targets[{name}]: not a mapping; skipped")
            continue
        try:
            maxlen = spec["max_model_len"]
            budget = spec["kv_budget"]
            urls = spec["base_urls"]
            if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen <= 0:
                raise ValueError("max_model_len must be a positive int")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise ValueError("kv_budget must be a positive int")
            if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u for u in urls):
                raise ValueError("base_urls must be a non-empty list of str")
            metrics = spec.get("metrics_url", "") or ""
            if not isinstance(metrics, str):
                raise ValueError("metrics_url must be str")
            model = spec.get("model", "") or ""
            if not isinstance(model, str):
                raise ValueError("model must be str")
            targets[str(name)] = Target(str(name), maxlen, budget, list(urls), metrics, model)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"targets[{name}]: {exc}; skipped")
    return targets


def load_config(get_setting: Callable[[str, Any], Any]) -> Config:
    """Build a Config from a settings reader; every invalid value warns, never raises."""
    warnings: List[str] = []
    values: Dict[str, Any] = {}
    for key, type_ in _KNOWN_KNOBS.items():
        value = get_setting(key, None)
        if value is None:
            continue
        if type_ is bool:
            if not isinstance(value, bool):
                warnings.append(f"{key}: expected bool, got {value!r}; using default")
                continue
            values[key] = value
        elif _valid_number(value, type_):
            values[key] = _coerce(value, type_)
        else:
            warnings.append(f"{key}: expected {type_.__name__}, got {value!r}; using default")
    cfg = Config(**values)
    cfg.warnings = warnings
    cfg.targets = _parse_targets(get_setting("targets", None), warnings)
    return cfg
