from config import load_config


def test_defaults():
    cfg = load_config(lambda k, d=None: d)
    assert cfg.race_reserve == 5000 and cfg.recheck_cadence == 1.5
    assert cfg.hold_escape == 300.0 and cfg.hit_dedupe == 10.0
    assert cfg.compact_above == 50_000 and cfg.escalate_window == 900.0
    assert cfg.throttle_duration == 1800.0 and cfg.max_hold == 600.0
    assert not cfg.log_only and not cfg.probe_authoritative
    assert not cfg.user_agent
    assert cfg.targets == {} and cfg.warnings == []


def test_settings_override_and_targets():
    def get(k, d=None):
        vals = {"race_reserve": 999,
                "user_agent": True,
                "targets": {"a": {"max_model_len": 100, "kv_budget": 120,
                                  "base_urls": ["http://x:1"], "metrics_url": "http://x:2/metrics"}}}
        return vals.get(k, d)

    cfg = load_config(get)
    assert cfg.race_reserve == 999
    assert cfg.user_agent
    t = cfg.targets["a"]
    assert t.max_model_len == 100 and t.kv_budget == 120 and t.metrics_url == "http://x:2/metrics"


def test_invalid_values_keep_default_and_warn():
    cfg = load_config(lambda k, d=None: "garbage" if k == "race_reserve" else d)
    assert cfg.race_reserve == 5000
    assert any("race_reserve" in w for w in cfg.warnings)


def test_target_model_field_parses():
    def get(k, d=None):
        return {"targets": {"a": {"max_model_len": 100, "kv_budget": 120,
                                  "base_urls": ["http://x:1"], "model": "alpha"}}}.get(k, d)

    cfg = load_config(get)
    assert cfg.targets["a"].model == "alpha"


def test_target_model_defaults_to_empty():
    def get(k, d=None):
        return {"targets": {"a": {"max_model_len": 100, "kv_budget": 120,
                                  "base_urls": ["http://x:1"]}}}.get(k, d)

    cfg = load_config(get)
    assert cfg.targets["a"].model == ""


def test_target_nonstr_model_skipped_with_warning():
    def get(k, d=None):
        return {"targets": {"bad": {"max_model_len": 100, "kv_budget": 120,
                                    "base_urls": ["http://x:1"], "model": ["alpha"]},
                            "ok": {"max_model_len": 100, "kv_budget": 120,
                                   "base_urls": ["http://x:2"]}}}.get(k, d)

    cfg = load_config(get)
    assert "bad" not in cfg.targets
    assert "ok" in cfg.targets and cfg.targets["ok"].model == ""
    assert any("targets[bad]" in w for w in cfg.warnings)
