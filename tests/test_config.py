import os
from pathlib import Path

from change_analyst.config import Settings


def _clear_ca_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key)


def test_defaults(monkeypatch):
    _clear_ca_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.prometheus_url == "http://localhost:9090"
    assert s.namespaces == ["demo"]
    assert s.llm_model == "sonnet"
    assert s.min_age_min == 5
    assert s.max_changes_per_run == 5
    assert s.state_path == Path("state.json")


def test_env_override(monkeypatch):
    _clear_ca_env(monkeypatch)
    monkeypatch.setenv("CA_NAMESPACES", '["demo", "shop"]')
    monkeypatch.setenv("CA_MIN_AGE_MIN", "3")
    s = Settings(_env_file=None)
    assert s.namespaces == ["demo", "shop"]
    assert s.min_age_min == 3
