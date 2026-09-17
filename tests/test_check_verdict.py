import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_verdict.py"


def load():
    spec = importlib.util.spec_from_file_location("check_verdict", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(kind, changed_at, verdict, change_id):
    return {"change": {"id": change_id, "kind": kind, "changed_at": changed_at},
            "verdict": {"verdict": verdict, "confidence": "high"}}


def test_checks_latest_change_of_kind_in_latest_report(tmp_path):
    (tmp_path / "20260916T100000Z.json").write_text(json.dumps({"results": [
        result("rollout", "2026-09-16T09:50:00Z", "regression", "old")]}))
    (tmp_path / "20260916T110000Z.json").write_text(json.dumps({"results": [
        result("rollout", "2026-09-16T10:40:00Z", "improvement", "reset"),
        result("rollout", "2026-09-16T10:50:00Z", "regression", "new"),
        result("configmap", "2026-09-16T10:50:00Z", "no_impact", "cm"),
    ]}))
    check = load()
    assert check.main("rollout", "regression", str(tmp_path)) == 0
    assert check.main("configmap", "regression", str(tmp_path)) == 1


def test_fails_without_reports(tmp_path):
    assert load().main("rollout", "regression", str(tmp_path)) == 1
