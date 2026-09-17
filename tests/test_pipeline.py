import json
import threading
from datetime import timedelta

from factories import T0, FakeApps, FakeCore, container, deployment, replica_set, template

from change_analyst.config import Settings
from change_analyst.llm.client import LLMResult
from change_analyst.pipeline import Runtime, execute_run
from change_analyst.state_store import load_state

NOW = T0 + timedelta(minutes=30)


class ScriptedLLM:
    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system, prompt, json_schema=None):
        with self._lock:
            self.calls += 1
        if "judge" in system.lower():
            return LLMResult(text="", structured={"verdict": "regression", "confidence": "high",
                                                  "summary": "Errors rose.", "evidence": ["5xx up"]})
        return LLMResult(text="", structured={"action": "final", "final": {
            "status": "ok", "observations": ["looked"], "anomalies": [], "data_gaps": []}})


class ExplodingApps(FakeApps):
    def list_namespaced_replica_set(self, namespace):
        raise RuntimeError("cluster unreachable")


def settings(tmp_path):
    return Settings(_env_file=None, reports_dir=tmp_path / "reports", state_path=tmp_path / "state.json",
                    lock_path=tmp_path / "lock", namespaces=["demo"], first_run_lookback_min=60)


def cluster():
    apps = FakeApps(
        replica_sets=[
            replica_set(revision=1, created=T0 - timedelta(days=1)),
            replica_set(revision=2, created=NOW - timedelta(minutes=10),
                        tmpl=template([container(args=["--random-error=true"])])),
        ],
        deployments=[deployment()],
    )
    return FakeCore(), apps


def test_run_analyzes_change_writes_report_and_state(tmp_path):
    s = settings(tmp_path)
    core, apps = cluster()
    llm = ScriptedLLM()
    path = execute_run(s, Runtime(core=core, apps=apps, prom=object(), llm=llm), now_fn=lambda: NOW)

    assert path.exists()
    report = json.loads(path.with_suffix(".json").read_text())
    assert [r["verdict"]["verdict"] for r in report["results"]] == ["regression"]
    assert report["llm_calls"] == 3
    state = load_state(s.state_path)
    assert state.last_run_at == NOW
    assert state.analyzed_change_ids == ["rollout:demo/podinfo:rev2"]
    assert state.pending == []

    later = NOW + timedelta(minutes=10)
    path2 = execute_run(s, Runtime(core=core, apps=apps, prom=object(), llm=ScriptedLLM()), now_fn=lambda: later)
    assert json.loads(path2.with_suffix(".json").read_text())["results"] == []


def test_detection_failure_keeps_last_run_and_reports_error(tmp_path):
    s = settings(tmp_path)
    path = execute_run(s, Runtime(core=FakeCore(), apps=ExplodingApps(), prom=object(), llm=ScriptedLLM()),
                       now_fn=lambda: NOW)
    report = json.loads(path.with_suffix(".json").read_text())
    assert "cluster unreachable" in report["errors"][0]
    assert load_state(s.state_path).last_run_at is None
