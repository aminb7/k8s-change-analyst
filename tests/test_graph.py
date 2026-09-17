import threading
import time
from datetime import datetime, timezone

from change_analyst.graph import Deps, build_graph
from change_analyst.models import Change, Findings, Verdict

T = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def mk(i: int) -> Change:
    return Change(id=f"c{i}", kind="rollout", namespace="demo", workload="podinfo", changed_at=T, description="d")


def fake_deps(deadline=float("inf"), clock=time.monotonic, active=None):
    lock = threading.Lock()
    active = active if active is not None else {"now": 0, "max": 0}

    def agent(name):
        def run(change):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.05)
            with lock:
                active["now"] -= 1
            return Findings(agent=name, status="ok", observations=[f"{name} {change.id}"])
        return run

    def judge(change, metrics, logs):
        assert metrics.agent == "metrics" and logs.agent == "logs"
        return Verdict(change_id=change.id, verdict="no_impact", confidence="high", summary="fine")

    return Deps(investigate_metrics=agent("metrics"), investigate_logs=agent("logs"),
                judge=judge, deadline=deadline, clock=clock)


def test_every_ready_change_gets_a_result():
    out = build_graph(fake_deps()).invoke({"ready": [mk(1), mk(2), mk(3)], "results": [], "deferred": []},
                                          {"max_concurrency": 2})
    assert sorted(r.change.id for r in out["results"]) == ["c1", "c2", "c3"]
    assert all(r.metrics.observations == [f"metrics {r.change.id}"] for r in out["results"])
    assert out["deferred"] == []


def test_metrics_and_logs_agents_run_in_parallel():
    active = {"now": 0, "max": 0}
    build_graph(fake_deps(active=active)).invoke({"ready": [mk(1)], "results": [], "deferred": []})
    assert active["max"] == 2


def test_no_ready_changes_finishes_cleanly():
    out = build_graph(fake_deps()).invoke({"ready": [], "results": [], "deferred": []})
    assert out["results"] == [] and out["deferred"] == []


def test_changes_after_deadline_are_deferred():
    out = build_graph(fake_deps(deadline=10.0, clock=lambda: 11.0)).invoke(
        {"ready": [mk(1), mk(2)], "results": [], "deferred": []})
    assert out["results"] == []
    assert sorted(c.id for c in out["deferred"]) == ["c1", "c2"]
