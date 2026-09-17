import json
from datetime import timedelta

from factories import T0, FakeApps, FakeCore, deployment, pod

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.logs import error_signatures, make_log_tools, normalize_line

W = Windows(T0 - timedelta(minutes=15), T0, T0 + timedelta(minutes=2), T0 + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T0, description="d")


def test_normalize_line_masks_variable_parts():
    line = "2026-09-16T10:00:01Z error dial tcp 10.0.3.7:6379 req=3f2a9c1e-1111-2222-3333-444455556666 after 250ms"
    assert normalize_line(line) == "<n>-<n>-<n>T<n>:<n>:<n>Z error dial tcp <ip> req=<uuid> after <n>ms"


def test_error_signatures_groups_and_counts():
    lines = [
        '{"level":"error","msg":"request 17 failed"}',
        '{"level":"error","msg":"request 99 failed"}',
        '{"level":"info","msg":"ok"}',
        "panic: runtime error",
    ]
    sigs = error_signatures(lines)
    assert sigs[0] == {"signature": '{"level":"error","msg":"request <n> failed"}', "count": 2,
                       "example": '{"level":"error","msg":"request 17 failed"}'}
    assert sigs[1]["signature"] == "panic: runtime error"
    assert len(sigs) == 2


def test_get_error_signatures_reads_all_pods_and_reports_gaps():
    core = FakeCore(
        pods=[pod(name="podinfo-abc12-aaaaa"), pod(name="podinfo-abc12-bbbbb")],
        logs={("podinfo-abc12-aaaaa", False): "Error: boom 1\nfine\nError: boom 2\n"},
    )
    tools = {t.name: t for t in make_log_tools(core, FakeApps(deployments=[deployment()]), CHANGE, W,
                                               now_fn=lambda: T0 + timedelta(minutes=10))}
    out = json.loads(tools["get_error_signatures"].invoke({}))
    assert out["pods_read"] == 1
    assert out["signatures"][0] == {"signature": "Error: boom <n>", "count": 2, "example": "Error: boom 1"}
    assert out["unavailable"][0].startswith("podinfo-abc12-bbbbb: logs unavailable")


def test_get_pod_logs_previous_and_error():
    core = FakeCore(logs={("podinfo-abc12-aaaaa", True): "Error: invalid argument\n"})
    tools = {t.name: t for t in make_log_tools(core, FakeApps(), CHANGE, W)}
    assert "invalid argument" in tools["get_pod_logs"].invoke({"pod": "podinfo-abc12-aaaaa", "previous": True})
    assert tools["get_pod_logs"].invoke({"pod": "podinfo-abc12-aaaaa"}).startswith("ERROR:")
