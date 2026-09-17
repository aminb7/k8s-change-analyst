import json
from datetime import timedelta

from factories import T0, FakeApps, FakeCore, deployment, event, pod

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.k8s import make_k8s_tools, summarize_events, summarize_pod

W = Windows(T0 - timedelta(minutes=15), T0, T0 + timedelta(minutes=2), T0 + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T0, description="d")


def test_summarize_pod_reports_crashloop():
    summary = summarize_pod(pod(ready=False, restarts=4, waiting="CrashLoopBackOff", last_terminated="Error"))
    assert summary["ready"] is False
    assert summary["restarts"] == 4
    assert summary["waiting_reasons"] == ["CrashLoopBackOff"]
    assert summary["last_termination_reasons"] == ["Error"]
    assert summary["images"] == ["ghcr.io/stefanprodan/podinfo:6.15.0"]


def test_summarize_events_filters_and_groups():
    events = [
        event(name="podinfo-abc12-aaaaa", reason="BackOff", count=3),
        event(name="podinfo-abc12-bbbbb", reason="BackOff", count=2),
        event(name="podinfo-abc12-aaaaa", reason="Pulled", type_="Normal"),
        event(name="other-abc12-aaaaa", reason="BackOff"),
        event(name="podinfo-abc12-aaaaa", reason="Unhealthy", when=T0 - timedelta(hours=1)),
    ]
    groups = summarize_events(events, "podinfo", since=W.before_start)
    assert groups == [{
        "kind": "Pod", "reason": "BackOff", "count": 5,
        "objects": ["podinfo-abc12-aaaaa", "podinfo-abc12-bbbbb"],
        "example": "Back-off restarting failed container",
    }]


def test_tools_use_deployment_selector_and_report_errors():
    core = FakeCore(pods=[pod()], events=[event()])
    apps = FakeApps(deployments=[deployment()])
    tools = {t.name: t for t in make_k8s_tools(core, apps, CHANGE, W)}
    pods = json.loads(tools["get_pod_status"].invoke({}))
    assert pods[0]["name"] == "podinfo-abc12-xyz34"
    assert core.label_selectors == ["app=podinfo"]
    assert json.loads(tools["get_events"].invoke({}))[0]["reason"] == "BackOff"

    missing = {t.name: t for t in make_k8s_tools(core, FakeApps(), CHANGE, W)}
    assert missing["get_pod_status"].invoke({}).startswith("ERROR:")
