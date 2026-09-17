import json
from datetime import datetime, timedelta, timezone

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.metrics import METRIC_QUERIES, make_metrics_tools, pod_selector
from change_analyst.tools.prometheus import PromError

T = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
W = Windows(T - timedelta(minutes=15), T, T + timedelta(minutes=2), T + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T, description="d")


class FakeProm:
    def __init__(self, before=(1.0, 1.0), after=(2.0, 2.0), series=1, fail=False):
        self.before, self.after, self.series, self.fail = before, after, series, fail
        self.queries = []

    def query_range(self, query, start, end, step_s=15):
        self.queries.append((query, start, end))
        if self.fail:
            raise PromError("connection refused")
        values = self.before if start == W.before_start else self.after
        return [{"metric": {"i": str(i)}, "values": [[n, str(v)] for n, v in enumerate(values)]}
                for i in range(self.series)]


def tools_by_name(prom):
    return {t.name: t for t in make_metrics_tools(prom, CHANGE, W)}


def test_pod_selector_matches_deployment_pods_only():
    assert pod_selector("demo", "podinfo") == 'namespace="demo",pod=~"podinfo-[a-z0-9]+-[a-z0-9]+"'


def test_all_metric_queries_render():
    for name, template in METRIC_QUERIES.items():
        rendered = template.format(sel=pod_selector("demo", "podinfo"), ns="demo", wl="podinfo")
        assert "{" in rendered and "{sel}" not in rendered, name


def test_compare_windows_reports_before_after_and_change():
    prom = FakeProm(before=(1.0, 1.0), after=(3.0, 3.0))
    out = json.loads(tools_by_name(prom)["compare_windows"].invoke({"metrics": ["request_rate"]}))
    entry = out[0]
    assert entry["metric"] == "request_rate"
    assert entry["before"]["avg"] == 1.0
    assert entry["after"]["avg"] == 3.0
    assert entry["avg_change_pct"] == 200.0
    assert (W.before_start, W.before_end) in [(q[1], q[2]) for q in prom.queries]
    assert (W.after_start, W.after_end) in [(q[1], q[2]) for q in prom.queries]


def test_compare_windows_handles_unknown_metric_and_errors():
    out = json.loads(tools_by_name(FakeProm(fail=True))["compare_windows"].invoke(
        {"metrics": ["bogus", "cpu"]}))
    assert out[0]["error"].startswith("unknown metric")
    assert "connection refused" in out[1]["error"]


def test_query_promql_caps_series():
    prom = FakeProm(series=12)
    out = json.loads(tools_by_name(prom)["query_promql"].invoke({"query": "up"}))
    assert len(out["series"]) == 10
    assert out["total_series"] == 12
    assert prom.queries[0][1:] == (W.before_start, W.after_end)


def test_query_promql_returns_error_text():
    text = tools_by_name(FakeProm(fail=True))["query_promql"].invoke({"query": "up"})
    assert text.startswith("ERROR:")
