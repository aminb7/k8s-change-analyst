import json

from langchain_core.tools import BaseTool, tool

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.prometheus import summarize_values

MAX_SERIES = 10

# {sel} = namespace + pod regex for the Deployment, {ns} = namespace, {wl} = Deployment name.
METRIC_QUERIES: dict[str, str] = {
    "request_rate": 'sum(rate(http_requests_total{{{sel}}}[1m]))',
    "error_rate_5xx": ('(sum(rate(http_requests_total{{{sel},status=~"5.."}}[1m])) or vector(0))'
                       ' / clamp_min(sum(rate(http_requests_total{{{sel}}}[1m])), 1e-9)'),
    "latency_p95": ('histogram_quantile(0.95, sum by (le) '
                    '(rate(http_request_duration_seconds_bucket{{{sel}}}[1m])))'),
    "cpu": 'sum(rate(container_cpu_usage_seconds_total{{{sel},container!="",container!="POD"}}[1m]))',
    "memory": 'sum(container_memory_working_set_bytes{{{sel},container!="",container!="POD"}})',
    "restarts": 'sum(kube_pod_container_status_restarts_total{{{sel}}})',
    "ready_replicas": 'sum(kube_deployment_status_replicas_ready{{namespace="{ns}",deployment="{wl}"}})',
}


def pod_selector(namespace: str, workload: str) -> str:
    return f'namespace="{namespace}",pod=~"{workload}-[a-z0-9]+-[a-z0-9]+"'


def _first_series_values(result: list[dict]) -> list:
    return result[0]["values"] if result else []


def make_metrics_tools(prom, change: Change, windows: Windows) -> list[BaseTool]:
    selector = pod_selector(change.namespace, change.workload)

    @tool
    def compare_windows(metrics: list[str]) -> str:
        """Compare metrics of the changed Deployment between the before-change window and the
        after-change window. Returns avg/p95/min/max/last for each window and the percent change
        of the average. Valid metric names: request_rate (req/s), error_rate_5xx (fraction 0-1),
        latency_p95 (seconds), cpu (cores), memory (bytes), restarts (total container restarts),
        ready_replicas."""
        report = []
        for metric in metrics:
            if metric not in METRIC_QUERIES:
                report.append({"metric": metric, "error": f"unknown metric; choose from {sorted(METRIC_QUERIES)}"})
                continue
            query = METRIC_QUERIES[metric].format(sel=selector, ns=change.namespace, wl=change.workload)
            try:
                before = summarize_values(_first_series_values(
                    prom.query_range(query, windows.before_start, windows.before_end)))
                after = summarize_values(_first_series_values(
                    prom.query_range(query, windows.after_start, windows.after_end)))
            except Exception as exc:
                report.append({"metric": metric, "error": f"query failed: {exc}"})
                continue
            change_pct = None
            if before and after and before["avg"] != 0:
                change_pct = round((after["avg"] - before["avg"]) / abs(before["avg"]) * 100, 1)
            report.append({"metric": metric, "before": before, "after": after, "avg_change_pct": change_pct})
        return json.dumps(report)

    @tool
    def query_promql(query: str) -> str:
        """Run a custom PromQL range query over the whole analysis period (before-window start to
        after-window end). Use only when compare_windows is not enough. Returns min/avg/max/last
        per series (at most 10 series)."""
        try:
            result = prom.query_range(query, windows.before_start, windows.after_end)
        except Exception as exc:
            return f"ERROR: query failed: {exc}"
        series = [{"labels": s["metric"], "summary": summarize_values(s["values"])} for s in result[:MAX_SERIES]]
        return json.dumps({"total_series": len(result), "series": series})

    return [compare_windows, query_promql]
