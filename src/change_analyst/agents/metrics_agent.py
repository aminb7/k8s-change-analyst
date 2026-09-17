import logging

from change_analyst.agents.runner import build_task, run_tool_agent
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.models import Change, Findings, findings_schema
from change_analyst.planning import Windows
from change_analyst.tools.metrics import make_metrics_tools

log = logging.getLogger(__name__)

METRICS_SYSTEM = """You are the metrics investigator in a Kubernetes change-impact analysis system.
You investigate ONE change to ONE Deployment by comparing Prometheus metrics in the window before the change with the window after it.

How to work:
1. Call compare_windows once with all relevant metrics, normally:
   ["request_rate", "error_rate_5xx", "latency_p95", "restarts", "ready_replicas", "cpu", "memory"].
2. Only if something is unclear, use query_promql for a targeted follow-up.
3. Then finish.

Your final answer:
- observations: factual statements with numbers, e.g. "error_rate_5xx avg 0.0 before -> 0.31 after".
- anomalies: only things that got clearly worse after the change (ignore changes within roughly ±20% for rates, latency, cpu and memory).
- data_gaps: metrics that returned no data or errors.
- status: "ok" if the key metrics were available, "partial" if some were missing, "failed" if no data could be retrieved.
Never invent numbers."""


def investigate_metrics(model: CLIChatModel, prom, change: Change, windows: Windows, max_steps: int) -> Findings:
    tools = make_metrics_tools(prom, change, windows)
    try:
        data = run_tool_agent(model, tools, METRICS_SYSTEM, build_task(change, windows),
                              max_steps, findings_schema())
        return Findings.from_agent_output("metrics", data)
    except Exception as exc:  # the run must go on even if this agent fails
        log.warning("metrics agent failed for %s: %s", change.id, exc)
        return Findings.failed("metrics", f"metrics agent failed: {exc}")
