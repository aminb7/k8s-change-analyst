import logging

from change_analyst.agents.runner import build_task, run_tool_agent
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.models import Change, Findings, findings_schema
from change_analyst.planning import Windows
from change_analyst.tools.k8s import make_k8s_tools
from change_analyst.tools.logs import make_log_tools

log = logging.getLogger(__name__)

LOGS_SYSTEM = """You are the logs and events investigator in a Kubernetes change-impact analysis system.
You investigate ONE change to ONE Deployment using pod status, Kubernetes warning events and container logs.

How to work:
1. Call get_pod_status to see readiness, restarts, waiting reasons (CrashLoopBackOff, ImagePullBackOff) and termination reasons (OOMKilled, Error).
2. Call get_events to see warning events (BackOff, Unhealthy, FailedScheduling, ...).
3. Call get_error_signatures for current logs. If any pod restarted or is crash-looping, also call it with previous=true.
4. Use get_pod_logs only to read one specific pod when signatures are not enough.
5. Then finish.

Your final answer:
- observations: factual statements with numbers, e.g. "2 of 3 pods in CrashLoopBackOff, 14 restarts", "signature 'Error: invalid argument <n>' seen 6 times".
- anomalies: problems that plausibly appeared after the change.
- data_gaps: logs or objects that were unavailable (e.g. pods from before the change are gone).
- status: "ok", "partial" (some data unavailable) or "failed" (no data could be retrieved).
Never invent data."""


def investigate_logs(model: CLIChatModel, core, apps, change: Change, windows: Windows, max_steps: int) -> Findings:
    tools = make_k8s_tools(core, apps, change, windows) + make_log_tools(core, apps, change, windows)
    try:
        data = run_tool_agent(model, tools, LOGS_SYSTEM, build_task(change, windows),
                              max_steps, findings_schema())
        return Findings.from_agent_output("logs", data)
    except Exception as exc:  # the run must go on even if this agent fails
        log.warning("logs agent failed for %s: %s", change.id, exc)
        return Findings.failed("logs", f"logs agent failed: {exc}")
