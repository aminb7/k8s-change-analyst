import json
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from change_analyst.models import Change
from change_analyst.planning import Windows


def short_error(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if status is not None:
        return f"HTTP {status} {getattr(exc, 'reason', '')}".strip()
    return str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__


def list_workload_pods(core, apps, namespace: str, workload: str) -> list:
    deploy = apps.read_namespaced_deployment(workload, namespace)
    labels = deploy.spec.selector.match_labels or {}
    selector = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return core.list_namespaced_pod(namespace, label_selector=selector).items


def summarize_pod(pod) -> dict:
    statuses = pod.status.container_statuses or []
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": bool(statuses) and all(s.ready for s in statuses),
        "restarts": sum(s.restart_count or 0 for s in statuses),
        "waiting_reasons": [s.state.waiting.reason for s in statuses if s.state and s.state.waiting],
        "last_termination_reasons": [s.last_state.terminated.reason for s in statuses
                                     if s.last_state and s.last_state.terminated],
        "images": [c.image for c in pod.spec.containers],
        "created": pod.metadata.creation_timestamp.isoformat(),
    }


def summarize_events(events, workload: str, since: datetime) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for ev in events:
        if ev.type != "Warning":
            continue
        obj = ev.involved_object
        if not (obj.name == workload or obj.name.startswith(workload + "-")):
            continue
        when = ev.last_timestamp or ev.event_time or ev.metadata.creation_timestamp
        if when is not None and when < since:
            continue
        group = groups.setdefault((obj.kind, ev.reason), {
            "kind": obj.kind, "reason": ev.reason, "count": 0, "objects": set(), "example": ev.message,
        })
        group["count"] += ev.count or 1
        group["objects"].add(obj.name)
    result = [{**g, "objects": sorted(g["objects"])[:5]} for g in groups.values()]
    return sorted(result, key=lambda g: (-g["count"], g["reason"]))


def make_k8s_tools(core, apps, change: Change, windows: Windows) -> list[BaseTool]:
    @tool
    def get_pod_status() -> str:
        """List the current pods of the changed Deployment with phase, readiness, restart count,
        waiting reasons (e.g. CrashLoopBackOff), last termination reasons (e.g. OOMKilled, Error)
        and images."""
        try:
            pods = list_workload_pods(core, apps, change.namespace, change.workload)
        except Exception as exc:
            return f"ERROR: could not list pods: {short_error(exc)}"
        return json.dumps([summarize_pod(p) for p in pods])

    @tool
    def get_events() -> str:
        """Warning events since the start of the before-change window for the changed Deployment,
        its ReplicaSets and its pods, grouped by object kind and reason with counts."""
        try:
            events = core.list_namespaced_event(change.namespace).items
        except Exception as exc:
            return f"ERROR: could not list events: {short_error(exc)}"
        return json.dumps(summarize_events(events, change.workload, windows.before_start))

    return [get_pod_status, get_events]
