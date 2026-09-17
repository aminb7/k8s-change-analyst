import json
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone

from langchain_core.tools import BaseTool, tool

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.k8s import list_workload_pods, short_error

MAX_LOG_CHARS = 8000
MAX_SIGNATURE_CHARS = 300

ERROR_LINE = re.compile(
    r'(?i)\b(error|exception|fatal|panic|fail(ed|ure)?|traceback)\b|"level":"(error|fatal|panic)"'
)
NORMALIZERS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<hex>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<n>"),
]


def normalize_line(line: str) -> str:
    text = line.strip()
    for pattern, replacement in NORMALIZERS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text)[:MAX_SIGNATURE_CHARS]


def error_signatures(lines: list[str], top: int = 15) -> list[dict]:
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for line in lines:
        if not ERROR_LINE.search(line):
            continue
        signature = normalize_line(line)
        counts[signature] += 1
        examples.setdefault(signature, line.strip()[:MAX_SIGNATURE_CHARS])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"signature": s, "count": c, "example": examples[s]} for s, c in ranked]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_log_tools(
    core, apps, change: Change, windows: Windows, now_fn: Callable[[], datetime] = _utcnow
) -> list[BaseTool]:
    @tool
    def get_error_signatures(previous: bool = False) -> str:
        """Group error-like log lines from the changed Deployment's current pods into normalized
        signatures with counts and one example each (top 15), covering the analysis period.
        Set previous=true to read the previous (crashed) container instance of each pod instead."""
        try:
            pods = list_workload_pods(core, apps, change.namespace, change.workload)
        except Exception as exc:
            return f"ERROR: could not list pods: {short_error(exc)}"
        since = max(60, int((now_fn() - windows.before_start).total_seconds()))
        lines: list[str] = []
        unavailable: list[str] = []
        for p in pods:
            name = p.metadata.name
            kwargs = {"previous": True} if previous else {"since_seconds": since}
            try:
                text = core.read_namespaced_pod_log(name, change.namespace, tail_lines=2000, **kwargs)
            except Exception as exc:
                unavailable.append(f"{name}: logs unavailable ({short_error(exc)})")
                continue
            lines.extend(text.splitlines())
        return json.dumps({
            "pods_read": len(pods) - len(unavailable),
            "signatures": error_signatures(lines),
            "unavailable": unavailable,
        })

    @tool
    def get_pod_logs(pod: str, previous: bool = False, tail: int = 50) -> str:
        """Return the last `tail` log lines (max 200) of one pod of the changed Deployment.
        Set previous=true for the previous (crashed) container instance."""
        try:
            text = core.read_namespaced_pod_log(pod, change.namespace, previous=previous,
                                                tail_lines=min(max(tail, 1), 200))
        except Exception as exc:
            return f"ERROR: could not read logs for {pod}: {short_error(exc)}"
        return text[-MAX_LOG_CHARS:] or "(empty log)"

    return [get_error_signatures, get_pod_logs]
