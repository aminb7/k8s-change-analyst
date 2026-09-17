"""One analysis run: detect -> plan -> investigate (graph) -> report -> save state."""
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from change_analyst.agents.judge import judge_change
from change_analyst.agents.logs_agent import investigate_logs
from change_analyst.agents.metrics_agent import investigate_metrics
from change_analyst.changes.detect import detect_changes
from change_analyst.config import Settings
from change_analyst.graph import Deps, build_graph
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.llm.client import LLMClient
from change_analyst.models import Change, PendingChange
from change_analyst.planning import compute_windows, plan_run
from change_analyst.report import RunReport, write_report
from change_analyst.state_store import State, load_state, remember_analyzed, save_state

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    core: Any
    apps: Any
    prom: Any
    llm: LLMClient


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def execute_run(settings: Settings, runtime: Runtime, now_fn: Callable[[], datetime] = _utcnow) -> Path:
    started = now_fn()
    t0 = time.monotonic()
    state = load_state(settings.state_path)
    errors: list[str] = []

    since = state.last_run_at or started - timedelta(minutes=settings.first_run_lookback_min)
    detection_ok = True
    try:
        detected, hashes = detect_changes(runtime.core, runtime.apps, settings.namespaces,
                                          since, state.configmap_hashes)
    except Exception as exc:
        log.exception("change detection failed")
        errors.append(f"change detection failed: {exc}")
        detected, hashes, detection_ok = [], state.configmap_hashes, False

    ready, waiting = plan_run(detected, state, started, settings)
    first_seen = {p.change.id: p.first_seen for p in ready}
    log.info("detected=%d ready=%d waiting=%d", len(detected), len(ready), len(waiting))

    model = CLIChatModel(client=runtime.llm)

    def windows_for(change: Change):
        return compute_windows(change.changed_at, now_fn(), settings)

    deps = Deps(
        investigate_metrics=lambda c: investigate_metrics(model, runtime.prom, c, windows_for(c),
                                                          settings.max_agent_steps),
        investigate_logs=lambda c: investigate_logs(model, runtime.core, runtime.apps, c, windows_for(c),
                                                    settings.max_agent_steps),
        judge=lambda c, m, l: judge_change(runtime.llm, c, m, l),
        deadline=t0 + settings.run_timeout_s,
    )
    out = build_graph(deps).invoke(
        {"ready": [p.change for p in ready], "results": [], "deferred": []},
        {"max_concurrency": settings.max_concurrency},
    )
    results = out["results"]
    pending = waiting + [PendingChange(change=c, first_seen=first_seen[c.id]) for c in out["deferred"]]

    report = RunReport(
        started_at=started,
        duration_s=round(time.monotonic() - t0, 1),
        llm_calls=getattr(runtime.llm, "calls", 0),
        results=results,
        pending=pending,
        errors=errors,
    )
    path = write_report(report, settings.reports_dir)
    save_state(settings.state_path, State(
        last_run_at=started if detection_ok else state.last_run_at,
        analyzed_change_ids=remember_analyzed(state.analyzed_change_ids, [r.change.id for r in results]),
        configmap_hashes=hashes,
        pending=pending,
    ))
    return path
