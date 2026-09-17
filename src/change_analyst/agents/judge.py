import json
import logging

from change_analyst.llm.client import LLMClient
from change_analyst.models import Change, Findings, Verdict, verdict_schema

log = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are the judge in a Kubernetes change-impact analysis system.
You receive one change and the findings of two investigators (metrics, and logs/events). Decide whether the change hurt the workload.

Verdicts:
- regression: something clearly got worse after the change (higher error rate or latency, restarts, crash loops, fewer ready replicas, new error signatures).
- improvement: something clearly got better and nothing got worse.
- no_impact: the after-window looks like the before-window within normal noise (roughly ±20% on rates, latency, cpu, memory; no new errors or restarts).
- inconclusive: not enough data to decide.

Confidence: "high" only when metrics and logs agree; "medium" when one source is clear and the other is silent; "low" otherwise.
summary: 2-4 sentences a busy engineer can act on.
evidence: each item must cite a specific observation (with numbers) from the findings. Do not invent data."""


def judge_change(client: LLMClient, change: Change, metrics: Findings, logs: Findings) -> Verdict:
    if metrics.status == "failed" and logs.status == "failed":
        return Verdict(
            change_id=change.id, verdict="inconclusive", confidence="low",
            summary="Both investigations failed, so no verdict could be reached.",
            evidence=metrics.data_gaps + logs.data_gaps,
        )
    prompt = json.dumps({
        "change": change.model_dump(mode="json"),
        "metrics_findings": metrics.model_dump(),
        "logs_findings": logs.model_dump(),
    }, indent=2)
    try:
        result = client.complete(JUDGE_SYSTEM, prompt, json_schema=verdict_schema())
        fields = {k: v for k, v in (result.structured or {}).items() if k != "change_id"}
        return Verdict(change_id=change.id, **fields)
    except Exception as exc:  # the run must go on even if the judge fails
        log.warning("judge failed for %s: %s", change.id, exc)
        return Verdict(change_id=change.id, verdict="inconclusive", confidence="low",
                       summary=f"Judge failed: {exc}", evidence=[])
