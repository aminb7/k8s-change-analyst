from datetime import timedelta

from factories import T0, FakeApps, FakeCore, deployment, pod

from change_analyst.agents.judge import JUDGE_SYSTEM, judge_change
from change_analyst.agents.logs_agent import LOGS_SYSTEM, investigate_logs
from change_analyst.agents.metrics_agent import METRICS_SYSTEM, investigate_metrics
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.llm.client import LLMError, LLMResult
from change_analyst.models import Change, Findings
from change_analyst.planning import Windows

W = Windows(T0 - timedelta(minutes=15), T0, T0 + timedelta(minutes=2), T0 + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T0, description="args changed")
FINAL = {"status": "ok", "observations": ["error_rate_5xx 0.0 -> 0.3"], "anomalies": ["errors up"], "data_gaps": []}


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, system, prompt, json_schema=None):
        self.requests.append((system, prompt, json_schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text="", structured=reply)


class FakeProm:
    def __init__(self):
        self.queries = []

    def query_range(self, query, start, end, step_s=15):
        self.queries.append(query)
        return [{"metric": {}, "values": [[1, "0.5"]]}]


def test_only_judge_prompt_mentions_judge():
    assert "judge" in JUDGE_SYSTEM.lower()
    assert "judge" not in METRICS_SYSTEM.lower()
    assert "judge" not in LOGS_SYSTEM.lower()


def test_investigate_metrics_uses_tools_and_returns_findings():
    client = ScriptedClient([
        {"action": "call_tool", "tool": "compare_windows", "args": {"metrics": ["error_rate_5xx"]}},
        {"action": "final", "final": FINAL},
    ])
    prom = FakeProm()
    findings = investigate_metrics(CLIChatModel(client=client), prom, CHANGE, W, max_steps=6)
    assert findings == Findings(agent="metrics", **FINAL)
    assert len(prom.queries) == 2
    assert "rollout:demo/podinfo:rev2" in client.requests[0][1]


def test_investigate_metrics_failure_becomes_failed_findings():
    client = ScriptedClient([LLMError("cli down")])
    findings = investigate_metrics(CLIChatModel(client=client), FakeProm(), CHANGE, W, max_steps=6)
    assert findings.status == "failed"
    assert "cli down" in findings.data_gaps[0]


def test_investigate_logs_uses_pod_status_tool():
    client = ScriptedClient([
        {"action": "call_tool", "tool": "get_pod_status", "args": {}},
        {"action": "final", "final": FINAL},
    ])
    core = FakeCore(pods=[pod(waiting="CrashLoopBackOff")])
    findings = investigate_logs(CLIChatModel(client=client), core, FakeApps(deployments=[deployment()]),
                                CHANGE, W, max_steps=6)
    assert findings.agent == "logs"
    assert "CrashLoopBackOff" in client.requests[1][1]


def test_judge_returns_verdict_with_change_id():
    client = ScriptedClient([{"verdict": "regression", "confidence": "high",
                              "summary": "Errors rose.", "evidence": ["error_rate_5xx 0 -> 0.3"]}])
    verdict = judge_change(client, CHANGE, Findings(agent="metrics", **FINAL), Findings(agent="logs", **FINAL))
    assert verdict.change_id == CHANGE.id
    assert verdict.verdict == "regression"
    system, prompt, schema = client.requests[0]
    assert system == JUDGE_SYSTEM
    assert "metrics_findings" in prompt
    assert "change_id" not in schema["properties"]


def test_judge_skips_llm_when_both_investigations_failed():
    client = ScriptedClient([])
    verdict = judge_change(client, CHANGE, Findings.failed("metrics", "a"), Findings.failed("logs", "b"))
    assert verdict.verdict == "inconclusive"
    assert verdict.confidence == "low"
    assert client.requests == []


def test_judge_failure_is_inconclusive():
    client = ScriptedClient([LLMError("timeout")])
    verdict = judge_change(client, CHANGE, Findings(agent="metrics", **FINAL), Findings.failed("logs", "b"))
    assert verdict.verdict == "inconclusive"
    assert "timeout" in verdict.summary
