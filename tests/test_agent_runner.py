from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from change_analyst.agents.runner import build_task, run_tool_agent
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.llm.client import LLMResult
from change_analyst.models import Change
from change_analyst.planning import Windows


class ToolHappyClient:
    """Calls the `ping` tool whenever tools are offered; finishes otherwise."""

    def __init__(self):
        self.requests = 0

    def complete(self, system, prompt, json_schema=None):
        self.requests += 1
        if "call_tool" in json_schema["properties"]["action"]["enum"]:
            return LLMResult(text="", structured={"action": "call_tool", "tool": "ping", "args": {"n": self.requests}})
        return LLMResult(text="", structured={"action": "final", "final": {"status": "ok", "pings": prompt.count("[TOOL RESULT ping]")}})


def make_ping(calls):
    @tool
    def ping(n: int) -> str:
        """Ping."""
        calls.append(n)
        return f"pong {n}"
    return ping


def test_agent_stops_calling_tools_after_budget():
    calls = []
    client = ToolHappyClient()
    result = run_tool_agent(
        CLIChatModel(client=client), [make_ping(calls)], "system", "task",
        max_steps=2, final_schema={"type": "object"},
    )
    assert calls == [1, 2]
    assert result == {"status": "ok", "pings": 2}
    assert client.requests == 3


class OneShotClient:
    def complete(self, system, prompt, json_schema=None):
        if "[TOOL RESULT ping]" in prompt:
            return LLMResult(text="", structured={"action": "final", "final": {"seen": "pong 7" in prompt}})
        return LLMResult(text="", structured={"action": "call_tool", "tool": "ping", "args": {"n": 7}})


def test_agent_feeds_tool_result_back_to_model():
    calls = []
    result = run_tool_agent(CLIChatModel(client=OneShotClient()), [make_ping(calls)],
                            "system", "task", max_steps=5, final_schema={"type": "object"})
    assert calls == [7]
    assert result == {"seen": True}


def test_build_task_contains_change_and_windows():
    t = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
    change = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                    workload="podinfo", changed_at=t, description="image bump")
    windows = Windows(t - timedelta(minutes=15), t, t + timedelta(minutes=2), t + timedelta(minutes=10))
    task = build_task(change, windows)
    assert "rollout:demo/podinfo:rev2" in task
    assert "image bump" in task
    assert windows.after_end.isoformat() in task
