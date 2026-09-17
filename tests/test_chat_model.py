import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from change_analyst.llm.chat_model import CLIChatModel, LLMOutputError
from change_analyst.llm.client import LLMResult


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, system, prompt, json_schema=None):
        self.requests.append({"system": system, "prompt": prompt, "schema": json_schema})
        return LLMResult(text="", structured=self.replies.pop(0))


@tool
def get_pod_status() -> str:
    """Return pod status."""
    return "[]"


@tool
def compare_windows(metrics: list[str]) -> str:
    """Compare metrics."""
    return "{}"


def test_tool_call_reply_becomes_ai_tool_call():
    client = ScriptedClient([{"action": "call_tool", "tool": "compare_windows",
                              "args": {"metrics": ["cpu"]}}])
    model = CLIChatModel(client=client).bind_tools([get_pod_status, compare_windows])
    msg = model.invoke([SystemMessage("You are X."), HumanMessage("Investigate.")])
    assert isinstance(msg, AIMessage)
    assert msg.tool_calls[0]["name"] == "compare_windows"
    assert msg.tool_calls[0]["args"] == {"metrics": ["cpu"]}
    request = client.requests[0]
    assert request["system"].startswith("You are X.")
    assert "compare_windows" in request["system"]
    assert "[USER]\nInvestigate." in request["prompt"]
    assert request["schema"]["properties"]["action"]["enum"] == ["final", "call_tool"]


def test_final_reply_becomes_json_content_with_final_schema():
    final_schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    client = ScriptedClient([{"action": "final", "final": {"status": "ok"}}])
    model = CLIChatModel(client=client).bind_tools([], final_schema=final_schema)
    msg = model.invoke([HumanMessage("done?")])
    assert json.loads(msg.content) == {"status": "ok"}
    assert msg.tool_calls == []
    schema = client.requests[0]["schema"]
    assert schema["properties"]["action"]["enum"] == ["final"]
    assert schema["properties"]["final"] == final_schema
    assert "no tools available" in client.requests[0]["system"]


def test_transcript_includes_previous_tool_calls_and_results():
    client = ScriptedClient([{"action": "final", "final": {}}])
    model = CLIChatModel(client=client).bind_tools([get_pod_status])
    history = [
        HumanMessage("go"),
        AIMessage(content="", tool_calls=[{"name": "get_pod_status", "args": {}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content='[{"name": "p1"}]', tool_call_id="c1", name="get_pod_status"),
    ]
    model.invoke(history)
    prompt = client.requests[0]["prompt"]
    assert "[ASSISTANT called tool get_pod_status with args {}]" in prompt
    assert '[TOOL RESULT get_pod_status]\n[{"name": "p1"}]' in prompt


def test_invalid_reply_is_retried_with_error_feedback():
    client = ScriptedClient([
        {"action": "call_tool", "tool": "delete_cluster", "args": {}},
        {"action": "call_tool", "tool": "get_pod_status", "args": {}},
    ])
    model = CLIChatModel(client=client).bind_tools([get_pod_status])
    msg = model.invoke([HumanMessage("go")])
    assert msg.tool_calls[0]["name"] == "get_pod_status"
    assert "INVALID" in client.requests[1]["prompt"]
    assert "delete_cluster" in client.requests[1]["prompt"]


def test_two_invalid_replies_raise():
    client = ScriptedClient([{"action": "dance"}, {"action": "final"}])
    model = CLIChatModel(client=client).bind_tools([get_pod_status])
    with pytest.raises(LLMOutputError):
        model.invoke([HumanMessage("go")])
