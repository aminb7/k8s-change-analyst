"""Generic tool-using agent loop: model -> tools -> model ... -> final JSON."""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.models import Change
from change_analyst.planning import Windows


def run_tool_agent(
    model: CLIChatModel,
    tools: list[BaseTool],
    system: str,
    task: str,
    max_steps: int,
    final_schema: dict,
) -> dict:
    with_tools = model.bind_tools(tools, final_schema=final_schema)
    finish_only = model.bind_tools([], final_schema=final_schema)

    def call_model(state: MessagesState) -> dict:
        used = sum(isinstance(m, ToolMessage) for m in state["messages"])
        llm = with_tools if used < max_steps else finish_only
        return {"messages": [llm.invoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", END])
    graph.add_edge("tools", "model")

    out = graph.compile().invoke(
        {"messages": [SystemMessage(system), HumanMessage(task)]},
        {"recursion_limit": 2 * max_steps + 5},
    )
    return json.loads(out["messages"][-1].content)


def build_task(change: Change, windows: Windows) -> str:
    return (
        "Investigate this change.\n\nCHANGE:\n"
        + json.dumps(change.model_dump(mode="json"), indent=2)
        + "\n\nANALYSIS WINDOWS (UTC):\n"
        + json.dumps(windows.as_dict(), indent=2)
    )
