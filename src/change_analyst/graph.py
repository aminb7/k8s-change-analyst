"""LangGraph wiring: fan out one investigation per change, each with parallel agents + judge."""
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from change_analyst.models import Change, ChangeResult, Findings, Verdict


class RunState(TypedDict):
    ready: list[Change]
    results: Annotated[list[ChangeResult], operator.add]
    deferred: Annotated[list[Change], operator.add]


class InvestigationState(TypedDict, total=False):
    change: Change
    metrics: Findings
    logs: Findings
    verdict: Verdict


@dataclass
class Deps:
    investigate_metrics: Callable[[Change], Findings]
    investigate_logs: Callable[[Change], Findings]
    judge: Callable[[Change, Findings, Findings], Verdict]
    deadline: float
    clock: Callable[[], float] = field(default=time.monotonic)


def build_investigation_graph(deps: Deps):
    graph = StateGraph(InvestigationState)
    graph.add_node("metrics_agent", lambda s: {"metrics": deps.investigate_metrics(s["change"])})
    graph.add_node("logs_agent", lambda s: {"logs": deps.investigate_logs(s["change"])})
    graph.add_node("judge", lambda s: {"verdict": deps.judge(s["change"], s["metrics"], s["logs"])})
    graph.add_edge(START, "metrics_agent")
    graph.add_edge(START, "logs_agent")
    graph.add_edge(["metrics_agent", "logs_agent"], "judge")
    graph.add_edge("judge", END)
    return graph.compile()


def build_graph(deps: Deps):
    investigation = build_investigation_graph(deps)

    def fan_out(state: RunState):
        return [Send("investigate", {"change": c}) for c in state["ready"]] or ["finish"]

    def investigate(state):  # receives the Send payload {"change": Change}
        change = state["change"]
        if deps.clock() > deps.deadline:
            return {"deferred": [change]}
        out = investigation.invoke({"change": change})
        return {"results": [ChangeResult(change=change, metrics=out["metrics"],
                                         logs=out["logs"], verdict=out["verdict"])]}

    graph = StateGraph(RunState)
    graph.add_node("plan", lambda s: {})
    graph.add_node("investigate", investigate)
    graph.add_node("finish", lambda s: {})
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["investigate", "finish"])
    graph.add_edge("investigate", "finish")
    graph.add_edge("finish", END)
    return graph.compile()
