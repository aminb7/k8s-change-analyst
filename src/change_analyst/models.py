from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Change(BaseModel):
    id: str
    kind: Literal["rollout", "configmap"]
    namespace: str
    workload: str
    changed_at: datetime
    description: str
    diff: dict[str, Any] = Field(default_factory=dict)


class PendingChange(BaseModel):
    change: Change
    first_seen: datetime


class Findings(BaseModel):
    agent: Literal["metrics", "logs"]
    status: Literal["ok", "partial", "failed"]
    observations: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)

    @classmethod
    def failed(cls, agent: str, reason: str) -> "Findings":
        return cls(agent=agent, status="failed", data_gaps=[reason])

    @classmethod
    def from_agent_output(cls, agent: str, data: dict[str, Any]) -> "Findings":
        fields = {k: v for k, v in data.items() if k != "agent"}
        return cls(agent=agent, **fields)


class Verdict(BaseModel):
    change_id: str
    verdict: Literal["regression", "improvement", "no_impact", "inconclusive"]
    confidence: Literal["low", "medium", "high"]
    summary: str
    evidence: list[str] = Field(default_factory=list)


class ChangeResult(BaseModel):
    change: Change
    metrics: Findings
    logs: Findings
    verdict: Verdict


def _schema_without(model: type[BaseModel], field: str) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["properties"].pop(field, None)
    if field in schema.get("required", []):
        schema["required"].remove(field)
    return schema


def findings_schema() -> dict[str, Any]:
    """JSON schema an investigator agent must produce (agent is filled in by code)."""
    return _schema_without(Findings, "agent")


def verdict_schema() -> dict[str, Any]:
    """JSON schema the judge must produce (change_id is filled in by code)."""
    return _schema_without(Verdict, "change_id")
