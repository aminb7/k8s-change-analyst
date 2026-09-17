from datetime import datetime, timezone

from change_analyst.models import (
    Change,
    Findings,
    findings_schema,
    verdict_schema,
)


def test_change_json_roundtrip():
    change = Change(
        id="rollout:demo/podinfo:rev2",
        kind="rollout",
        namespace="demo",
        workload="podinfo",
        changed_at=datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc),
        description="image bump",
        diff={"podinfo.image": ["a", "b"]},
    )
    assert Change.model_validate_json(change.model_dump_json()) == change


def test_findings_failed_records_reason():
    findings = Findings.failed("logs", "boom")
    assert findings.agent == "logs"
    assert findings.status == "failed"
    assert findings.data_gaps == ["boom"]
    assert findings.observations == []


def test_findings_from_agent_output_ignores_agent_key():
    findings = Findings.from_agent_output(
        "metrics", {"agent": "logs", "status": "ok", "observations": ["x"]}
    )
    assert findings.agent == "metrics"
    assert findings.observations == ["x"]


def test_schema_helpers_drop_fixed_fields():
    fs = findings_schema()
    assert "agent" not in fs["properties"]
    assert "agent" not in fs.get("required", [])
    assert "status" in fs["required"]
    vs = verdict_schema()
    assert "change_id" not in vs["properties"]
    assert "change_id" not in vs.get("required", [])
    assert set(vs["properties"]["verdict"]["enum"]) == {
        "regression", "improvement", "no_impact", "inconclusive"
    }
