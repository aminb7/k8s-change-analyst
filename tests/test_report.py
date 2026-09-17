import json
from datetime import datetime, timezone

from change_analyst.models import Change, ChangeResult, Findings, PendingChange, Verdict
from change_analyst.report import RunReport, render_markdown, write_report

T = datetime(2026, 9, 16, 10, 0, 5, tzinfo=timezone.utc)
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo", workload="podinfo",
                changed_at=T, description="Deployment podinfo rolled out revision 2: podinfo.args changed")


def sample_report(**overrides):
    result = ChangeResult(
        change=CHANGE,
        metrics=Findings(agent="metrics", status="ok", observations=["error_rate_5xx 0.0 -> 0.31"],
                         anomalies=["5xx errors appeared"]),
        logs=Findings(agent="logs", status="partial", data_gaps=["old pods gone"]),
        verdict=Verdict(change_id=CHANGE.id, verdict="regression", confidence="high",
                        summary="The rollout introduced 5xx errors.", evidence=["error_rate_5xx 0.0 -> 0.31"]),
    )
    data = dict(started_at=T, duration_s=42.3, llm_calls=7, results=[result],
                pending=[PendingChange(change=CHANGE.model_copy(update={"id": "young"}), first_seen=T)],
                errors=["something minor"])
    data.update(overrides)
    return RunReport(**data)


def test_render_markdown_contains_key_sections():
    md = render_markdown(sample_report())
    assert md.startswith("# Change impact report — 2026-09-16 10:00 UTC")
    assert "| `rollout:demo/podinfo:rev2` | demo/podinfo | **regression** | high |" in md
    assert "The rollout introduced 5xx errors." in md
    assert "### Metrics findings (ok)" in md
    assert "- 5xx errors appeared" in md
    assert "### Logs findings (partial)" in md
    assert "- old pods gone" in md
    assert "## Pending" in md and "`young`" in md
    assert "## Errors" in md


def test_render_markdown_without_results():
    md = render_markdown(sample_report(results=[], pending=[], errors=[]))
    assert "No changes were ready for analysis." in md
    assert "## Pending" not in md


def test_write_report_creates_files_and_latest_link(tmp_path):
    path = write_report(sample_report(), tmp_path / "reports")
    assert path.name == "20260916T100005Z.md"
    data = json.loads((tmp_path / "reports" / "20260916T100005Z.json").read_text())
    assert data["results"][0]["verdict"]["verdict"] == "regression"
    latest = tmp_path / "reports" / "latest.md"
    assert latest.is_symlink() and latest.read_text() == path.read_text()
    write_report(sample_report(started_at=T.replace(second=9)), tmp_path / "reports")
    assert latest.resolve().name == "20260916T100009Z.md"
