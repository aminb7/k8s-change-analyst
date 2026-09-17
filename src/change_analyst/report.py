from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from change_analyst.models import ChangeResult, PendingChange


class RunReport(BaseModel):
    started_at: datetime
    duration_s: float
    llm_calls: int
    results: list[ChangeResult] = Field(default_factory=list)
    pending: list[PendingChange] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _bullets(title: str, items: list[str]) -> list[str]:
    return [f"**{title}:**", ""] + ([f"- {i}" for i in items] or ["- (none)"]) + [""]


def render_markdown(report: RunReport) -> str:
    out = [
        f"# Change impact report — {report.started_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"Analyzed {len(report.results)} change(s) · {len(report.pending)} pending · "
        f"{report.llm_calls} LLM calls · {report.duration_s:.0f}s",
        "",
    ]
    if report.results:
        out += ["| Change | Workload | Verdict | Confidence |", "|---|---|---|---|"]
        for r in report.results:
            out.append(f"| `{r.change.id}` | {r.change.namespace}/{r.change.workload} | "
                       f"**{r.verdict.verdict}** | {r.verdict.confidence} |")
        out.append("")
    else:
        out += ["No changes were ready for analysis.", ""]

    for r in report.results:
        out += [
            f"## {r.change.id}",
            "",
            f"**Verdict:** {r.verdict.verdict} ({r.verdict.confidence} confidence)  ",
            f"**Changed at:** {r.change.changed_at:%Y-%m-%d %H:%M:%S} UTC",
            "",
            r.change.description,
            "",
            r.verdict.summary,
            "",
            "### Evidence",
            "",
        ]
        out += [f"- {e}" for e in r.verdict.evidence] or ["- (none)"]
        out.append("")
        for findings in (r.metrics, r.logs):
            out += [f"### {findings.agent.capitalize()} findings ({findings.status})", ""]
            out += _bullets("Observations", findings.observations)
            out += _bullets("Anomalies", findings.anomalies)
            out += _bullets("Data gaps", findings.data_gaps)

    if report.pending:
        out += ["## Pending", ""]
        out += [f"- `{p.change.id}` (changed {p.change.changed_at:%H:%M:%S} UTC, "
                f"first seen {p.first_seen:%H:%M:%S} UTC)" for p in report.pending]
        out.append("")
    if report.errors:
        out += ["## Errors", ""] + [f"- {e}" for e in report.errors] + [""]
    return "\n".join(out)


def write_report(report: RunReport, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = reports_dir / f"{stamp}.json"
    md_path = reports_dir / f"{stamp}.md"
    json_path.write_text(report.model_dump_json(indent=2))
    md_path.write_text(render_markdown(report))
    latest = reports_dir / "latest.md"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(md_path.name)
    return md_path
