# k8s-change-analyst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cron-run, LLM-powered multi-agent CLI that detects Kubernetes changes (Deployment rollouts, ConfigMap edits), investigates each with metrics/logs agents, and writes a verdict report.

**Architecture:** Deterministic change detection feeds a LangGraph graph that fans out one investigation per change (Send). Each investigation runs a metrics agent and a logs agent in parallel (tool-calling loops built on a custom `CLIChatModel` that drives `claude -p` via subprocess with prompt-based tool calling), then a judge produces a structured verdict. Plain code renders Markdown/JSON reports and persists run state.

**Tech Stack:** Python 3.12 (uv), LangGraph 1.2, langchain-core 1.6, kubernetes client, httpx, pydantic / pydantic-settings, typer, pytest. Demo: kind, Helm kube-prometheus-stack, podinfo 6.15.0.

**Spec:** `docs/superpowers/specs/2026-09-16-k8s-change-analyst-design.md`

## Global Constraints

- Python `>=3.12`, managed with `uv`; package name `change_analyst` under `src/`; console script `k8s-change-analyst`.
- Dependencies: `langgraph>=1.2,<2`, `langchain-core>=1.6,<2`, `kubernetes>=36`, `httpx>=0.28`, `pydantic>=2.13`, `pydantic-settings>=2.15`, `typer>=0.27`; dev: `pytest>=9`.
- All datetimes are timezone-aware UTC.
- LLM backend: Claude CLI via subprocess: `claude -p --output-format json --tools "" --no-session-persistence --strict-mcp-config --disable-slash-commands --setting-sources "" --system-prompt <system> [--model <model>] [--json-schema <schema>]`, prompt on stdin. Result JSON fields used: `is_error`, `subtype` (`"success"`), `result`, `structured_output`.
- Settings come from env vars with prefix `CA_` (or a `.env` file). Defaults: `prometheus_url=http://localhost:9090`, `namespaces=["demo"]`, `llm_model="sonnet"`, `llm_timeout_s=180`, `before_window_min=15`, `settle_delay_min=2`, `min_age_min=5`, `max_after_window_min=30`, `max_pending_min=60`, `first_run_lookback_min=60`, `max_changes_per_run=5`, `max_agent_steps=6`, `max_concurrency=2`, `run_timeout_s=1200`, `reports_dir=reports`, `state_path=state.json`, `lock_path=.change-analyst.lock`.
- Tools never raise into the graph: failures are returned as strings starting with `ERROR:`.
- Verdict values: `regression | improvement | no_impact | inconclusive`; confidence: `low | medium | high`.
- Demo namespace `demo`, Deployment `podinfo`, ConfigMap `podinfo-config`, Prometheus reachable on host port 9090 (kind port mapping to NodePort 30090).
- Unit tests never touch a real cluster, Prometheus, or the Claude CLI.

**Refinements to the spec made while planning (intentional):**
- Config is env/`.env` only (no `config.yaml`) — one mechanism is enough.
- Agent tools are bound to the change being investigated (namespace, workload, windows are closed over), so the LLM only passes the interesting arguments (e.g. `metrics: list[str]`). This reduces LLM argument errors and tool-call count.
- `compare_windows` accepts a list of metrics so the metrics agent needs one call, not seven.
- Default model `sonnet` (≈$0.007/call measured) — override with `CA_LLM_MODEL`.
- ConfigMap `Change.diff` contains the new data values (truncated) so agents know what changed; state still stores only hashes.
- On the very first run, rollouts from the last `first_run_lookback_min` minutes are considered.

---

## File Structure

```
k8s-change-analyst/
├── pyproject.toml                     # Task 1
├── .gitignore                         # Task 1
├── Makefile                           # Task 3 (run/test/e2e targets added in Tasks 14–15)
├── README.md                          # Task 14
├── deploy/demo/
│   ├── kind-config.yaml               # Task 3
│   ├── prometheus-values.yaml         # Task 3
│   ├── app.yaml                       # Task 3: namespace, ConfigMap, podinfo, Service, ServiceMonitor, loadgen
│   └── scenarios/{harmless,errors,crashloop,config,reset}.sh   # Task 3
├── scripts/
│   ├── e2e.sh                         # Task 15
│   └── check_verdict.py               # Task 15
├── src/change_analyst/
│   ├── __init__.py                    # Task 1
│   ├── models.py                      # Task 1: Change, PendingChange, Findings, Verdict, ChangeResult, schema helpers
│   ├── config.py                      # Task 1: Settings
│   ├── state_store.py                 # Task 2: State, load/save, remember_analyzed
│   ├── lock.py                        # Task 2: run_lock, LockHeld
│   ├── changes/__init__.py            # Task 4
│   ├── changes/detect.py              # Task 4: rollouts + ConfigMaps
│   ├── planning.py                    # Task 5: plan_run, Windows, compute_windows
│   ├── llm/__init__.py                # Task 6
│   ├── llm/client.py                  # Task 6: LLMClient, LLMResult, LLMError, ClaudeCLIClient
│   ├── llm/chat_model.py              # Task 7: CLIChatModel, LLMOutputError
│   ├── agents/__init__.py             # Task 8
│   ├── agents/runner.py               # Task 8: run_tool_agent, build_task
│   ├── tools/__init__.py              # Task 9
│   ├── tools/prometheus.py            # Task 9: PromClient, PromError, summarize_values
│   ├── tools/metrics.py               # Task 9: METRIC_QUERIES, make_metrics_tools
│   ├── tools/k8s.py                   # Task 10: pods/events helpers + tools
│   ├── tools/logs.py                  # Task 10: error signatures + log tools
│   ├── agents/metrics_agent.py        # Task 11
│   ├── agents/logs_agent.py           # Task 11
│   ├── agents/judge.py                # Task 11
│   ├── graph.py                       # Task 12
│   ├── report.py                      # Task 13
│   ├── pipeline.py                    # Task 14: execute_run, Runtime
│   └── cli.py                         # Task 14
└── tests/
    ├── factories.py                   # Task 4: fake k8s objects/APIs
    └── test_*.py
```

---

### Task 1: Tooling, project skeleton, models, settings

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/change_analyst/__init__.py`, `src/change_analyst/models.py`, `src/change_analyst/config.py`
- Test: `tests/test_models.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `models.Change(id: str, kind: Literal["rollout","configmap"], namespace: str, workload: str, changed_at: datetime, description: str, diff: dict = {})`
  - `models.PendingChange(change: Change, first_seen: datetime)`
  - `models.Findings(agent: Literal["metrics","logs"], status: Literal["ok","partial","failed"], observations: list[str], anomalies: list[str], data_gaps: list[str])` with `Findings.failed(agent, reason) -> Findings` and `Findings.from_agent_output(agent, data: dict) -> Findings`
  - `models.Verdict(change_id: str, verdict: Literal[...], confidence: Literal[...], summary: str, evidence: list[str])`
  - `models.ChangeResult(change: Change, metrics: Findings, logs: Findings, verdict: Verdict)`
  - `models.findings_schema() -> dict` (Findings JSON schema without `agent`), `models.verdict_schema() -> dict` (Verdict JSON schema without `change_id`)
  - `config.Settings` (pydantic-settings, env prefix `CA_`) with fields listed in Global Constraints.

- [ ] **Step 1: Install host tooling**

Run:
```bash
brew install uv kind helm kubectl
uv --version && kind version && helm version --short && kubectl version --client
```
Expected: all four print versions (kubectl ≥ 1.30 so it is within skew of kind's node image).

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "k8s-change-analyst"
version = "0.1.0"
description = "LLM multi-agent analyst that judges the impact of Kubernetes changes"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "langgraph>=1.2,<2",
  "langchain-core>=1.6,<2",
  "kubernetes>=36",
  "httpx>=0.28",
  "pydantic>=2.13",
  "pydantic-settings>=2.15",
  "typer>=0.27",
]

[project.scripts]
k8s-change-analyst = "change_analyst.cli:app"

[dependency-groups]
dev = ["pytest>=9"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/change_analyst"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
```

Also create an empty `README.md` (filled in Task 14) — hatchling requires it to exist:
```bash
echo "# k8s-change-analyst" > README.md
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
reports/
logs/
state.json
state.json.tmp
.change-analyst.lock
.env
.e2e/
.e2e-run.log
```

- [ ] **Step 4: Pin Python and sync**

```bash
uv python pin 3.12
mkdir -p src/change_analyst tests
touch src/change_analyst/__init__.py
uv sync
```
Expected: `.python-version` contains `3.12`, `.venv` created, dependencies installed.

- [ ] **Step 5: Write failing tests**

`tests/test_models.py`:
```python
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
```

`tests/test_config.py`:
```python
import os
from pathlib import Path

from change_analyst.config import Settings


def _clear_ca_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key)


def test_defaults(monkeypatch):
    _clear_ca_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.prometheus_url == "http://localhost:9090"
    assert s.namespaces == ["demo"]
    assert s.llm_model == "sonnet"
    assert s.min_age_min == 5
    assert s.max_changes_per_run == 5
    assert s.state_path == Path("state.json")


def test_env_override(monkeypatch):
    _clear_ca_env(monkeypatch)
    monkeypatch.setenv("CA_NAMESPACES", '["demo", "shop"]')
    monkeypatch.setenv("CA_MIN_AGE_MIN", "3")
    s = Settings(_env_file=None)
    assert s.namespaces == ["demo", "shop"]
    assert s.min_age_min == 3
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.models'`

- [ ] **Step 7: Implement `src/change_analyst/models.py`**

```python
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
```

- [ ] **Step 8: Implement `src/change_analyst/config.py`**

```python
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CA_", env_file=".env", extra="ignore")

    prometheus_url: str = "http://localhost:9090"
    namespaces: list[str] = ["demo"]
    kube_context: str | None = None

    llm_backend: Literal["claude_cli"] = "claude_cli"
    llm_model: str | None = "sonnet"
    llm_timeout_s: int = 180

    before_window_min: int = 15
    settle_delay_min: int = 2
    min_age_min: int = 5
    max_after_window_min: int = 30
    max_pending_min: int = 60
    first_run_lookback_min: int = 60

    max_changes_per_run: int = 5
    max_agent_steps: int = 6
    max_concurrency: int = 2
    run_timeout_s: int = 1200

    reports_dir: Path = Path("reports")
    state_path: Path = Path("state.json")
    lock_path: Path = Path(".change-analyst.lock")
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore README.md src tests
git commit -m "feat: project skeleton with core models and settings"
```

---

### Task 2: State store and run lock

**Files:**
- Create: `src/change_analyst/state_store.py`, `src/change_analyst/lock.py`
- Test: `tests/test_state_store.py`, `tests/test_lock.py`

**Interfaces:**
- Consumes: `models.PendingChange`, `models.Change`.
- Produces:
  - `state_store.State(last_run_at: datetime | None, analyzed_change_ids: list[str], configmap_hashes: dict[str, str], pending: list[PendingChange])`
  - `state_store.load_state(path: Path) -> State`
  - `state_store.save_state(path: Path, state: State) -> None` (atomic)
  - `state_store.remember_analyzed(existing: list[str], new: list[str], limit: int = 1000) -> list[str]`
  - `lock.run_lock(path: Path)` context manager; `lock.LockHeld` exception.

- [ ] **Step 1: Write failing tests**

`tests/test_state_store.py`:
```python
from datetime import datetime, timezone

from change_analyst.models import Change, PendingChange
from change_analyst.state_store import State, load_state, remember_analyzed, save_state

T0 = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def test_load_missing_returns_empty_state(tmp_path):
    state = load_state(tmp_path / "state.json")
    assert state.last_run_at is None
    assert state.analyzed_change_ids == []
    assert state.configmap_hashes == {}
    assert state.pending == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "state.json"
    change = Change(
        id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
        workload="podinfo", changed_at=T0, description="d",
    )
    state = State(
        last_run_at=T0,
        analyzed_change_ids=["a"],
        configmap_hashes={"demo/podinfo-config": "abc"},
        pending=[PendingChange(change=change, first_seen=T0)],
    )
    save_state(path, state)
    assert load_state(path) == state
    assert not (path.parent / "state.json.tmp").exists()


def test_remember_analyzed_dedupes_and_trims():
    result = remember_analyzed(["a", "b", "c"], ["c", "d"], limit=3)
    assert result == ["b", "c", "d"]
```

`tests/test_lock.py`:
```python
import pytest

from change_analyst.lock import LockHeld, run_lock


def test_second_lock_is_refused(tmp_path):
    path = tmp_path / "run.lock"
    with run_lock(path):
        with pytest.raises(LockHeld):
            with run_lock(path):
                pass


def test_lock_is_released_after_exit(tmp_path):
    path = tmp_path / "run.lock"
    with run_lock(path):
        pass
    with run_lock(path):
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state_store.py tests/test_lock.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/change_analyst/state_store.py`**

```python
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from change_analyst.models import PendingChange


class State(BaseModel):
    last_run_at: datetime | None = None
    analyzed_change_ids: list[str] = Field(default_factory=list)
    configmap_hashes: dict[str, str] = Field(default_factory=dict)
    pending: list[PendingChange] = Field(default_factory=list)


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    return State.model_validate_json(path.read_text())


def save_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, path)


def remember_analyzed(existing: list[str], new: list[str], limit: int = 1000) -> list[str]:
    merged = [i for i in existing if i not in set(new)] + list(dict.fromkeys(new))
    return merged[-limit:]
```

- [ ] **Step 4: Implement `src/change_analyst/lock.py`**

```python
import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockHeld(Exception):
    """Another run holds the lock."""


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockHeld(str(path)) from exc
        yield
    finally:
        handle.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_state_store.py tests/test_lock.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/change_analyst/state_store.py src/change_analyst/lock.py tests/test_state_store.py tests/test_lock.py
git commit -m "feat: persistent run state and run lock"
```

---

### Task 3: Demo environment (kind + Prometheus + podinfo + scenarios)

**Files:**
- Create: `deploy/demo/kind-config.yaml`, `deploy/demo/prometheus-values.yaml`, `deploy/demo/app.yaml`, `deploy/demo/scenarios/harmless.sh`, `deploy/demo/scenarios/errors.sh`, `deploy/demo/scenarios/crashloop.sh`, `deploy/demo/scenarios/config.sh`, `deploy/demo/scenarios/reset.sh`, `Makefile`

**Interfaces:**
- Consumes: nothing.
- Produces: a cluster where Prometheus at `http://localhost:9090` has series `http_requests_total{namespace="demo",pod=~"podinfo-..."}`, `http_request_duration_seconds_bucket`, `container_cpu_usage_seconds_total`, `container_memory_working_set_bytes`, `kube_pod_container_status_restarts_total`, `kube_deployment_status_replicas_ready{namespace="demo",deployment="podinfo"}`. Scenario scripts callable as `deploy/demo/scenarios/<name>.sh`.

Prerequisite: Docker Desktop is running (`docker info` succeeds).

- [ ] **Step 1: Create `deploy/demo/kind-config.yaml`**

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: change-analyst
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30090   # Prometheus NodePort
        hostPort: 9090
        protocol: TCP
```

- [ ] **Step 2: Create `deploy/demo/prometheus-values.yaml`**

```yaml
grafana:
  enabled: false
alertmanager:
  enabled: false
prometheus:
  service:
    type: NodePort
    nodePort: 30090
  prometheusSpec:
    scrapeInterval: 15s
    evaluationInterval: 15s
    retention: 1d
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
```

- [ ] **Step 3: Create `deploy/demo/app.yaml`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: demo
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: podinfo-config
  namespace: demo
data:
  PODINFO_RANDOM_ERROR: "false"
  PODINFO_UI_MESSAGE: "hello from the change-analyst demo"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: podinfo
  namespace: demo
  labels:
    app: podinfo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: podinfo
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: podinfo
    spec:
      containers:
        - name: podinfo
          image: ghcr.io/stefanprodan/podinfo:6.15.0
          command: ["./podinfo"]
          args: ["--port=9898", "--level=info"]
          envFrom:
            - configMapRef:
                name: podinfo-config
          ports:
            - name: http
              containerPort: 9898
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            periodSeconds: 5
            timeoutSeconds: 2
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
            timeoutSeconds: 2
          resources:
            requests:
              cpu: 50m
              memory: 32Mi
            limits:
              memory: 128Mi
---
apiVersion: v1
kind: Service
metadata:
  name: podinfo
  namespace: demo
  labels:
    app: podinfo
spec:
  selector:
    app: podinfo
  ports:
    - name: http
      port: 9898
      targetPort: http
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: podinfo
  namespace: demo
spec:
  selector:
    matchLabels:
      app: podinfo
  namespaceSelector:
    matchNames: ["demo"]
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loadgen
  namespace: demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loadgen
  template:
    metadata:
      labels:
        app: loadgen
    spec:
      containers:
        - name: loadgen
          image: curlimages/curl:8.10.1
          command: ["/bin/sh", "-c"]
          args:
            - |
              while true; do
                for i in 1 2 3 4 5; do
                  curl -s -o /dev/null --max-time 3 http://podinfo.demo:9898/ &
                  curl -s -o /dev/null --max-time 3 http://podinfo.demo:9898/api/info &
                done
                wait
                sleep 1
              done
```

- [ ] **Step 4: Create scenario scripts**

`deploy/demo/scenarios/harmless.sh`:
```bash
#!/usr/bin/env bash
# Rollout that changes only a pod annotation: expected verdict no_impact.
set -euo pipefail
kubectl -n demo patch deployment podinfo --type merge \
  -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"demo/harmless-change\":\"$(date +%s)\"}}}}}"
kubectl -n demo rollout status deployment/podinfo --timeout=180s
```

`deploy/demo/scenarios/errors.sh`:
```bash
#!/usr/bin/env bash
# Rollout enabling random 5xx errors and 200-800ms delays: expected verdict regression.
set -euo pipefail
kubectl -n demo patch deployment podinfo --type json -p '[
  {"op": "replace", "path": "/spec/template/spec/containers/0/args",
   "value": ["--port=9898", "--level=info", "--random-error=true",
             "--random-delay=true", "--random-delay-unit=ms",
             "--random-delay-min=200", "--random-delay-max=800"]}
]'
kubectl -n demo rollout status deployment/podinfo --timeout=180s || true
```

`deploy/demo/scenarios/crashloop.sh`:
```bash
#!/usr/bin/env bash
# Rollout with an invalid flag: new pods exit with code 2 and crash-loop. Expected verdict regression.
set -euo pipefail
kubectl -n demo patch deployment podinfo --type json -p '[
  {"op": "replace", "path": "/spec/template/spec/containers/0/args",
   "value": ["--port=not-a-number", "--level=info"]}
]'
echo "crashloop rollout started (it will not complete; that is expected)"
```

`deploy/demo/scenarios/config.sh`:
```bash
#!/usr/bin/env bash
# ConfigMap edit enabling random errors, picked up by a restart: expected verdict regression.
set -euo pipefail
kubectl -n demo patch configmap podinfo-config --type merge -p '{"data":{"PODINFO_RANDOM_ERROR":"true"}}'
kubectl -n demo rollout restart deployment/podinfo
kubectl -n demo rollout status deployment/podinfo --timeout=180s || true
```

`deploy/demo/scenarios/reset.sh`:
```bash
#!/usr/bin/env bash
# Restore the known-good manifests and restart podinfo.
set -euo pipefail
cd "$(dirname "$0")/../../.."
kubectl apply -f deploy/demo/app.yaml
kubectl -n demo rollout restart deployment/podinfo
kubectl -n demo rollout status deployment/podinfo --timeout=180s
```

Make them executable:
```bash
chmod +x deploy/demo/scenarios/*.sh
```

- [ ] **Step 5: Create `Makefile`** (recipe lines MUST start with a TAB character)

```make
CLUSTER := change-analyst

.PHONY: demo-up demo-down scenario-%

demo-up:
	kind create cluster --config deploy/demo/kind-config.yaml
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm upgrade --install kps prometheus-community/kube-prometheus-stack \
		--namespace monitoring --create-namespace \
		-f deploy/demo/prometheus-values.yaml --wait --timeout 10m
	kubectl apply -f deploy/demo/app.yaml
	kubectl -n demo rollout status deployment/podinfo --timeout=180s
	kubectl -n demo rollout status deployment/loadgen --timeout=180s

demo-down:
	kind delete cluster --name $(CLUSTER)

scenario-%:
	deploy/demo/scenarios/$*.sh
```

- [ ] **Step 6: Bring the demo up and verify metrics**

Run:
```bash
docker info >/dev/null && make demo-up
sleep 120
for q in \
  'sum(rate(http_requests_total{namespace="demo"}[1m]))' \
  'sum(container_memory_working_set_bytes{namespace="demo",pod=~"podinfo-.*",container!=""})' \
  'sum(kube_pod_container_status_restarts_total{namespace="demo",pod=~"podinfo-.*"})' \
  'kube_deployment_status_replicas_ready{namespace="demo",deployment="podinfo"}'; do
  echo "== $q"; curl -s --get localhost:9090/api/v1/query --data-urlencode "query=$q" | head -c 300; echo
done
```
Expected: each query returns `"status":"success"` with a non-empty `result` array (request rate > 0, ready replicas `"2"`).
If `http_requests_total` is empty, check the ServiceMonitor is discovered: `curl -s localhost:9090/api/v1/targets | grep -o 'serviceMonitor/demo/podinfo[^"]*' | head -1`.

- [ ] **Step 7: Verify one scenario end-to-end manually**

Run:
```bash
make scenario-errors
sleep 90
curl -s --get localhost:9090/api/v1/query --data-urlencode 'query=sum(rate(http_requests_total{namespace="demo",status=~"5.."}[1m]))' | head -c 300; echo
make scenario-reset
```
Expected: the 5xx rate result is > 0 after the errors scenario; reset completes with `successfully rolled out`.

- [ ] **Step 8: Commit**

```bash
git add deploy Makefile
git commit -m "feat: kind demo environment with podinfo, Prometheus and fault scenarios"
```

---

### Task 4: Change detection

**Files:**
- Create: `src/change_analyst/changes/__init__.py` (empty), `src/change_analyst/changes/detect.py`, `tests/factories.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `models.Change`.
- Produces:
  - `detect.diff_templates(old_template, new_template) -> dict[str, list]`
  - `detect.detect_rollouts(replica_sets: list, since: datetime) -> list[Change]`
  - `detect.configmap_hash(cm) -> str`
  - `detect.detect_configmap_changes(configmaps: list, deployments: list, previous_hashes: dict[str, str]) -> tuple[list[Change], dict[str, str]]`
  - `detect.detect_changes(core, apps, namespaces: list[str], since: datetime, previous_hashes: dict[str, str]) -> tuple[list[Change], dict[str, str]]` — `core` is a `kubernetes.client.CoreV1Api`-like object, `apps` an `AppsV1Api`-like object.
  - `tests/factories.py`: `T0`, `container`, `template`, `replica_set`, `deployment`, `configmap`, `env_from_configmap`, `pod`, `event`, `FakeCore`, `FakeApps` (used by Tasks 10, 11, 14).

- [ ] **Step 1: Create `tests/factories.py`**

```python
"""Lightweight stand-ins for kubernetes client objects and APIs (attribute access only)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from kubernetes.client.exceptions import ApiException

T0 = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
IMAGE = "ghcr.io/stefanprodan/podinfo:6.15.0"


def container(name="podinfo", image=IMAGE, args=None, env=None, env_from=None, resources=None):
    return NS(name=name, image=image, args=args, command=None, env=env,
              env_from=env_from, resources=resources)


def template(containers=None, annotations=None, volumes=None):
    return NS(
        metadata=NS(annotations=annotations),
        spec=NS(containers=containers or [container()], init_containers=None, volumes=volumes),
    )


def replica_set(deployment="podinfo", revision=1, created=T0, tmpl=None, namespace="demo"):
    owners = [NS(kind="Deployment", name=deployment)] if deployment else []
    return NS(
        metadata=NS(
            name=f"{deployment or 'orphan'}-rev{revision}",
            namespace=namespace,
            annotations={"deployment.kubernetes.io/revision": str(revision)},
            owner_references=owners,
            creation_timestamp=created,
        ),
        spec=NS(template=tmpl or template()),
    )


def deployment(name="podinfo", namespace="demo", tmpl=None, match_labels=None):
    return NS(
        metadata=NS(name=name, namespace=namespace),
        spec=NS(template=tmpl or template(), selector=NS(match_labels=match_labels or {"app": name})),
    )


def configmap(name="podinfo-config", namespace="demo", data=None, modified=T0):
    return NS(
        metadata=NS(name=name, namespace=namespace, managed_fields=[NS(time=modified)],
                    creation_timestamp=T0 - timedelta(days=1)),
        data=data or {},
        binary_data=None,
    )


def env_from_configmap(name):
    return [NS(config_map_ref=NS(name=name))]


def pod(name="podinfo-abc12-xyz34", namespace="demo", phase="Running", ready=True, restarts=0,
        waiting=None, last_terminated=None, image=IMAGE, created=T0):
    status = NS(
        name="podinfo", ready=ready, restart_count=restarts,
        state=NS(waiting=NS(reason=waiting) if waiting else None),
        last_state=NS(terminated=NS(reason=last_terminated) if last_terminated else None),
    )
    return NS(
        metadata=NS(name=name, namespace=namespace, creation_timestamp=created),
        status=NS(phase=phase, container_statuses=[status]),
        spec=NS(containers=[NS(name="podinfo", image=image)]),
    )


def event(name="podinfo-abc12-xyz34", kind="Pod", reason="BackOff", type_="Warning",
          message="Back-off restarting failed container", count=3, when=T0):
    return NS(type=type_, reason=reason, message=message, count=count,
              involved_object=NS(kind=kind, name=name), last_timestamp=when,
              event_time=None, metadata=NS(creation_timestamp=when))


class FakeApps:
    def __init__(self, replica_sets=(), deployments=()):
        self.replica_sets = list(replica_sets)
        self.deployments = list(deployments)

    def list_namespaced_replica_set(self, namespace):
        return NS(items=[r for r in self.replica_sets if r.metadata.namespace == namespace])

    def list_namespaced_deployment(self, namespace):
        return NS(items=[d for d in self.deployments if d.metadata.namespace == namespace])

    def read_namespaced_deployment(self, name, namespace):
        for d in self.deployments:
            if d.metadata.name == name and d.metadata.namespace == namespace:
                return d
        raise ApiException(status=404, reason="Not Found")


class FakeCore:
    def __init__(self, configmaps=(), pods=(), events=(), logs=None):
        self.configmaps = list(configmaps)
        self.pods = list(pods)
        self.events = list(events)
        self.logs = logs or {}  # (pod_name, previous) -> text
        self.label_selectors = []

    def list_namespaced_config_map(self, namespace):
        return NS(items=[c for c in self.configmaps if c.metadata.namespace == namespace])

    def list_namespaced_pod(self, namespace, label_selector=None):
        self.label_selectors.append(label_selector)
        return NS(items=[p for p in self.pods if p.metadata.namespace == namespace])

    def list_namespaced_event(self, namespace):
        return NS(items=list(self.events))

    def read_namespaced_pod_log(self, name, namespace, previous=False, since_seconds=None, tail_lines=None):
        key = (name, previous)
        if key not in self.logs:
            raise ApiException(status=400, reason="Bad Request")
        return self.logs[key]
```

- [ ] **Step 2: Write failing tests**

`tests/test_detect.py`:
```python
from datetime import timedelta

from factories import (
    T0, FakeApps, FakeCore, configmap, container, deployment, env_from_configmap,
    replica_set, template,
)

from change_analyst.changes.detect import (
    configmap_hash, detect_changes, detect_configmap_changes, detect_rollouts, diff_templates,
)


def test_diff_templates_reports_changed_fields_only():
    old = template([container(image="podinfo:6.14.0", args=["--port=9898"])])
    new = template([container(image="podinfo:6.15.0", args=["--port=9898"])],
                   annotations={"demo/x": "1"})
    diff = diff_templates(old, new)
    assert diff == {
        "podinfo.image": ["podinfo:6.14.0", "podinfo:6.15.0"],
        "pod_annotations": [{}, {"demo/x": "1"}],
    }


def test_detects_rollout_after_since():
    old = replica_set(revision=1, created=T0 - timedelta(hours=2),
                      tmpl=template([container(image="podinfo:6.14.0")]))
    new = replica_set(revision=2, created=T0 + timedelta(minutes=5),
                      tmpl=template([container(image="podinfo:6.15.0")]))
    changes = detect_rollouts([new, old], since=T0)
    assert len(changes) == 1
    change = changes[0]
    assert change.id == "rollout:demo/podinfo:rev2"
    assert change.kind == "rollout"
    assert change.workload == "podinfo"
    assert change.changed_at == T0 + timedelta(minutes=5)
    assert change.diff == {"podinfo.image": ["podinfo:6.14.0", "podinfo:6.15.0"]}
    assert "podinfo.image" in change.description


def test_ignores_old_first_revision_and_orphan_replica_sets():
    first = replica_set(revision=1, created=T0 + timedelta(minutes=1))
    orphan = replica_set(deployment=None, revision=2, created=T0 + timedelta(minutes=1))
    old_pair = [replica_set(deployment="api", revision=1, created=T0 - timedelta(hours=3)),
                replica_set(deployment="api", revision=2, created=T0 - timedelta(hours=1))]
    assert detect_rollouts([first, orphan, *old_pair], since=T0) == []


def test_configmap_first_sighting_only_records_hash():
    cm = configmap(data={"A": "1"})
    changes, hashes = detect_configmap_changes([cm], [], previous_hashes={})
    assert changes == []
    assert hashes == {"demo/podinfo-config": configmap_hash(cm)}


def test_configmap_change_links_referencing_deployments():
    old = configmap(data={"PODINFO_RANDOM_ERROR": "false"})
    new = configmap(data={"PODINFO_RANDOM_ERROR": "true"}, modified=T0 + timedelta(minutes=3))
    user = deployment(tmpl=template([container(env_from=env_from_configmap("podinfo-config"))]))
    bystander = deployment(name="other")
    changes, hashes = detect_configmap_changes(
        [new], [user, bystander], previous_hashes={"demo/podinfo-config": configmap_hash(old)}
    )
    assert len(changes) == 1
    change = changes[0]
    assert change.kind == "configmap"
    assert change.workload == "podinfo"
    assert change.changed_at == T0 + timedelta(minutes=3)
    assert change.id == f"configmap:demo/podinfo-config:{configmap_hash(new)}:podinfo"
    assert change.diff["data"] == {"PODINFO_RANDOM_ERROR": "true"}
    assert hashes["demo/podinfo-config"] == configmap_hash(new)


def test_unchanged_configmap_produces_no_change():
    cm = configmap(data={"A": "1"})
    user = deployment(tmpl=template([container(env_from=env_from_configmap("podinfo-config"))]))
    changes, _ = detect_configmap_changes([cm], [user], {"demo/podinfo-config": configmap_hash(cm)})
    assert changes == []


def test_detect_changes_queries_each_namespace():
    apps = FakeApps(replica_sets=[
        replica_set(revision=1, created=T0 - timedelta(hours=1)),
        replica_set(revision=2, created=T0 + timedelta(minutes=1),
                    tmpl=template([container(args=["--level=debug"])])),
    ])
    core = FakeCore(configmaps=[configmap()])
    changes, hashes = detect_changes(core, apps, ["demo", "empty"], T0, {})
    assert [c.id for c in changes] == ["rollout:demo/podinfo:rev2"]
    assert "demo/podinfo-config" in hashes
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.changes'`

- [ ] **Step 4: Implement `src/change_analyst/changes/detect.py`**

Also create empty `src/change_analyst/changes/__init__.py`.

```python
"""Deterministic change detection from the Kubernetes API (no LLM)."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from change_analyst.models import Change

REVISION_ANNOTATION = "deployment.kubernetes.io/revision"
MAX_VALUE_CHARS = 200


def _revision(rs) -> int:
    return int((rs.metadata.annotations or {}).get(REVISION_ANNOTATION, "0"))


def _owner_deployment(rs) -> str | None:
    for ref in rs.metadata.owner_references or []:
        if ref.kind == "Deployment":
            return ref.name
    return None


def _resources(resources) -> dict[str, dict[str, str]]:
    if resources is None:
        return {}
    return {"limits": dict(resources.limits or {}), "requests": dict(resources.requests or {})}


def _container_summary(pod_template) -> dict[str, dict[str, Any]]:
    summary = {}
    for c in pod_template.spec.containers or []:
        env = {e.name: (e.value if e.value is not None else "<valueFrom>") for e in c.env or []}
        summary[c.name] = {
            "image": c.image,
            "args": list(c.args or []),
            "command": list(c.command or []),
            "env": env,
            "resources": _resources(c.resources),
        }
    return summary


def _template_annotations(pod_template) -> dict[str, str]:
    metadata = pod_template.metadata
    return dict(metadata.annotations or {}) if metadata else {}


def diff_templates(old_template, new_template) -> dict[str, list]:
    diff: dict[str, list] = {}
    old_c, new_c = _container_summary(old_template), _container_summary(new_template)
    for name in sorted(set(old_c) | set(new_c)):
        old, new = old_c.get(name, {}), new_c.get(name, {})
        for field in ("image", "args", "command", "env", "resources"):
            if old.get(field) != new.get(field):
                diff[f"{name}.{field}"] = [old.get(field), new.get(field)]
    old_a, new_a = _template_annotations(old_template), _template_annotations(new_template)
    if old_a != new_a:
        diff["pod_annotations"] = [old_a, new_a]
    return diff


def _describe_diff(diff: dict[str, list]) -> str:
    if not diff:
        return "pod template changed (no container-level difference detected)"
    parts = []
    for key, (old, new) in diff.items():
        parts.append(f"{key}: {json.dumps(old)[:300]} -> {json.dumps(new)[:300]}")
    return "; ".join(parts)


def detect_rollouts(replica_sets: list, since: datetime) -> list[Change]:
    by_deployment: dict[tuple[str, str], list] = defaultdict(list)
    for rs in replica_sets:
        name = _owner_deployment(rs)
        if name:
            by_deployment[(rs.metadata.namespace, name)].append(rs)

    changes = []
    for (namespace, name), rss in by_deployment.items():
        rss.sort(key=_revision)
        for previous, current in zip(rss, rss[1:]):
            if current.metadata.creation_timestamp <= since:
                continue
            revision = _revision(current)
            diff = diff_templates(previous.spec.template, current.spec.template)
            changes.append(Change(
                id=f"rollout:{namespace}/{name}:rev{revision}",
                kind="rollout",
                namespace=namespace,
                workload=name,
                changed_at=current.metadata.creation_timestamp,
                description=f"Deployment {name} rolled out revision {revision}: {_describe_diff(diff)}",
                diff=diff,
            ))
    return sorted(changes, key=lambda c: c.changed_at)


def configmap_hash(cm) -> str:
    payload = json.dumps({"data": cm.data or {}, "binary_data": cm.binary_data or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _last_modified(cm) -> datetime:
    times = [f.time for f in cm.metadata.managed_fields or [] if f.time]
    return max(times) if times else cm.metadata.creation_timestamp


def _uses_configmap(deploy, cm_name: str) -> bool:
    spec = deploy.spec.template.spec
    for volume in spec.volumes or []:
        if volume.config_map and volume.config_map.name == cm_name:
            return True
    for c in list(spec.containers or []) + list(spec.init_containers or []):
        for source in c.env_from or []:
            if source.config_map_ref and source.config_map_ref.name == cm_name:
                return True
        for env in c.env or []:
            ref = env.value_from.config_map_key_ref if env.value_from else None
            if ref and ref.name == cm_name:
                return True
    return False


def detect_configmap_changes(
    configmaps: list, deployments: list, previous_hashes: dict[str, str]
) -> tuple[list[Change], dict[str, str]]:
    hashes: dict[str, str] = {}
    changes = []
    for cm in configmaps:
        namespace, name = cm.metadata.namespace, cm.metadata.name
        key = f"{namespace}/{name}"
        digest = configmap_hash(cm)
        hashes[key] = digest
        old = previous_hashes.get(key)
        if old is None or old == digest:
            continue
        data = {k: str(v)[:MAX_VALUE_CHARS] for k, v in (cm.data or {}).items()}
        for d in deployments:
            if d.metadata.namespace != namespace or not _uses_configmap(d, name):
                continue
            workload = d.metadata.name
            changes.append(Change(
                id=f"configmap:{key}:{digest}:{workload}",
                kind="configmap",
                namespace=namespace,
                workload=workload,
                changed_at=_last_modified(cm),
                description=(f"ConfigMap {key} data changed (hash {old} -> {digest}); "
                             f"used by Deployment {workload}"),
                diff={"data": data},
            ))
    return changes, hashes


def detect_changes(
    core, apps, namespaces: list[str], since: datetime, previous_hashes: dict[str, str]
) -> tuple[list[Change], dict[str, str]]:
    changes: list[Change] = []
    hashes: dict[str, str] = {}
    for namespace in namespaces:
        replica_sets = apps.list_namespaced_replica_set(namespace).items
        deployments = apps.list_namespaced_deployment(namespace).items
        configmaps = core.list_namespaced_config_map(namespace).items
        changes.extend(detect_rollouts(replica_sets, since))
        cm_changes, cm_hashes = detect_configmap_changes(configmaps, deployments, previous_hashes)
        changes.extend(cm_changes)
        hashes.update(cm_hashes)
    return sorted(changes, key=lambda c: c.changed_at), hashes
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_detect.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/change_analyst/changes tests/factories.py tests/test_detect.py
git commit -m "feat: detect deployment rollouts and configmap changes"
```

---

### Task 5: Run planning and analysis windows

**Files:**
- Create: `src/change_analyst/planning.py`
- Test: `tests/test_planning.py`

**Interfaces:**
- Consumes: `models.Change`, `models.PendingChange`, `state_store.State`, `config.Settings`.
- Produces:
  - `planning.Windows(before_start, before_end, after_start, after_end: datetime)` frozen dataclass with `as_dict() -> dict[str, str]`
  - `planning.compute_windows(changed_at: datetime, now: datetime, settings: Settings) -> Windows`
  - `planning.plan_run(detected: list[Change], state: State, now: datetime, settings: Settings) -> tuple[list[PendingChange], list[PendingChange]]` — returns `(ready, waiting)`; ready is oldest-first and capped at `max_changes_per_run`.

- [ ] **Step 1: Write failing tests**

`tests/test_planning.py`:
```python
from datetime import datetime, timedelta, timezone

from change_analyst.config import Settings
from change_analyst.models import Change, PendingChange
from change_analyst.planning import compute_windows, plan_run
from change_analyst.state_store import State

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(
    _env_file=None, before_window_min=15, settle_delay_min=2, min_age_min=5,
    max_after_window_min=30, max_pending_min=60, max_changes_per_run=2,
)


def mk(change_id: str, minutes_ago: float) -> Change:
    return Change(id=change_id, kind="rollout", namespace="demo", workload="podinfo",
                  changed_at=NOW - timedelta(minutes=minutes_ago), description="d")


def ids(items):
    return [p.change.id for p in items]


def test_young_change_waits_and_old_change_is_ready():
    ready, waiting = plan_run([mk("young", 2), mk("old", 10)], State(), NOW, SETTINGS)
    assert ids(ready) == ["old"]
    assert ids(waiting) == ["young"]
    assert waiting[0].first_seen == NOW


def test_already_analyzed_changes_are_skipped():
    ready, waiting = plan_run([mk("done", 10)], State(analyzed_change_ids=["done"]), NOW, SETTINGS)
    assert ready == [] and waiting == []


def test_pending_change_keeps_first_seen_and_is_not_duplicated():
    first_seen = NOW - timedelta(minutes=8)
    state = State(pending=[PendingChange(change=mk("p", 9), first_seen=first_seen)])
    ready, waiting = plan_run([mk("p", 9)], state, NOW, SETTINGS)
    assert ids(ready) == ["p"]
    assert ready[0].first_seen == first_seen
    assert waiting == []


def test_change_pending_too_long_is_forced_ready():
    state = State(pending=[PendingChange(change=mk("stuck", 1), first_seen=NOW - timedelta(minutes=61))])
    ready, _ = plan_run([], state, NOW, SETTINGS)
    assert ids(ready) == ["stuck"]


def test_ready_changes_are_capped_oldest_first():
    ready, waiting = plan_run([mk("a", 30), mk("b", 20), mk("c", 10)], State(), NOW, SETTINGS)
    assert ids(ready) == ["a", "b"]
    assert ids(waiting) == ["c"]


def test_compute_windows():
    changed = NOW - timedelta(minutes=10)
    w = compute_windows(changed, NOW, SETTINGS)
    assert w.before_start == changed - timedelta(minutes=15)
    assert w.before_end == changed
    assert w.after_start == changed + timedelta(minutes=2)
    assert w.after_end == NOW
    assert w.as_dict()["after_end"] == NOW.isoformat()


def test_compute_windows_caps_after_window():
    changed = NOW - timedelta(hours=2)
    w = compute_windows(changed, NOW, SETTINGS)
    assert w.after_end == changed + timedelta(minutes=32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_planning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.planning'`

- [ ] **Step 3: Implement `src/change_analyst/planning.py`**

```python
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from change_analyst.config import Settings
from change_analyst.models import Change, PendingChange
from change_analyst.state_store import State


@dataclass(frozen=True)
class Windows:
    before_start: datetime
    before_end: datetime
    after_start: datetime
    after_end: datetime

    def as_dict(self) -> dict[str, str]:
        return {k: v.isoformat() for k, v in asdict(self).items()}


def compute_windows(changed_at: datetime, now: datetime, settings: Settings) -> Windows:
    after_start = changed_at + timedelta(minutes=settings.settle_delay_min)
    after_end = min(now, after_start + timedelta(minutes=settings.max_after_window_min))
    return Windows(
        before_start=changed_at - timedelta(minutes=settings.before_window_min),
        before_end=changed_at,
        after_start=after_start,
        after_end=max(after_end, after_start),
    )


def plan_run(
    detected: list[Change], state: State, now: datetime, settings: Settings
) -> tuple[list[PendingChange], list[PendingChange]]:
    analyzed = set(state.analyzed_change_ids)
    candidates: dict[str, PendingChange] = {p.change.id: p for p in state.pending}
    for change in detected:
        if change.id not in analyzed and change.id not in candidates:
            candidates[change.id] = PendingChange(change=change, first_seen=now)

    min_age = timedelta(minutes=settings.min_age_min)
    max_pending = timedelta(minutes=settings.max_pending_min)
    ready: list[PendingChange] = []
    waiting: list[PendingChange] = []
    for item in sorted(candidates.values(), key=lambda p: p.change.changed_at):
        old_enough = now - item.change.changed_at >= min_age
        waited_too_long = now - item.first_seen >= max_pending
        if (old_enough or waited_too_long) and len(ready) < settings.max_changes_per_run:
            ready.append(item)
        else:
            waiting.append(item)
    return ready, waiting
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_planning.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/change_analyst/planning.py tests/test_planning.py
git commit -m "feat: plan ready/pending changes and compute analysis windows"
```

---

### Task 6: Claude CLI client

**Files:**
- Create: `src/change_analyst/llm/__init__.py` (empty), `src/change_analyst/llm/client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `client.LLMResult(text: str, structured: dict | None)`
  - `client.LLMError(RuntimeError)`
  - `client.LLMClient` Protocol: `complete(system: str, prompt: str, json_schema: dict | None = None) -> LLMResult`
  - `client.ClaudeCLIClient(model: str | None = None, timeout_s: int = 180, executable: str = "claude", workdir: str | None = None, retries: int = 1, runner=subprocess.run)` with `.complete(...)`, `.build_command(system, json_schema) -> list[str]`, and `.calls: int` (number of subprocess invocations).

- [ ] **Step 1: Write failing tests**

`tests/test_llm_client.py`:
```python
import json
import subprocess

import pytest

from change_analyst.llm.client import ClaudeCLIClient, LLMError


def proc(payload, returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=json.dumps(payload), stderr=stderr)


SUCCESS = {"type": "result", "subtype": "success", "is_error": False,
           "result": '{"answer": 5}', "structured_output": {"answer": 5}}


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_builds_isolated_command_and_returns_structured_output():
    runner = FakeRunner([proc(SUCCESS)])
    client = ClaudeCLIClient(model="sonnet", timeout_s=30, workdir="/tmp", runner=runner)
    schema = {"type": "object"}
    result = client.complete("SYS", "PROMPT", json_schema=schema)

    assert result.structured == {"answer": 5}
    assert result.text == '{"answer": 5}'
    cmd, kwargs = runner.calls[0]
    assert cmd[:3] == ["claude", "-p", "--output-format"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in cmd
    assert "--strict-mcp-config" in cmd
    assert "--disable-slash-commands" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema
    assert kwargs["input"] == "PROMPT"
    assert kwargs["timeout"] == 30
    assert kwargs["cwd"] == "/tmp"
    assert client.calls == 1


def test_omits_model_and_schema_when_not_given():
    runner = FakeRunner([proc({**SUCCESS, "structured_output": None})])
    client = ClaudeCLIClient(model=None, runner=runner)
    result = client.complete("SYS", "PROMPT")
    cmd, _ = runner.calls[0]
    assert "--model" not in cmd and "--json-schema" not in cmd
    assert result.structured is None


def test_retries_once_after_timeout():
    runner = FakeRunner([subprocess.TimeoutExpired(cmd="claude", timeout=1), proc(SUCCESS)])
    client = ClaudeCLIClient(runner=runner)
    assert client.complete("S", "P", json_schema={"type": "object"}).structured == {"answer": 5}
    assert client.calls == 2


@pytest.mark.parametrize("bad", [
    proc(SUCCESS, returncode=1, stderr="auth failed"),
    subprocess.CompletedProcess(args=[], returncode=0, stdout="not json", stderr=""),
    proc({**SUCCESS, "is_error": True, "subtype": "error_during_execution"}),
    proc({**SUCCESS, "structured_output": None}),
])
def test_raises_after_two_failures(bad):
    runner = FakeRunner([bad, bad])
    client = ClaudeCLIClient(runner=runner)
    with pytest.raises(LLMError):
        client.complete("S", "P", json_schema={"type": "object"})
    assert len(runner.calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.llm'`

- [ ] **Step 3: Implement `src/change_analyst/llm/client.py`**

```python
"""LLM backends. The Claude CLI backend shells out to `claude -p`."""
import json
import logging
import subprocess
import tempfile
import threading
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

log = logging.getLogger(__name__)


class LLMResult(BaseModel):
    text: str
    structured: dict[str, Any] | None = None


class LLMError(RuntimeError):
    """The LLM backend failed to produce a usable answer."""


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult: ...


class ClaudeCLIClient:
    def __init__(
        self,
        model: str | None = None,
        timeout_s: int = 180,
        executable: str = "claude",
        workdir: str | None = None,
        retries: int = 1,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.executable = executable
        # A neutral directory so no project CLAUDE.md is picked up.
        self.workdir = workdir or tempfile.gettempdir()
        self.retries = retries
        self.runner = runner
        self.calls = 0
        self._lock = threading.Lock()

    def build_command(self, system: str, json_schema: dict | None) -> list[str]:
        cmd = [
            self.executable, "-p",
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "",
            "--system-prompt", system,
        ]
        if self.model:
            cmd += ["--model", self.model]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema)]
        return cmd

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult:
        last_error: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._call_once(system, prompt, json_schema)
            except LLMError as exc:
                last_error = exc
                log.warning("claude CLI attempt %d failed: %s", attempt + 1, exc)
        assert last_error is not None
        raise last_error

    def _call_once(self, system: str, prompt: str, json_schema: dict | None) -> LLMResult:
        with self._lock:
            self.calls += 1
        cmd = self.build_command(system, json_schema)
        try:
            proc = self.runner(cmd, input=prompt, capture_output=True, text=True,
                               timeout=self.timeout_s, cwd=self.workdir)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude CLI timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            raise LLMError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON output: {proc.stdout[:200]!r}") from exc
        if data.get("is_error") or data.get("subtype") != "success":
            raise LLMError(f"claude CLI reported an error ({data.get('subtype')}): "
                           f"{str(data.get('result'))[:300]}")
        structured = data.get("structured_output")
        if json_schema is not None and not isinstance(structured, dict):
            raise LLMError("claude CLI returned no structured_output for a schema request")
        return LLMResult(text=data.get("result") or "", structured=structured)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: 7 passed.

- [ ] **Step 5: Smoke-test against the real CLI (manual, costs ~1 cent)**

Run:
```bash
uv run python -c "
from change_analyst.llm.client import ClaudeCLIClient
c = ClaudeCLIClient(model='sonnet')
print(c.complete('You are a calculator.', 'What is 2+3?', json_schema={'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer']}))
"
```
Expected: `text='{"answer":5}' structured={'answer': 5}` (text formatting may differ).

- [ ] **Step 6: Commit**

```bash
git add src/change_analyst/llm tests/test_llm_client.py
git commit -m "feat: Claude CLI LLM client with structured output and retry"
```

---

### Task 7: CLIChatModel with prompt-based tool calling

**Files:**
- Create: `src/change_analyst/llm/chat_model.py`
- Test: `tests/test_chat_model.py`

**Interfaces:**
- Consumes: `llm.client.LLMClient`, `llm.client.LLMResult`.
- Produces:
  - `chat_model.CLIChatModel(client: LLMClient, tool_specs: list[dict] = [], final_schema: dict = {"type": "object"})` — a LangChain `BaseChatModel`.
  - `CLIChatModel.bind_tools(tools, *, final_schema: dict | None = None) -> CLIChatModel` (returns a copy).
  - `CLIChatModel.response_schema() -> dict`
  - Invocation returns `AIMessage` with either `tool_calls=[{"name", "args", "id", "type": "tool_call"}]` and empty content, or `content=json.dumps(final_object)`.
  - `chat_model.LLMOutputError(RuntimeError)` raised after two invalid replies.

- [ ] **Step 1: Write failing tests**

`tests/test_chat_model.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.llm.chat_model'`

- [ ] **Step 3: Implement `src/change_analyst/llm/chat_model.py`**

```python
"""A LangChain chat model that runs prompt-based tool calling over any LLMClient."""
import json
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

PROTOCOL = """You are running inside an automated agent loop. Reply ONLY with a JSON object matching the response schema.
- To call a tool: {"action": "call_tool", "tool": "<tool name>", "args": {<arguments>}}. Call exactly one tool per reply.
- To finish: {"action": "final", "final": <your final answer object>}.
Base every statement on tool results you have actually received."""


class LLMOutputError(RuntimeError):
    """The model did not produce a valid action."""


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content)


class CLIChatModel(BaseChatModel):
    client: Any
    tool_specs: list[dict] = Field(default_factory=list)
    final_schema: dict = Field(default_factory=lambda: {"type": "object"})

    @property
    def _llm_type(self) -> str:
        return "cli-chat-model"

    def bind_tools(self, tools, *, final_schema: dict | None = None, **kwargs: Any) -> "CLIChatModel":
        update: dict[str, Any] = {"tool_specs": [convert_to_openai_tool(t)["function"] for t in tools]}
        if final_schema is not None:
            update["final_schema"] = final_schema
        return self.model_copy(update=update)

    def response_schema(self) -> dict:
        actions = ["final"] + (["call_tool"] if self.tool_specs else [])
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": actions},
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "final": self.final_schema,
            },
            "required": ["action"],
        }

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        system, prompt = self._render(messages)
        schema = self.response_schema()
        error: str | None = None
        for _ in range(2):
            attempt_prompt = prompt if error is None else (
                f"{prompt}\n\nYOUR PREVIOUS REPLY WAS INVALID: {error}\nReply again with a valid JSON object."
            )
            result = self.client.complete(system, attempt_prompt, json_schema=schema)
            try:
                message = self._to_message(result.structured)
            except LLMOutputError as exc:
                error = str(exc)
                continue
            return ChatResult(generations=[ChatGeneration(message=message)])
        raise LLMOutputError(f"model produced an invalid reply twice: {error}")

    def _render(self, messages: list[BaseMessage]) -> tuple[str, str]:
        system_parts: list[str] = []
        lines: list[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(_text(m.content))
            elif isinstance(m, HumanMessage):
                lines.append(f"[USER]\n{_text(m.content)}")
            elif isinstance(m, AIMessage):
                for call in m.tool_calls:
                    lines.append(f"[ASSISTANT called tool {call['name']} with args {json.dumps(call['args'])}]")
                if m.content:
                    lines.append(f"[ASSISTANT]\n{_text(m.content)}")
            elif isinstance(m, ToolMessage):
                lines.append(f"[TOOL RESULT {m.name or ''}]\n{_text(m.content)}")
        if self.tool_specs:
            tools_block = "AVAILABLE TOOLS:\n" + json.dumps(self.tool_specs, indent=2)
        else:
            tools_block = "AVAILABLE TOOLS: (no tools available — you must reply with action=final now)"
        system = "\n\n".join(system_parts + [PROTOCOL, tools_block])
        return system, "\n\n".join(lines)

    def _to_message(self, data: Any) -> AIMessage:
        if not isinstance(data, dict):
            raise LLMOutputError("reply is not a JSON object")
        action = data.get("action")
        if action == "call_tool" and self.tool_specs:
            names = {spec["name"] for spec in self.tool_specs}
            tool_name = data.get("tool")
            args = data.get("args") or {}
            if tool_name not in names:
                raise LLMOutputError(f"unknown tool {tool_name!r}; available tools: {sorted(names)}")
            if not isinstance(args, dict):
                raise LLMOutputError("args must be a JSON object")
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            return AIMessage(content="", tool_calls=[
                {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
            ])
        if action == "final":
            if "final" not in data:
                raise LLMOutputError("action=final requires a 'final' field")
            return AIMessage(content=json.dumps(data["final"]))
        raise LLMOutputError(f"unsupported action {action!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_model.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/change_analyst/llm/chat_model.py tests/test_chat_model.py
git commit -m "feat: CLIChatModel with prompt-based tool calling"
```

---

### Task 8: Tool-agent runner (LangGraph loop with step budget)

**Files:**
- Create: `src/change_analyst/agents/__init__.py` (empty), `src/change_analyst/agents/runner.py`
- Test: `tests/test_agent_runner.py`

**Interfaces:**
- Consumes: `llm.chat_model.CLIChatModel`, `models.Change`, `planning.Windows`.
- Produces:
  - `runner.run_tool_agent(model: CLIChatModel, tools: list[BaseTool], system: str, task: str, max_steps: int, final_schema: dict) -> dict` — runs model→ToolNode loop; after `max_steps` tool results the model is re-bound with no tools so it must finish; returns the parsed final JSON object. Raises on LLM failure (callers catch).
  - `runner.build_task(change: Change, windows: Windows) -> str`

- [ ] **Step 1: Write failing tests**

`tests/test_agent_runner.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.agents'`

- [ ] **Step 3: Implement `src/change_analyst/agents/runner.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/change_analyst/agents tests/test_agent_runner.py
git commit -m "feat: LangGraph tool-agent loop with step budget"
```

---

### Task 9: Prometheus client and metrics tools

**Files:**
- Create: `src/change_analyst/tools/__init__.py` (empty), `src/change_analyst/tools/prometheus.py`, `src/change_analyst/tools/metrics.py`
- Test: `tests/test_prometheus.py`, `tests/test_metrics_tools.py`

**Interfaces:**
- Consumes: `models.Change`, `planning.Windows`.
- Produces:
  - `prometheus.PromError(RuntimeError)`
  - `prometheus.PromClient(base_url: str, timeout_s: float = 15.0, http: httpx.Client | None = None)` with `query_range(query: str, start: datetime, end: datetime, step_s: int = 15) -> list[dict]` (Prometheus `data.result`: `[{"metric": {...}, "values": [[ts, "val"], ...]}]`)
  - `prometheus.summarize_values(values: list) -> dict | None` → `{"avg", "p95", "min", "max", "last", "samples"}` rounded to 6 decimals
  - `metrics.METRIC_QUERIES: dict[str, str]` keys: `request_rate, error_rate_5xx, latency_p95, cpu, memory, restarts, ready_replicas`
  - `metrics.pod_selector(namespace: str, workload: str) -> str`
  - `metrics.make_metrics_tools(prom, change: Change, windows: Windows) -> list[BaseTool]` → tools `compare_windows(metrics: list[str])`, `query_promql(query: str)`. `prom` is any object with `query_range(query, start, end)`.

- [ ] **Step 1: Write failing tests**

`tests/test_prometheus.py`:
```python
from datetime import datetime, timezone

import httpx
import pytest

from change_analyst.tools.prometheus import PromClient, PromError, summarize_values

T = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def test_query_range_sends_params_and_returns_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "success", "data": {"resultType": "matrix", "result": [
            {"metric": {}, "values": [[1, "1"], [2, "3"]]}]}})

    client = PromClient("http://prom:9090", http=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.query_range("up", T, T, step_s=30)
    assert result == [{"metric": {}, "values": [[1, "1"], [2, "3"]]}]
    assert seen["path"] == "/api/v1/query_range"
    assert seen["query"] == "up"
    assert seen["step"] == "30s"
    assert float(seen["start"]) == T.timestamp()


def test_query_range_raises_on_prometheus_error():
    def handler(request):
        return httpx.Response(400, json={"status": "error", "error": "parse error"})

    client = PromClient("http://prom:9090", http=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(PromError, match="parse error"):
        client.query_range("sum(", T, T)


def test_summarize_values_skips_nan():
    summary = summarize_values([[1, "1"], [2, "NaN"], [3, "3"], [4, "2"]])
    assert summary == {"avg": 2.0, "p95": 3.0, "min": 1.0, "max": 3.0, "last": 2.0, "samples": 3}
    assert summarize_values([[1, "NaN"]]) is None
```

`tests/test_metrics_tools.py`:
```python
import json
from datetime import datetime, timedelta, timezone

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.metrics import METRIC_QUERIES, make_metrics_tools, pod_selector
from change_analyst.tools.prometheus import PromError

T = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
W = Windows(T - timedelta(minutes=15), T, T + timedelta(minutes=2), T + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T, description="d")


class FakeProm:
    def __init__(self, before=(1.0, 1.0), after=(2.0, 2.0), series=1, fail=False):
        self.before, self.after, self.series, self.fail = before, after, series, fail
        self.queries = []

    def query_range(self, query, start, end, step_s=15):
        self.queries.append((query, start, end))
        if self.fail:
            raise PromError("connection refused")
        values = self.before if start == W.before_start else self.after
        return [{"metric": {"i": str(i)}, "values": [[n, str(v)] for n, v in enumerate(values)]}
                for i in range(self.series)]


def tools_by_name(prom):
    return {t.name: t for t in make_metrics_tools(prom, CHANGE, W)}


def test_pod_selector_matches_deployment_pods_only():
    assert pod_selector("demo", "podinfo") == 'namespace="demo",pod=~"podinfo-[a-z0-9]+-[a-z0-9]+"'


def test_all_metric_queries_render():
    for name, template in METRIC_QUERIES.items():
        rendered = template.format(sel=pod_selector("demo", "podinfo"), ns="demo", wl="podinfo")
        assert "{" in rendered and "{sel}" not in rendered, name


def test_compare_windows_reports_before_after_and_change():
    prom = FakeProm(before=(1.0, 1.0), after=(3.0, 3.0))
    out = json.loads(tools_by_name(prom)["compare_windows"].invoke({"metrics": ["request_rate"]}))
    entry = out[0]
    assert entry["metric"] == "request_rate"
    assert entry["before"]["avg"] == 1.0
    assert entry["after"]["avg"] == 3.0
    assert entry["avg_change_pct"] == 200.0
    assert (W.before_start, W.before_end) in [(q[1], q[2]) for q in prom.queries]
    assert (W.after_start, W.after_end) in [(q[1], q[2]) for q in prom.queries]


def test_compare_windows_handles_unknown_metric_and_errors():
    out = json.loads(tools_by_name(FakeProm(fail=True))["compare_windows"].invoke(
        {"metrics": ["bogus", "cpu"]}))
    assert out[0]["error"].startswith("unknown metric")
    assert "connection refused" in out[1]["error"]


def test_query_promql_caps_series():
    prom = FakeProm(series=12)
    out = json.loads(tools_by_name(prom)["query_promql"].invoke({"query": "up"}))
    assert len(out["series"]) == 10
    assert out["total_series"] == 12
    assert prom.queries[0][1:] == (W.before_start, W.after_end)


def test_query_promql_returns_error_text():
    text = tools_by_name(FakeProm(fail=True))["query_promql"].invoke({"query": "up"})
    assert text.startswith("ERROR:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prometheus.py tests/test_metrics_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.tools'`

- [ ] **Step 3: Implement `src/change_analyst/tools/prometheus.py`**

Also create empty `src/change_analyst/tools/__init__.py`.

```python
import math
from datetime import datetime

import httpx


class PromError(RuntimeError):
    """Prometheus query failed."""


class PromClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0, http: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=timeout_s)

    def query_range(self, query: str, start: datetime, end: datetime, step_s: int = 15) -> list[dict]:
        try:
            response = self.http.get(f"{self.base_url}/api/v1/query_range", params={
                "query": query, "start": start.timestamp(), "end": end.timestamp(), "step": f"{step_s}s",
            })
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PromError(f"prometheus request failed: {exc}") from exc
        if body.get("status") != "success":
            raise PromError(body.get("error") or f"HTTP {response.status_code}")
        return body["data"]["result"]


def summarize_values(values: list) -> dict | None:
    numbers = []
    for _, raw in values:
        value = float(raw)
        if math.isfinite(value):
            numbers.append(value)
    if not numbers:
        return None
    ordered = sorted(numbers)
    p95 = ordered[round(0.95 * (len(ordered) - 1))]
    return {
        "avg": round(sum(numbers) / len(numbers), 6),
        "p95": round(p95, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "last": round(numbers[-1], 6),
        "samples": len(numbers),
    }
```

- [ ] **Step 4: Implement `src/change_analyst/tools/metrics.py`**

```python
import json

from langchain_core.tools import BaseTool, tool

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.prometheus import summarize_values

MAX_SERIES = 10

# {sel} = namespace + pod regex for the Deployment, {ns} = namespace, {wl} = Deployment name.
METRIC_QUERIES: dict[str, str] = {
    "request_rate": 'sum(rate(http_requests_total{{{sel}}}[1m]))',
    "error_rate_5xx": ('(sum(rate(http_requests_total{{{sel},status=~"5.."}}[1m])) or vector(0))'
                       ' / clamp_min(sum(rate(http_requests_total{{{sel}}}[1m])), 1e-9)'),
    "latency_p95": ('histogram_quantile(0.95, sum by (le) '
                    '(rate(http_request_duration_seconds_bucket{{{sel}}}[1m])))'),
    "cpu": 'sum(rate(container_cpu_usage_seconds_total{{{sel},container!="",container!="POD"}}[1m]))',
    "memory": 'sum(container_memory_working_set_bytes{{{sel},container!="",container!="POD"}})',
    "restarts": 'sum(kube_pod_container_status_restarts_total{{{sel}}})',
    "ready_replicas": 'sum(kube_deployment_status_replicas_ready{{namespace="{ns}",deployment="{wl}"}})',
}


def pod_selector(namespace: str, workload: str) -> str:
    return f'namespace="{namespace}",pod=~"{workload}-[a-z0-9]+-[a-z0-9]+"'


def _first_series_values(result: list[dict]) -> list:
    return result[0]["values"] if result else []


def make_metrics_tools(prom, change: Change, windows: Windows) -> list[BaseTool]:
    selector = pod_selector(change.namespace, change.workload)

    @tool
    def compare_windows(metrics: list[str]) -> str:
        """Compare metrics of the changed Deployment between the before-change window and the
        after-change window. Returns avg/p95/min/max/last for each window and the percent change
        of the average. Valid metric names: request_rate (req/s), error_rate_5xx (fraction 0-1),
        latency_p95 (seconds), cpu (cores), memory (bytes), restarts (total container restarts),
        ready_replicas."""
        report = []
        for metric in metrics:
            if metric not in METRIC_QUERIES:
                report.append({"metric": metric, "error": f"unknown metric; choose from {sorted(METRIC_QUERIES)}"})
                continue
            query = METRIC_QUERIES[metric].format(sel=selector, ns=change.namespace, wl=change.workload)
            try:
                before = summarize_values(_first_series_values(
                    prom.query_range(query, windows.before_start, windows.before_end)))
                after = summarize_values(_first_series_values(
                    prom.query_range(query, windows.after_start, windows.after_end)))
            except Exception as exc:
                report.append({"metric": metric, "error": f"query failed: {exc}"})
                continue
            change_pct = None
            if before and after and before["avg"] != 0:
                change_pct = round((after["avg"] - before["avg"]) / abs(before["avg"]) * 100, 1)
            report.append({"metric": metric, "before": before, "after": after, "avg_change_pct": change_pct})
        return json.dumps(report)

    @tool
    def query_promql(query: str) -> str:
        """Run a custom PromQL range query over the whole analysis period (before-window start to
        after-window end). Use only when compare_windows is not enough. Returns min/avg/max/last
        per series (at most 10 series)."""
        try:
            result = prom.query_range(query, windows.before_start, windows.after_end)
        except Exception as exc:
            return f"ERROR: query failed: {exc}"
        series = [{"labels": s["metric"], "summary": summarize_values(s["values"])} for s in result[:MAX_SERIES]]
        return json.dumps({"total_series": len(result), "series": series})

    return [compare_windows, query_promql]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_prometheus.py tests/test_metrics_tools.py -v`
Expected: 9 passed.

- [ ] **Step 6: Verify queries against the demo cluster (manual)**

Run (demo cluster from Task 3 up):
```bash
uv run python -c "
from datetime import datetime, timedelta, timezone
from change_analyst.tools.prometheus import PromClient
from change_analyst.tools.metrics import METRIC_QUERIES, pod_selector
prom = PromClient('http://localhost:9090'); now = datetime.now(timezone.utc)
for name, q in METRIC_QUERIES.items():
    res = prom.query_range(q.format(sel=pod_selector('demo','podinfo'), ns='demo', wl='podinfo'), now - timedelta(minutes=10), now)
    print(name, 'series=', len(res), 'last=', res[0]['values'][-1] if res else None)
"
```
Expected: every metric prints `series= 1` with a numeric last value (latency_p95 may be very small). Fix any query that returns `series= 0` before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/change_analyst/tools tests/test_prometheus.py tests/test_metrics_tools.py
git commit -m "feat: Prometheus client and metrics comparison tools"
```

---

### Task 10: Kubernetes and log tools

**Files:**
- Create: `src/change_analyst/tools/k8s.py`, `src/change_analyst/tools/logs.py`
- Test: `tests/test_k8s_tools.py`, `tests/test_log_tools.py`

**Interfaces:**
- Consumes: `models.Change`, `planning.Windows`, `tests/factories.py` (`pod`, `event`, `deployment`, `FakeCore`, `FakeApps`, `T0`).
- Produces:
  - `k8s.list_workload_pods(core, apps, namespace: str, workload: str) -> list`
  - `k8s.summarize_pod(pod) -> dict` keys: `name, phase, ready, restarts, waiting_reasons, last_termination_reasons, images, created`
  - `k8s.summarize_events(events, workload: str, since: datetime) -> list[dict]` keys: `kind, reason, count, objects, example`
  - `k8s.make_k8s_tools(core, apps, change, windows) -> list[BaseTool]` → `get_pod_status()`, `get_events()`
  - `logs.normalize_line(line: str) -> str`, `logs.error_signatures(lines: list[str], top: int = 15) -> list[dict]` keys: `signature, count, example`
  - `logs.make_log_tools(core, apps, change, windows, now_fn=utc now) -> list[BaseTool]` → `get_error_signatures(previous: bool = False)`, `get_pod_logs(pod: str, previous: bool = False, tail: int = 50)`

- [ ] **Step 1: Write failing tests**

`tests/test_k8s_tools.py`:
```python
import json
from datetime import timedelta

from factories import T0, FakeApps, FakeCore, deployment, event, pod

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.k8s import make_k8s_tools, summarize_events, summarize_pod

W = Windows(T0 - timedelta(minutes=15), T0, T0 + timedelta(minutes=2), T0 + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T0, description="d")


def test_summarize_pod_reports_crashloop():
    summary = summarize_pod(pod(ready=False, restarts=4, waiting="CrashLoopBackOff", last_terminated="Error"))
    assert summary["ready"] is False
    assert summary["restarts"] == 4
    assert summary["waiting_reasons"] == ["CrashLoopBackOff"]
    assert summary["last_termination_reasons"] == ["Error"]
    assert summary["images"] == ["ghcr.io/stefanprodan/podinfo:6.15.0"]


def test_summarize_events_filters_and_groups():
    events = [
        event(name="podinfo-abc12-aaaaa", reason="BackOff", count=3),
        event(name="podinfo-abc12-bbbbb", reason="BackOff", count=2),
        event(name="podinfo-abc12-aaaaa", reason="Pulled", type_="Normal"),
        event(name="other-abc12-aaaaa", reason="BackOff"),
        event(name="podinfo-abc12-aaaaa", reason="Unhealthy", when=T0 - timedelta(hours=1)),
    ]
    groups = summarize_events(events, "podinfo", since=W.before_start)
    assert groups == [{
        "kind": "Pod", "reason": "BackOff", "count": 5,
        "objects": ["podinfo-abc12-aaaaa", "podinfo-abc12-bbbbb"],
        "example": "Back-off restarting failed container",
    }]


def test_tools_use_deployment_selector_and_report_errors():
    core = FakeCore(pods=[pod()], events=[event()])
    apps = FakeApps(deployments=[deployment()])
    tools = {t.name: t for t in make_k8s_tools(core, apps, CHANGE, W)}
    pods = json.loads(tools["get_pod_status"].invoke({}))
    assert pods[0]["name"] == "podinfo-abc12-xyz34"
    assert core.label_selectors == ["app=podinfo"]
    assert json.loads(tools["get_events"].invoke({}))[0]["reason"] == "BackOff"

    missing = {t.name: t for t in make_k8s_tools(core, FakeApps(), CHANGE, W)}
    assert missing["get_pod_status"].invoke({}).startswith("ERROR:")
```

`tests/test_log_tools.py`:
```python
import json
from datetime import timedelta

from factories import T0, FakeApps, FakeCore, deployment, pod

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.logs import error_signatures, make_log_tools, normalize_line

W = Windows(T0 - timedelta(minutes=15), T0, T0 + timedelta(minutes=2), T0 + timedelta(minutes=10))
CHANGE = Change(id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
                workload="podinfo", changed_at=T0, description="d")


def test_normalize_line_masks_variable_parts():
    line = "2026-09-16T10:00:01Z error dial tcp 10.0.3.7:6379 req=3f2a9c1e-1111-2222-3333-444455556666 after 250ms"
    assert normalize_line(line) == "<n>-<n>-<n>T<n>:<n>:<n>Z error dial tcp <ip> req=<uuid> after <n>ms"


def test_error_signatures_groups_and_counts():
    lines = [
        '{"level":"error","msg":"request 17 failed"}',
        '{"level":"error","msg":"request 99 failed"}',
        '{"level":"info","msg":"ok"}',
        "panic: runtime error",
    ]
    sigs = error_signatures(lines)
    assert sigs[0] == {"signature": '{"level":"error","msg":"request <n> failed"}', "count": 2,
                       "example": '{"level":"error","msg":"request 17 failed"}'}
    assert sigs[1]["signature"] == "panic: runtime error"
    assert len(sigs) == 2


def test_get_error_signatures_reads_all_pods_and_reports_gaps():
    core = FakeCore(
        pods=[pod(name="podinfo-abc12-aaaaa"), pod(name="podinfo-abc12-bbbbb")],
        logs={("podinfo-abc12-aaaaa", False): "Error: boom 1\nfine\nError: boom 2\n"},
    )
    tools = {t.name: t for t in make_log_tools(core, FakeApps(deployments=[deployment()]), CHANGE, W,
                                               now_fn=lambda: T0 + timedelta(minutes=10))}
    out = json.loads(tools["get_error_signatures"].invoke({}))
    assert out["pods_read"] == 1
    assert out["signatures"][0] == {"signature": "Error: boom <n>", "count": 2, "example": "Error: boom 1"}
    assert out["unavailable"][0].startswith("podinfo-abc12-bbbbb: logs unavailable")


def test_get_pod_logs_previous_and_error():
    core = FakeCore(logs={("podinfo-abc12-aaaaa", True): "Error: invalid argument\n"})
    tools = {t.name: t for t in make_log_tools(core, FakeApps(), CHANGE, W)}
    assert "invalid argument" in tools["get_pod_logs"].invoke({"pod": "podinfo-abc12-aaaaa", "previous": True})
    assert tools["get_pod_logs"].invoke({"pod": "podinfo-abc12-aaaaa"}).startswith("ERROR:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_k8s_tools.py tests/test_log_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.tools.k8s'`

- [ ] **Step 3: Implement `src/change_analyst/tools/k8s.py`**

```python
import json
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from change_analyst.models import Change
from change_analyst.planning import Windows


def short_error(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if status is not None:
        return f"HTTP {status} {getattr(exc, 'reason', '')}".strip()
    return str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__


def list_workload_pods(core, apps, namespace: str, workload: str) -> list:
    deploy = apps.read_namespaced_deployment(workload, namespace)
    labels = deploy.spec.selector.match_labels or {}
    selector = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return core.list_namespaced_pod(namespace, label_selector=selector).items


def summarize_pod(pod) -> dict:
    statuses = pod.status.container_statuses or []
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": bool(statuses) and all(s.ready for s in statuses),
        "restarts": sum(s.restart_count or 0 for s in statuses),
        "waiting_reasons": [s.state.waiting.reason for s in statuses if s.state and s.state.waiting],
        "last_termination_reasons": [s.last_state.terminated.reason for s in statuses
                                     if s.last_state and s.last_state.terminated],
        "images": [c.image for c in pod.spec.containers],
        "created": pod.metadata.creation_timestamp.isoformat(),
    }


def summarize_events(events, workload: str, since: datetime) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for ev in events:
        if ev.type != "Warning":
            continue
        obj = ev.involved_object
        if not (obj.name == workload or obj.name.startswith(workload + "-")):
            continue
        when = ev.last_timestamp or ev.event_time or ev.metadata.creation_timestamp
        if when is not None and when < since:
            continue
        group = groups.setdefault((obj.kind, ev.reason), {
            "kind": obj.kind, "reason": ev.reason, "count": 0, "objects": set(), "example": ev.message,
        })
        group["count"] += ev.count or 1
        group["objects"].add(obj.name)
    result = [{**g, "objects": sorted(g["objects"])[:5]} for g in groups.values()]
    return sorted(result, key=lambda g: (-g["count"], g["reason"]))


def make_k8s_tools(core, apps, change: Change, windows: Windows) -> list[BaseTool]:
    @tool
    def get_pod_status() -> str:
        """List the current pods of the changed Deployment with phase, readiness, restart count,
        waiting reasons (e.g. CrashLoopBackOff), last termination reasons (e.g. OOMKilled, Error)
        and images."""
        try:
            pods = list_workload_pods(core, apps, change.namespace, change.workload)
        except Exception as exc:
            return f"ERROR: could not list pods: {short_error(exc)}"
        return json.dumps([summarize_pod(p) for p in pods])

    @tool
    def get_events() -> str:
        """Warning events since the start of the before-change window for the changed Deployment,
        its ReplicaSets and its pods, grouped by object kind and reason with counts."""
        try:
            events = core.list_namespaced_event(change.namespace).items
        except Exception as exc:
            return f"ERROR: could not list events: {short_error(exc)}"
        return json.dumps(summarize_events(events, change.workload, windows.before_start))

    return [get_pod_status, get_events]
```

- [ ] **Step 4: Implement `src/change_analyst/tools/logs.py`**

```python
import json
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone

from langchain_core.tools import BaseTool, tool

from change_analyst.models import Change
from change_analyst.planning import Windows
from change_analyst.tools.k8s import list_workload_pods, short_error

MAX_LOG_CHARS = 8000
MAX_SIGNATURE_CHARS = 300

ERROR_LINE = re.compile(
    r'(?i)\b(error|exception|fatal|panic|fail(ed|ure)?|traceback)\b|"level":"(error|fatal|panic)"'
)
NORMALIZERS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<hex>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<n>"),
]


def normalize_line(line: str) -> str:
    text = line.strip()
    for pattern, replacement in NORMALIZERS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text)[:MAX_SIGNATURE_CHARS]


def error_signatures(lines: list[str], top: int = 15) -> list[dict]:
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for line in lines:
        if not ERROR_LINE.search(line):
            continue
        signature = normalize_line(line)
        counts[signature] += 1
        examples.setdefault(signature, line.strip()[:MAX_SIGNATURE_CHARS])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"signature": s, "count": c, "example": examples[s]} for s, c in ranked]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_log_tools(
    core, apps, change: Change, windows: Windows, now_fn: Callable[[], datetime] = _utcnow
) -> list[BaseTool]:
    @tool
    def get_error_signatures(previous: bool = False) -> str:
        """Group error-like log lines from the changed Deployment's current pods into normalized
        signatures with counts and one example each (top 15), covering the analysis period.
        Set previous=true to read the previous (crashed) container instance of each pod instead."""
        try:
            pods = list_workload_pods(core, apps, change.namespace, change.workload)
        except Exception as exc:
            return f"ERROR: could not list pods: {short_error(exc)}"
        since = max(60, int((now_fn() - windows.before_start).total_seconds()))
        lines: list[str] = []
        unavailable: list[str] = []
        for p in pods:
            name = p.metadata.name
            kwargs = {"previous": True} if previous else {"since_seconds": since}
            try:
                text = core.read_namespaced_pod_log(name, change.namespace, tail_lines=2000, **kwargs)
            except Exception as exc:
                unavailable.append(f"{name}: logs unavailable ({short_error(exc)})")
                continue
            lines.extend(text.splitlines())
        return json.dumps({
            "pods_read": len(pods) - len(unavailable),
            "signatures": error_signatures(lines),
            "unavailable": unavailable,
        })

    @tool
    def get_pod_logs(pod: str, previous: bool = False, tail: int = 50) -> str:
        """Return the last `tail` log lines (max 200) of one pod of the changed Deployment.
        Set previous=true for the previous (crashed) container instance."""
        try:
            text = core.read_namespaced_pod_log(pod, change.namespace, previous=previous,
                                                tail_lines=min(max(tail, 1), 200))
        except Exception as exc:
            return f"ERROR: could not read logs for {pod}: {short_error(exc)}"
        return text[-MAX_LOG_CHARS:] or "(empty log)"

    return [get_error_signatures, get_pod_logs]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_k8s_tools.py tests/test_log_tools.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/change_analyst/tools/k8s.py src/change_analyst/tools/logs.py tests/test_k8s_tools.py tests/test_log_tools.py
git commit -m "feat: kubernetes pod/event tools and log error signatures"
```

---

### Task 11: Investigator agents and judge

**Files:**
- Create: `src/change_analyst/agents/metrics_agent.py`, `src/change_analyst/agents/logs_agent.py`, `src/change_analyst/agents/judge.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `agents.runner.run_tool_agent`, `agents.runner.build_task`, `tools.metrics.make_metrics_tools`, `tools.k8s.make_k8s_tools`, `tools.logs.make_log_tools`, `models.*`, `llm.client.LLMClient`, `llm.chat_model.CLIChatModel`.
- Produces:
  - `metrics_agent.investigate_metrics(model: CLIChatModel, prom, change: Change, windows: Windows, max_steps: int) -> Findings` (never raises)
  - `logs_agent.investigate_logs(model: CLIChatModel, core, apps, change: Change, windows: Windows, max_steps: int) -> Findings` (never raises)
  - `judge.judge_change(client: LLMClient, change: Change, metrics: Findings, logs: Findings) -> Verdict` (never raises)
  - Prompt constants `METRICS_SYSTEM`, `LOGS_SYSTEM`, `JUDGE_SYSTEM`. Only `JUDGE_SYSTEM` contains the word "judge" (Task 14's test relies on this).

- [ ] **Step 1: Write failing tests**

`tests/test_agents.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.agents.judge'`

- [ ] **Step 3: Implement `src/change_analyst/agents/metrics_agent.py`**

```python
import logging

from change_analyst.agents.runner import build_task, run_tool_agent
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.models import Change, Findings, findings_schema
from change_analyst.planning import Windows
from change_analyst.tools.metrics import make_metrics_tools

log = logging.getLogger(__name__)

METRICS_SYSTEM = """You are the metrics investigator in a Kubernetes change-impact analysis system.
You investigate ONE change to ONE Deployment by comparing Prometheus metrics in the window before the change with the window after it.

How to work:
1. Call compare_windows once with all relevant metrics, normally:
   ["request_rate", "error_rate_5xx", "latency_p95", "restarts", "ready_replicas", "cpu", "memory"].
2. Only if something is unclear, use query_promql for a targeted follow-up.
3. Then finish.

Your final answer:
- observations: factual statements with numbers, e.g. "error_rate_5xx avg 0.0 before -> 0.31 after".
- anomalies: only things that got clearly worse after the change (ignore changes within roughly ±20% for rates, latency, cpu and memory).
- data_gaps: metrics that returned no data or errors.
- status: "ok" if the key metrics were available, "partial" if some were missing, "failed" if no data could be retrieved.
Never invent numbers."""


def investigate_metrics(model: CLIChatModel, prom, change: Change, windows: Windows, max_steps: int) -> Findings:
    tools = make_metrics_tools(prom, change, windows)
    try:
        data = run_tool_agent(model, tools, METRICS_SYSTEM, build_task(change, windows),
                              max_steps, findings_schema())
        return Findings.from_agent_output("metrics", data)
    except Exception as exc:  # the run must go on even if this agent fails
        log.warning("metrics agent failed for %s: %s", change.id, exc)
        return Findings.failed("metrics", f"metrics agent failed: {exc}")
```

- [ ] **Step 4: Implement `src/change_analyst/agents/logs_agent.py`**

```python
import logging

from change_analyst.agents.runner import build_task, run_tool_agent
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.models import Change, Findings, findings_schema
from change_analyst.planning import Windows
from change_analyst.tools.k8s import make_k8s_tools
from change_analyst.tools.logs import make_log_tools

log = logging.getLogger(__name__)

LOGS_SYSTEM = """You are the logs and events investigator in a Kubernetes change-impact analysis system.
You investigate ONE change to ONE Deployment using pod status, Kubernetes warning events and container logs.

How to work:
1. Call get_pod_status to see readiness, restarts, waiting reasons (CrashLoopBackOff, ImagePullBackOff) and termination reasons (OOMKilled, Error).
2. Call get_events to see warning events (BackOff, Unhealthy, FailedScheduling, ...).
3. Call get_error_signatures for current logs. If any pod restarted or is crash-looping, also call it with previous=true.
4. Use get_pod_logs only to read one specific pod when signatures are not enough.
5. Then finish.

Your final answer:
- observations: factual statements with numbers, e.g. "2 of 3 pods in CrashLoopBackOff, 14 restarts", "signature 'Error: invalid argument <n>' seen 6 times".
- anomalies: problems that plausibly appeared after the change.
- data_gaps: logs or objects that were unavailable (e.g. pods from before the change are gone).
- status: "ok", "partial" (some data unavailable) or "failed" (no data could be retrieved).
Never invent data."""


def investigate_logs(model: CLIChatModel, core, apps, change: Change, windows: Windows, max_steps: int) -> Findings:
    tools = make_k8s_tools(core, apps, change, windows) + make_log_tools(core, apps, change, windows)
    try:
        data = run_tool_agent(model, tools, LOGS_SYSTEM, build_task(change, windows),
                              max_steps, findings_schema())
        return Findings.from_agent_output("logs", data)
    except Exception as exc:  # the run must go on even if this agent fails
        log.warning("logs agent failed for %s: %s", change.id, exc)
        return Findings.failed("logs", f"logs agent failed: {exc}")
```

- [ ] **Step 5: Implement `src/change_analyst/agents/judge.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_agents.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add src/change_analyst/agents tests/test_agents.py
git commit -m "feat: metrics and logs investigator agents and judge"
```

---

### Task 12: LangGraph run graph

**Files:**
- Create: `src/change_analyst/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `models.Change`, `models.Findings`, `models.Verdict`, `models.ChangeResult`.
- Produces:
  - `graph.Deps(investigate_metrics: Callable[[Change], Findings], investigate_logs: Callable[[Change], Findings], judge: Callable[[Change, Findings, Findings], Verdict], deadline: float, clock: Callable[[], float] = time.monotonic)` dataclass
  - `graph.build_graph(deps: Deps)` → compiled graph; `invoke({"ready": list[Change], "results": [], "deferred": []}, {"max_concurrency": n})` returns dict with `results: list[ChangeResult]` and `deferred: list[Change]` (changes skipped because the deadline passed).

- [ ] **Step 1: Write failing tests**

`tests/test_graph.py`:
```python
import threading
import time
from datetime import datetime, timezone

from change_analyst.graph import Deps, build_graph
from change_analyst.models import Change, Findings, Verdict

T = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def mk(i: int) -> Change:
    return Change(id=f"c{i}", kind="rollout", namespace="demo", workload="podinfo", changed_at=T, description="d")


def fake_deps(deadline=float("inf"), clock=time.monotonic, active=None):
    lock = threading.Lock()
    active = active if active is not None else {"now": 0, "max": 0}

    def agent(name):
        def run(change):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.05)
            with lock:
                active["now"] -= 1
            return Findings(agent=name, status="ok", observations=[f"{name} {change.id}"])
        return run

    def judge(change, metrics, logs):
        assert metrics.agent == "metrics" and logs.agent == "logs"
        return Verdict(change_id=change.id, verdict="no_impact", confidence="high", summary="fine")

    return Deps(investigate_metrics=agent("metrics"), investigate_logs=agent("logs"),
                judge=judge, deadline=deadline, clock=clock)


def test_every_ready_change_gets_a_result():
    out = build_graph(fake_deps()).invoke({"ready": [mk(1), mk(2), mk(3)], "results": [], "deferred": []},
                                          {"max_concurrency": 2})
    assert sorted(r.change.id for r in out["results"]) == ["c1", "c2", "c3"]
    assert all(r.metrics.observations == [f"metrics {r.change.id}"] for r in out["results"])
    assert out["deferred"] == []


def test_metrics_and_logs_agents_run_in_parallel():
    active = {"now": 0, "max": 0}
    build_graph(fake_deps(active=active)).invoke({"ready": [mk(1)], "results": [], "deferred": []})
    assert active["max"] == 2


def test_no_ready_changes_finishes_cleanly():
    out = build_graph(fake_deps()).invoke({"ready": [], "results": [], "deferred": []})
    assert out["results"] == [] and out["deferred"] == []


def test_changes_after_deadline_are_deferred():
    out = build_graph(fake_deps(deadline=10.0, clock=lambda: 11.0)).invoke(
        {"ready": [mk(1), mk(2)], "results": [], "deferred": []})
    assert out["results"] == []
    assert sorted(c.id for c in out["deferred"]) == ["c1", "c2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.graph'`

- [ ] **Step 3: Implement `src/change_analyst/graph.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/change_analyst/graph.py tests/test_graph.py
git commit -m "feat: LangGraph run graph with parallel investigations"
```

---

### Task 13: Report rendering

**Files:**
- Create: `src/change_analyst/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `models.ChangeResult`, `models.PendingChange`.
- Produces:
  - `report.RunReport(started_at: datetime, duration_s: float, llm_calls: int, results: list[ChangeResult], pending: list[PendingChange], errors: list[str])`
  - `report.render_markdown(report: RunReport) -> str`
  - `report.write_report(report: RunReport, reports_dir: Path) -> Path` — writes `<stamp>.json` and `<stamp>.md` (stamp `%Y%m%dT%H%M%SZ`), points `latest.md` symlink at the markdown file, returns the markdown path.

- [ ] **Step 1: Write failing tests**

`tests/test_report.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.report'`

- [ ] **Step 3: Implement `src/change_analyst/report.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/change_analyst/report.py tests/test_report.py
git commit -m "feat: markdown and json run reports"
```

---

### Task 14: Pipeline, CLI, README and cron

**Files:**
- Create: `src/change_analyst/pipeline.py`, `src/change_analyst/cli.py`
- Modify: `README.md` (replace placeholder), `Makefile` (add `run` and `test` targets)
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above: `Settings`, `load_state`, `save_state`, `State`, `remember_analyzed`, `detect_changes`, `plan_run`, `compute_windows`, `CLIChatModel`, `ClaudeCLIClient`, `investigate_metrics`, `investigate_logs`, `judge_change`, `Deps`, `build_graph`, `RunReport`, `write_report`, `PromClient`, `run_lock`, `LockHeld`.
- Produces:
  - `pipeline.Runtime(core, apps, prom, llm)` dataclass
  - `pipeline.execute_run(settings: Settings, runtime: Runtime, now_fn: Callable[[], datetime] = utc now) -> Path` (report markdown path)
  - `cli.app` (Typer) with command `run`; `cli.build_runtime(settings) -> Runtime`

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline.py`:
```python
import json
import threading
from datetime import timedelta

from factories import T0, FakeApps, FakeCore, container, deployment, replica_set, template

from change_analyst.config import Settings
from change_analyst.llm.client import LLMResult
from change_analyst.pipeline import Runtime, execute_run
from change_analyst.state_store import load_state

NOW = T0 + timedelta(minutes=30)


class ScriptedLLM:
    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system, prompt, json_schema=None):
        with self._lock:
            self.calls += 1
        if "judge" in system.lower():
            return LLMResult(text="", structured={"verdict": "regression", "confidence": "high",
                                                  "summary": "Errors rose.", "evidence": ["5xx up"]})
        return LLMResult(text="", structured={"action": "final", "final": {
            "status": "ok", "observations": ["looked"], "anomalies": [], "data_gaps": []}})


class ExplodingApps(FakeApps):
    def list_namespaced_replica_set(self, namespace):
        raise RuntimeError("cluster unreachable")


def settings(tmp_path):
    return Settings(_env_file=None, reports_dir=tmp_path / "reports", state_path=tmp_path / "state.json",
                    lock_path=tmp_path / "lock", namespaces=["demo"], first_run_lookback_min=60)


def cluster():
    apps = FakeApps(
        replica_sets=[
            replica_set(revision=1, created=T0 - timedelta(days=1)),
            replica_set(revision=2, created=NOW - timedelta(minutes=10),
                        tmpl=template([container(args=["--random-error=true"])])),
        ],
        deployments=[deployment()],
    )
    return FakeCore(), apps


def test_run_analyzes_change_writes_report_and_state(tmp_path):
    s = settings(tmp_path)
    core, apps = cluster()
    llm = ScriptedLLM()
    path = execute_run(s, Runtime(core=core, apps=apps, prom=object(), llm=llm), now_fn=lambda: NOW)

    assert path.exists()
    report = json.loads(path.with_suffix(".json").read_text())
    assert [r["verdict"]["verdict"] for r in report["results"]] == ["regression"]
    assert report["llm_calls"] == 3
    state = load_state(s.state_path)
    assert state.last_run_at == NOW
    assert state.analyzed_change_ids == ["rollout:demo/podinfo:rev2"]
    assert state.pending == []

    later = NOW + timedelta(minutes=10)
    path2 = execute_run(s, Runtime(core=core, apps=apps, prom=object(), llm=ScriptedLLM()), now_fn=lambda: later)
    assert json.loads(path2.with_suffix(".json").read_text())["results"] == []


def test_detection_failure_keeps_last_run_and_reports_error(tmp_path):
    s = settings(tmp_path)
    path = execute_run(s, Runtime(core=FakeCore(), apps=ExplodingApps(), prom=object(), llm=ScriptedLLM()),
                       now_fn=lambda: NOW)
    report = json.loads(path.with_suffix(".json").read_text())
    assert "cluster unreachable" in report["errors"][0]
    assert load_state(s.state_path).last_run_at is None
```

`tests/test_cli.py`:
```python
from typer.testing import CliRunner

from change_analyst import cli
from change_analyst.lock import run_lock


def test_run_exits_quietly_when_lock_is_held(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock"
    monkeypatch.setenv("CA_LOCK_PATH", str(lock_path))

    def must_not_build(settings):
        raise AssertionError("runtime should not be built while another run holds the lock")

    monkeypatch.setattr(cli, "build_runtime", must_not_build)
    with run_lock(lock_path):
        result = CliRunner().invoke(cli.app, ["run"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'change_analyst.pipeline'`

- [ ] **Step 3: Implement `src/change_analyst/pipeline.py`**

```python
"""One analysis run: detect -> plan -> investigate (graph) -> report -> save state."""
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from change_analyst.agents.judge import judge_change
from change_analyst.agents.logs_agent import investigate_logs
from change_analyst.agents.metrics_agent import investigate_metrics
from change_analyst.changes.detect import detect_changes
from change_analyst.config import Settings
from change_analyst.graph import Deps, build_graph
from change_analyst.llm.chat_model import CLIChatModel
from change_analyst.llm.client import LLMClient
from change_analyst.models import Change, PendingChange
from change_analyst.planning import compute_windows, plan_run
from change_analyst.report import RunReport, write_report
from change_analyst.state_store import State, load_state, remember_analyzed, save_state

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    core: Any
    apps: Any
    prom: Any
    llm: LLMClient


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def execute_run(settings: Settings, runtime: Runtime, now_fn: Callable[[], datetime] = _utcnow) -> Path:
    started = now_fn()
    t0 = time.monotonic()
    state = load_state(settings.state_path)
    errors: list[str] = []

    since = state.last_run_at or started - timedelta(minutes=settings.first_run_lookback_min)
    detection_ok = True
    try:
        detected, hashes = detect_changes(runtime.core, runtime.apps, settings.namespaces,
                                          since, state.configmap_hashes)
    except Exception as exc:
        log.exception("change detection failed")
        errors.append(f"change detection failed: {exc}")
        detected, hashes, detection_ok = [], state.configmap_hashes, False

    ready, waiting = plan_run(detected, state, started, settings)
    first_seen = {p.change.id: p.first_seen for p in ready}
    log.info("detected=%d ready=%d waiting=%d", len(detected), len(ready), len(waiting))

    model = CLIChatModel(client=runtime.llm)

    def windows_for(change: Change):
        return compute_windows(change.changed_at, now_fn(), settings)

    deps = Deps(
        investigate_metrics=lambda c: investigate_metrics(model, runtime.prom, c, windows_for(c),
                                                          settings.max_agent_steps),
        investigate_logs=lambda c: investigate_logs(model, runtime.core, runtime.apps, c, windows_for(c),
                                                    settings.max_agent_steps),
        judge=lambda c, m, l: judge_change(runtime.llm, c, m, l),
        deadline=t0 + settings.run_timeout_s,
    )
    out = build_graph(deps).invoke(
        {"ready": [p.change for p in ready], "results": [], "deferred": []},
        {"max_concurrency": settings.max_concurrency},
    )
    results = out["results"]
    pending = waiting + [PendingChange(change=c, first_seen=first_seen[c.id]) for c in out["deferred"]]

    report = RunReport(
        started_at=started,
        duration_s=round(time.monotonic() - t0, 1),
        llm_calls=getattr(runtime.llm, "calls", 0),
        results=results,
        pending=pending,
        errors=errors,
    )
    path = write_report(report, settings.reports_dir)
    save_state(settings.state_path, State(
        last_run_at=started if detection_ok else state.last_run_at,
        analyzed_change_ids=remember_analyzed(state.analyzed_change_ids, [r.change.id for r in results]),
        configmap_hashes=hashes,
        pending=pending,
    ))
    return path
```

- [ ] **Step 4: Implement `src/change_analyst/cli.py`**

```python
import logging

import typer

from change_analyst.config import Settings
from change_analyst.llm.client import ClaudeCLIClient
from change_analyst.lock import LockHeld, run_lock
from change_analyst.pipeline import Runtime, execute_run
from change_analyst.tools.prometheus import PromClient

app = typer.Typer(help="Judge the impact of recent Kubernetes changes.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """k8s-change-analyst: LLM multi-agent change-impact analysis."""


def build_runtime(settings: Settings) -> Runtime:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config(context=settings.kube_context)
    return Runtime(
        core=client.CoreV1Api(),
        apps=client.AppsV1Api(),
        prom=PromClient(settings.prometheus_url),
        llm=ClaudeCLIClient(model=settings.llm_model or None, timeout_s=settings.llm_timeout_s),
    )


@app.command()
def run() -> None:
    """Detect recent changes, investigate them with agents, and write a report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    try:
        with run_lock(settings.lock_path):
            path = execute_run(settings, build_runtime(settings))
    except LockHeld:
        typer.echo("Another run is still in progress; exiting.", err=True)
        return
    typer.echo(f"Report written to {path}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass (73 so far).

- [ ] **Step 7: Add Makefile targets**

Append to `Makefile` (recipe lines start with a TAB) and extend the `.PHONY` line to `.PHONY: demo-up demo-down scenario-% run test e2e`:
```make
run:
	uv run k8s-change-analyst run

test:
	uv run pytest -q
```

- [ ] **Step 8: Write `README.md`**

````markdown
# k8s-change-analyst

An LLM-powered multi-agent system that runs periodically and answers: **did a recent change to the cluster hurt anything?**

Each run it:
1. detects Deployment rollouts and ConfigMap edits since the last run (plain code),
2. investigates each change with two agents running in parallel: a **metrics agent** (Prometheus before/after comparison) and a **logs agent** (pod status, warning events, log error signatures),
3. asks a **judge** for a verdict (`regression`, `improvement`, `no_impact`, `inconclusive`) with evidence,
4. writes `reports/<timestamp>.md` / `.json` (and `reports/latest.md`).

Agents are LangGraph tool-calling loops. The LLM backend is the Claude CLI (`claude -p`) called through a subprocess, behind a swappable `LLMClient` interface.

## Requirements

- macOS/Linux with Docker, `uv`, `kind`, `helm`, `kubectl` (`brew install uv kind helm kubectl`)
- The Claude CLI, logged in (`claude` works in your terminal)

## Quick start

```bash
uv sync
make demo-up            # kind cluster + Prometheus + podinfo + load generator (~5 min)
make scenario-errors    # roll out a version that returns random 5xx errors
sleep 360               # changes are analyzed once they are at least 5 minutes old
make run
cat reports/latest.md
make scenario-reset
```

Scenarios: `harmless` (no_impact), `errors`, `crashloop`, `config` (regression), `reset` (restore).

## Configuration

Environment variables (or a `.env` file), prefix `CA_`:

| Variable | Default | Meaning |
|---|---|---|
| `CA_PROMETHEUS_URL` | `http://localhost:9090` | Prometheus base URL |
| `CA_NAMESPACES` | `["demo"]` | JSON list of namespaces to watch |
| `CA_KUBE_CONTEXT` | current context | kubeconfig context |
| `CA_LLM_MODEL` | `sonnet` | model passed to `claude --model` (empty = CLI default) |
| `CA_LLM_TIMEOUT_S` | `180` | per LLM call timeout |
| `CA_BEFORE_WINDOW_MIN` / `CA_SETTLE_DELAY_MIN` / `CA_MIN_AGE_MIN` / `CA_MAX_AFTER_WINDOW_MIN` | `15` / `2` / `5` / `30` | analysis windows |
| `CA_MAX_CHANGES_PER_RUN` / `CA_MAX_AGENT_STEPS` / `CA_MAX_CONCURRENCY` / `CA_RUN_TIMEOUT_S` | `5` / `6` / `2` / `1200` | limits |
| `CA_REPORTS_DIR` / `CA_STATE_PATH` / `CA_LOCK_PATH` | `reports` / `state.json` / `.change-analyst.lock` | files |

## Running periodically

cron does not load your shell profile, so give it a `PATH` that contains `uv`, `kubectl` and `claude`
(find them with `dirname "$(command -v claude)"` etc.). Example `crontab -e` entry, every 10 minutes:

```cron
*/10 * * * * cd /Users/you/Projects/k8s-change-analyst && PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin uv run k8s-change-analyst run >> logs/cron.log 2>&1
```

Create the log directory once: `mkdir -p logs`. Overlapping runs are prevented by a lock file.

On macOS you can use launchd instead: save as `~/Library/LaunchAgents/dev.change-analyst.plist`, adjust paths, then `launchctl load ~/Library/LaunchAgents/dev.change-analyst.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.change-analyst</string>
  <key>WorkingDirectory</key><string>/Users/you/Projects/k8s-change-analyst</string>
  <key>ProgramArguments</key>
  <array><string>/opt/homebrew/bin/uv</string><string>run</string><string>k8s-change-analyst</string><string>run</string></array>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/Users/you/.local/bin:/usr/bin:/bin</string></dict>
  <key>StartInterval</key><integer>600</integer>
  <key>StandardOutPath</key><string>/Users/you/Projects/k8s-change-analyst/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>/Users/you/Projects/k8s-change-analyst/logs/launchd.log</string>
</dict>
</plist>
```

## Development

```bash
make test     # unit tests (no cluster, no LLM)
make e2e      # full scenario run against the demo cluster with the real CLI (~45 min)
```

Design: `docs/superpowers/specs/2026-09-16-k8s-change-analyst-design.md`
````

- [ ] **Step 9: Real run against the demo cluster (manual)**

Run (demo cluster up):
```bash
make scenario-errors && sleep 360 && make run && cat reports/latest.md
make scenario-reset
```
Expected: the report lists `rollout:demo/podinfo:revN` with verdict `regression` and evidence mentioning the 5xx error rate and/or latency. If an agent shows `status: failed`, read the `data_gaps` reason and the run log before moving on.

- [ ] **Step 10: Commit**

```bash
git add src/change_analyst/pipeline.py src/change_analyst/cli.py tests/test_pipeline.py tests/test_cli.py README.md Makefile
git commit -m "feat: run pipeline, CLI entrypoint, README with cron setup"
```

---

### Task 15: End-to-end scenario check

**Files:**
- Create: `scripts/check_verdict.py`, `scripts/e2e.sh`
- Modify: `Makefile` (add `e2e` target)
- Test: `tests/test_check_verdict.py`

**Interfaces:**
- Consumes: report JSON format from Task 13 (`results[].change.kind`, `results[].change.changed_at`, `results[].change.id`, `results[].verdict.verdict`, `results[].verdict.confidence`); CLI from Task 14; scenario scripts from Task 3.
- Produces: `scripts/check_verdict.py <kind> <expected> [reports_dir]` exit code 0 on match, 1 otherwise; `make e2e`.

- [ ] **Step 1: Write failing test**

`tests/test_check_verdict.py`:
```python
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_verdict.py"


def load():
    spec = importlib.util.spec_from_file_location("check_verdict", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(kind, changed_at, verdict, change_id):
    return {"change": {"id": change_id, "kind": kind, "changed_at": changed_at},
            "verdict": {"verdict": verdict, "confidence": "high"}}


def test_checks_latest_change_of_kind_in_latest_report(tmp_path):
    (tmp_path / "20260916T100000Z.json").write_text(json.dumps({"results": [
        result("rollout", "2026-09-16T09:50:00Z", "regression", "old")]}))
    (tmp_path / "20260916T110000Z.json").write_text(json.dumps({"results": [
        result("rollout", "2026-09-16T10:40:00Z", "improvement", "reset"),
        result("rollout", "2026-09-16T10:50:00Z", "regression", "new"),
        result("configmap", "2026-09-16T10:50:00Z", "no_impact", "cm"),
    ]}))
    check = load()
    assert check.main("rollout", "regression", str(tmp_path)) == 0
    assert check.main("configmap", "regression", str(tmp_path)) == 1


def test_fails_without_reports(tmp_path):
    assert load().main("rollout", "regression", str(tmp_path)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_check_verdict.py -v`
Expected: FAIL with `FileNotFoundError` for `scripts/check_verdict.py`.

- [ ] **Step 3: Implement `scripts/check_verdict.py`**

```python
"""Assert the verdict of the most recent change of a given kind in the newest report."""
import json
import sys
from pathlib import Path


def main(kind: str, expected: str, reports_dir: str = "reports") -> int:
    files = sorted(Path(reports_dir).glob("*.json"))
    if not files:
        print(f"FAIL: no reports in {reports_dir}")
        return 1
    report = json.loads(files[-1].read_text())
    candidates = [r for r in report["results"] if r["change"]["kind"] == kind]
    if not candidates:
        print(f"FAIL: no {kind} change analyzed in {files[-1].name}")
        return 1
    latest = max(candidates, key=lambda r: r["change"]["changed_at"])
    got = latest["verdict"]["verdict"]
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}: {latest['change']['id']} expected={expected} got={got} "
          f"(confidence {latest['verdict']['confidence']})")
    return 0 if got == expected else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_check_verdict.py -v`
Expected: 2 passed.

- [ ] **Step 5: Implement `scripts/e2e.sh`**

```bash
#!/usr/bin/env bash
# Runs each demo scenario, analyzes it with the real Claude CLI and checks the verdict.
# Requires: `make demo-up` done, Prometheus on localhost:9090, `claude` logged in. Takes ~45 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."

export CA_BEFORE_WINDOW_MIN=5 CA_SETTLE_DELAY_MIN=1 CA_MIN_AGE_MIN=3
export CA_STATE_PATH=.e2e/state.json CA_REPORTS_DIR=.e2e/reports CA_LOCK_PATH=.e2e/lock
WAIT_AFTER_CHANGE=${WAIT_AFTER_CHANGE:-240}
WAIT_QUIET=${WAIT_QUIET:-120}

rm -rf .e2e && mkdir -p .e2e
failures=0

analyze() { uv run k8s-change-analyst run; }

check() {  # <scenario> <change kind> <expected verdict>
  echo "=== scenario $1 (expect $2 -> $3)"
  "deploy/demo/scenarios/$1.sh"
  sleep "$WAIT_AFTER_CHANGE"
  analyze
  uv run python scripts/check_verdict.py "$2" "$3" .e2e/reports || failures=$((failures + 1))
  deploy/demo/scenarios/reset.sh
  sleep "$WAIT_AFTER_CHANGE"
  analyze > /dev/null   # consume the reset changes so they don't pollute the next check
  sleep "$WAIT_QUIET"
}

analyze > /dev/null     # baseline: record ConfigMap hashes and clear old rollouts
check harmless rollout no_impact
check errors rollout regression
check crashloop rollout regression
check config configmap regression

echo "=== e2e finished with $failures failing scenario(s); reports in .e2e/reports"
exit $(( failures > 0 ? 1 : 0 ))
```

```bash
chmod +x scripts/e2e.sh
```

Add to `Makefile` (TAB-indented recipe):
```make
e2e:
	scripts/e2e.sh
```

- [ ] **Step 6: Run the end-to-end check (manual, ~45 min, real CLI)**

Run: `make e2e 2>&1 | tee .e2e-run.log`
Expected: four `PASS:` lines and `e2e finished with 0 failing scenario(s)`.
LLM verdicts are not deterministic: if a single scenario fails, open its report in `.e2e/reports/`, check whether the evidence supports the expected verdict, and tune `METRICS_SYSTEM` / `LOGS_SYSTEM` / `JUDGE_SYSTEM` wording (not the thresholds in code) before re-running. Record any remaining failure honestly in the commit message.

- [ ] **Step 7: Commit**

```bash
git add scripts tests/test_check_verdict.py Makefile
git commit -m "feat: end-to-end scenario verification script"
```
