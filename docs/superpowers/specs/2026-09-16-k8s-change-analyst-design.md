# k8s-change-analyst — Design

- **Date:** 2026-09-16
- **Status:** Draft for review
- **Scope:** About one week of work for one developer

## 1. Problem and purpose

Teams can already look at metrics (Prometheus/Grafana), logs, and a history of changes. What they can't easily get is an answer to: **"Did something we changed recently break something?"** Answering it means lining up a change (a rollout, a config edit) with what happened afterwards in metrics, logs, and events, which is usually done by hand.

`k8s-change-analyst` is a multi-agent system that runs periodically. On each run it:

1. finds the changes made in the cluster since the last run,
2. for each change, uses LLM-powered agents to compare metrics, logs and events from before and after the change,
3. gives each change a verdict (`regression`, `improvement`, `no_impact`, `inconclusive`) with a confidence level and the evidence behind it,
4. writes a Markdown and JSON report.

### How it differs from existing tools

| Tool | What it does | What this project adds |
|---|---|---|
| Prometheus/Grafana, log stacks | Show raw signals | Connects signals to a specific change and reaches a conclusion |
| k8sgpt | Explains objects that are broken right now | Asks whether a change caused harm, using data over time |
| Komodor-style change timelines | Lists changes | Gives each change a verdict with evidence |
| Robusta | Adds context to alerts once they fire | Needs no alert, and also catches problems below alert thresholds |

### Goals

- Detect Deployment rollouts and ConfigMap changes since the last run.
- Produce a verdict with evidence for each change, combining metrics, logs and events.
- Run unattended from cron on the developer's machine against a local kind cluster.
- Show a working demo with scenarios that can be reproduced.

### Non-goals (for the one-week version)

- Accurate verdicts at production quality. A working pipeline matters more.
- Scale (large clusters, many changes per run).
- HPA/scaling, Secret, or node changes. Loki. Alerting or Slack output.
- Running inside the cluster (stretch goal, see §11).

## 2. Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Agent framework | **LangGraph** (Python) | The pipeline is a fixed graph: detect → investigate each change in parallel → judge → report. Explicit state and `Send` make it easy to explain and debug. |
| LLM backend | **Claude CLI through subprocess**, behind a swappable `LLMClient` interface | Avoids the low per-minute and per-day limits of the Gemini free tier. The interface lets a Gemini or API client be added later. |
| Tool-calling approach | **Mix:** code detects changes; investigator agents use tools through **prompt-based tool calling** in LangGraph; judge and report work on structured data | Tool-using agents are kept to where they add value, and the number of CLI calls stays low. Tools run inside LangGraph (visible, backend-independent), not inside the CLI. |
| Environment | **Local kind + kube-prometheus-stack**, logs from the Kubernetes API | The fastest way to a working demo. Loki can be added later without changing the agents. |
| Scheduling | **Host cron/launchd** runs a Python command | Uses the host's existing `claude` login. The code works with either kubeconfig or in-cluster config, so a CronJob can be added later. |
| Output | Markdown + JSON in `reports/`, state in `state.json` | Simple, easy to inspect, easy to test. |

## 3. Architecture

```
cron (on host) ──► k8s-change-analyst run
                        │
             ┌──────────▼───────────┐
             │ 1. detect_changes    │  plain code, no LLM
             └──────────┬───────────┘
                        │ LangGraph Send: one investigation per change
        ┌───────────────┴────────────────┐
        ▼                                ▼
 ┌──────────────┐                ┌──────────────┐
 │ metrics_agent│                │  logs_agent  │   ReAct-style agents with tools
 └──────┬───────┘                └──────┬───────┘   (CLIChatModel + ToolNode)
        └──────────────┬────────────────┘
                       ▼
             ┌────────────────────┐
             │ judge              │  LLM, no tools, structured verdict
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ report + save state│  plain code
             └────────────────────┘
```

Within one change, the metrics agent and logs agent run in parallel, and different changes are investigated in parallel. A setting caps how many run at once (default 2) so the CLI isn't overloaded.

## 4. Components

### 4.1 Change detection (`changes/detect.py`), plain code

Produces `Change` objects:

```python
class Change(BaseModel):
    id: str                 # stable, e.g. "rollout:default/podinfo:rev7" or "configmap:default/podinfo-cfg:<hash>"
    kind: Literal["rollout", "configmap"]
    namespace: str
    workload: str           # Deployment name (for ConfigMaps: Deployments that mount/reference it)
    changed_at: datetime
    description: str        # human-readable diff summary
    diff: dict              # e.g. {"image": ["podinfo:6.5.0", "podinfo:6.5.1"], "env": {...}, "args": [...]}
```

- **Rollouts:** for each Deployment, list its ReplicaSets and sort them by the `deployment.kubernetes.io/revision` annotation. Any ReplicaSet created after the last run's timestamp is a rollout. The diff compares its pod template (images, env, args, resources) with the previous revision's.
- **ConfigMaps:** hash the `data` of each ConfigMap in the watched namespaces and compare with the hashes saved in state. The change time is taken from the latest `managedFields` timestamp. The change is linked to Deployments that use the ConfigMap as a volume, `envFrom`, or `env.valueFrom`. On the first run, only a baseline of hashes is saved and no changes are reported.
- **Scope:** watched namespaces come from settings (default: `demo`).
- **Skipped:** IDs already analyzed (saved in state).

### 4.2 LLM layer (`llm/`)

**`LLMClient` interface (`client.py`)**

```python
class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult: ...

class LLMResult(BaseModel):
    text: str
    structured: dict | None
```

**`ClaudeCLIClient`** runs:

```
claude -p --output-format json [--json-schema <schema>] --system-prompt <system>
       --tools "" --no-session-persistence [--model <model>]
```

- The prompt is passed on stdin.
- It reads the JSON result from the CLI and gets the result text or the structured output.
- It has a per-call timeout (default 180s) and retries once on a non-zero exit, a timeout, or invalid output.
- It runs in a neutral working directory so no project `CLAUDE.md` gets loaded.
- The exact CLI flags are checked against the installed version (v2.1.252) during implementation.

**`CLIChatModel` (`chat_model.py`)** is a LangChain `BaseChatModel` wrapped around an `LLMClient`.

- `bind_tools(tools)` saves the tool schemas.
- `_generate(messages)` turns the conversation into a text transcript and adds a list of the tools with their argument schemas. It calls `complete` with a response schema:

  ```json
  {"oneOf": [
    {"action": "call_tool", "tool": "<name>", "args": {...}},
    {"action": "final", "content": "<findings JSON or text>"}
  ]}
  ```

  - `call_tool` becomes an `AIMessage` with `tool_calls=[...]`.
  - `final` becomes an `AIMessage` with `content`.
- One tool call per step keeps the parsing simple.
- If the tool name is unknown or the arguments are invalid, the model is asked once more with the error message. If that also fails, the reply becomes a final message saying the agent failed.

This lets the investigator agents use LangGraph's standard `create_react_agent` / `ToolNode` loop without changes. Swapping in a native chat model (for example Gemini) later means changing only a setting.

### 4.3 Tools (`tools/`)

All tools are plain functions with typed arguments. **They return compact summaries, never raw data.** Errors come back as text (`"ERROR: prometheus unreachable: ..."`) and are never raised.

| Tool | File | Returns |
|---|---|---|
| `compare_windows(workload, namespace, metric, change_at)` | metrics.py | Before and after values (avg and p95), percent change. `metric` is one of: `request_rate`, `error_rate_5xx`, `latency_p95`, `cpu`, `memory`, `restarts`, `ready_replicas` |
| `query_promql(query, start, end)` | metrics.py | A fallback for other queries. Returns a summary (min/avg/max/last per series, at most 10 series) |
| `get_pod_status(workload, namespace)` | k8s.py | Pods with phase, ready status, restart count, last termination reason, image |
| `get_events(workload, namespace, since)` | k8s.py | Warning events for the Deployment, its ReplicaSets and pods, grouped by reason, with counts |
| `get_error_signatures(workload, namespace, since, previous=False)` | logs.py | Error types found in logs (error/exception/fatal lines, with numbers, IDs and IPs replaced by placeholders), with counts and one example line each. Top 15 |
| `get_pod_logs(pod, namespace, tail=50, previous=False)` | logs.py | The last lines of the log, cut to a maximum size |

PromQL templates live in `metrics.py` and target podinfo's HTTP metrics, cAdvisor, and kube-state-metrics.

### 4.4 Agents (`agents/`)

Both investigators run `create_react_agent(CLIChatModel, tools, prompt)` with a step limit (default 6 tool calls). Their final message must match:

```python
class Findings(BaseModel):
    agent: Literal["metrics", "logs"]
    status: Literal["ok", "partial", "failed"]
    observations: list[str]      # factual, each with numbers
    anomalies: list[str]         # things that got worse after the change
    data_gaps: list[str]         # e.g. "old pods gone, no pre-change logs"
```

- **metrics_agent** is given the `Change` and the windows. It uses `compare_windows` (and `query_promql` if it needs to).
- **logs_agent** is given the `Change`. It uses `get_pod_status`, `get_events`, `get_error_signatures`, and `get_pod_logs`.
- **judge** is a direct `LLMClient.complete` call with a JSON schema and no tools. It receives the `Change` plus both `Findings` and returns:

```python
class Verdict(BaseModel):
    change_id: str
    verdict: Literal["regression", "improvement", "no_impact", "inconclusive"]
    confidence: Literal["low", "medium", "high"]
    summary: str                 # 2–4 sentences
    evidence: list[str]          # each citing a specific observation
```

  If both findings have `status == "failed"`, the judge is skipped and the verdict is `inconclusive` / `low`.

### 4.5 Graph (`graph.py`)

- **Run state:** `changes`, `pending`, `results: Annotated[list[ChangeResult], operator.add]`.
- **Nodes:**
  - `detect` splits changes into ready vs. pending (by age).
  - For each ready change, a `Send` starts the `investigate` subgraph: `metrics_agent` and `logs_agent` in parallel, then `judge`. The subgraph returns a `ChangeResult`.
  - `report` writes the output files and then saves state.

### 4.6 Report and state (`report.py`, `state_store.py`)

**Reports:**
- `reports/<run-timestamp>.json`: all `ChangeResult`s, the pending list, run info (duration, number of LLM calls, errors).
- `reports/<run-timestamp>.md`: a table of verdicts, then one section per change (description, verdict, summary, evidence, observations from each agent, data gaps).
- `reports/latest.md` is a symlink to the newest report.

**`state.json`:**

```json
{"last_run_at": "...", "analyzed_change_ids": ["..."], "configmap_hashes": {"ns/name": "sha"},
 "pending": [{"change": {...}, "first_seen": "..."}]}
```

The state file is written atomically (temp file, then rename), and only after the report has been saved.

## 5. Analysis windows

- **Before:** `[changed_at − 15m, changed_at]`
- **After:** `[changed_at + 2m, now]`, capped at 30 minutes.
- If a change is less than 5 minutes old, it becomes **pending** and is re-checked on a later run.
- If a change stays pending for more than 60 minutes (for example, when runs were skipped), it's analyzed with whatever data exists.
- All these values can be changed in settings.

## 6. Configuration (`config.py`)

`pydantic-settings`, read from environment variables or `config.yaml`:

- **Connections:** `prometheus_url` (default `http://localhost:9090`), `namespaces` (default `["demo"]`), `kube_context`.
- **LLM backend:** `llm_backend` (`claude_cli`), `llm_model` (optional), `llm_timeout_s`.
- **Windows:** `before_window`, `settle_delay`, `min_age`, `max_after_window`.
- **Limits:** `max_changes_per_run` (5), `max_agent_steps` (6), `max_concurrency` (2), `run_timeout_s` (1200).
- **Paths:** `reports_dir`, `state_path`, `lock_path`.

## 7. Error handling

| Failure | Behavior |
|---|---|
| CLI timeout, non-zero exit, or invalid JSON | Retry once. If it still fails, the agent reports `status="failed"` and the change is still reported |
| Tool error (Prometheus down, pod gone) | Error text goes back to the agent, which records it in `data_gaps` |
| Both investigators failed | Judge is skipped, verdict is `inconclusive` / `low` |
| More changes than `max_changes_per_run` | Extra changes go to pending |
| Run hits `run_timeout_s` | Results so far are reported, unfinished changes stay pending |
| A previous run is still going | A lock file makes the new run exit right away with a log message |
| Crash before the report is written | State is not updated, so the next run analyzes the same changes again |

## 8. Demo environment (`deploy/demo/`, `Makefile`)

**`make demo-up`:**
- a kind cluster
- kube-prometheus-stack (Helm), with Prometheus exposed on `localhost:9090` through port-forward or NodePort
- namespace `demo`, with the podinfo Deployment, Service, and ServiceMonitor
- a load generator Deployment

**`make demo-down`** deletes the cluster.

**Scenarios:**

| Target | Change applied | Expected verdict |
|---|---|---|
| `scenario-harmless` | New image tag or pod annotation, behavior unchanged | `no_impact` |
| `scenario-errors` | Rollout with podinfo random-error and random-delay turned on | `regression` (5xx rate and p95 up, error types in logs) |
| `scenario-crashloop` | Rollout with invalid args or env | `regression` (restarts, CrashLoopBackOff, ready < desired) |
| `scenario-config` | Breaking ConfigMap edit, then restart | `regression` linked to the ConfigMap change |

The exact podinfo flags or env vars for fault injection are checked during implementation.

**Scheduling:** the README documents a crontab line (`*/10 * * * * cd <repo> && uv run k8s-change-analyst run >> logs/cron.log 2>&1`) and a macOS launchd option. It also notes that cron needs `PATH` set so it can find `claude` and `kubectl`.

## 9. Testing

- **Unit tests** (no cluster, no LLM): change detection with example objects, grouping log lines into error types, building PromQL queries and windows, saving and loading state, rendering reports.
- **`CLIChatModel` tests:** a fake `LLMClient` returns scripted responses. They check that tool calls are parsed, that the loop ends on `final`, that bad output is retried, and that an unknown tool is handled.
- **`ClaudeCLIClient` tests:** `subprocess.run` is replaced with a fake. The tests check the command built, how output is parsed, and timeout and retry behavior.
- **Graph test:** the whole LangGraph run with fake tools and a fake LLM. It checks the parallel investigations, the pending path, and that the report and state are written.
- **End-to-end (`make e2e`, manual, needs the real CLI and a cluster):** for each scenario, apply it, wait, run, and check the verdict in the JSON report. Not part of CI.

## 10. Project layout

```
k8s-change-analyst/
├── pyproject.toml          # uv; langgraph, langchain-core, kubernetes, httpx, pydantic(-settings), typer
├── Makefile
├── README.md
├── deploy/demo/            # kind config, helm values, podinfo, loadgen, ServiceMonitor, scenarios/
├── docs/superpowers/specs/
├── src/change_analyst/
│   ├── cli.py
│   ├── config.py
│   ├── models.py           # Change, Findings, Verdict, ChangeResult
│   ├── state_store.py
│   ├── changes/detect.py
│   ├── llm/client.py
│   ├── llm/chat_model.py
│   ├── tools/metrics.py
│   ├── tools/logs.py
│   ├── tools/k8s.py
│   ├── agents/metrics_agent.py
│   ├── agents/logs_agent.py
│   ├── agents/judge.py
│   ├── graph.py
│   └── report.py
└── tests/
```

## 11. Plan and stretch goals

| Day | Work |
|---|---|
| 1 | Project skeleton, demo cluster and scenarios, change detection + tests |
| 2 | `ClaudeCLIClient`, `CLIChatModel` with tool calling + tests |
| 3 | Tools (Prometheus, logs, events, pod status), metrics and logs agents |
| 4 | Judge, graph wiring, report, saved state, CLI command, lock file |
| 5 | End-to-end runs of all scenarios, prompt tuning, cron setup, README |

**Stretch goals (only after the above works):**
- a Kubernetes CronJob version (image with the CLI, Secret for the API key, RBAC)
- a `GeminiClient` or native chat model backend
- HPA and scaling changes
- tracking new error types between runs
- Loki for logs from before the change
