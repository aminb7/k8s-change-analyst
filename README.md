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
| `CA_MAX_PENDING_MIN` | `60` | force-analyze a pending change once it has waited this long |
| `CA_FIRST_RUN_LOOKBACK_MIN` | `60` | lookback window for detecting changes when no state is saved yet |
| `CA_MAX_CHANGES_PER_RUN` | `5` | max changes analyzed in one run |
| `CA_MAX_AGENT_STEPS` | `6` | max tool-calling steps per agent |
| `CA_MAX_CONCURRENCY` | `2` | max concurrent investigations; at 1, a change's two agents also run one after another |
| `CA_RUN_TIMEOUT_S` | `1200` | overall run deadline |
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

## Troubleshooting

If Prometheus has no `http_requests_total` series for the `demo` namespace a few minutes after `make demo-up`, restart the Prometheus pod so its config reloader picks up the podinfo `ServiceMonitor`:

```bash
kubectl -n monitoring delete pod -l app.kubernetes.io/name=prometheus
```

## Development

```bash
make test     # unit tests (no cluster, no LLM)
make e2e      # full scenario run against the demo cluster with the real CLI (~45 min)
```

Design: `docs/superpowers/specs/2026-09-16-k8s-change-analyst-design.md`
