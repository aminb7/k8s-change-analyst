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
