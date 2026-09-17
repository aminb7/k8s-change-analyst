#!/usr/bin/env bash
# Rollout that changes only a pod annotation: expected verdict no_impact.
set -euo pipefail
kubectl -n demo patch deployment podinfo --type merge \
  -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"demo/harmless-change\":\"$(date +%s)\"}}}}}"
kubectl -n demo rollout status deployment/podinfo --timeout=180s
