#!/usr/bin/env bash
# Restore the known-good manifests and restart podinfo.
set -euo pipefail
cd "$(dirname "$0")/../../.."
kubectl apply -f deploy/demo/app.yaml
kubectl -n demo rollout restart deployment/podinfo
kubectl -n demo rollout status deployment/podinfo --timeout=180s
