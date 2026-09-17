#!/usr/bin/env bash
# ConfigMap edit enabling random errors, picked up by a restart: expected verdict regression.
set -euo pipefail
kubectl -n demo patch configmap podinfo-config --type merge -p '{"data":{"PODINFO_RANDOM_ERROR":"true"}}'
kubectl -n demo rollout restart deployment/podinfo
kubectl -n demo rollout status deployment/podinfo --timeout=180s || true
