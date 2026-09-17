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
