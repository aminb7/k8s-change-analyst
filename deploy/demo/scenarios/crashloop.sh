#!/usr/bin/env bash
# Rollout with an invalid flag: new pods exit with code 2 and crash-loop. Expected verdict regression.
set -euo pipefail
kubectl -n demo patch deployment podinfo --type json -p '[
  {"op": "replace", "path": "/spec/template/spec/containers/0/args",
   "value": ["--port=not-a-number", "--level=info"]}
]'
echo "crashloop rollout started (it will not complete; that is expected)"
