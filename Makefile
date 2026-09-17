CLUSTER := change-analyst

.PHONY: demo-up demo-down scenario-% run test e2e

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

run:
	uv run k8s-change-analyst run

test:
	uv run pytest -q
