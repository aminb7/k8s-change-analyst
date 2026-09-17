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
