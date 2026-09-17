"""Deterministic change detection from the Kubernetes API (no LLM)."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from change_analyst.models import Change

REVISION_ANNOTATION = "deployment.kubernetes.io/revision"
MAX_VALUE_CHARS = 200


def _revision(rs) -> int:
    return int((rs.metadata.annotations or {}).get(REVISION_ANNOTATION, "0"))


def _owner_deployment(rs) -> str | None:
    for ref in rs.metadata.owner_references or []:
        if ref.kind == "Deployment":
            return ref.name
    return None


def _resources(resources) -> dict[str, dict[str, str]]:
    if resources is None:
        return {}
    return {"limits": dict(resources.limits or {}), "requests": dict(resources.requests or {})}


def _container_summary(pod_template) -> dict[str, dict[str, Any]]:
    summary = {}
    for c in pod_template.spec.containers or []:
        env = {e.name: (e.value if e.value is not None else "<valueFrom>") for e in c.env or []}
        summary[c.name] = {
            "image": c.image,
            "args": list(c.args or []),
            "command": list(c.command or []),
            "env": env,
            "resources": _resources(c.resources),
        }
    return summary


def _template_annotations(pod_template) -> dict[str, str]:
    metadata = pod_template.metadata
    return dict(metadata.annotations or {}) if metadata else {}


def diff_templates(old_template, new_template) -> dict[str, list]:
    diff: dict[str, list] = {}
    old_c, new_c = _container_summary(old_template), _container_summary(new_template)
    for name in sorted(set(old_c) | set(new_c)):
        old, new = old_c.get(name, {}), new_c.get(name, {})
        for field in ("image", "args", "command", "env", "resources"):
            if old.get(field) != new.get(field):
                diff[f"{name}.{field}"] = [old.get(field), new.get(field)]
    old_a, new_a = _template_annotations(old_template), _template_annotations(new_template)
    if old_a != new_a:
        diff["pod_annotations"] = [old_a, new_a]
    return diff


def _describe_diff(diff: dict[str, list]) -> str:
    if not diff:
        return "pod template changed (no container-level difference detected)"
    parts = []
    for key, (old, new) in diff.items():
        parts.append(f"{key}: {json.dumps(old)[:300]} -> {json.dumps(new)[:300]}")
    return "; ".join(parts)


def detect_rollouts(replica_sets: list, since: datetime) -> list[Change]:
    by_deployment: dict[tuple[str, str], list] = defaultdict(list)
    for rs in replica_sets:
        name = _owner_deployment(rs)
        if name:
            by_deployment[(rs.metadata.namespace, name)].append(rs)

    changes = []
    for (namespace, name), rss in by_deployment.items():
        rss.sort(key=_revision)
        for previous, current in zip(rss, rss[1:]):
            if current.metadata.creation_timestamp <= since:
                continue
            revision = _revision(current)
            diff = diff_templates(previous.spec.template, current.spec.template)
            changes.append(Change(
                id=f"rollout:{namespace}/{name}:rev{revision}",
                kind="rollout",
                namespace=namespace,
                workload=name,
                changed_at=current.metadata.creation_timestamp,
                description=f"Deployment {name} rolled out revision {revision}: {_describe_diff(diff)}",
                diff=diff,
            ))
    return sorted(changes, key=lambda c: c.changed_at)


def configmap_hash(cm) -> str:
    payload = json.dumps({"data": cm.data or {}, "binary_data": cm.binary_data or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _last_modified(cm) -> datetime:
    times = [f.time for f in cm.metadata.managed_fields or [] if f.time]
    return max(times) if times else cm.metadata.creation_timestamp


def _uses_configmap(deploy, cm_name: str) -> bool:
    spec = deploy.spec.template.spec
    for volume in spec.volumes or []:
        if volume.config_map and volume.config_map.name == cm_name:
            return True
    for c in list(spec.containers or []) + list(spec.init_containers or []):
        for source in c.env_from or []:
            if source.config_map_ref and source.config_map_ref.name == cm_name:
                return True
        for env in c.env or []:
            ref = env.value_from.config_map_key_ref if env.value_from else None
            if ref and ref.name == cm_name:
                return True
    return False


def detect_configmap_changes(
    configmaps: list, deployments: list, previous_hashes: dict[str, str]
) -> tuple[list[Change], dict[str, str]]:
    hashes: dict[str, str] = {}
    changes = []
    for cm in configmaps:
        namespace, name = cm.metadata.namespace, cm.metadata.name
        key = f"{namespace}/{name}"
        digest = configmap_hash(cm)
        hashes[key] = digest
        old = previous_hashes.get(key)
        if old is None or old == digest:
            continue
        data = {k: str(v)[:MAX_VALUE_CHARS] for k, v in (cm.data or {}).items()}
        for d in deployments:
            if d.metadata.namespace != namespace or not _uses_configmap(d, name):
                continue
            workload = d.metadata.name
            changes.append(Change(
                id=f"configmap:{key}:{digest}:{workload}",
                kind="configmap",
                namespace=namespace,
                workload=workload,
                changed_at=_last_modified(cm),
                description=(f"ConfigMap {key} data changed (hash {old} -> {digest}); "
                             f"used by Deployment {workload}"),
                diff={"data": data},
            ))
    return changes, hashes


def detect_changes(
    core, apps, namespaces: list[str], since: datetime, previous_hashes: dict[str, str]
) -> tuple[list[Change], dict[str, str]]:
    changes: list[Change] = []
    hashes: dict[str, str] = {}
    for namespace in namespaces:
        replica_sets = apps.list_namespaced_replica_set(namespace).items
        deployments = apps.list_namespaced_deployment(namespace).items
        configmaps = core.list_namespaced_config_map(namespace).items
        changes.extend(detect_rollouts(replica_sets, since))
        cm_changes, cm_hashes = detect_configmap_changes(configmaps, deployments, previous_hashes)
        changes.extend(cm_changes)
        hashes.update(cm_hashes)
    return sorted(changes, key=lambda c: c.changed_at), hashes
