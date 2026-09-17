from datetime import timedelta

from factories import (
    T0, FakeApps, FakeCore, configmap, container, deployment, env_from_configmap,
    replica_set, template,
)

from change_analyst.changes.detect import (
    configmap_hash, detect_changes, detect_configmap_changes, detect_rollouts, diff_templates,
)


def test_diff_templates_reports_changed_fields_only():
    old = template([container(image="podinfo:6.14.0", args=["--port=9898"])])
    new = template([container(image="podinfo:6.15.0", args=["--port=9898"])],
                   annotations={"demo/x": "1"})
    diff = diff_templates(old, new)
    assert diff == {
        "podinfo.image": ["podinfo:6.14.0", "podinfo:6.15.0"],
        "pod_annotations": [{}, {"demo/x": "1"}],
    }


def test_detects_rollout_after_since():
    old = replica_set(revision=1, created=T0 - timedelta(hours=2),
                      tmpl=template([container(image="podinfo:6.14.0")]))
    new = replica_set(revision=2, created=T0 + timedelta(minutes=5),
                      tmpl=template([container(image="podinfo:6.15.0")]))
    changes = detect_rollouts([new, old], since=T0)
    assert len(changes) == 1
    change = changes[0]
    assert change.id == "rollout:demo/podinfo:rev2"
    assert change.kind == "rollout"
    assert change.workload == "podinfo"
    assert change.changed_at == T0 + timedelta(minutes=5)
    assert change.diff == {"podinfo.image": ["podinfo:6.14.0", "podinfo:6.15.0"]}
    assert "podinfo.image" in change.description


def test_ignores_old_first_revision_and_orphan_replica_sets():
    first = replica_set(revision=1, created=T0 + timedelta(minutes=1))
    orphan = replica_set(deployment=None, revision=2, created=T0 + timedelta(minutes=1))
    old_pair = [replica_set(deployment="api", revision=1, created=T0 - timedelta(hours=3)),
                replica_set(deployment="api", revision=2, created=T0 - timedelta(hours=1))]
    assert detect_rollouts([first, orphan, *old_pair], since=T0) == []


def test_configmap_first_sighting_only_records_hash():
    cm = configmap(data={"A": "1"})
    changes, hashes = detect_configmap_changes([cm], [], previous_hashes={})
    assert changes == []
    assert hashes == {"demo/podinfo-config": configmap_hash(cm)}


def test_configmap_change_links_referencing_deployments():
    old = configmap(data={"PODINFO_RANDOM_ERROR": "false"})
    new = configmap(data={"PODINFO_RANDOM_ERROR": "true"}, modified=T0 + timedelta(minutes=3))
    user = deployment(tmpl=template([container(env_from=env_from_configmap("podinfo-config"))]))
    bystander = deployment(name="other")
    changes, hashes = detect_configmap_changes(
        [new], [user, bystander], previous_hashes={"demo/podinfo-config": configmap_hash(old)}
    )
    assert len(changes) == 1
    change = changes[0]
    assert change.kind == "configmap"
    assert change.workload == "podinfo"
    assert change.changed_at == T0 + timedelta(minutes=3)
    assert change.id == f"configmap:demo/podinfo-config:{configmap_hash(new)}:podinfo"
    assert change.diff["data"] == {"PODINFO_RANDOM_ERROR": "true"}
    assert hashes["demo/podinfo-config"] == configmap_hash(new)


def test_unchanged_configmap_produces_no_change():
    cm = configmap(data={"A": "1"})
    user = deployment(tmpl=template([container(env_from=env_from_configmap("podinfo-config"))]))
    changes, _ = detect_configmap_changes([cm], [user], {"demo/podinfo-config": configmap_hash(cm)})
    assert changes == []


def test_detect_changes_queries_each_namespace():
    apps = FakeApps(replica_sets=[
        replica_set(revision=1, created=T0 - timedelta(hours=1)),
        replica_set(revision=2, created=T0 + timedelta(minutes=1),
                    tmpl=template([container(args=["--level=debug"])])),
    ])
    core = FakeCore(configmaps=[configmap()])
    changes, hashes = detect_changes(core, apps, ["demo", "empty"], T0, {})
    assert [c.id for c in changes] == ["rollout:demo/podinfo:rev2"]
    assert "demo/podinfo-config" in hashes
