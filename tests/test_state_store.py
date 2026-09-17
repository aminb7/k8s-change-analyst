from datetime import datetime, timezone

from change_analyst.models import Change, PendingChange
from change_analyst.state_store import State, load_state, remember_analyzed, save_state

T0 = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def test_load_missing_returns_empty_state(tmp_path):
    state = load_state(tmp_path / "state.json")
    assert state.last_run_at is None
    assert state.analyzed_change_ids == []
    assert state.configmap_hashes == {}
    assert state.pending == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "state.json"
    change = Change(
        id="rollout:demo/podinfo:rev2", kind="rollout", namespace="demo",
        workload="podinfo", changed_at=T0, description="d",
    )
    state = State(
        last_run_at=T0,
        analyzed_change_ids=["a"],
        configmap_hashes={"demo/podinfo-config": "abc"},
        pending=[PendingChange(change=change, first_seen=T0)],
    )
    save_state(path, state)
    assert load_state(path) == state
    assert not (path.parent / "state.json.tmp").exists()


def test_remember_analyzed_dedupes_and_trims():
    result = remember_analyzed(["a", "b", "c"], ["c", "d"], limit=3)
    assert result == ["b", "c", "d"]
