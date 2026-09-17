from datetime import datetime, timedelta, timezone

from change_analyst.config import Settings
from change_analyst.models import Change, PendingChange
from change_analyst.planning import compute_windows, plan_run
from change_analyst.state_store import State

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(
    _env_file=None, before_window_min=15, settle_delay_min=2, min_age_min=5,
    max_after_window_min=30, max_pending_min=60, max_changes_per_run=2,
)


def mk(change_id: str, minutes_ago: float) -> Change:
    return Change(id=change_id, kind="rollout", namespace="demo", workload="podinfo",
                  changed_at=NOW - timedelta(minutes=minutes_ago), description="d")


def ids(items):
    return [p.change.id for p in items]


def test_young_change_waits_and_old_change_is_ready():
    ready, waiting = plan_run([mk("young", 2), mk("old", 10)], State(), NOW, SETTINGS)
    assert ids(ready) == ["old"]
    assert ids(waiting) == ["young"]
    assert waiting[0].first_seen == NOW


def test_already_analyzed_changes_are_skipped():
    ready, waiting = plan_run([mk("done", 10)], State(analyzed_change_ids=["done"]), NOW, SETTINGS)
    assert ready == [] and waiting == []


def test_pending_change_keeps_first_seen_and_is_not_duplicated():
    first_seen = NOW - timedelta(minutes=8)
    state = State(pending=[PendingChange(change=mk("p", 9), first_seen=first_seen)])
    ready, waiting = plan_run([mk("p", 9)], state, NOW, SETTINGS)
    assert ids(ready) == ["p"]
    assert ready[0].first_seen == first_seen
    assert waiting == []


def test_change_pending_too_long_is_forced_ready():
    state = State(pending=[PendingChange(change=mk("stuck", 1), first_seen=NOW - timedelta(minutes=61))])
    ready, _ = plan_run([], state, NOW, SETTINGS)
    assert ids(ready) == ["stuck"]


def test_ready_changes_are_capped_oldest_first():
    ready, waiting = plan_run([mk("a", 30), mk("b", 20), mk("c", 10)], State(), NOW, SETTINGS)
    assert ids(ready) == ["a", "b"]
    assert ids(waiting) == ["c"]


def test_compute_windows():
    changed = NOW - timedelta(minutes=10)
    w = compute_windows(changed, NOW, SETTINGS)
    assert w.before_start == changed - timedelta(minutes=15)
    assert w.before_end == changed
    assert w.after_start == changed + timedelta(minutes=2)
    assert w.after_end == NOW
    assert w.as_dict()["after_end"] == NOW.isoformat()


def test_compute_windows_caps_after_window():
    changed = NOW - timedelta(hours=2)
    w = compute_windows(changed, NOW, SETTINGS)
    assert w.after_end == changed + timedelta(minutes=32)
