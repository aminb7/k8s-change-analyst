from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from change_analyst.config import Settings
from change_analyst.models import Change, PendingChange
from change_analyst.state_store import State


@dataclass(frozen=True)
class Windows:
    before_start: datetime
    before_end: datetime
    after_start: datetime
    after_end: datetime

    def as_dict(self) -> dict[str, str]:
        return {k: v.isoformat() for k, v in asdict(self).items()}


def compute_windows(changed_at: datetime, now: datetime, settings: Settings) -> Windows:
    after_start = changed_at + timedelta(minutes=settings.settle_delay_min)
    after_end = min(now, after_start + timedelta(minutes=settings.max_after_window_min))
    return Windows(
        before_start=changed_at - timedelta(minutes=settings.before_window_min),
        before_end=changed_at,
        after_start=after_start,
        after_end=max(after_end, after_start),
    )


def plan_run(
    detected: list[Change], state: State, now: datetime, settings: Settings
) -> tuple[list[PendingChange], list[PendingChange]]:
    analyzed = set(state.analyzed_change_ids)
    candidates: dict[str, PendingChange] = {p.change.id: p for p in state.pending}
    for change in detected:
        if change.id not in analyzed and change.id not in candidates:
            candidates[change.id] = PendingChange(change=change, first_seen=now)

    min_age = timedelta(minutes=settings.min_age_min)
    max_pending = timedelta(minutes=settings.max_pending_min)
    ready: list[PendingChange] = []
    waiting: list[PendingChange] = []
    for item in sorted(candidates.values(), key=lambda p: p.change.changed_at):
        old_enough = now - item.change.changed_at >= min_age
        waited_too_long = now - item.first_seen >= max_pending
        if (old_enough or waited_too_long) and len(ready) < settings.max_changes_per_run:
            ready.append(item)
        else:
            waiting.append(item)
    return ready, waiting
