import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from change_analyst.models import PendingChange


class State(BaseModel):
    last_run_at: datetime | None = None
    analyzed_change_ids: list[str] = Field(default_factory=list)
    configmap_hashes: dict[str, str] = Field(default_factory=dict)
    pending: list[PendingChange] = Field(default_factory=list)


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    return State.model_validate_json(path.read_text())


def save_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, path)


def remember_analyzed(existing: list[str], new: list[str], limit: int = 1000) -> list[str]:
    merged = [i for i in existing if i not in set(new)] + list(dict.fromkeys(new))
    return merged[-limit:]
