import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockHeld(Exception):
    """Another run holds the lock."""


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockHeld(str(path)) from exc
        yield
    finally:
        handle.close()
