import pytest

from change_analyst.lock import LockHeld, run_lock


def test_second_lock_is_refused(tmp_path):
    path = tmp_path / "run.lock"
    with run_lock(path):
        with pytest.raises(LockHeld):
            with run_lock(path):
                pass


def test_lock_is_released_after_exit(tmp_path):
    path = tmp_path / "run.lock"
    with run_lock(path):
        pass
    with run_lock(path):
        pass
