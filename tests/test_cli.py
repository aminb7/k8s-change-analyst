from typer.testing import CliRunner

from change_analyst import cli
from change_analyst.lock import run_lock


def test_run_exits_quietly_when_lock_is_held(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock"
    monkeypatch.setenv("CA_LOCK_PATH", str(lock_path))

    def must_not_build(settings):
        raise AssertionError("runtime should not be built while another run holds the lock")

    monkeypatch.setattr(cli, "build_runtime", must_not_build)
    with run_lock(lock_path):
        result = CliRunner().invoke(cli.app, ["run"])
    assert result.exit_code == 0
