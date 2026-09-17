import json
import subprocess

import pytest

from change_analyst.llm.client import ClaudeCLIClient, LLMError


def proc(payload, returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=json.dumps(payload), stderr=stderr)


SUCCESS = {"type": "result", "subtype": "success", "is_error": False,
           "result": '{"answer": 5}', "structured_output": {"answer": 5}}


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_builds_isolated_command_and_returns_structured_output():
    runner = FakeRunner([proc(SUCCESS)])
    client = ClaudeCLIClient(model="sonnet", timeout_s=30, workdir="/tmp", runner=runner)
    schema = {"type": "object"}
    result = client.complete("SYS", "PROMPT", json_schema=schema)

    assert result.structured == {"answer": 5}
    assert result.text == '{"answer": 5}'
    cmd, kwargs = runner.calls[0]
    assert cmd[:3] == ["claude", "-p", "--output-format"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in cmd
    assert "--strict-mcp-config" in cmd
    assert "--disable-slash-commands" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema
    assert kwargs["input"] == "PROMPT"
    assert kwargs["timeout"] == 30
    assert kwargs["cwd"] == "/tmp"
    assert client.calls == 1


def test_omits_model_and_schema_when_not_given():
    runner = FakeRunner([proc({**SUCCESS, "structured_output": None})])
    client = ClaudeCLIClient(model=None, runner=runner)
    result = client.complete("SYS", "PROMPT")
    cmd, _ = runner.calls[0]
    assert "--model" not in cmd and "--json-schema" not in cmd
    assert result.structured is None


def test_retries_once_after_timeout():
    runner = FakeRunner([subprocess.TimeoutExpired(cmd="claude", timeout=1), proc(SUCCESS)])
    client = ClaudeCLIClient(runner=runner)
    assert client.complete("S", "P", json_schema={"type": "object"}).structured == {"answer": 5}
    assert client.calls == 2


@pytest.mark.parametrize("bad", [
    proc(SUCCESS, returncode=1, stderr="auth failed"),
    subprocess.CompletedProcess(args=[], returncode=0, stdout="not json", stderr=""),
    proc({**SUCCESS, "is_error": True, "subtype": "error_during_execution"}),
    proc({**SUCCESS, "structured_output": None}),
])
def test_raises_after_two_failures(bad):
    runner = FakeRunner([bad, bad])
    client = ClaudeCLIClient(runner=runner)
    with pytest.raises(LLMError):
        client.complete("S", "P", json_schema={"type": "object"})
    assert len(runner.calls) == 2


def test_wraps_oserror_in_lmmerror_and_retries():
    runner = FakeRunner([FileNotFoundError(2, "No such file", "claude"), proc(SUCCESS)])
    client = ClaudeCLIClient(runner=runner)
    result = client.complete("S", "P", json_schema={"type": "object"})
    assert result.structured == {"answer": 5}
    assert client.calls == 2


def test_raises_lmmerror_after_two_oserrors():
    runner = FakeRunner([FileNotFoundError(2, "No such file", "claude"),
                         FileNotFoundError(2, "No such file", "claude")])
    client = ClaudeCLIClient(runner=runner)
    with pytest.raises(LLMError) as exc_info:
        client.complete("S", "P")
    assert "could not run claude CLI" in str(exc_info.value)
    assert len(runner.calls) == 2
