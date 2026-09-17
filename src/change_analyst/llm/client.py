"""LLM backends. The Claude CLI backend shells out to `claude -p`."""
import json
import logging
import subprocess
import tempfile
import threading
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

log = logging.getLogger(__name__)


class LLMResult(BaseModel):
    text: str
    structured: dict[str, Any] | None = None


class LLMError(RuntimeError):
    """The LLM backend failed to produce a usable answer."""


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult: ...


class ClaudeCLIClient:
    def __init__(
        self,
        model: str | None = None,
        timeout_s: int = 180,
        executable: str = "claude",
        workdir: str | None = None,
        retries: int = 1,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.executable = executable
        # A neutral directory so no project CLAUDE.md is picked up.
        self.workdir = workdir or tempfile.gettempdir()
        self.retries = retries
        self.runner = runner
        self.calls = 0
        self._lock = threading.Lock()

    def build_command(self, system: str, json_schema: dict | None) -> list[str]:
        cmd = [
            self.executable, "-p",
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "",
            "--system-prompt", system,
        ]
        if self.model:
            cmd += ["--model", self.model]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema)]
        return cmd

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult:
        last_error: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._call_once(system, prompt, json_schema)
            except LLMError as exc:
                last_error = exc
                log.warning("claude CLI attempt %d failed: %s", attempt + 1, exc)
        assert last_error is not None
        raise last_error

    def _call_once(self, system: str, prompt: str, json_schema: dict | None) -> LLMResult:
        with self._lock:
            self.calls += 1
        cmd = self.build_command(system, json_schema)
        try:
            proc = self.runner(cmd, input=prompt, capture_output=True, text=True,
                               timeout=self.timeout_s, cwd=self.workdir)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude CLI timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            raise LLMError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON output: {proc.stdout[:200]!r}") from exc
        if data.get("is_error") or data.get("subtype") != "success":
            raise LLMError(f"claude CLI reported an error ({data.get('subtype')}): "
                           f"{str(data.get('result'))[:300]}")
        structured = data.get("structured_output")
        if json_schema is not None and not isinstance(structured, dict):
            raise LLMError("claude CLI returned no structured_output for a schema request")
        return LLMResult(text=data.get("result") or "", structured=structured)
