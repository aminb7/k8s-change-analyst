from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CA_", env_file=".env", extra="ignore")

    prometheus_url: str = "http://localhost:9090"
    namespaces: list[str] = ["demo"]
    kube_context: str | None = None

    llm_backend: Literal["claude_cli"] = "claude_cli"
    llm_model: str | None = "sonnet"
    llm_timeout_s: int = 180

    before_window_min: int = 15
    settle_delay_min: int = 2
    min_age_min: int = 5
    max_after_window_min: int = 30
    max_pending_min: int = 60
    first_run_lookback_min: int = 60

    max_changes_per_run: int = 5
    max_agent_steps: int = 6
    max_concurrency: int = 2
    run_timeout_s: int = 1200

    reports_dir: Path = Path("reports")
    state_path: Path = Path("state.json")
    lock_path: Path = Path(".change-analyst.lock")
