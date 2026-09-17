import logging

import typer

from change_analyst.config import Settings
from change_analyst.llm.client import ClaudeCLIClient
from change_analyst.lock import LockHeld, run_lock
from change_analyst.pipeline import Runtime, execute_run
from change_analyst.tools.prometheus import PromClient

app = typer.Typer(help="Judge the impact of recent Kubernetes changes.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """k8s-change-analyst: LLM multi-agent change-impact analysis."""


def build_runtime(settings: Settings) -> Runtime:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config(context=settings.kube_context)
    return Runtime(
        core=client.CoreV1Api(),
        apps=client.AppsV1Api(),
        prom=PromClient(settings.prometheus_url),
        llm=ClaudeCLIClient(model=settings.llm_model or None, timeout_s=settings.llm_timeout_s),
    )


@app.command()
def run() -> None:
    """Detect recent changes, investigate them with agents, and write a report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    try:
        with run_lock(settings.lock_path):
            path = execute_run(settings, build_runtime(settings))
    except LockHeld:
        typer.echo("Another run is still in progress; exiting.", err=True)
        return
    typer.echo(f"Report written to {path}")
