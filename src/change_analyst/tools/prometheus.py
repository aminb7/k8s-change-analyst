import math
from datetime import datetime

import httpx


class PromError(RuntimeError):
    """Prometheus query failed."""


class PromClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0, http: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=timeout_s)

    def query_range(self, query: str, start: datetime, end: datetime, step_s: int = 15) -> list[dict]:
        try:
            response = self.http.get(f"{self.base_url}/api/v1/query_range", params={
                "query": query, "start": start.timestamp(), "end": end.timestamp(), "step": f"{step_s}s",
            })
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PromError(f"prometheus request failed: {exc}") from exc
        if body.get("status") != "success":
            raise PromError(body.get("error") or f"HTTP {response.status_code}")
        return body["data"]["result"]


def summarize_values(values: list) -> dict | None:
    numbers = []
    for _, raw in values:
        value = float(raw)
        if math.isfinite(value):
            numbers.append(value)
    if not numbers:
        return None
    ordered = sorted(numbers)
    p95 = ordered[round(0.95 * (len(ordered) - 1))]
    return {
        "avg": round(sum(numbers) / len(numbers), 6),
        "p95": round(p95, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "last": round(numbers[-1], 6),
        "samples": len(numbers),
    }
