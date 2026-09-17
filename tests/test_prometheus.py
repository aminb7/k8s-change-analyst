from datetime import datetime, timezone

import httpx
import pytest

from change_analyst.tools.prometheus import PromClient, PromError, summarize_values

T = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def test_query_range_sends_params_and_returns_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "success", "data": {"resultType": "matrix", "result": [
            {"metric": {}, "values": [[1, "1"], [2, "3"]]}]}})

    client = PromClient("http://prom:9090", http=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.query_range("up", T, T, step_s=30)
    assert result == [{"metric": {}, "values": [[1, "1"], [2, "3"]]}]
    assert seen["path"] == "/api/v1/query_range"
    assert seen["query"] == "up"
    assert seen["step"] == "30s"
    assert float(seen["start"]) == T.timestamp()


def test_query_range_raises_on_prometheus_error():
    def handler(request):
        return httpx.Response(400, json={"status": "error", "error": "parse error"})

    client = PromClient("http://prom:9090", http=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(PromError, match="parse error"):
        client.query_range("sum(", T, T)


def test_summarize_values_skips_nan():
    summary = summarize_values([[1, "1"], [2, "NaN"], [3, "3"], [4, "2"]])
    assert summary == {"avg": 2.0, "p95": 3.0, "min": 1.0, "max": 3.0, "last": 2.0, "samples": 3}
    assert summarize_values([[1, "NaN"]]) is None
