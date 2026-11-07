"""Offline resilience tests for the Apify and Exa clients.

Every test monkeypatches the shared ``requests.Session`` methods, so no request
leaves the process and no paid Actor run is started.
"""

from __future__ import annotations

import json
import logging

import pytest
import requests
from requests.adapters import HTTPAdapter

from sentpred import apify, news
from sentpred.apify import ApifyError, run_actor
from sentpred.config import Asset
from sentpred.sources import POST_COLUMNS


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


def install(monkeypatch, module, *, get=None, post=None):
    """Replace session methods on the module's shared Session instance."""
    if get is not None:
        monkeypatch.setattr(module.SESSION, "get", get)
    if post is not None:
        monkeypatch.setattr(module.SESSION, "post", post)


def no_sleep(monkeypatch, module) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(module.time, "sleep", delays.append)
    return delays


def status_response(status: str, dataset_id: str = "dataset-1") -> FakeResponse:
    return FakeResponse(200, {"data": {"id": "run-1", "status": status, "defaultDatasetId": dataset_id}})


def test_sessions_configured_with_retry_adapter():
    expected = {429, 500, 502, 503, 504}
    for module in (apify, news):
        adapter = module.SESSION.get_adapter("https://api.example.test/")
        assert isinstance(adapter, HTTPAdapter)
        retry = adapter.max_retries
        assert retry.total == 4
        assert retry.backoff_factor == 0.5
        assert set(retry.status_forcelist) == expected
        assert {"GET", "POST"} <= set(retry.allowed_methods)


def test_apify_retries_transient_status_then_succeeds(monkeypatch):
    calls: list[str] = []
    responses = [FakeResponse(503, text="busy"), FakeResponse(200, {"data": {"id": "run-1"}})]

    def fake_post(url, **kwargs):
        calls.append(url)
        return responses.pop(0)

    def fake_get(url, **kwargs):
        if "/actor-runs/run-1" in url:
            return status_response("SUCCEEDED")
        if "/datasets/dataset-1/items" in url:
            return FakeResponse(200, [{"title": "one"}, {"title": "two"}])
        raise AssertionError(f"unexpected GET {url}")

    install(monkeypatch, apify, get=fake_get, post=fake_post)
    delays = no_sleep(monkeypatch, apify)

    assert run_actor("user~actor", {"q": 1}, "token", timeout_s=30) == [{"title": "one"}, {"title": "two"}]
    assert len(calls) == 2
    assert delays == [0.5]


def test_apify_network_failure_is_wrapped_as_apify_error(monkeypatch):
    def fake_post(url, **kwargs):
        raise requests.ConnectionError("connection reset")

    install(monkeypatch, apify, post=fake_post)
    delays = no_sleep(monkeypatch, apify)

    with pytest.raises(ApifyError, match="failed after 3 attempts"):
        run_actor("user~actor", {}, "token")
    assert delays == [0.5, 1.0]


def test_apify_local_timeout_aborts_run_before_raising(monkeypatch):
    post_urls: list[str] = []

    def fake_post(url, **kwargs):
        post_urls.append(url)
        if url.endswith("/abort"):
            return FakeResponse(200, {"data": {}})
        return FakeResponse(200, {"data": {"id": "run-1"}})

    def fake_get(url, **kwargs):
        return status_response("RUNNING")

    install(monkeypatch, apify, get=fake_get, post=fake_post)
    no_sleep(monkeypatch, apify)

    with pytest.raises(ApifyError, match="did not finish"):
        run_actor("user~actor", {}, "token", timeout_s=-1, poll_s=0)

    assert post_urls[-1] == f"{apify.API}/actor-runs/run-1/abort"


@pytest.mark.parametrize("status", ["FAILED", "ABORTED", "TIMED-OUT"])
def test_apify_terminal_failure_returns_non_empty_partial_dataset(monkeypatch, status):
    def fake_post(url, **kwargs):
        return FakeResponse(200, {"data": {"id": "run-1"}})

    def fake_get(url, **kwargs):
        if "/actor-runs/run-1" in url:
            return status_response(status)
        if "/datasets/dataset-1/items" in url:
            return FakeResponse(200, [{"title": "partial"}])
        raise AssertionError(f"unexpected GET {url}")

    install(monkeypatch, apify, get=fake_get, post=fake_post)

    assert run_actor("user~actor", {}, "token", timeout_s=30) == [{"title": "partial"}]


def test_apify_terminal_failure_with_empty_dataset_raises(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(200, {"data": {"id": "run-1"}})

    def fake_get(url, **kwargs):
        if "/actor-runs/run-1" in url:
            return status_response("FAILED")
        if "/datasets/dataset-1/items" in url:
            return FakeResponse(200, [])
        raise AssertionError(f"unexpected GET {url}")

    install(monkeypatch, apify, get=fake_get, post=fake_post)

    with pytest.raises(ApifyError, match="FAILED and produced no data"):
        run_actor("user~actor", {}, "token", timeout_s=30)


def _news_asset(ticker: str = "0700.HK", name: str = "Tencent Holdings") -> Asset:
    return Asset("HKEX", ticker, name, (name,))


def _exa_payload(title: str) -> dict:
    return {
        "results": [{
            "title": title,
            "highlights": ["a price-relevant highlight"],
            "publishedDate": "2026-11-06T03:00:00Z",
            "url": "https://example.test/news",
            "author": "reporter",
        }]
    }


def test_exa_retries_429_then_succeeds(monkeypatch):
    responses = [FakeResponse(429, text="slow down"), FakeResponse(200, _exa_payload("Tencent beats"))]
    calls = 0

    def fake_post(url, **kwargs):
        nonlocal calls
        calls += 1
        return responses.pop(0)

    install(monkeypatch, news, post=fake_post)
    delays = no_sleep(monkeypatch, news)

    df = news.fetch_news((_news_asset(),), "exa-key", days=10)

    assert calls == 2
    assert delays == [0.5]
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "0700.HK"
    assert "Tencent beats" in df.iloc[0]["text"]


def test_exa_failure_does_not_crash_other_assets(monkeypatch, caplog):
    first = _news_asset()
    second = _news_asset("005930.KS", "Samsung Electronics")
    calls = 0
    sleeps: list[float] = []

    def fake_post(url, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:  # all attempts for the first asset fail
            return FakeResponse(500, text="server error")
        return FakeResponse(200, _exa_payload("Samsung wins"))

    install(monkeypatch, news, post=fake_post)
    monkeypatch.setattr(news.time, "sleep", sleeps.append)

    with caplog.at_level(logging.ERROR, logger="sentpred.news"):
        df = news.fetch_news((first, second), "exa-key", days=10)

    assert calls == 4
    assert sleeps == [0.5, 1.0]
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "005930.KS"
    assert list(df.columns) == POST_COLUMNS
    assert any("Exa error for 0700.HK" in record.message for record in caplog.records)


def test_exa_network_failure_is_isolated_per_asset(monkeypatch):
    def fake_post(url, **kwargs):
        raise requests.Timeout("read timed out")

    install(monkeypatch, news, post=fake_post)
    no_sleep(monkeypatch, news)

    df = news.fetch_news((_news_asset(),), "exa-key", days=10)

    assert df.empty
    assert list(df.columns) == POST_COLUMNS
