"""Minimal Apify REST client: start an Actor run, wait for it, download dataset items.

Network calls share one ``requests.Session`` configured with urllib3 retries. Every
``requests.RequestException`` is translated into :class:`ApifyError` so the pipeline's
per-source handler in ``cli.stage_scrape`` can skip a broken source instead of dying.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.apify.com/v2"
log = logging.getLogger(__name__)

#: Statuses worth retrying when a server or rate limiter is momentarily unhappy.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: Total calls made by our own retry loop (the Session adapter also retries underneath).
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 0.5


class ApifyError(RuntimeError):
    pass


def _build_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


def _backoff(attempt: int) -> None:
    time.sleep(BACKOFF_BASE_S * (2**attempt))


def _request(
    method: str,
    url: str,
    *,
    attempts: int = MAX_ATTEMPTS,
    **kwargs: Any,
) -> requests.Response:
    """Call the shared session, retrying transient errors and wrapping network failures."""
    last_exc: requests.RequestException | None = None
    for attempt in range(attempts):
        try:
            resp = getattr(SESSION, method)(url, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            log.warning("Apify %s %s failed (%s); retrying", method.upper(), url, exc)
            _backoff(attempt)
            continue
        if resp.status_code in RETRY_STATUS and attempt + 1 < attempts:
            log.warning("Apify %s %s returned HTTP %s; retrying", method.upper(), url, resp.status_code)
            _backoff(attempt)
            continue
        return resp
    raise ApifyError(
        f"Apify {method.upper()} {url} failed after {attempts} attempts: {last_exc}"
    ) from last_exc


def run_actor(
    actor_id: str,
    run_input: dict[str, Any],
    token: str,
    max_charge_usd: float | None = None,
    timeout_s: int = 900,
    poll_s: int = 10,
) -> list[dict[str, Any]]:
    """Run an Actor (id as ``user~name``) and return its default dataset items.

    A local timeout aborts the server-side run before raising so paid compute stops.
    A run that ends FAILED/ABORTED/TIMED-OUT still returns whatever made it into the
    default dataset; only an empty dataset is treated as a hard failure.
    """
    params: dict[str, Any] = {"token": token}
    if max_charge_usd is not None:
        # Cost cap is a query parameter, not part of the Actor input.
        params["maxTotalChargeUsd"] = max_charge_usd

    resp = _request("post", f"{API}/acts/{actor_id}/runs", params=params, json=run_input, timeout=60)
    _raise(resp)
    run_id = _json(resp, f"{actor_id} start response")["data"]["id"]
    log.info("started %s run %s", actor_id, run_id)

    deadline = time.monotonic() + timeout_s
    while True:
        status_resp = _request("get", f"{API}/actor-runs/{run_id}", params={"token": token}, timeout=60)
        _raise(status_resp)
        run = _json(status_resp, f"run {run_id} status response")["data"]
        status = run["status"]
        if status == "SUCCEEDED":
            log.info("run %s finished; usageTotalUsd=%s", run_id, run.get("usageTotalUsd"))
            break
        if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
            return _partial_or_raise(actor_id, run_id, status, run, token)
        if time.monotonic() > deadline:
            _abort_run(run_id, token)
            raise ApifyError(f"{actor_id} run {run_id} did not finish within {timeout_s}s")
        time.sleep(poll_s)

    return _dataset_items(run["defaultDatasetId"], token)


def _partial_or_raise(
    actor_id: str,
    run_id: str,
    status: str,
    run: dict[str, Any],
    token: str,
) -> list[dict[str, Any]]:
    """Return partial results from a failed run when the dataset has anything usable."""
    dataset_id = run.get("defaultDatasetId")
    if dataset_id:
        try:
            items = _dataset_items(dataset_id, token)
        except ApifyError as exc:
            raise ApifyError(
                f"{actor_id} run {run_id} ended with {status} and its dataset could not be read: {exc}"
            ) from exc
        if items:
            log.warning(
                "%s run %s ended with %s; returning %d partial dataset items",
                actor_id, run_id, status, len(items),
            )
            return items
    raise ApifyError(f"{actor_id} run {run_id} ended with {status} and produced no data")


def _abort_run(run_id: str, token: str) -> None:
    """Best-effort abort so a locally timed-out run is not left consuming paid compute."""
    try:
        resp = _request(
            "post",
            f"{API}/actor-runs/{run_id}/abort",
            params={"token": token},
            timeout=30,
            attempts=1,
        )
        _raise(resp)
        log.warning("aborted Apify run %s after local timeout", run_id)
    except ApifyError as exc:
        log.warning("failed to abort Apify run %s after local timeout: %s", run_id, exc)


def _dataset_items(dataset_id: str, token: str, page: int = 1000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = _request(
            "get",
            f"{API}/datasets/{dataset_id}/items",
            params={"token": token, "format": "json", "clean": "true", "limit": page, "offset": offset},
            timeout=120,
        )
        _raise(resp)
        batch = _json(resp, f"dataset {dataset_id} response")
        items.extend(batch)
        if len(batch) < page:
            return items
        offset += page


def _json(resp: requests.Response, context: str) -> Any:
    try:
        return resp.json()
    except ValueError as exc:
        raise ApifyError(f"{context} was not valid JSON: {exc}") from exc


def _raise(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        raise ApifyError(f"Apify HTTP {resp.status_code}: {resp.text[:300]}")
