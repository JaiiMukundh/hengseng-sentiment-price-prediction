"""Minimal Apify REST client: start an Actor run, wait for it, download dataset items."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

API = "https://api.apify.com/v2"
log = logging.getLogger(__name__)


class ApifyError(RuntimeError):
    pass


def run_actor(
    actor_id: str,
    run_input: dict[str, Any],
    token: str,
    max_charge_usd: float | None = None,
    timeout_s: int = 900,
    poll_s: int = 10,
) -> list[dict[str, Any]]:
    """Run an Actor (id as ``user~name``) and return its default dataset items."""
    params: dict[str, Any] = {"token": token}
    if max_charge_usd is not None:
        # Cost cap is a query parameter, not part of the Actor input.
        params["maxTotalChargeUsd"] = max_charge_usd

    resp = requests.post(f"{API}/acts/{actor_id}/runs", params=params, json=run_input, timeout=60)
    _raise(resp)
    run_id = resp.json()["data"]["id"]
    log.info("started %s run %s", actor_id, run_id)

    deadline = time.monotonic() + timeout_s
    while True:
        status_resp = requests.get(f"{API}/actor-runs/{run_id}", params={"token": token}, timeout=60)
        _raise(status_resp)
        run = status_resp.json()["data"]
        if run["status"] == "SUCCEEDED":
            log.info("run %s finished; usageTotalUsd=%s", run_id, run.get("usageTotalUsd"))
            break
        if run["status"] in {"FAILED", "ABORTED", "TIMED-OUT"}:
            raise ApifyError(f"{actor_id} run {run_id} ended with {run['status']}")
        if time.monotonic() > deadline:
            raise ApifyError(f"{actor_id} run {run_id} did not finish within {timeout_s}s")
        time.sleep(poll_s)

    return _dataset_items(run["defaultDatasetId"], token)


def _dataset_items(dataset_id: str, token: str, page: int = 1000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = requests.get(
            f"{API}/datasets/{dataset_id}/items",
            params={"token": token, "format": "json", "clean": "true", "limit": page, "offset": offset},
            timeout=120,
        )
        _raise(resp)
        batch = resp.json()
        items.extend(batch)
        if len(batch) < page:
            return items
        offset += page


def _raise(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        raise ApifyError(f"Apify HTTP {resp.status_code}: {resp.text[:300]}")
