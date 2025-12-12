"""Helpers for querying already-scraped worker windows and reconciling proxy vs Luxor data.

Rules:
  a) If a worker exists in both sources, only the proxy hashrate is kept.
  b) If a worker exists in proxy but not Luxor (e.g., merged "not hotkeyed"), keep the proxy value.
  c) If a worker exists in Luxor but not proxy (not switched yet), keep the Luxor value.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Path fragment lives here; host/base comes from settings/env.
WORKERS_ENDPOINT = "/api/v1/workers"


def _build_workers_url(base_url: str) -> str:
    return base_url.rstrip("/") + WORKERS_ENDPOINT


def _iso_seconds(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.UTC)
    return ts.isoformat(timespec="seconds")


def _split_worker_id(worker_id: str) -> tuple[str, str]:
    """Split worker identifier into (hotkey, worker_suffix).

    Proxy workers: ``<subaccount>.<hotkey>.<worker_suffix>``.
    Luxor scraped workers: ``<hotkey>.<worker_suffix>``.
    Hotkey is truncated to 48 chars for consistency with snapshot parsing.
    """
    if not worker_id:
        return "", ""

    parts = worker_id.split(".")
    if len(parts) >= 3:
        hotkey = parts[1]
        worker_suffix = ".".join(parts[2:])
    elif len(parts) == 2:
        hotkey = parts[0]
        worker_suffix = parts[1]
    else:
        hotkey = worker_id
        worker_suffix = ""

    return hotkey[:48], worker_suffix


def records_to_worker_hashrates(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Convert worker records into hotkey -> worker_suffix -> hashrate (single value)."""
    out: dict[str, dict[str, int]] = {}
    for rec in records:
        hotkey, worker_suffix = _split_worker_id(rec.get("worker_id", ""))
        if not hotkey or not worker_suffix:
            continue
        hr = rec.get("hashrate") or 0
        try:
            hr_int = int(hr)
        except (ValueError, TypeError):
            continue
        out.setdefault(hotkey, {})[worker_suffix] = hr_int
    return out


async def fetch_workers_window(
    *,
    base_url: str | None = None,
    start: dt.datetime,
    end: dt.datetime,
    pattern: str = "*",
    limit: int = 500,
    offset: int = 0,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Call the workers API for a given time window (uses already-scraped data).

    The API is paginated; this helper fetches all pages starting from ``offset``.
    """
    if not base_url:
        return []

    url = _build_workers_url(base_url)

    params_base = {
        "pattern": pattern,
        "limit": limit,
        "start_time": _iso_seconds(start),
        "end_time": _iso_seconds(end),
    }

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
        close_client = True

    workers: list[dict[str, Any]] = []
    page_offset = offset

    try:
        while True:
            params = {**params_base, "offset": page_offset}
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            page_workers = payload.get("workers", []) or []
            workers.extend(page_workers)

            logger.debug(
                "Fetched workers window page",
                base_url=url,
                start=params["start_time"],
                end=params["end_time"],
                offset=page_offset,
                returned=len(page_workers),
                total=len(workers),
            )

            if len(page_workers) < limit:
                break

            page_offset += limit

        return workers
    finally:
        if close_client:
            await client.aclose()


async def fetch_proxy_hashrate_data(
    *,
    start: dt.datetime,
    end: dt.datetime,
    proxy_url: str | None,
    pattern: str = "*",
    timeout: float = 10.0,
) -> dict[str, dict[str, int]]:
    """Fetch proxy workers window and convert to worker-level hashrates by hotkey.

    Proxy is expected to return a single value per worker for the window.
    """
    if not proxy_url:
        return {}

    try:
        proxy_records = await fetch_workers_window(
            base_url=proxy_url,
            start=start,
            end=end,
            pattern=pattern,
            timeout=timeout,
        )
        return records_to_worker_hashrates(proxy_records)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to fetch proxy workers window",
            proxy_url=proxy_url,
            start=start.isoformat(),
            end=end.isoformat(),
            error=str(exc),
        )
        return {}


def merge_worker_hashrates(
    *, luxor_hashrates: Mapping[str, Mapping[str, int]], proxy_hashrates: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    """Apply rules a/b/c to worker-level averaged hashrates."""
    merged: dict[str, dict[str, int]] = {}

    for hotkey, workers in luxor_hashrates.items():
        merged[hotkey] = {**workers}

    for hotkey, workers in proxy_hashrates.items():
        dest = merged.setdefault(hotkey, {})
        for worker_suffix, avg in workers.items():
            dest[worker_suffix] = avg  # proxy wins or adds

    return merged
