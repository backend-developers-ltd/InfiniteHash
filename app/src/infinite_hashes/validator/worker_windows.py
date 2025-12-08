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

DEFAULT_WORKERS_API = "http://172.236.7.39:8000/api/v1/workers"


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
    base_url: str = DEFAULT_WORKERS_API,
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
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            page_workers = payload.get("workers", []) or []
            workers.extend(page_workers)

            logger.debug(
                "Fetched workers window page",
                base_url=base_url,
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


def reconcile_workers(
    *,
    luxor_workers: Iterable[Mapping[str, Any]],
    proxy_workers: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge Luxor and proxy workers per rules a/b/c.

    Returns mapping worker_id -> record with a 'source' field set to 'proxy' or 'luxor'.
    """
    combined: dict[str, dict[str, Any]] = {}

    # Prefer proxy when overlapping (rule a) and include proxy-only (rule b)
    for worker in proxy_workers:
        hotkey, _full_worker_id = _split_worker_id(worker.get("worker_id", ""))
        if not hotkey:
            continue
        combined[hotkey] = {**worker, "source": "proxy"}

    # Add Luxor-only (rule c); skip if proxy already present
    for worker in luxor_workers:
        hotkey, _full_worker_id = _split_worker_id(worker.get("worker_id", ""))
        if not hotkey or hotkey in combined:
            continue
        combined[hotkey] = {**worker, "source": "luxor"}

    return combined


async def fetch_reconciled_workers(
    *,
    start: dt.datetime,
    end: dt.datetime,
    luxor_base_url: str = DEFAULT_WORKERS_API,
    proxy_base_url: str | None = None,
    pattern: str = "*",
    limit: int = 500,
    offset: int = 0,
    timeout: float = 10.0,
) -> dict[str, dict[str, Any]]:
    """Fetch Luxor and proxy worker windows and merge according to rules a/b/c."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        luxor = await fetch_workers_window(
            base_url=luxor_base_url,
            start=start,
            end=end,
            pattern=pattern,
            limit=limit,
            offset=offset,
            timeout=timeout,
            client=client,
        )

        proxy: list[dict[str, Any]] = []
        if proxy_base_url:
            proxy = await fetch_workers_window(
                base_url=proxy_base_url,
                start=start,
                end=end,
                pattern=pattern,
                limit=limit,
                offset=offset,
                timeout=timeout,
                client=client,
            )

    merged = reconcile_workers(luxor_workers=luxor, proxy_workers=proxy)
    logger.debug(
        "Reconciled workers from luxor and proxy",
        start=_iso_seconds(start),
        end=_iso_seconds(end),
        luxor_count=len(luxor),
        proxy_count=len(proxy),
        merged_count=len(merged),
    )
    return merged


async def fetch_proxy_hashrate_data(
    *,
    start: dt.datetime,
    end: dt.datetime,
    proxy_url: str | None,
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
