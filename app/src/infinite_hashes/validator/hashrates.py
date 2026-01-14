import collections
import datetime
from typing import Any

import httpx
import structlog
from django.conf import settings

from infinite_hashes.validator.models import LuxorSnapshot
from infinite_hashes.validator.worker_windows import _split_worker_id

logger = structlog.get_logger(__name__)


async def get_hashrates_from_snapshots_async(
    subaccount_name: str,
    start: datetime.datetime,
    end: datetime.datetime,
) -> dict[str, dict[str, int]]:
    """Get hashrates from scraped Luxor snapshots as averaged per (hotkey, worker).

    For each hotkey, groups snapshots by timestamp and collects each worker's
    hashrate, returning the average over the window per worker.
    """
    logger.debug(
        "Querying scraped Luxor snapshots",
        subaccount=subaccount_name,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    # Query snapshots in the time window
    snapshots = []
    async for snapshot in LuxorSnapshot.objects.filter(
        subaccount_name=subaccount_name,
        snapshot_time__gte=start,
        snapshot_time__lte=end,
    ).order_by("worker_name", "snapshot_time"):
        snapshots.append(snapshot)

    logger.debug(
        "Found snapshots in time range",
        subaccount=subaccount_name,
        count=len(snapshots),
        sample_workers=list(set(s.worker_name for s in snapshots))[:3] if snapshots else [],
        query_range=(start.isoformat(), end.isoformat()),
    )

    # Group by (hotkey, timestamp) and collect hashrates per worker
    by_hotkey_time: dict[str, dict[datetime.datetime, dict[str, int]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )

    for snapshot in snapshots:
        hotkey, worker_suffix = _split_worker_id(snapshot.worker_name)
        if not hotkey or not worker_suffix:
            continue
        by_hotkey_time[hotkey][snapshot.snapshot_time][worker_suffix] = snapshot.hashrate

    # Build averages per worker
    hashrates: dict[str, dict[str, int]] = {}
    for hotkey, time_dict in by_hotkey_time.items():
        worker_sums: dict[str, int] = collections.defaultdict(int)
        worker_counts: dict[str, int] = collections.defaultdict(int)
        for ts_data in time_dict.values():
            for worker_suffix, hr in ts_data.items():
                worker_sums[worker_suffix] += hr
                worker_counts[worker_suffix] += 1

        worker_avg: dict[str, int] = {}
        for worker_suffix, total in worker_sums.items():
            count = worker_counts.get(worker_suffix, 0)
            if count:
                worker_avg[worker_suffix] = int(total / count)

        if worker_avg:
            hashrates[hotkey] = worker_avg

    if not hashrates and snapshots:
        logger.warning(
            "Snapshots found but no hashrates extracted",
            snapshots_count=len(snapshots),
            sample_snapshot={"worker": snapshots[0].worker_name, "hotkey_extracted": snapshots[0].worker_name[:48]}
            if snapshots
            else None,
        )

    logger.debug(
        "Retrieved hashrates from snapshots",
        subaccount=subaccount_name,
        hotkeys=len(hashrates),
        workers_per_hotkey={k: len(v) for k, v in hashrates.items()},
        total_datapoints=sum(1 for workers in hashrates.values() for _v in workers.values()),
        hotkeys_list=list(hashrates.keys())[:5],
    )

    return hashrates


async def get_hashrates_async(
    subaccount_name: str,
    start: datetime.datetime,
    end: datetime.datetime,
    page_size: int = 100,
    tick_size: Any | None = None,
) -> dict[str, list[int]]:
    """Fetch average hashrates per worker hotkey over [start, end].

    Args:
        subaccount_name: Luxor subaccount name
        start: Start datetime
        end: End datetime
        page_size: Number of results per page
        tick_size: Optional tick size with `.label` and `.timedelta` attributes

    Returns:
        Mapping of hotkey -> list[int] of H/s samples (averaged per tick)
    """
    # Look up API key for this subaccount
    api_key = settings.LUXOR_API_KEY_BY_SUBACCOUNT.get(subaccount_name)
    if not api_key:
        logger.error(
            "No API key configured for subaccount",
            subaccount=subaccount_name,
            configured_subaccounts=list(settings.LUXOR_API_KEY_BY_SUBACCOUNT.keys()),
        )
        raise ValueError(f"No API key configured for subaccount: {subaccount_name}")

    hashrates: dict[str, list[int]] = collections.defaultdict(list)

    if tick_size is None:
        tick_label = "1h"
        tick_delta = datetime.timedelta(hours=1)
    else:
        tick_label = getattr(tick_size, "label", "1h")
        tick_delta = getattr(tick_size, "timedelta", datetime.timedelta(hours=1))

    logger.info(
        "Validator: Querying Luxor for hashrates",
        subaccount=subaccount_name,
        start=start.isoformat(),
        end=end.isoformat(),
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        tick_label=tick_label,
    )

    async with httpx.AsyncClient(
        base_url=settings.LUXOR_API_URL,
        headers={
            "Authorization": api_key,
        },
    ) as luxor:
        samples = int((end - start) / tick_delta) if tick_delta.total_seconds() else 1

        page_url = httpx.URL(
            f"/v2/pool/workers-hashrate-efficiency/BTC/{subaccount_name}",
            params={
                "end_date": end.date().isoformat(),
                "page_number": 1,
                "page_size": page_size,
                "start_date": start.date().isoformat(),
                "tick_size": tick_label,
            },
        )

        while True:
            response = await luxor.get(page_url)
            response.raise_for_status()
            response_json = response.json()

            by_worker = response_json.get("hashrate_efficiency_revenue", {})
            for worker_name, worker_hashrates in by_worker.items():
                worker_hotkey = worker_name[:48]
                worker_vals = [
                    int(h.get("hashrate", 0)) for h in worker_hashrates if h.get("date_time", "") >= start.isoformat()
                ]

                if len(worker_vals) < samples:
                    worker_vals += [0] * (samples - len(worker_vals))

                avg = int(sum(worker_vals) / max(1, len(worker_vals)))
                hashrates[worker_hotkey].append(avg)

            next_url = response_json.get("pagination", {}).get("next_page_url")
            if not next_url:
                break
            page_url = next_url

    logger.info(
        "Validator: Received hashrates from Luxor",
        subaccount=subaccount_name,
        hotkeys=list(hashrates.keys()),
        samples_per_hotkey={k: len(v) for k, v in hashrates.items()},
        total_hashrates=sum(len(v) for v in hashrates.values()),
    )

    return hashrates
