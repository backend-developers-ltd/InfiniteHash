import collections
import datetime
from typing import Any

import httpx
import structlog
from django.conf import settings

from infinite_hashes.validator.models import LuxorSnapshot

logger = structlog.get_logger(__name__)


async def get_hashrates_from_snapshots_async(
    subaccount_name: str,
    start: datetime.datetime,
    end: datetime.datetime,
) -> dict[str, list[list[int]]]:
    """Get hashrates from scraped Luxor snapshots.

    For each hotkey, groups snapshots by timestamp and collects all workers'
    hashrates for that minute as separate values, returning one sample (list) per minute.

    Args:
        subaccount_name: Luxor subaccount name
        start: Start datetime (inclusive)
        end: End datetime (inclusive)

    Returns:
        Dictionary mapping hotkeys to lists of hashrate samples (in H/s).
        Each sample is a list of individual worker hashrates at that timestamp.
        Worker names are truncated to first 48 chars (hotkey) for consistency.
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

    # Group by (hotkey, timestamp) and collect all workers' hashrates as separate values
    # This allows matching individual worker hashrates to individual commitments
    by_hotkey_time: dict[str, dict[datetime.datetime, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for snapshot in snapshots:
        hotkey = snapshot.worker_name[:48]
        # Collect each worker's hashrate separately for this timestamp
        by_hotkey_time[hotkey][snapshot.snapshot_time].append(snapshot.hashrate)

    # Convert to list of samples per hotkey (sorted by time)
    # Each sample is a list of worker hashrates at that timestamp
    hashrates: dict[str, list[list[int]]] = {}
    for hotkey, time_dict in by_hotkey_time.items():
        hashrates[hotkey] = [hrs for _ts, hrs in sorted(time_dict.items())]

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
        total_datapoints=sum(len(v) for v in hashrates.values()),
        samples_per_hotkey={k: len(v) for k, v in hashrates.items()},
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
            f"/v1/pool/workers-hashrate-efficiency/BTC/{subaccount_name}",
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
