"""Validator worker event loop helpers used by integration tests."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from django.utils import timezone

from infinite_hashes.testutils.integration.cached_bittensor import close_cached_bittensor, with_cached_bittensor
from infinite_hashes.validator import tasks as price_tasks
from infinite_hashes.validator.models import AuctionResult, PriceObservation, WeightsBatch
from infinite_hashes.validator.tasks import (
    calculate_auction_weights,
    calculate_weights,
    process_auctions,
    set_auction_weights,
    set_weights,
)

logger = logging.getLogger(__name__)


def _response(
    worker_id: int,
    command: str,
    *,
    success: bool,
    data: Any | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "RESPONSE",
        "worker_id": worker_id,
        "command": command,
        "success": success,
    }
    if data is not None:
        payload["data"] = data
    if error is not None:
        payload["error"] = error
    return payload


def _validator_cmd_ping(_params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    return {"pong": True}


def _validator_cmd_scrape_prices(params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    now = timezone.now()

    # Default prices
    samples = {
        "TAO_USDC": (45_000_000_000_000_000_000, "binance"),  # 45 USDC per TAO
        "ALPHA_TAO": (250_000_000_000_000_000, "subtensor"),  # 0.25 ALPHA per TAO
        "HASHP_USDC": (50_000_000_000_000_000_000, "hashrateindex"),  # 50 USDC per PH per day
    }

    # Allow overriding specific prices (e.g., ALPHA_TAO for budget control)
    for metric in ["TAO_USDC", "ALPHA_TAO", "HASHP_USDC"]:
        if metric in params:
            price_fp18 = params[metric]
            source = samples[metric][1]
            samples[metric] = (price_fp18, source)

    created = 0
    for metric, (price_fp18, source) in samples.items():
        PriceObservation.objects.create(
            metric=metric,
            price_fp18=price_fp18,
            observed_at=now,
            source=source,
        )
        created += 1
    return {"created": created}


def _validator_cmd_publish_local_commitment(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    event_loop = context.get("event_loop")
    # Note: publish_local_commitment creates its own wallet internally from settings
    # The wallet is cached automatically by turbobt.Bittensor monkey-patching
    result = with_cached_bittensor(price_tasks.publish_local_commitment, event_loop)
    return {"success": True, "result": result}


def _validator_cmd_process_auction(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    event_loop = context.get("event_loop")
    before = set(AuctionResult.objects.values_list("id", flat=True))
    processed = with_cached_bittensor(process_auctions, event_loop)

    if processed <= 0:
        return {"processed": 0, "new_results": []}
    new_results = [
        {
            "id": result.id,
            "start_block": int(result.start_block),
            "end_block": int(result.end_block),
            "epoch_start": int(result.epoch_start),
            "winners": result.winners,
        }
        for result in AuctionResult.objects.order_by("-end_block")
        if result.id not in before
    ]
    return {"processed": processed, "new_results": new_results}


def _validator_cmd_get_db_state(_params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    batches = [
        {
            "id": batch.id,
            "epoch_start": int(batch.epoch_start),
            "mechanism_id": int(batch.mechanism_id),
            "scored": bool(batch.scored),
            "should_be_scored": bool(batch.should_be_scored),
        }
        for batch in WeightsBatch.objects.order_by("-created_at")[:20]
    ]
    results = [
        {
            "start_block": int(result.start_block),
            "end_block": int(result.end_block),
            "epoch_start": int(result.epoch_start),
            "winners": result.winners,
        }
        for result in AuctionResult.objects.order_by("-end_block")[:20]
    ]
    return {"weights_batches": batches, "auction_results": results}


def _validator_cmd_get_auction_results(_params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    payload = [
        {
            "start_block": int(result.start_block),
            "end_block": int(result.end_block),
            "epoch_start": int(result.epoch_start),
            "winners": result.winners,
        }
        for result in AuctionResult.objects.order_by("start_block")
    ]
    return {"results": payload}


def _validator_cmd_calculate_weights(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    event_loop = context.get("event_loop")
    weights = with_cached_bittensor(calculate_weights, event_loop)
    return {"weights": weights}


def _validator_cmd_set_weights(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    event_loop = context.get("event_loop")
    # Note: set_weights creates its own wallet internally from settings
    # The wallet is cached automatically by turbobt.Bittensor monkey-patching
    updated = with_cached_bittensor(set_weights, event_loop)
    return {"updated": bool(updated)}


def _validator_cmd_calculate_auction_weights(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    event_loop = context.get("event_loop")
    weights = with_cached_bittensor(calculate_auction_weights, event_loop)
    return {"weights": weights}


def _validator_cmd_set_auction_weights(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    event_loop = context.get("event_loop")
    # Note: set_auction_weights creates its own wallet internally from settings
    # The wallet is cached automatically by turbobt.Bittensor monkey-patching
    updated = with_cached_bittensor(set_auction_weights, event_loop)
    return {"updated": bool(updated)}


def _validator_cmd_insert_hashrate_samples(params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """Insert hashrate snapshot data into validator's own database."""
    from datetime import datetime

    from django.db import transaction
    from django.utils import timezone

    from infinite_hashes.validator.models import LuxorSnapshot

    snapshots_data = params.get("snapshots", [])
    if not snapshots_data:
        return {"created": 0}

    # Deserialize datetime strings and create LuxorSnapshot objects
    snapshots = []
    for data in snapshots_data:
        # Parse ISO format datetimes - these should be timezone-aware
        snapshot_time = datetime.fromisoformat(data["snapshot_time"])
        last_updated = datetime.fromisoformat(data["last_updated"])

        # Ensure timezone-aware (Django requires it when USE_TZ=True)
        if timezone.is_naive(snapshot_time):
            snapshot_time = timezone.make_aware(snapshot_time, timezone.utc)
        if timezone.is_naive(last_updated):
            last_updated = timezone.make_aware(last_updated, timezone.utc)

        snapshots.append(
            LuxorSnapshot(
                snapshot_time=snapshot_time,
                subaccount_name=data["subaccount_name"],
                worker_name=data["worker_name"],
                hashrate=data["hashrate"],
                efficiency=data["efficiency"],
                revenue=data["revenue"],
                last_updated=last_updated,
            )
        )

    # Bulk insert in validator's database (with correct schema) and commit
    # Use atomic block to ensure the data is committed and visible to subsequent queries
    with transaction.atomic():
        created_objs = LuxorSnapshot.objects.bulk_create(snapshots)

    return {"created": len(created_objs)}


def _validator_cmd_insert_scraping_events(params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    """Insert scraping event heartbeats into validator's own database."""
    from datetime import datetime

    from django.db import transaction
    from django.utils import timezone

    from infinite_hashes.validator.models import ValidatorScrapingEvent

    events_data = params.get("events", [])
    if not events_data:
        return {"created": 0}

    # Deserialize datetime strings and create ValidatorScrapingEvent objects
    events = []
    for data in events_data:
        # Parse ISO format datetimes - these should be timezone-aware
        scraped_at = datetime.fromisoformat(data["scraped_at"])

        # Ensure timezone-aware (Django requires it when USE_TZ=True)
        if timezone.is_naive(scraped_at):
            scraped_at = timezone.make_aware(scraped_at, timezone.utc)

        events.append(
            ValidatorScrapingEvent(
                scraped_at=scraped_at,
                block_number=data["block_number"],
                worker_count=data["worker_count"],
            )
        )

    # Bulk insert in validator's database (with correct schema) and commit
    # Use atomic block to ensure the data is committed and visible to subsequent queries
    with transaction.atomic():
        created_objs = ValidatorScrapingEvent.objects.bulk_create(events)

    return {"created": len(created_objs)}


_VALIDATOR_COMMAND_HANDLERS: dict[str, Any] = {
    "PING": _validator_cmd_ping,
    "SCRAPE_PRICES": _validator_cmd_scrape_prices,
    "PUBLISH_LOCAL_COMMITMENT": _validator_cmd_publish_local_commitment,
    "PROCESS_AUCTION": _validator_cmd_process_auction,
    "GET_DB_STATE": _validator_cmd_get_db_state,
    "GET_AUCTION_RESULTS": _validator_cmd_get_auction_results,
    "CALCULATE_WEIGHTS": _validator_cmd_calculate_weights,
    "SET_WEIGHTS": _validator_cmd_set_weights,
    "CALCULATE_AUCTION_WEIGHTS": _validator_cmd_calculate_auction_weights,
    "SET_AUCTION_WEIGHTS": _validator_cmd_set_auction_weights,
    "INSERT_HASHRATE_SAMPLES": _validator_cmd_insert_hashrate_samples,
    "INSERT_SCRAPING_EVENTS": _validator_cmd_insert_scraping_events,
}


def run_validator_event_loop(
    command_queue: Any,
    response_queue: Any,
    *,
    worker_id: int,
    context: dict[str, Any] | None = None,
) -> None:
    context = dict(context or {})

    db_schema = context.get("db_schema")
    if db_schema:
        from django.conf import settings
        from django.db import connections

        for db_config in settings.DATABASES.values():
            if "OPTIONS" not in db_config:
                db_config["OPTIONS"] = {}
            db_config["OPTIONS"]["options"] = f"-c search_path={db_schema}"
        for conn in connections.all():
            conn.close()

    # TEST SPEEDUP: Create persistent event loop for caching Bittensor connection
    import os

    if os.environ.get("TEST_DB_PATH"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        context["event_loop"] = loop

    try:
        while True:
            command = command_queue.get()
            if not isinstance(command, dict):
                continue
            command_type = command.get("type")
            if command_type == "STOP":
                response_queue.put(_response(worker_id, "STOP", success=True))
                break
            if not isinstance(command_type, str):
                response_queue.put(_response(worker_id, str(command_type), success=False, error="invalid command"))
                continue
            handler = _VALIDATOR_COMMAND_HANDLERS.get(command_type)
            if handler is None:
                response_queue.put(_response(worker_id, command_type, success=False, error="unknown command"))
                continue
            params = command.get("params") or {}
            try:
                result = handler(params, context)
                response_queue.put(_response(worker_id, command_type, success=True, data=result))
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{str(exc)}\n{traceback.format_exc()}"
                response_queue.put(_response(worker_id, command_type, success=False, error=error_msg))
    finally:
        # Clean up the event loop if we created one
        if "event_loop" in context:
            loop = context["event_loop"]
            # Close the cached Bittensor connection if it exists
            try:
                loop.run_until_complete(close_cached_bittensor())
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close cached Bittensor connection")
            loop.close()
