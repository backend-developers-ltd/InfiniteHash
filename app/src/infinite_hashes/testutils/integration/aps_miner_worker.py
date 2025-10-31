"""APS miner worker event loop helpers used by integration tests.

This is the APS miner equivalent of miner_worker.py, using APScheduler tasks
instead of Django/Celery tasks.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from infinite_hashes.aps_miner.tasks import compute_current_auction, ensure_worker_commitment
from infinite_hashes.testutils.integration.cached_bittensor import close_cached_bittensor, with_cached_bittensor

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


def _aps_miner_cmd_ping(_params: dict[str, Any], _context: dict[str, Any]) -> dict[str, Any]:
    return {"pong": True}


def _aps_miner_cmd_submit_commitment(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Submit commitment using APS miner task."""
    force = bool(params.get("force", False))
    config_path = context.get("config_path")
    if not config_path:
        raise ValueError("config_path not found in context")

    # Load network from TOML config for caching
    from infinite_hashes.aps_miner.config import MinerConfig

    config = MinerConfig.load(config_path)
    network = config.bittensor.network

    event_loop = context.get("event_loop")
    updated = with_cached_bittensor(
        ensure_worker_commitment, event_loop, network=network, config_path=config_path, force=force
    )
    return {"committed": bool(updated)}


def _aps_miner_cmd_compute_current_auction(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Compute current auction using APS miner task."""
    config_path = context.get("config_path")
    if not config_path:
        raise ValueError("config_path not found in context")

    # Load network from TOML config for caching
    from infinite_hashes.aps_miner.config import MinerConfig

    config = MinerConfig.load(config_path)
    network = config.bittensor.network

    event_loop = context.get("event_loop")
    result = with_cached_bittensor(compute_current_auction, event_loop, network=network, config_path=config_path)
    return result


def _aps_miner_cmd_get_config(_params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Get current config path (for debugging)."""
    return {"config_path": context.get("config_path")}


_APS_MINER_COMMAND_HANDLERS: dict[str, Any] = {
    "PING": _aps_miner_cmd_ping,
    "SUBMIT_COMMITMENT": _aps_miner_cmd_submit_commitment,
    "COMPUTE_CURRENT_AUCTION": _aps_miner_cmd_compute_current_auction,
    "GET_CONFIG": _aps_miner_cmd_get_config,
}


def run_aps_miner_event_loop(
    command_queue: Any,
    response_queue: Any,
    *,
    worker_id: int,
    context: dict[str, Any] | None = None,
) -> None:
    """Run APS miner worker event loop.

    Unlike the Django miner, this doesn't need database setup.
    Creates persistent event loop for caching Bittensor connection in tests.
    """
    context = dict(context or {})

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
            handler = _APS_MINER_COMMAND_HANDLERS.get(command_type)
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
