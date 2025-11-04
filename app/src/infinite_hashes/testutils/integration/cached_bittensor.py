"""Test-only utilities for caching Bittensor connections to speed up tests."""

from __future__ import annotations

import asyncio
from typing import Any

# Save original Bittensor class before any monkey-patching
import turbobt

_original_bittensor_class = turbobt.Bittensor

# TEST-ONLY: Cache bittensor connections to avoid metadata re-parsing overhead
# Dictionary keyed by wallet identifier (hotkey) or None for walletless connections
_test_bittensor_connections: dict[str | None, Any] = {}
_test_connection_lock = None


def _get_wallet_key(wallet) -> str | None:
    """Get a unique key for caching wallet-configured connections."""
    if wallet is None:
        return None
    # Use hotkey address as unique identifier
    return wallet.hotkey.ss58_address if hasattr(wallet, "hotkey") else None


async def _get_cached_bittensor(wallet=None, network=None):
    """Get or create a cached Bittensor connection for test speedup.

    Args:
        wallet: Optional wallet to configure the connection with.
                If None, creates a walletless connection.
        network: Optional network string. If None, falls back to Django settings.

    Returns:
        Cached Bittensor connection (with wallet configured if provided)
    """
    global _test_bittensor_connections, _test_connection_lock

    if _test_connection_lock is None:
        _test_connection_lock = asyncio.Lock()

    wallet_key = _get_wallet_key(wallet)

    async with _test_connection_lock:
        if wallet_key not in _test_bittensor_connections:
            # Get network from parameter or Django settings
            if network is None:
                from django.conf import settings

                network = settings.BITTENSOR_NETWORK

            # Use original class to avoid infinite recursion during monkey-patching
            connection = _original_bittensor_class(network)
            await connection.__aenter__()

            # Configure wallet if provided
            if wallet is not None:
                connection.wallet = wallet

            _test_bittensor_connections[wallet_key] = connection

        return _test_bittensor_connections[wallet_key]


def with_cached_bittensor(func, event_loop, wallet=None, network=None, **kwargs):
    """Execute a function with cached bittensor connection if event loop is provided.

    This monkey-patches turbobt.Bittensor to return the cached connection,
    making the caching transparent to the function being called.

    Args:
        func: The function to execute
        event_loop: Optional event loop for test speedup
        wallet: Optional wallet for wallet-configured connections
        network: Optional network string (if None, uses Django settings)
        **kwargs: Arguments to pass to the function

    Returns:
        The return value from the function
    """
    import os

    # Only use caching in test mode with persistent loop
    if not (os.environ.get("TEST_DB_PATH") and event_loop):
        return func(**kwargs)

    async def _run_with_cache():
        bittensor = await _get_cached_bittensor(wallet=wallet, network=network)

        # Clear turbobt's RPC cache to get fresh data from simulator
        storage = getattr(bittensor.subtensor, "_transport", None)
        if storage:
            storage = getattr(storage, "_storage", None)
            if storage and hasattr(storage, "_cache"):
                storage._cache.clear()

        # Monkey-patch turbobt.Bittensor to return cached connection
        import turbobt

        original_bittensor_class = turbobt.Bittensor

        class CachedBittensor:
            def __init__(self, *args, **inner_kwargs):
                # If a wallet is passed to the constructor, cache by that wallet
                self._init_wallet = inner_kwargs.get("wallet")

            async def __aenter__(self):
                # If constructor received a wallet, get/create connection for that wallet
                if self._init_wallet:
                    conn = await _get_cached_bittensor(wallet=self._init_wallet, network=network)
                else:
                    conn = bittensor

                # Clear turbobt's RPC cache
                storage = getattr(conn.subtensor, "_transport", None)
                if storage:
                    storage = getattr(storage, "_storage", None)
                    if storage and hasattr(storage, "_cache"):
                        storage._cache.clear()

                return conn

            async def __aexit__(self, *args):
                pass  # Don't close the cached connection

        try:
            turbobt.Bittensor = CachedBittensor
            # Run the synchronous function in a thread to avoid Django async context issues
            return await asyncio.to_thread(func, event_loop=event_loop, **kwargs)
        finally:
            turbobt.Bittensor = original_bittensor_class

    return event_loop.run_until_complete(_run_with_cache())


async def close_cached_bittensor(wallet=None):
    """Close cached Bittensor connection(s).

    Args:
        wallet: Optional wallet. If provided, closes only the connection for that wallet.
                If None, closes all cached connections.
    """
    global _test_bittensor_connections

    if wallet is not None:
        # Close specific wallet connection
        wallet_key = _get_wallet_key(wallet)
        connection = _test_bittensor_connections.pop(wallet_key, None)
        if connection:
            await connection.__aexit__(None, None, None)
    else:
        # Close all connections
        for connection in _test_bittensor_connections.values():
            await connection.__aexit__(None, None, None)
        _test_bittensor_connections.clear()
