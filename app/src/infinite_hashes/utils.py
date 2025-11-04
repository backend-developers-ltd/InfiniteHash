"""Common utilities for the InfiniteHash application."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro_func: Callable[..., Coroutine[Any, Any, T]], *args, event_loop=None, **kwargs) -> T:
    """Run an async function, using event_loop if provided, otherwise async_to_sync.

    This helper properly handles the case where an event loop is already running,
    using run_coroutine_threadsafe instead of run_until_complete.

    Args:
        coro_func: The async function to call
        *args: Positional arguments for the function
        event_loop: Optional event loop for test speedup
        **kwargs: Keyword arguments for the function

    Returns:
        The return value from the async function
    """
    if event_loop:
        coro = coro_func(*args, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, event_loop)
        return future.result()
    else:
        from asgiref.sync import async_to_sync

        return async_to_sync(coro_func)(*args, **kwargs)
