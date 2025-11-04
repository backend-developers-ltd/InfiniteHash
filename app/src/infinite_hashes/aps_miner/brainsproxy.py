"""
Integration helpers for Braiins Farm Proxy configuration.

This module updates the routing weights inside the active Braiins configuration
profile and signals the configurator sidecar to reapply the profile.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import structlog
import tomlkit

from .models import BidResult

logger = structlog.get_logger(__name__)

ACTIVE_PROFILE_PATH = Path(os.environ.get("BRAIINSPROXY_ACTIVE_PROFILE", "/root/src/brainsproxy/active_profile.toml"))
RELOAD_SENTINEL_PATH = Path(os.environ.get("BRAIINSPROXY_RELOAD_SENTINEL", "/root/src/brainsproxy/.reconfigure"))

GOAL_LUXOR = "InfiniteHashLuxorGoal"
GOAL_DEFAULT = "MinerDefaultGoal"
WEIGHT_SCALE = 2**32


def _sum_hashrates(bids: Iterable[BidResult]) -> float:
    total = 0.0
    for bid in bids:
        try:
            total += float(bid.hashrate)
        except (TypeError, ValueError):
            logger.warning("Failed to parse bid hashrate for Braiins integration", hashrate=bid.hashrate)
    return total


def _find_goal(doc: tomlkit.TOMLDocument, goal_name: str) -> tomlkit.items.Table | None:
    routing = doc.get("routing")
    if not isinstance(routing, list):
        return None

    for route in routing:
        goals = route.get("goal")
        if not isinstance(goals, list):
            continue
        for goal in goals:
            if isinstance(goal, dict) and goal.get("name") == goal_name:
                return goal
    return None


def update_routing_weights(won_bids: Iterable[BidResult], lost_bids: Iterable[BidResult]) -> None:
    """
    Update Braiins routing weights based on APS miner outcome.

    Args:
        won_bids: Iterable of bids that won the current auction window.
        lost_bids: Iterable of bids that lost the current auction window.
    """
    if not ACTIVE_PROFILE_PATH.exists():
        logger.info("Braiins profile not present – skipping update", profile=str(ACTIVE_PROFILE_PATH))
        return

    try:
        with ACTIVE_PROFILE_PATH.open("r", encoding="utf-8") as f:
            doc = tomlkit.load(f)
    except Exception:
        logger.exception("Unable to load Braiins profile")
        return

    luxor_goal = _find_goal(doc, GOAL_LUXOR)
    default_goal = _find_goal(doc, GOAL_DEFAULT)

    if luxor_goal is None or default_goal is None:
        logger.warning("Expected Braiins goals not found", luxor_found=bool(luxor_goal), default_found=bool(default_goal))
        return

    won_total = _sum_hashrates(won_bids)
    lost_total = _sum_hashrates(lost_bids)
    total = won_total + lost_total

    if total <= 0:
        logger.debug("No hashrate totals available for Braiins routing update; skipping")
        return

    luxor_weight = int(round(WEIGHT_SCALE * (won_total / total)))
    luxor_weight = max(0, min(WEIGHT_SCALE, luxor_weight))
    default_weight = WEIGHT_SCALE - luxor_weight
    default_weight = max(0, default_weight)

    updated = False

    old_luxor = luxor_goal.get("hr_weight")
    if old_luxor != luxor_weight:
        luxor_goal["hr_weight"] = luxor_weight
        updated = True
        logger.info(
            "Updated Braiins Luxor goal weight",
            old_weight=old_luxor,
            new_weight=luxor_weight,
        )

    old_default = default_goal.get("hr_weight")
    if old_default != default_weight:
        default_goal["hr_weight"] = default_weight
        updated = True
        logger.info(
            "Updated Braiins default goal weight",
            old_weight=old_default,
            new_weight=default_weight,
        )

    if not updated:
        logger.debug("Braiins routing weights already up to date")
        return

    try:
        with ACTIVE_PROFILE_PATH.open("w", encoding="utf-8") as f:
            tomlkit.dump(doc, f)
    except Exception:
        logger.exception("Failed to persist Braiins profile updates")
        return

    try:
        RELOAD_SENTINEL_PATH.touch()
    except Exception:
        logger.exception("Failed to signal Braiins configurator", sentinel=str(RELOAD_SENTINEL_PATH))
