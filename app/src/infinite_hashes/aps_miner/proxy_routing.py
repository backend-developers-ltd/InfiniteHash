"""
Integration helpers for InfiniteHash Proxy pool routing.

This module updates subnet pool target hashrate in pools.toml so winning bids
receive an absolute allocation while private pools can consume the remainder.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import structlog
import tomlkit

from .models import BidResult

logger = structlog.get_logger(__name__)

DEFAULT_POOLS_CONFIG_PATH = "/root/src/proxy/pools.toml"
DEFAULT_SUBNET_POOL_NAME = "central-proxy"
DEFAULT_RELOAD_SENTINEL_PATH = "/root/src/proxy/.reload-ihp"


def pools_config_path() -> Path:
    configured_path = os.environ.get("APS_MINER_POOLS_CONFIG_PATH", DEFAULT_POOLS_CONFIG_PATH)
    return Path(configured_path)


def reload_sentinel_path() -> Path:
    configured_path = os.environ.get("APS_MINER_IHP_RELOAD_SENTINEL", DEFAULT_RELOAD_SENTINEL_PATH)
    return Path(configured_path)


def _sum_hashrates_ph(bids: Iterable[BidResult]) -> float:
    total = 0.0
    for bid in bids:
        try:
            total += float(bid.hashrate)
        except (TypeError, ValueError):
            logger.warning("Failed to parse bid hashrate for proxy routing", hashrate=bid.hashrate)
    return total


def _format_target_hashrate_from_ph(total_ph: float) -> str:
    target_th = max(total_ph, 0.0) * 1000.0
    target_str = f"{target_th:.3f}".rstrip("0").rstrip(".")
    if not target_str:
        target_str = "0"
    return f"{target_str}TH/s"


def _is_subnet_pool(pool: dict, subnet_pool_name: str) -> bool:
    pool_name = str(pool.get("name", "")).strip().lower()
    return pool_name == subnet_pool_name


def update_subnet_target_hashrate(won_bids: Iterable[BidResult], _lost_bids: Iterable[BidResult]) -> None:
    """
    Update the subnet pool target hashrate in pools.toml.

    Won hashrate is assigned to subnet pools as absolute `target_hashrate`.
    Remaining hashrate continues to flow to private pools through their weights.
    """
    config_path = pools_config_path()
    if not config_path.exists():
        logger.info("InfiniteHash proxy pools config not present - skipping update", path=str(config_path))
        return

    subnet_pool_name = os.environ.get("APS_MINER_SUBNET_POOL_NAME", DEFAULT_SUBNET_POOL_NAME).strip().lower()
    target_hashrate = _format_target_hashrate_from_ph(_sum_hashrates_ph(won_bids))

    try:
        with config_path.open("r", encoding="utf-8") as f:
            doc = tomlkit.load(f)
    except Exception:
        logger.exception("Unable to load InfiniteHash proxy pools config", path=str(config_path))
        return

    pools_section = doc.get("pools")
    if not isinstance(pools_section, dict):
        logger.warning("Missing [pools] section in proxy config", path=str(config_path))
        return

    main_pools = pools_section.get("main")
    if not isinstance(main_pools, list):
        logger.warning("Missing [[pools.main]] entries in proxy config", path=str(config_path))
        return

    updated = False
    matched = 0
    for pool in main_pools:
        if not isinstance(pool, dict):
            continue
        if not _is_subnet_pool(pool, subnet_pool_name):
            continue

        matched += 1
        old_target_hashrate = pool.get("target_hashrate")
        if old_target_hashrate != target_hashrate:
            pool["target_hashrate"] = target_hashrate
            updated = True

        if "weight" in pool:
            del pool["weight"]
            updated = True

    if matched == 0:
        logger.warning(
            "No subnet pool entries matched for target hashrate update",
            name=subnet_pool_name,
            path=str(config_path),
        )
        return

    if not updated:
        logger.debug("Subnet target hashrate already up to date", target_hashrate=target_hashrate)
        return

    try:
        with config_path.open("w", encoding="utf-8") as f:
            tomlkit.dump(doc, f)
    except Exception:
        logger.exception("Failed to persist proxy pools config updates", path=str(config_path))
        return

    sentinel_path = reload_sentinel_path()
    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.touch()
    except Exception:
        logger.exception("Failed to signal IHP proxy reload", sentinel=str(sentinel_path))

    logger.info(
        "Updated subnet target hashrate",
        name=subnet_pool_name,
        target_hashrate=target_hashrate,
        matched_pools=matched,
    )
