"""
Configuration management for APScheduler-based miner.

Loads configuration from TOML file and provides structured access.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli

from infinite_hashes.consensus.bidding import (
    MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18,
    MAX_BIDDING_COMMITMENT_WORKERS,
)
from infinite_hashes.consensus.price import _parse_decimal_to_fp18_int


@dataclass
class BittensorConfig:
    """Bittensor network configuration."""

    network: str
    netuid: int


@dataclass
class WalletConfig:
    """Wallet configuration."""

    name: str
    hotkey_name: str
    directory: str

    @property
    def directory_path(self) -> Path:
        """Resolve wallet directory path with ~ expansion."""
        return Path(os.path.expanduser(self.directory))


@dataclass
class WorkersConfig:
    """Workers configuration."""

    price_multiplier: str
    hashrates: list[str] | None = None
    worker_sizes: dict[str, int] | None = None

    def total_workers(self) -> int:
        if self.worker_sizes:
            return sum(self.worker_sizes.values())
        return len(self.hashrates or [])


@dataclass
class MinerConfig:
    """Complete miner configuration."""

    bittensor: BittensorConfig
    wallet: WalletConfig
    workers: WorkersConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinerConfig":
        """Create configuration from parsed TOML dictionary."""
        workers_data = data["workers"]

        hashrates_raw = workers_data.get("hashrates")
        worker_sizes_raw = workers_data.get("worker_sizes")

        if hashrates_raw is not None and worker_sizes_raw is not None:
            raise ValueError("workers config must specify only one of 'hashrates' or 'worker_sizes'")
        if hashrates_raw is None and worker_sizes_raw is None:
            raise ValueError(
                "workers config must specify 'hashrates' (list for v1, dict for v2) or 'worker_sizes' (v2)"
            )

        hashrates: list[str] | None = None
        worker_sizes: dict[str, Any] | None = None

        if worker_sizes_raw is not None:
            if not isinstance(worker_sizes_raw, dict):
                raise ValueError("workers.worker_sizes must be a dict of hashrate_str -> count_int")
            worker_sizes = worker_sizes_raw
        else:
            if isinstance(hashrates_raw, list):
                hashrates = hashrates_raw
            elif isinstance(hashrates_raw, dict):
                worker_sizes = hashrates_raw
            else:
                raise ValueError("workers.hashrates must be a list (v1) or dict (v2)")

        normalized_worker_sizes: dict[str, int] | None = None
        if worker_sizes is not None:
            normalized_worker_sizes = {}
            for k, v in worker_sizes.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError("workers.worker_sizes has empty/non-string hashrate key")
                hr_key = k.strip()
                try:
                    hr_fp18 = _parse_decimal_to_fp18_int(hr_key)
                except ValueError as e:
                    raise ValueError(f"invalid workers.worker_sizes hashrate: {hr_key}") from e
                if hr_fp18 > MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18:
                    raise ValueError(f"workers.worker_sizes hashrate exceeds v2 max worker size (0.45 PH): {hr_key}")
                try:
                    c = int(v)
                except (TypeError, ValueError):
                    raise ValueError(f"invalid workers.worker_sizes count for {hr_key}: {v!r}")
                if c <= 0:
                    raise ValueError(f"workers.worker_sizes count must be > 0 for {hr_key}")
                normalized_worker_sizes[hr_key] = c
            if not normalized_worker_sizes:
                raise ValueError("workers.worker_sizes is empty or invalid")
            total_workers = sum(normalized_worker_sizes.values())
            if total_workers > MAX_BIDDING_COMMITMENT_WORKERS:
                raise ValueError(
                    f"workers.worker_sizes total worker count exceeds v2 commitment limit ({MAX_BIDDING_COMMITMENT_WORKERS}): {total_workers}"
                )

        if hashrates is not None:
            if not isinstance(hashrates, list):
                raise ValueError("workers.hashrates must be a list of strings")
            normalized_hashrates: list[str] = []
            for hr in hashrates:
                if not isinstance(hr, str) or not hr.strip():
                    raise ValueError("workers.hashrates contains empty/non-string hashrate")
                hr_s = hr.strip()
                try:
                    _parse_decimal_to_fp18_int(hr_s)
                except ValueError as e:
                    raise ValueError(f"invalid workers.hashrates hashrate: {hr_s}") from e
                normalized_hashrates.append(hr_s)
            if not normalized_hashrates:
                raise ValueError("workers.hashrates is empty")
            hashrates = normalized_hashrates

        workers_config = WorkersConfig(
            price_multiplier=workers_data["price_multiplier"],
            hashrates=hashrates,
            worker_sizes=normalized_worker_sizes,
        )
        return cls(
            bittensor=BittensorConfig(**data["bittensor"]),
            wallet=WalletConfig(**data["wallet"]),
            workers=workers_config,
        )

    @classmethod
    def load(cls, config_path: str | Path) -> "MinerConfig":
        """Load configuration from TOML file."""
        path = Path(config_path)
        with path.open("rb") as f:
            data = tomli.load(f)
        return cls.from_dict(data)
