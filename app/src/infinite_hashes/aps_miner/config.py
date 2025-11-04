"""
Configuration management for APScheduler-based miner.

Loads configuration from TOML file and provides structured access.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli


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
    hashrates: list[str]


@dataclass
class MinerConfig:
    """Complete miner configuration."""

    bittensor: BittensorConfig
    wallet: WalletConfig
    workers: WorkersConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinerConfig":
        """Create configuration from parsed TOML dictionary."""
        return cls(
            bittensor=BittensorConfig(**data["bittensor"]),
            wallet=WalletConfig(**data["wallet"]),
            workers=WorkersConfig(**data["workers"]),
        )

    @classmethod
    def load(cls, config_path: str | Path) -> "MinerConfig":
        """Load configuration from TOML file."""
        path = Path(config_path)
        with path.open("rb") as f:
            data = tomli.load(f)
        return cls.from_dict(data)
