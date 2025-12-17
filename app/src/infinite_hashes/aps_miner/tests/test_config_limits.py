from __future__ import annotations

import pytest

from infinite_hashes.aps_miner.config import MinerConfig
from infinite_hashes.consensus.bidding import MAX_BIDDING_COMMITMENT_WORKERS


def _base_config() -> dict[str, object]:
    return {
        "bittensor": {"network": "finney", "netuid": 89},
        "wallet": {"name": "default", "hotkey_name": "default", "directory": "~/.bittensor/wallets"},
        "workers": {"price_multiplier": "1.0"},
    }


def test_worker_sizes_total_count_limit_enforced() -> None:
    cfg = _base_config()
    cfg["workers"] = {
        "price_multiplier": "1.0",
        "worker_sizes": {"0.1": MAX_BIDDING_COMMITMENT_WORKERS + 1},
    }
    with pytest.raises(ValueError, match="total worker count exceeds v2 commitment limit"):
        MinerConfig.from_dict(cfg)  # type: ignore[arg-type]


def test_hashrates_dict_total_count_limit_enforced() -> None:
    cfg = _base_config()
    cfg["workers"] = {
        "price_multiplier": "1.0",
        "hashrates": {"0.1": MAX_BIDDING_COMMITMENT_WORKERS + 1},
    }
    with pytest.raises(ValueError, match="total worker count exceeds v2 commitment limit"):
        MinerConfig.from_dict(cfg)  # type: ignore[arg-type]


def test_worker_sizes_total_count_at_limit_ok() -> None:
    cfg = _base_config()
    cfg["workers"] = {
        "price_multiplier": "1.0",
        "worker_sizes": {"0.1": MAX_BIDDING_COMMITMENT_WORKERS},
    }
    parsed = MinerConfig.from_dict(cfg)  # type: ignore[arg-type]
    assert parsed.workers.total_workers() == MAX_BIDDING_COMMITMENT_WORKERS
