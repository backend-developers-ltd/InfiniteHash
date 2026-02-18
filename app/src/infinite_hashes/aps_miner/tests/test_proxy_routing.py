from __future__ import annotations

import tomli

from infinite_hashes.aps_miner.models import BidResult
from infinite_hashes.aps_miner.proxy_routing import update_subnet_target_hashrate


def test_update_subnet_target_hashrate_sets_absolute_target_and_keeps_private_weights(monkeypatch, tmp_path) -> None:
    pools_file = tmp_path / "pools.toml"
    sentinel_file = tmp_path / ".reload-ihp"
    pools_file.write_text(
        """
[pools]
backup = { name = "backup", host = "backup.pool", port = 3333 }

[[pools.main]]
name = "subnet"
host = "stratum.infinitehash.xyz"
port = 9332
weight = 1

[[pools.main]]
name = "private"
host = "btc.global.luxor.tech"
port = 700
weight = 7
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("APS_MINER_POOLS_CONFIG_PATH", str(pools_file))
    monkeypatch.setenv("APS_MINER_SUBNET_POOL_NAME", "subnet")
    monkeypatch.setenv("APS_MINER_IHP_RELOAD_SENTINEL", str(sentinel_file))

    won_bids = [
        BidResult(hashrate="0.1", price_fp18=1, won=True),
        BidResult(hashrate="0.2", price_fp18=1, won=True),
    ]
    lost_bids = [BidResult(hashrate="0.3", price_fp18=1, won=False)]

    update_subnet_target_hashrate(won_bids, lost_bids)

    with pools_file.open("rb") as f:
        data = tomli.load(f)

    subnet_pool = data["pools"]["main"][0]
    private_pool = data["pools"]["main"][1]

    assert subnet_pool["target_hashrate"] == "300TH/s"
    assert "weight" not in subnet_pool
    assert private_pool["weight"] == 7
    assert sentinel_file.exists()


def test_update_subnet_target_hashrate_sets_zero_when_no_wins(monkeypatch, tmp_path) -> None:
    pools_file = tmp_path / "pools.toml"
    sentinel_file = tmp_path / ".reload-ihp"
    pools_file.write_text(
        """
[pools]
backup = { name = "backup", host = "backup.pool", port = 3333 }

[[pools.main]]
name = "subnet"
host = "stratum.infinitehash.xyz"
port = 9332
weight = 1
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("APS_MINER_POOLS_CONFIG_PATH", str(pools_file))
    monkeypatch.setenv("APS_MINER_SUBNET_POOL_NAME", "subnet")
    monkeypatch.setenv("APS_MINER_IHP_RELOAD_SENTINEL", str(sentinel_file))

    update_subnet_target_hashrate([], [BidResult(hashrate="0.3", price_fp18=1, won=False)])

    with pools_file.open("rb") as f:
        data = tomli.load(f)

    subnet_pool = data["pools"]["main"][0]
    assert subnet_pool["target_hashrate"] == "0TH/s"
    assert sentinel_file.exists()
