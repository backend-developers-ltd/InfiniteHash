from __future__ import annotations

from datetime import UTC, datetime

from infinite_hashes.aps_miner import callbacks
from infinite_hashes.aps_miner.models import AuctionResult, BidResult


def _build_result() -> AuctionResult:
    now = datetime.now(tz=UTC)
    return AuctionResult(
        epoch_start=1,
        start_block=10,
        end_block=20,
        window_start_time=now,
        window_end_time=now,
        commitments_count=2,
        all_winners=[],
        my_bids=[
            BidResult(hashrate="0.1", price_fp18=1, won=True),
            BidResult(hashrate="0.2", price_fp18=1, won=False),
        ],
    )


def test_handle_auction_results_uses_ihp_proxy_when_configured(monkeypatch) -> None:
    calls = {"ihp": 0, "braiins": 0}

    def _ihp_handler(*_args, **_kwargs) -> None:
        calls["ihp"] += 1

    def _braiins_handler(*_args, **_kwargs) -> None:
        calls["braiins"] += 1

    monkeypatch.setenv("APS_MINER_PROXY_MODE", "ihp")
    monkeypatch.setattr(callbacks, "update_subnet_target_hashrate", _ihp_handler)
    monkeypatch.setattr(callbacks, "update_routing_weights", _braiins_handler)

    callbacks.handle_auction_results(_build_result())

    assert calls["ihp"] == 1
    assert calls["braiins"] == 0


def test_handle_auction_results_auto_detects_ihp_proxy_from_pools_file(monkeypatch, tmp_path) -> None:
    calls = {"ihp": 0, "braiins": 0}

    def _ihp_handler(*_args, **_kwargs) -> None:
        calls["ihp"] += 1

    def _braiins_handler(*_args, **_kwargs) -> None:
        calls["braiins"] += 1

    pools_file = tmp_path / "pools.toml"
    pools_file.write_text("[pools]\n", encoding="utf-8")

    monkeypatch.delenv("APS_MINER_PROXY_MODE", raising=False)
    monkeypatch.setenv("APS_MINER_POOLS_CONFIG_PATH", str(pools_file))
    monkeypatch.setattr(callbacks, "update_subnet_target_hashrate", _ihp_handler)
    monkeypatch.setattr(callbacks, "update_routing_weights", _braiins_handler)

    callbacks.handle_auction_results(_build_result())

    assert calls["ihp"] == 1
    assert calls["braiins"] == 0
