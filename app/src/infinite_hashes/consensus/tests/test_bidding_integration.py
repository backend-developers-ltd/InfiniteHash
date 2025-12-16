import pytest

from infinite_hashes.consensus import bidding as bidding_mod
from infinite_hashes.consensus.bidding import _build_worker_items, select_auction_winners_async

FP = 10**18


def test_build_worker_items_mixed_v1_v2_counts():
    # v1: (hr, price)
    # v2: (hr, price, count)
    bids_by_hotkey = {
        "h1": [("1.0", 1 * FP)],  # v1 (1 worker)
        "h2": [("2.0", 1 * FP, 2)],  # v2 (2 workers)
        "h3": [("3.0", 1 * FP, 1)],  # v2 (1 worker)
    }

    items = _build_worker_items(bids_by_hotkey)
    names = sorted([i.name for i in items])

    assert len(items) == 4  # 1 + 2 + 1
    assert names.count("h1:1") == 1
    assert names.count("h2:2") == 2
    assert names.count("h3:3") == 1


@pytest.mark.asyncio
async def test_select_auction_winners_smoke(monkeypatch):
    async def fake_fetch_prices(*_args, **_kwargs):
        return {"TAO_USDC": FP, "ALPHA_TAO": FP, "HASHP_USDC": FP}

    def fake_solve_ilp_indices(workers, *_args, **_kwargs):
        return set(range(len(workers)))

    monkeypatch.setattr(bidding_mod, "_fetch_prices", fake_fetch_prices)
    monkeypatch.setattr(bidding_mod, "_solve_ilp_indices", fake_solve_ilp_indices)

    bids_by_hotkey = {
        "h1": [("1.0", 1 * FP)],
        "h2": [("2.0", 1 * FP, 2)],
    }

    winners, budget_ph = await select_auction_winners_async(
        bt=None,
        netuid=1,
        start_block=100,
        end_block=160,
        bids_by_hotkey=bids_by_hotkey,
        miners_share_fp18=int(0.41 * FP),
    )

    assert budget_ph > 0
    assert len(winners) == 3
    assert {w["hotkey"] for w in winners} == {"h1", "h2"}
