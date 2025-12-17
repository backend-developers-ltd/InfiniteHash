from __future__ import annotations

from infinite_hashes.aps_miner.tasks import _reconcile_miner_bids

FP = 10**18


def _winners(hotkey: str, hashrate: str, price_fp18: int, n: int) -> list[dict[str, object]]:
    return [{"hotkey": hotkey, "hashrate": hashrate, "price": price_fp18} for _ in range(n)]


def test_reconcile_v2_partial_win_same_size() -> None:
    hotkey = "hk"
    price = 1 * FP
    commitment_bids = [("0.1", price, 10)]
    winners = _winners(hotkey, "0.1", price, 3)

    my_bids = _reconcile_miner_bids(hotkey=hotkey, commitment_bids=commitment_bids, winners=winners)
    assert len(my_bids) == 10
    assert sum(1 for b in my_bids if b.won) == 3
    assert sum(1 for b in my_bids if not b.won) == 7


def test_reconcile_v1_duplicates_partial_win() -> None:
    hotkey = "hk"
    price = 1 * FP
    commitment_bids = [("0.1", price), ("0.1", price), ("0.1", price)]
    winners = _winners(hotkey, "0.1", price, 2)

    my_bids = _reconcile_miner_bids(hotkey=hotkey, commitment_bids=commitment_bids, winners=winners)
    assert len(my_bids) == 3
    assert sum(1 for b in my_bids if b.won) == 2
    assert sum(1 for b in my_bids if not b.won) == 1


def test_reconcile_price_mismatch_does_not_mark_won() -> None:
    hotkey = "hk"
    commitment_price = 101 * FP
    winner_price = 100 * FP
    commitment_bids = [("0.1", commitment_price, 5)]
    winners = _winners(hotkey, "0.1", winner_price, 5)

    my_bids = _reconcile_miner_bids(hotkey=hotkey, commitment_bids=commitment_bids, winners=winners)
    assert len(my_bids) == 5
    assert all(not b.won for b in my_bids)
