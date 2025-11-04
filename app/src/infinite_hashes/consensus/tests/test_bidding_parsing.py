import pytest

from infinite_hashes.consensus.bidding import BiddingCommitment


def test_bidding_from_compact_parses_groups():
    raw = "1;b;1=0.5,1.25;2=2;"
    c = BiddingCommitment.from_compact(raw)
    assert c.t == "b" and c.v == 1
    assert c.bids == [("0.5", 10**18), ("1.25", 10**18), ("2", 2 * 10**18)]


def test_bidding_from_compact_rejects_wrong_token():
    with pytest.raises(ValueError):
        BiddingCommitment.from_compact("1;x;100=a;")


def test_bidding_compact_skips_invalid_keys():
    # Non-decimal keys should be ignored
    raw = "1;b;100=1.0,bad,2.00;"
    c = BiddingCommitment.from_compact(raw)
    assert c.bids == [("1", 100 * 10**18), ("2", 100 * 10**18)]


def test_bidding_to_compact_groups_and_orders():
    bids = [("1.00", 10**18), ("0.5", 10**18), ("2.000", 2 * 10**18)]
    c = BiddingCommitment(t="b", bids=list(bids), v=1)
    s = c.to_compact()
    # Groups identical values, sorts values ascending and keys lexicographically, no trailing ';'
    assert s == "1;b;1=0.5,1;2=2"


def test_bidding_to_compact_decimal_minimal_and_roundtrip():
    bids = [("1.25", int(1.25 * 10**18)), ("0.5", int(0.5 * 10**18))]
    c = BiddingCommitment(t="b", bids=list(bids), v=1)
    s = c.to_compact()
    # Expect minimal decimals in values and deterministic key order within a group
    assert s == "1;b;0.5=0.5;1.25=1.25"
    c2 = BiddingCommitment.from_compact(s)
    assert c2.bids == sorted(bids, key=lambda t: t[0])


def test_bidding_to_compact_skips_invalid_keys_on_serialize():
    bids = [("1", 10**18), ("bad;key", 10**18), ("also=bad", 10**18), ("bad,too", 10**18)]
    c = BiddingCommitment(t="b", bids=list(bids), v=1)
    s = c.to_compact()
    # Only 'ok' remains
    assert s == "1;b;1=1"


def test_bidding_from_compact_empty_and_whitespace():
    assert BiddingCommitment.from_compact("1;b; ").bids == []
    assert BiddingCommitment.from_compact("1;b; ;").bids == []
