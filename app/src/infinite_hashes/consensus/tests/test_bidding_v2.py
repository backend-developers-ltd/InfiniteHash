from infinite_hashes.consensus.bidding import BiddingCommitment
from infinite_hashes.consensus.parser import parse_commitment


def test_bidding_v2_compact_cycle():
    # Setup
    bids = [
        ("BTC", 100 * 10**18, {"0.1": 10, "0.2": 20}),
        ("BTC", 50 * 10**18, {"0.3": 5}),
    ]

    commit = BiddingCommitment(t="b", bids=bids, v=2)

    # Test compact serialization
    compact = commit.to_compact()

    # Expected format roughly:
    # 2;b;BTC,100=0.1:10,0.2:20;KAS,50=1.5:5

    # Check parts
    parts = compact.split(";")
    assert parts[0] == "2"
    assert parts[1] == "b"

    # Parse back
    parsed = BiddingCommitment.from_compact(compact)
    assert isinstance(parsed, BiddingCommitment)
    assert parsed.t == "b"
    assert parsed.v == 2
    assert len(parsed.bids) == 2

    # Check content
    # Convert bids to dict for easier comparison
    bids_dict = {}  # (algo, price) -> map
    for algo, price, hr_map in parsed.bids:
        bids_dict[(algo, price)] = hr_map

    btc_key = ("BTC", 100 * 10**18)
    btc_key_2 = ("BTC", 50 * 10**18)

    assert btc_key in bids_dict
    assert btc_key_2 in bids_dict

    assert bids_dict[btc_key]["0.1"] == 10
    assert bids_dict[btc_key]["0.2"] == 20
    assert bids_dict[btc_key_2]["0.3"] == 5


def test_bidding_v2_parsing_via_generic():
    bids = [("BTC", 1 * 10**18, {"0.4": 100})]
    commit = BiddingCommitment(t="b", bids=bids, v=2)
    compact = commit.to_compact()

    parsed = parse_commitment(compact)
    assert isinstance(parsed, BiddingCommitment)
    assert parsed.bids[0][0] == "BTC"
    assert parsed.bids[0][2]["0.4"] == 100


def test_bidding_v2_aggregation():
    # Test that same algo/price are merged
    bids = [
        ("BTC", 100 * 10**18, {"0.1": 10}),
        ("BTC", 100 * 10**18, {"0.1": 5, "0.2": 8}),
    ]

    commit = BiddingCommitment(t="b", bids=bids, v=2)
    compact = commit.to_compact()

    # Should result in single BTC,100 entry with aggregated counts
    parsed = BiddingCommitment.from_compact(compact)
    assert len(parsed.bids) == 1

    algo, price, hr_map = parsed.bids[0]
    assert algo == "BTC"
    assert price == 100 * 10**18
    assert hr_map["0.1"] == 15  # 10 + 5
    assert hr_map["0.2"] == 8


def test_bidding_v2_hashrate_sorting():
    bids = [("BTC", 1 * 10**18, {"0.2": 1, "0.1": 1})]
    compact = BiddingCommitment(t="b", bids=bids, v=2).to_compact()

    # d part should list 0.1 before 0.2
    d_part = compact.split(";", 2)[2]
    assert d_part.startswith("BTC,1=0.1:1,0.2:1")


def test_bidding_v2_rejects_non_btc_algo():
    # Only BTC is supported for now.
    non_btc = "2;b;KAS,1=0.1:1"
    assert parse_commitment(non_btc) is None


def test_bidding_v2_strict_rejects_empty_group():
    bad = "2;b;BTC,1=0.1:1;;BTC,1=0.2:1"
    assert parse_commitment(bad) is None


def test_bidding_v2_rejects_excessive_worker_count():
    ok = "2;b;BTC,1=0.1:1000"
    assert isinstance(parse_commitment(ok), BiddingCommitment)

    too_many = "2;b;BTC,1=0.1:1001"
    assert parse_commitment(too_many) is None


def test_bidding_v2_rejects_excessive_worker_size():
    ok = "2;b;BTC,1=0.45:1"
    assert isinstance(parse_commitment(ok), BiddingCommitment)

    too_big = "2;b;BTC,1=0.451:1"
    assert parse_commitment(too_big) is None
