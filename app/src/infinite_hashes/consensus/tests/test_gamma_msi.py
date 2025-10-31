from infinite_hashes.consensus.price import _gamma_msi_fp18


def fp18(x: float) -> int:
    return int(x * 1e18)


def test_gamma_msi_basic_window_and_median():
    # Three hotkeys: A,B,C with prices around 10 and one outlier
    prices = {
        "A": fp18(10.0),
        "B": fp18(10.2),
        "C": fp18(7.0),
        "D": fp18(13.0),
    }
    stakes = {"A": 40, "B": 30, "C": 15, "D": 15}
    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    assert res is not None
    # Weighted median inside minimal window [10.0, 10.2] falls at 10.0
    assert res == fp18(10.0)


def test_gamma_msi_no_values():
    assert _gamma_msi_fp18({}, {}, gamma=0.67) is None


def test_gamma_msi_tie_breaker_and_ordering():
    # Prices at same distance; ensure deterministic ordering by (x, hotkey)
    prices = {"A": fp18(10.0), "B": fp18(10.0), "C": fp18(10.0)}
    stakes = {"A": 34, "B": 33, "C": 33}
    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    assert res == fp18(10.0)


def test_gamma_msi_weighted_median_exact_edge_two_points():
    # Two stakers with equal stake at different prices; median lies exactly between them.
    # Algorithm should pick the first x where acc*2 >= Ww, i.e., the lower price.
    prices = {"A": fp18(10.0), "B": fp18(11.0)}
    stakes = {"A": 50, "B": 50}
    res = _gamma_msi_fp18(prices, stakes, gamma=1.0)
    assert res == fp18(10.0)


def test_gamma_msi_weighted_median_exact_edge_inside_three_points():
    # Three points: at cumulative 1, 2, 4 with total Ww=4, half is 2.
    # Median hits exactly at second point; algorithm picks that point's price.
    prices = {"A": fp18(10.0), "B": fp18(11.0), "C": fp18(12.0)}
    stakes = {"A": 1, "B": 1, "C": 2}
    res = _gamma_msi_fp18(prices, stakes, gamma=1.0)
    assert res == fp18(11.0)


def test_gamma_msi_all_stakes_zero_or_missing():
    prices = {"A": fp18(10.0), "B": fp18(10.2)}
    stakes = {"A": 0}  # B missing, A zero
    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    assert res is None


def test_gamma_msi_gamma_one_full_distribution():
    prices = {
        "A": fp18(10.0),
        "B": fp18(10.2),
        "C": fp18(7.0),
        "D": fp18(13.0),
    }
    stakes = {"A": 40, "B": 30, "C": 15, "D": 15}
    res = _gamma_msi_fp18(prices, stakes, gamma=1.0)
    # Weighted median over all values is at 10.0
    assert res == fp18(10.0)


def test_gamma_msi_majority_honest_vs_attack():
    # Honest cluster ~100 has 80% stake; attacker at 1000 has 20% stake
    prices = {
        "H1": fp18(100.0),
        "H2": fp18(101.0),
        "A1": fp18(1000.0),
    }
    stakes = {"H1": 50, "H2": 30, "A1": 20}
    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    # Expect consensus stays in honest window, weighted median at 100
    assert res == fp18(100.0)


def test_gamma_msi_ignores_zero_price():
    prices = {"A": 0, "B": fp18(10.0)}
    stakes = {"A": 100, "B": 1}
    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    assert res == fp18(10.0)


def test_gamma_msi_exact_gamma_threshold_edge():
    prices = {"A": fp18(10.0), "B": fp18(100.0)}
    stakes = {"A": 67, "B": 33}
    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    # Target is exactly 67; minimal window can be just A (width=0)
    assert res == fp18(10.0)


def test_gamma_msi_tie_same_width_two_pairs():
    # Two prices 10 and 11, multiple hotkeys making equal-width candidate windows
    prices = {
        "A": fp18(10.0),
        "B": fp18(10.0),
        "C": fp18(11.0),
        "D": fp18(11.0),
    }
    stakes = {"A": 26, "B": 24, "C": 25, "D": 25}
    res = _gamma_msi_fp18(prices, stakes, gamma=0.5)  # target = 50
    # Minimal window width ties between [10,11] pairs; weighted median falls at 10.0
    assert res == fp18(10.0)


def test_gamma_msi_many_sybils_low_stake():
    # 80 stake honest around 100; 20 total stake split across many sybils at 1000
    prices = {"H1": fp18(100.0), "H2": fp18(100.5)}
    stakes = {"H1": 50, "H2": 30}
    for i in range(20):
        prices[f"S{i}"] = fp18(1000.0)
        stakes[f"S{i}"] = 1

    res = _gamma_msi_fp18(prices, stakes, gamma=0.67)
    # Expect the honest cluster to dominate consensus
    assert res in (fp18(100.0), fp18(100.5))


def test_gamma_msi_gamma_gt_one_no_window():
    prices = {"A": fp18(10.0), "B": fp18(11.0)}
    stakes = {"A": 50, "B": 50}
    # gamma > 1 means target stake exceeds total stake; expect None
    res = _gamma_msi_fp18(prices, stakes, gamma=1.5)
    assert res is None
