from infinite_hashes.consensus.price import (
    PriceCommitment,
    _determine_metrics,
    _parse_price_commitments,
)


def test_prices_int_skips_invalid_values():
    model = PriceCommitment(
        t="p",
        prices={
            "TAO_USDC": "10",  # decimal semantics
            "BAD": "foo",
            "ALPHA_TAO": "0.5",
        },
        v=1,
    )
    out = model.prices_int()
    assert "TAO_USDC" in out and isinstance(out["TAO_USDC"], int)
    assert "ALPHA_TAO" in out and isinstance(out["ALPHA_TAO"], int)
    assert "BAD" not in out


def test_parse_price_commitments_skips_invalid_commitments():
    compact_valid = "1;p;X=1;Y=2;"  # compact v;t;k=v; format (decimal values)
    wrong_t = "1;x;X=1;"  # wrong type token
    invalid_compact = "1;p"  # incomplete

    out = _parse_price_commitments(
        {
            "HK2": wrong_t,
            "HK3": invalid_compact,
            "HK4": compact_valid,
        }
    )

    assert "HK4" in out and out["HK4"] == {"X": int(1 * 10**18), "Y": int(2 * 10**18)}
    assert "HK2" not in out
    assert "HK3" not in out


def test_compact_decimal_values_are_parsed_to_fp18():
    # 1.25 and 0.5 encoded as decimal strings
    compact = "1;p;A=1.25;B=.5;C=10;"
    out = _parse_price_commitments({"HK": compact})
    assert out["HK"]["A"] == int(1.25 * 10**18)
    assert out["HK"]["B"] == int(0.5 * 10**18)
    assert out["HK"]["C"] == int(10 * 10**18)


def test_determine_metrics_none_discovers_all():
    prices_map = {
        "HK1": {"A": 1, "B": 2},
        "HK2": {"B": 3, "C": 4},
    }
    metrics = _determine_metrics(prices_map, None)
    assert metrics == ["A", "B", "C"]
