"""Helper functions for computing test scenario budgets.

Allows easy control of available PH budget by adjusting ALPHA_TAO price.
"""

MECHANISM_SPLIT_DENOMINATOR = 65535
DEFAULT_MECHANISM_SPLIT = (52428, 13107)  # 80% -> mechanism 0, 20% -> mechanism 1
DEFAULT_MECHANISM_0_SHARE = DEFAULT_MECHANISM_SPLIT[0] / MECHANISM_SPLIT_DENOMINATOR
DEFAULT_MECHANISM_1_SHARE = DEFAULT_MECHANISM_SPLIT[1] / MECHANISM_SPLIT_DENOMINATOR


def compute_alpha_tao_for_budget(
    target_budget_ph: float,
    tao_usdc: float = 45.0,
    hashp_usdc: float = 50.0,
    blocks_per_day: int = 7200,
    alpha_per_block: float = 0.41,
    mechanism_share: float = 1.0,
) -> float:
    """Compute ALPHA_TAO price needed to achieve target PH budget.

    Budget formula (from bidding.py):
        Daily_ALPHA = blocks_per_day × alpha_per_block
        Daily_USDC = Daily_ALPHA × ALPHA_TAO × TAO_USDC
        Budget_PH = Daily_USDC / HASHP_USDC

    Solving for ALPHA_TAO:
        ALPHA_TAO = (Budget_PH × HASHP_USDC) / (Daily_ALPHA × TAO_USDC)

    Args:
        target_budget_ph: Desired PH budget
        tao_usdc: TAO price in USDC (default: 45)
        hashp_usdc: Hashprice in USDC/PH/day (default: 50)
        blocks_per_day: Blocks per day (default: 7200 = 86400s / 12s)
        alpha_per_block: ALPHA reward per block for miners (default: 0.41)

    Args:
        mechanism_share: Fraction of miner emission allocated to the mechanism (default: 1.0).

    Returns:
        Required ALPHA_TAO price (ALPHA per TAO)

    Example:
        >>> # For 100 PH budget
        >>> alpha_tao = compute_alpha_tao_for_budget(100.0)
        >>> alpha_tao
        0.037639265281541705
        >>> # Use in test
        >>> alpha_tao_fp18 = int(alpha_tao * 10**18)
        >>> alpha_tao_fp18
        37639265281541704
    """
    if mechanism_share <= 0:
        raise ValueError("mechanism_share must be positive")

    effective_alpha_per_block = alpha_per_block * mechanism_share
    daily_alpha = blocks_per_day * effective_alpha_per_block
    alpha_tao = (target_budget_ph * hashp_usdc) / (daily_alpha * tao_usdc)
    return alpha_tao


def alpha_tao_to_fp18(alpha_tao: float) -> int:
    """Convert ALPHA_TAO to FP18 format for validator_worker.py.

    Args:
        alpha_tao: ALPHA_TAO price (ALPHA per TAO)

    Returns:
        FP18 integer (alpha_tao * 10^18)

    Example:
        >>> alpha_tao_to_fp18(0.25)
        250000000000000000
    """
    return int(alpha_tao * 10**18)


def compute_budget_summary(
    alpha_tao: float,
    tao_usdc: float = 45.0,
    hashp_usdc: float = 50.0,
    blocks_per_day: int = 7200,
    alpha_per_block: float = 0.41,
    mechanism_share: float = 1.0,
) -> dict[str, float]:
    """Compute budget details for given prices.

    Args:
        alpha_tao: ALPHA_TAO price (ALPHA per TAO)
        tao_usdc: TAO price in USDC
        hashp_usdc: Hashprice in USDC/PH/day
        mechanism_share: Fraction of miner emission allocated to the mechanism.

    Returns:
        Dictionary with budget breakdown:
        - daily_alpha: Daily ALPHA rewards
        - alpha_usdc: ALPHA price in USDC
        - daily_usdc: Daily USDC budget
        - budget_ph: PH budget

    Example:
        >>> summary = compute_budget_summary(0.25)
        >>> summary['budget_ph']
        664.2
    """
    daily_alpha = blocks_per_day * alpha_per_block * mechanism_share
    alpha_usdc = alpha_tao * tao_usdc
    daily_usdc = daily_alpha * alpha_usdc
    budget_ph = daily_usdc / hashp_usdc

    return {
        "daily_alpha": daily_alpha,
        "alpha_usdc": alpha_usdc,
        "daily_usdc": daily_usdc,
        "budget_ph": budget_ph,
    }


# Common budget presets for testing
BUDGET_PRESETS = {
    "tiny": 50.0,  # For minimal tests
    "small": 100.0,  # Small scenarios
    "default": 664.2,  # Current default (ALPHA_TAO=0.25)
    "medium": 1000.0,  # Medium scenarios
    "large": 5000.0,  # Large scenarios
}


def get_alpha_tao_for_preset(preset: str) -> float:
    """Get ALPHA_TAO price for a budget preset.

    Args:
        preset: One of "tiny", "small", "default", "medium", "large"

    Returns:
        ALPHA_TAO price

    Example:
        >>> get_alpha_tao_for_preset("small")
        0.037639265281541705
    """
    if preset not in BUDGET_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(BUDGET_PRESETS.keys())}")

    target_budget = BUDGET_PRESETS[preset]
    return compute_alpha_tao_for_budget(target_budget)
