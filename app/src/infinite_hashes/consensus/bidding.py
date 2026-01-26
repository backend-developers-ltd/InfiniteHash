from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import structlog
from pydantic import Field

from infinite_hashes.auctions.mechanism_split import fetch_mechanism_share_fraction

from .commitment import CompactCommitment
from .price import (
    BUDGET_CAP_FIELD,
    _fp18_to_min_decimal_str,
    _parse_decimal_to_fp18_int,
    compute_price_consensus,
)

FP = 10**18
logger = structlog.get_logger(__name__)

# v2 safety limits:
# - Max expanded number of workers (sum of COUNT) to bound decoding and ILP size
# - Max worker hashrate (in PH) to enforce "small worker" commitments (450 TH = 0.45 PH)
MAX_BIDDING_COMMITMENT_WORKERS = 1000
MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18 = 450 * 10**15
MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18 = 50 * 10**15

# Optional budget cap from validator commitments (defaults to no cap).
DEFAULT_BUDGET_CAP = 1.0
ILP_BIG_M_SWITCH_BLOCK = 7405572


class BiddingCommitment(CompactCommitment):
    """Compact commitment for bids (versioned).

    v=1 (legacy)
        Logical payload `bids`: list[(hashrate_decimal_str, price_fp18_int)]
        Compact `d`: price_decimal=hr1,hr2;price_decimal_2=hr3;...

    v=2
        Logical payload `bids`: list[(algo_str, price_fp18_int, {hashrate_decimal_str: count_int})]
        Compact `d`: ALGO,PRICE=HR:COUNT,HR:COUNT;...

    - Keys (hashrates) are comma-separated within each group and encoded as
      minimal decimal strings (e.g., "1.5", "2").
    - Groups are semicolon-separated.
    - Values are decimal strings (minimal) representing FP18 amounts.
    - Invalid keys/values (including any separator collisions like ';', '=', ',') are rejected
      by raising `ValueError` (i.e., the whole commitment is invalid rather than "best-effort"
      skipping individual entries).
    - Keys inside groups and groups themselves are sorted deterministically
      (numeric ascending for values, lexicographic for keys).
    """

    t: Literal["b"]
    # v=1: list[(hashrate_decimal_str, price_fp18)]
    # v=2: list[(algo_str, price_fp18, {hashrate_decimal_str: count_int})]
    bids: list[tuple[Any, ...]] = Field(default_factory=list)

    # Always use the shortest token
    def _compact_t(self) -> str:
        return "b"

    def _d_compact(self) -> str:
        if int(getattr(self, "v", 1) or 1) >= 2:
            parts: list[str] = []
            grouped: dict[tuple[str, int], dict[int, int]] = {}
            total_count = 0

            for entry in self.bids or []:
                if not isinstance(entry, tuple | list) or len(entry) != 3:
                    raise ValueError("invalid v2 bidding payload entry")
                algo, price, hr_map = entry
                if not isinstance(algo, str):
                    raise ValueError("invalid v2 bidding algorithm")
                if algo != "BTC":
                    raise ValueError("only BTC bidding is supported")
                try:
                    price_fp18 = int(price)
                except (TypeError, ValueError):
                    raise ValueError("invalid v2 bidding price")
                if not isinstance(hr_map, dict):
                    raise ValueError("invalid v2 bidding hashrate map")

                key = (algo, price_fp18)
                if key not in grouped:
                    grouped[key] = {}

                for hr, count in hr_map.items():
                    if not isinstance(hr, str):
                        raise ValueError("invalid v2 bidding hashrate")
                    try:
                        hr_fp18 = _parse_decimal_to_fp18_int(hr.strip())
                        c = int(count)
                    except (ValueError, TypeError):
                        raise ValueError("invalid v2 bidding hashrate/count")
                    if hr_fp18 <= 0:
                        raise ValueError("invalid v2 bidding hashrate")
                    if hr_fp18 > MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18:
                        raise ValueError("v2 bidding hashrate exceeds max worker size")
                    if hr_fp18 < MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18:
                        raise ValueError("v2 bidding hashrate below min worker size")
                    if c <= 0:
                        raise ValueError("invalid v2 bidding worker count")
                    grouped[key][hr_fp18] = grouped[key].get(hr_fp18, 0) + c
                    total_count += c
                    if total_count > MAX_BIDDING_COMMITMENT_WORKERS:
                        raise ValueError("bidding commitment exceeds worker count limit")

            if not grouped:
                return ""

            for algo, price_fp18 in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
                hr_map = grouped[(algo, price_fp18)]
                if not hr_map:
                    continue
                price_str = _fp18_to_min_decimal_str(price_fp18)
                hr_parts = [
                    f"{_fp18_to_min_decimal_str(hr_fp18)}:{hr_map[hr_fp18]}" for hr_fp18 in sorted(hr_map.keys())
                ]
                if hr_parts:
                    parts.append(f"{algo},{price_str}={','.join(hr_parts)}")

            return ";".join(parts)

        # Invert: price -> list[keys], preserving duplicates
        groups: dict[int, list[str]] = {}
        for entry in self.bids or []:
            if not isinstance(entry, tuple | list) or len(entry) != 2:
                raise ValueError("invalid v1 bidding payload entry")
            k, v = entry
            try:
                iv = int(v)
            except (TypeError, ValueError):
                raise ValueError("invalid v1 bidding price")
            if not isinstance(k, str):
                raise ValueError("invalid v1 bidding hashrate")
            # Normalize key as minimal decimal; reject unsafe
            if not k or (";" in k or "=" in k or "," in k):
                raise ValueError("invalid v1 bidding hashrate")
            try:
                key_fp = _parse_decimal_to_fp18_int(k.strip())
            except ValueError:
                raise ValueError("invalid v1 bidding hashrate")
            norm_key = _fp18_to_min_decimal_str(key_fp)
            groups.setdefault(iv, []).append(norm_key)

        if not groups:
            return ""

        parts: list[str] = []
        for value in sorted(groups.keys()):
            keys = sorted(groups[value])  # duplicates retained
            parts.append(f"{_fp18_to_min_decimal_str(value)}={','.join(keys)}")
        return ";".join(parts)

    @classmethod
    def _from_d_compact(cls, v: int, d: str) -> BiddingCommitment:
        if int(v or 1) >= 2:
            bids_v2: list[tuple[str, int, dict[str, int]]] = []
            total_count = 0
            if not d or not d.strip():
                return cls(t="b", bids=[], v=v)

            d = d.strip()
            for group in d.split(";"):
                if not group:
                    raise ValueError("invalid v2 bidding compact format")
                if "=" not in group:
                    raise ValueError("invalid v2 bidding compact format")

                left, right = group.split("=", 1)
                left = left.strip()
                if not left or "," not in left:
                    raise ValueError("invalid v2 bidding compact format")

                algo, price_s = left.rsplit(",", 1)
                algo = algo.strip()
                price_s = price_s.strip()
                if algo != "BTC":
                    raise ValueError("only BTC bidding is supported")

                try:
                    price = _parse_decimal_to_fp18_int(price_s)
                except ValueError:
                    raise ValueError("invalid v2 bidding price")

                hr_map: dict[str, int] = {}
                for item in right.split(","):
                    item = item.strip()
                    if not item or ":" not in item:
                        raise ValueError("invalid v2 bidding compact format")
                    hr_s, count_s = item.split(":", 1)
                    hr_s = hr_s.strip()
                    count_s = count_s.strip()
                    try:
                        count = int(count_s)
                        hr_fp = _parse_decimal_to_fp18_int(hr_s)
                        if hr_fp <= 0:
                            raise ValueError("invalid v2 bidding hashrate")
                        if hr_fp > MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18:
                            raise ValueError("v2 bidding hashrate exceeds max worker size")
                        if hr_fp < MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18:
                            raise ValueError("v2 bidding hashrate below min worker size")
                        hr = _fp18_to_min_decimal_str(hr_fp)
                    except (ValueError, TypeError):
                        raise ValueError("invalid v2 bidding hashrate/count")
                    if count <= 0:
                        raise ValueError("invalid v2 bidding worker count")
                    hr_map[hr] = hr_map.get(hr, 0) + count
                    total_count += count
                    if total_count > MAX_BIDDING_COMMITMENT_WORKERS:
                        raise ValueError("bidding commitment exceeds worker count limit")

                if not hr_map:
                    raise ValueError("invalid v2 bidding compact format")
                bids_v2.append((algo, price, hr_map))

            return cls(t="b", bids=bids_v2, v=v)

        bids: list[tuple[str, int]] = []
        if d and d.strip():
            d = d.strip()
            for seg in d.split(";"):
                if not seg:
                    raise ValueError("invalid v1 bidding compact format")
                if "=" not in seg:
                    raise ValueError("invalid v1 bidding compact format")
                vs, ks = seg.split("=", 1)
                vs = vs.strip()
                ks = ks.strip()
                if not vs or not ks:
                    raise ValueError("invalid v1 bidding compact format")
                try:
                    value = _parse_decimal_to_fp18_int(vs)
                except ValueError:
                    raise ValueError("invalid v1 bidding price")
                for key in ks.split(","):
                    if not key:
                        raise ValueError("invalid v1 bidding compact format")
                    if ";" in key or "=" in key or "," in key:
                        raise ValueError("invalid v1 bidding compact format")
                    try:
                        key_fp = _parse_decimal_to_fp18_int(key.strip())
                    except ValueError:
                        raise ValueError("invalid v1 bidding hashrate")
                    bids.append((_fp18_to_min_decimal_str(key_fp), value))
        return cls(t="b", bids=bids, v=v)


# --- Auction selection (ILP-based with safe fallback) ---


def _fp_mul(a: int, b: int) -> int:
    return (a * b) // FP


def _fp_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by zero in _fp_div")
    return (a * FP) // b


def _budget_cap_from_fp18(cap_fp18: int | None) -> float:
    if cap_fp18 is None:
        return DEFAULT_BUDGET_CAP
    cap = float(cap_fp18) / FP
    if cap <= 0:
        return DEFAULT_BUDGET_CAP
    return cap


async def select_auction_winners_async(
    bt: Any,
    netuid: int,
    start_block: int,
    end_block: int,
    bids_by_hotkey: dict[str, list[tuple[str, int] | tuple[str, int, int]]],
    *,
    miners_share_fp18: int | None = None,
    mechanism_id: int = 1,
    ilp_scale: int = 1_000_000,
    cbc_max_nodes: int = 0,
    cbc_seed: int | None = 1,
    max_price_multiplier: float = 1.05,
    ilp_use_big_m: bool | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Deterministically select winning bids within budget using ILP (PuLP/CBC).

    Returns (winners, budget_ph) where winners = list[{hotkey, hashrate, price}].
    """

    prices = await _fetch_prices(bt, netuid, start_block)
    if prices is None:
        logger.warning(
            "Price consensus unavailable for auction budget",
            netuid=netuid,
            start_block=start_block,
            end_block=end_block,
        )
        return [], 0.0

    BASE_MINER_SHARE = 0.41  # fraction of block emission allocated to miners pre-split
    share_fp18 = miners_share_fp18
    share_fraction: float | None = None
    if share_fp18 is None:
        share_fraction = await fetch_mechanism_share_fraction(
            bt,
            netuid,
            mechanism_id,
        )
        share_fp18 = int(share_fraction * BASE_MINER_SHARE * FP)
    miner_share_per_block = float(share_fp18) / FP

    budget_ph = _compute_budget_ph(
        prices["ALPHA_TAO"],
        prices["TAO_USDC"],
        prices["HASHP_USDC"],
        start_block,
        end_block,
        share_fp18,
    )
    budget_cap = _budget_cap_from_fp18(prices.get(BUDGET_CAP_FIELD))
    budget_ph_capped = budget_ph * budget_cap
    if budget_ph <= 0:
        logger.warning(
            "Computed non-positive auction budget",
            netuid=netuid,
            start_block=start_block,
            end_block=end_block,
            alpha_tao=float(prices["ALPHA_TAO"]) / FP,
            tao_usdc=float(prices["TAO_USDC"]) / FP,
            hashp_usdc=float(prices["HASHP_USDC"]) / FP,
            miner_share_per_block=miner_share_per_block,
            share_fraction=share_fraction,
        )
        return [], 0.0

    logger.info(
        "Auction budget computed",
        netuid=netuid,
        start_block=start_block,
        end_block=end_block,
        alpha_tao=float(prices["ALPHA_TAO"]) / FP,
        tao_usdc=float(prices["TAO_USDC"]) / FP,
        hashp_usdc=float(prices["HASHP_USDC"]) / FP,
        miner_share_per_block=miner_share_per_block,
        share_fraction=share_fraction,
        budget_ph=budget_ph,
        budget_cap=budget_cap,
        budget_ph_capped=budget_ph_capped,
        bid_hotkeys=len(bids_by_hotkey or {}),
    )
    budget_ph = budget_ph_capped

    workers = _build_worker_items(bids_by_hotkey, max_price_multiplier)
    if not workers:
        return [], budget_ph

    use_big_m = ilp_use_big_m if ilp_use_big_m is not None else start_block >= ILP_BIG_M_SWITCH_BLOCK
    winners_idx = _solve_ilp_indices(workers, budget_ph, ilp_scale, cbc_max_nodes, cbc_seed, use_big_m=use_big_m)
    sel = [workers[i] for i in sorted(winners_idx)]
    sel.sort(key=lambda t: (t.price_fp, t.hashrate_min, t.name))
    winners = [{"hotkey": w.name.split(":", 1)[0], "hashrate": w.hashrate_min, "price": w.price_fp} for w in sel]
    return winners, budget_ph


async def _fetch_prices(bt: Any, netuid: int, block_number: int) -> dict[str, int] | None:
    metrics = ["TAO_USDC", "ALPHA_TAO", "HASHP_USDC", BUDGET_CAP_FIELD]
    res = await compute_price_consensus(netuid, block_number, metrics, bt=bt)
    try:
        tao_usdc = int(res.get("TAO_USDC"))
        alpha_tao = int(res.get("ALPHA_TAO"))
        hashp_usdc = int(res.get("HASHP_USDC"))
    except (TypeError, ValueError):
        return None
    cap_raw = res.get(BUDGET_CAP_FIELD)
    cap_fp18 = int(DEFAULT_BUDGET_CAP * FP)
    if isinstance(cap_raw, int):
        cap_fp18 = cap_raw
    return {"TAO_USDC": tao_usdc, "ALPHA_TAO": alpha_tao, "HASHP_USDC": hashp_usdc, BUDGET_CAP_FIELD: cap_fp18}


def _compute_budget_ph(
    alpha_tao_fp: int,
    tao_usdc_fp: int,
    hashp_usdc_fp: int,
    start_block: int,
    end_block: int,
    miners_share_fp18: int | None,
) -> float:
    """Compute PH budget based on daily ALPHA revenue.

    Logic:
    1. Calculate daily ALPHA production (blocks_per_day * alpha_per_block)
    2. Convert to daily USDC budget
    3. Divide by hashprice (USDC per PH per day) to get affordable PH

    This gives us the continuous PH capacity we can afford with daily revenue,
    independent of window length.

    Note: start_block and end_block are not used in calculation, only for validation.
    """
    alpha_usdc = _fp_mul(alpha_tao_fp, tao_usdc_fp)

    # Daily ALPHA production
    # Block time = 12 seconds, so blocks_per_day = 86400 / 12 = 7200
    blocks_per_day = 7200
    share = miners_share_fp18 if miners_share_fp18 is not None else (41 * (10**16))
    daily_alpha_fp = share * blocks_per_day

    # Convert to daily USDC budget
    daily_usdc_fp = _fp_mul(alpha_usdc, daily_alpha_fp)

    # Divide by hashprice (USDC per PH per day) to get PH budget
    if hashp_usdc_fp == 0:
        return 0.0
    budget_ph_fp = _fp_div(daily_usdc_fp, hashp_usdc_fp)
    return float(budget_ph_fp) / FP


@dataclass
class _WorkerItem:
    name: str
    raw_ph: float
    margin_bid: float
    eff_cost_ph: float
    price_fp: int
    hashrate_min: str


def _build_worker_items(
    bids_by_hotkey: dict[str, list[tuple[str, int] | tuple[str, int, int]]],
    max_price_multiplier: float = 1.05,
) -> list[_WorkerItem]:
    items: list[_WorkerItem] = []
    max_price_fp18 = int(max_price_multiplier * FP)

    for hk, bids in (bids_by_hotkey or {}).items():
        for bid in bids or []:
            count = 1
            if len(bid) == 3:
                hr_str, price_fp, count = bid
            elif len(bid) == 2:
                hr_str, price_fp = bid
            else:
                continue

            try:
                hr_fp = _parse_decimal_to_fp18_int(hr_str)
            except ValueError:
                continue
            if hr_fp <= 0:
                continue

            try:
                count_int = int(count)
            except (TypeError, ValueError):
                continue
            if count_int <= 0:
                continue

            # Filter out bids with price > MAX_PRICE_MULTIPLIER
            if price_fp > max_price_fp18:
                continue

            margin = max(0.0, float(price_fp) / FP)
            raw_ph = float(hr_fp) / FP
            eff_cost = raw_ph * margin
            hashrate_min = _fp18_to_min_decimal_str(hr_fp)
            name = f"{hk}:{hashrate_min}"
            for _ in range(count_int):
                items.append(
                    _WorkerItem(
                        name=name,
                        raw_ph=raw_ph,
                        margin_bid=margin,
                        eff_cost_ph=eff_cost,
                        price_fp=int(price_fp),
                        hashrate_min=hashrate_min,
                    )
                )
    return items


def _solve_ilp_indices(
    workers: list[_WorkerItem],
    budget_ph: float,
    ilp_scale: int,
    cbc_max_nodes: int,
    cbc_seed: int | None,
    *,
    use_big_m: bool = False,
) -> set[int]:
    import pulp  # type: ignore

    winners_idx: set[int] = set()
    cap = int((budget_ph * ilp_scale) // 1)
    if cap <= 0:
        return winners_idx
    int_costs: list[int] = []
    int_vals: list[int] = []
    keep: list[int] = []
    for i, w in enumerate(workers):
        c = int(max(0, int(round(w.eff_cost_ph * ilp_scale + 0.0000001))))
        v = int(max(0, int(round(w.raw_ph * ilp_scale + 0.0000001))))
        if c > 0 and v > 0 and c <= cap:
            keep.append(i)
            int_costs.append(c)
            int_vals.append(v)
    if not keep:
        return winners_idx

    prob = pulp.LpProblem("auction_knap", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{k}", lowBound=0, upBound=1, cat=pulp.LpBinary) for k in range(len(keep))]
    cost_expr = pulp.lpSum(int_costs[k] * x[k] for k in range(len(keep)))
    value_expr = pulp.lpSum(int_vals[k] * x[k] for k in range(len(keep)))
    prob += cost_expr <= cap
    if use_big_m:
        m_weight = cap + 1
        prob += (value_expr * m_weight) - cost_expr
    else:
        prob += value_expr
    options: list[str] = ["threads 1"]
    if cbc_seed is not None and cbc_seed > 0:
        options += [f"randomSeed {int(cbc_seed)}"]
    if cbc_max_nodes and cbc_max_nodes > 0:
        options += [f"maxNodes {int(cbc_max_nodes)}"]
    solver = pulp.PULP_CBC_CMD(msg=False, options=options)
    prob.solve(solver)
    for k, i in enumerate(keep):
        val = x[k].value()
        if val is not None and val >= 0.5:
            winners_idx.add(i)
    return winners_idx
