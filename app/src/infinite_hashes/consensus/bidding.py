from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from infinite_hashes.auctions.mechanism_split import fetch_mechanism_share_fraction

from .commitment import CompactCommitment
from .price import (
    _fp18_to_min_decimal_str,
    _parse_decimal_to_fp18_int,
    compute_price_consensus,
)

FP = 10**18


class BiddingCommitment(CompactCommitment):
    """Compact commitment for bids.

    Logical payload `bids` maps hashrate (decimal str) -> price_fp18 (int).
    Compact `d` stores the inverted grouping to minimize repetition with
    minimal decimal notation for values:
        price_decimal=key1,key2;price_decimal_2=key3;...

    - Keys (hashrates) are comma-separated within each group and encoded as
      minimal decimal strings (e.g., "1.5", "2").
    - Groups are semicolon-separated.
    - Values are decimal strings (minimal) representing FP18 amounts.
    - Keys containing any of the separators (';', '=', ',') are skipped for safety.
    - Keys inside groups and groups themselves are sorted deterministically
      (numeric ascending for values, lexicographic for keys).
    """

    t: Literal["b"]
    # Preserve duplicates: list of (hashrate_decimal_str, price_fp18)
    bids: list[tuple[str, int]] = Field(default_factory=list)

    # Always use the shortest token
    def _compact_t(self) -> str:
        return "b"

    def _d_compact(self) -> str:
        # Invert: price -> list[keys], preserving duplicates
        groups: dict[int, list[str]] = {}
        for entry in self.bids or []:
            if not isinstance(entry, tuple | list) or len(entry) != 2:
                continue
            k, v = entry
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if not isinstance(k, str):
                continue
            # Normalize key as minimal decimal; skip unsafe
            if not k or (";" in k or "=" in k or "," in k):
                continue
            try:
                key_fp = _parse_decimal_to_fp18_int(k.strip())
            except ValueError:
                continue
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
        bids: list[tuple[str, int]] = []
        if d:
            for seg in d.split(";"):
                if not seg:
                    continue
                if "=" not in seg:
                    continue
                vs, ks = seg.split("=", 1)
                vs = vs.strip()
                try:
                    value = _parse_decimal_to_fp18_int(vs)
                except ValueError:
                    continue
                for key in ks.split(","):
                    if not key:
                        continue
                    if ";" in key or "=" in key or "," in key:
                        continue
                    try:
                        key_fp = _parse_decimal_to_fp18_int(key.strip())
                    except ValueError:
                        continue
                    bids.append((_fp18_to_min_decimal_str(key_fp), value))
        return cls(t="b", bids=bids, v=v)


# --- Auction selection (ILP-based with safe fallback) ---


def _fp_mul(a: int, b: int) -> int:
    return (a * b) // FP


def _fp_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by zero in _fp_div")
    return (a * FP) // b


async def select_auction_winners_async(
    bt: Any,
    netuid: int,
    start_block: int,
    end_block: int,
    bids_by_hotkey: dict[str, list[tuple[str, int]]],
    *,
    miners_share_fp18: int | None = None,
    mechanism_id: int = 1,
    ilp_scale: int = 1000,
    cbc_max_nodes: int = 0,
    cbc_seed: int | None = 1,
    max_price_multiplier: float = 1.05,
) -> list[dict[str, Any]]:
    """Deterministically select winning bids within budget using ILP (PuLP/CBC).

    Returns list of winners: {hotkey, hashrate, price}.
    """

    prices = await _fetch_prices(bt, netuid, start_block)
    if prices is None:
        return []

    BASE_MINER_SHARE = 0.41  # fraction of block emission allocated to miners pre-split
    share_fp18 = miners_share_fp18
    if share_fp18 is None:
        share_fraction = await fetch_mechanism_share_fraction(
            bt,
            netuid,
            mechanism_id,
        )
        share_fp18 = int(share_fraction * BASE_MINER_SHARE * FP)

    budget_ph = _compute_budget_ph(
        prices["ALPHA_TAO"],
        prices["TAO_USDC"],
        prices["HASHP_USDC"],
        start_block,
        end_block,
        share_fp18,
    )
    if budget_ph <= 0:
        return []

    workers = _build_worker_items(bids_by_hotkey, max_price_multiplier)
    if not workers:
        return []

    winners_idx = _solve_ilp_indices(workers, budget_ph, ilp_scale, cbc_max_nodes, cbc_seed)
    sel = [workers[i] for i in sorted(winners_idx)]
    sel.sort(key=lambda t: (t.price_fp, t.hashrate_min, t.name))
    return [{"hotkey": w.name.split(":", 1)[0], "hashrate": w.hashrate_min, "price": w.price_fp} for w in sel]


async def _fetch_prices(bt: Any, netuid: int, block_number: int) -> dict[str, int] | None:
    metrics = ["TAO_USDC", "ALPHA_TAO", "HASHP_USDC"]
    res = await compute_price_consensus(netuid, block_number, metrics, bt=bt)
    try:
        tao_usdc = int(res.get("TAO_USDC"))
        alpha_tao = int(res.get("ALPHA_TAO"))
        hashp_usdc = int(res.get("HASHP_USDC"))
    except (TypeError, ValueError):
        return None
    return {"TAO_USDC": tao_usdc, "ALPHA_TAO": alpha_tao, "HASHP_USDC": hashp_usdc}


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
    bids_by_hotkey: dict[str, list[tuple[str, int]]], max_price_multiplier: float = 1.05
) -> list[_WorkerItem]:
    items: list[_WorkerItem] = []
    max_price_fp18 = int(max_price_multiplier * FP)

    for hk, bids in (bids_by_hotkey or {}).items():
        for hr_str, price_fp in bids or []:
            try:
                hr_fp = _parse_decimal_to_fp18_int(hr_str)
            except ValueError:
                continue
            if hr_fp <= 0:
                continue

            # Filter out bids with price > MAX_PRICE_MULTIPLIER
            if price_fp > max_price_fp18:
                continue

            margin = max(0.0, float(price_fp) / FP)
            raw_ph = float(hr_fp) / FP
            eff_cost = raw_ph * margin
            items.append(
                _WorkerItem(
                    name=f"{hk}:{_fp18_to_min_decimal_str(hr_fp)}",
                    raw_ph=raw_ph,
                    margin_bid=margin,
                    eff_cost_ph=eff_cost,
                    price_fp=int(price_fp),
                    hashrate_min=_fp18_to_min_decimal_str(hr_fp),
                )
            )
    return items


def _solve_ilp_indices(
    workers: list[_WorkerItem],
    budget_ph: float,
    ilp_scale: int,
    cbc_max_nodes: int,
    cbc_seed: int | None,
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
    prob += pulp.lpSum(int_costs[k] * x[k] for k in range(len(keep))) <= cap
    prob += pulp.lpSum(int_vals[k] * x[k] for k in range(len(keep)))
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
