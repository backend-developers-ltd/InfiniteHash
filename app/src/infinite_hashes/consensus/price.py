from __future__ import annotations

import asyncio
import math
from decimal import Decimal
from typing import Any, Literal

import structlog
from pydantic import Field

from .commitment import CompactCommitment, join_commitment_binary, split_commitment_binary

logger = structlog.get_logger(__name__)

BUDGET_CAP_FIELD = "cap"


class PriceCommitment(CompactCommitment):
    # Compact type discriminator 'p' only
    t: Literal["p"]
    prices: dict[str, int | str] = Field(default_factory=dict)
    bans: bytes = Field(default=b"\x00" * 32)  # 256-bit bitmap (32 bytes)

    def prices_int(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for k, v in (self.prices or {}).items():
            # Accept ints or decimal strings and convert to FP18 ints; strings are decimals
            if isinstance(v, int):
                out[k] = v
                continue
            if isinstance(v, str):
                s = v.strip()
                try:
                    out[k] = _parse_decimal_to_fp18_int(s)
                    continue
                except ValueError:
                    pass
        return out

    def get_banned_uids(self) -> set[int]:
        """Extract set of banned UIDs from the 256-bit bitmap."""
        banned = set()
        for uid in range(256):
            byte_idx = uid // 8
            bit_idx = uid % 8
            if byte_idx < len(self.bans):
                if (self.bans[byte_idx] >> bit_idx) & 1:
                    banned.add(uid)
        return banned

    @staticmethod
    def create_ban_bitmap(banned_uids: set[int]) -> bytes:
        """Create 256-bit bitmap from set of banned UIDs."""
        bitmap = bytearray(32)  # 256 bits = 32 bytes
        for uid in banned_uids:
            if 0 <= uid < 256:
                byte_idx = uid // 8
                bit_idx = uid % 8
                bitmap[byte_idx] |= 1 << bit_idx
        return bytes(bitmap)

    # --- CompactCommitment interface ---
    def _compact_t(self) -> str:  # always use shortest token
        return "p"

    def _d_compact(self) -> str:
        """Serialize prices into `key=value;...` with keys sorted for determinism.

        Values are encoded as minimal decimal strings representing the FP18 value
        (e.g., 1.5 instead of 1500000000000000000), to minimize bytes.
        """
        items = []
        for k in sorted((self.prices or {}).keys()):
            v = (self.prices or {}).get(k)
            try:
                iv = int(v) if v is not None else None
            except (TypeError, ValueError):
                iv = None
            if iv is None:
                continue
            # Disallow separators inside keys to keep format unambiguous
            if "=" in k or ";" in k:
                continue
            items.append(f"{k}={_fp18_to_min_decimal_str(iv)}")
        return ";".join(items)

    @classmethod
    def _from_d_compact(cls, v: int, d: str) -> PriceCommitment:
        prices: dict[str, int] = {}
        if d:
            segs = [seg for seg in d.split(";") if seg]
            for seg in segs:
                if "=" not in seg:
                    continue
                k, sval = seg.split("=", 1)
                if not k:
                    continue
                s = sval.strip()
                try:
                    prices[k] = _parse_decimal_to_fp18_int(s)
                except ValueError:
                    continue
        return cls(t="p", prices=prices, v=v)

    def to_compact_bytes(self) -> bytes:
        """Serialize to compact format with binary ban bitmap suffix.

        Format: text_part + "|" + 32_bytes_bans
        where text_part is standard compact format (v;t;d)
        """
        text = self.to_compact()
        return join_commitment_binary(text, self.bans)

    @classmethod
    def from_compact_bytes(cls, raw: bytes | str) -> PriceCommitment:
        """Parse compact format with optional binary ban bitmap suffix.

        Handles both text-only (backwards compatible) and text+binary formats.
        """
        text, binary_suffix = split_commitment_binary(raw)

        # Parse text part using standard from_compact
        commit = cls.from_compact(text)

        # Set bans from binary suffix if present
        if binary_suffix and len(binary_suffix) == 32:
            # Use object.__setattr__ since model might be frozen/immutable
            object.__setattr__(commit, "bans", binary_suffix)

        return commit


async def compute_price_consensus(
    netuid: int,
    block_number: int,
    metrics: list[str] | None = None,
    *,
    gamma: float = 0.67,
    bt: Any,
) -> dict[str, int | None]:
    """
    Compute price consensus for one or more metrics at a specific block.

    - If `metrics` is provided, computes only those metrics.
    - If `metrics` is None, computes all metrics present in commitments.
    """
    block = await bt.block(block_number).get()
    subnet = bt.subnet(netuid)

    commits_raw = await subnet.commitments.fetch(block_hash=block.hash)
    neurons = await subnet.neurons.all(block_hash=block.hash)

    stakes_by_hotkey = _build_stakes_by_hotkey(neurons)
    prices_map_by_hotkey = _parse_price_commitments(commits_raw)

    metrics_to_compute = _determine_metrics(prices_map_by_hotkey, metrics)

    async def _run(metric: str) -> int | None:
        prices_by_hotkey = {hk: p[metric] for hk, p in prices_map_by_hotkey.items() if metric in p}
        return await asyncio.to_thread(_gamma_msi_fp18, prices_by_hotkey, stakes_by_hotkey, gamma)

    if not metrics_to_compute:
        return {}

    results_list = await asyncio.gather(*(_run(m) for m in metrics_to_compute), return_exceptions=False)
    return {m: res for m, res in zip(metrics_to_compute, results_list)}


def _to_fp18(value: float | int | str | Decimal) -> int:
    d = Decimal(str(value))
    return int((d * (10**18)).to_integral_value())


def _gamma_msi_fp18(
    prices_by_hotkey: dict[str, int],
    stakes_by_hotkey: dict[str, int],
    gamma: float,
) -> int | None:
    """Gamma‑MSI over FP18 prices.

    - prices_by_hotkey: mapping hotkey -> price in FP18
    - stakes_by_hotkey: mapping hotkey -> stake in FP18 or integer units
    - gamma in (0,1]: target coverage proportion
    Returns the FP18 price at the weighted median inside the minimal log‑width window
    whose cumulative stake covers at least gamma of total stake. Tie‑breakers are
    deterministic by (x, hotkey).
    """
    points: list[tuple[float, int, str, int]] = []  # (log_price, stake, hotkey, price_fp)
    for hotkey, price_fp in prices_by_hotkey.items():
        stake = stakes_by_hotkey.get(hotkey, 0)
        if price_fp and stake:
            log_price = math.log(price_fp / 1e18)
            points.append((log_price, stake, hotkey, price_fp))

    if not points:
        return None

    # Sort by log price, then hotkey for stability
    points.sort(key=lambda t: (t[0], t[2]))

    total_stake = sum(stake for _, stake, _, _ in points)
    target_stake = gamma * total_stake

    # Two‑pointer minimal window finder that covers target_stake
    best_window: tuple[float, int, int] | None = None  # (width, left_idx, right_idx)
    window_stake = 0
    left = 0

    for right in range(len(points)):
        window_stake += points[right][1]

        while left <= right and window_stake - points[left][1] >= target_stake:
            window_stake -= points[left][1]
            left += 1

        if window_stake >= target_stake:
            width = points[right][0] - points[left][0]
            if (
                best_window is None
                or width < best_window[0]
                or (
                    width == best_window[0]
                    and (points[right][0], points[left][0]) < (points[best_window[2]][0], points[best_window[1]][0])
                )
            ):
                best_window = (width, left, right)

    if best_window is None:
        return None

    _, left, right = best_window
    window = points[left : right + 1]
    window_total = sum(stake for _, stake, _, _ in window)

    acc = 0
    chosen_price_fp = window[-1][3]
    for _, stake, _, price_fp in window:
        acc += stake
        if acc * 2 >= window_total:
            chosen_price_fp = price_fp
            break

    return int(chosen_price_fp)


def _fp18_to_min_decimal_str(val: int) -> str:
    """Convert FP18 integer to minimal decimal string without trailing zeros.

    Examples:
    - 1000000000000000000 -> "1"
    - 1500000000000000000 -> "1.5"
    - 1250000000000000000 -> "1.25"
    - 0 -> "0"
    """
    base = 10**18
    if val == 0:
        return "0"
    neg = val < 0
    n = -val if neg else val
    int_part = n // base
    frac = n % base
    if frac == 0:
        out = str(int_part)
    else:
        frac_str = f"{frac:018d}".rstrip("0")
        out = f"{int_part}.{frac_str}"
    return f"-{out}" if neg else out


def _parse_decimal_to_fp18_int(s: str) -> int:
    """Parse a decimal string into FP18 integer.

    Accepts forms like "1", "1.5", ".5" and up to 18 fractional digits.
    Raises ValueError on invalid format or too many fractional digits.
    """
    txt = s.strip()
    if not txt:
        raise ValueError("empty numeric string")
    neg = False
    if txt[0] in "+-":
        neg = txt[0] == "-"
        txt = txt[1:]
    if not txt:
        raise ValueError("sign without digits")
    if "." in txt:
        left, right = txt.split(".", 1)
    else:
        left, right = txt, ""
    left = left or "0"
    if not left.isdigit() or (right and not right.isdigit()):
        raise ValueError("invalid decimal digits")
    right = right.rstrip("0")  # normalize to minimal form
    if len(right) > 18:
        raise ValueError("too many fractional digits")
    base = 10**18
    int_part = int(left) * base
    frac = int((right or "0").ljust(18, "0")) if right else 0
    out = int_part + frac
    return -out if neg else out


def _build_stakes_by_hotkey(neurons: list[Any]) -> dict[str, int]:
    """Convert neuron list to a hotkey->stake_fp18 map.

    Stake is a dict mapping coldkey -> stake_rao (from Vec<(AccountId, u64)>).
    We sum all stakes from different coldkeys to get total stake.

    Raises if a neuron has an invalid stake type; unexpected errors should propagate.
    """
    out: dict[str, int] = {}
    for n in neurons:
        # n.stake is a dict: {coldkey_address: stake_in_rao}
        # Sum all stake values and convert from RAO to FP18
        if isinstance(n.stake, dict):
            total_stake_rao = sum(n.stake.values())
        else:
            # Fallback for non-dict stake (shouldn't happen with real data)
            total_stake_rao = n.stake

        # Convert from RAO (1e-9 TAO) to FP18
        # stake_rao is already in smallest unit, convert to TAO first then to FP18
        stake_tao = total_stake_rao / 1e9  # RAO to TAO
        stake_fp = _to_fp18(stake_tao)

        if stake_fp > 0:
            out[n.hotkey] = stake_fp

    return out


def _parse_price_commitments(commits: dict[str, bytes | str]) -> dict[str, dict[str, int]]:
    """Parse only valid PriceCommitment payloads into hotkey->metric->price_fp18.

    Skips entries that fail PriceCommitment validation; unexpected errors propagate.
    """
    from .parser import parse_commitment

    out: dict[str, dict[str, int]] = {}
    for hotkey, raw in commits.items():
        # Use generic parser with type filtering (auto-discovers "p" token)
        model = parse_commitment(raw, expected_types=[PriceCommitment])
        if model is None:
            continue
        prices = model.prices_int()
        if prices:
            out[hotkey] = prices
    return out


def _determine_metrics(prices_map_by_hotkey: dict[str, dict[str, int]], metrics: list[str] | None) -> list[str]:
    if metrics is not None:
        return list(metrics)
    all_metrics: set[str] = set()
    for pmap in prices_map_by_hotkey.values():
        all_metrics.update(pmap.keys())
    return sorted(all_metrics)


async def compute_ban_consensus(
    netuid: int,
    block_number: int,
    *,
    bt: Any,
) -> set[str]:
    """
    Compute ban consensus at a specific block.

    Returns the set of banned hotkeys based on >50% stake consensus.

    Algorithm:
    1. Parse all price commitments (with or without ban bitmaps)
    2. For each UID (0-255), accumulate stake from validators who marked it as banned
    3. UIDs with >50% of total stake supporting the ban are considered banned
    4. Map banned UIDs to their current hotkeys

    Args:
        netuid: Subnet ID
        block_number: Block number to compute consensus at
        bt: Bittensor instance

    Returns:
        Set of banned hotkeys
    """
    block = await bt.block(block_number).get()
    subnet = bt.subnet(netuid)

    commits_raw = await subnet.commitments.fetch(block_hash=block.hash)
    neurons = await subnet.neurons.all(block_hash=block.hash)

    stakes_by_hotkey = _build_stakes_by_hotkey(neurons)

    # Parse all commitments to get ban bitmaps (including zero bitmaps)
    all_commitments_by_hotkey = _parse_all_price_commitments(commits_raw)

    # Build UID to hotkey mapping
    uid_to_hotkey = {n.uid: n.hotkey for n in neurons}

    # Compute total stake of validators who submitted any price commitment
    total_stake = sum(
        stakes_by_hotkey.get(hotkey, 0) for hotkey in all_commitments_by_hotkey.keys() if hotkey in stakes_by_hotkey
    )

    if total_stake == 0:
        logger.debug("No stake in commitments", netuid=netuid, block=block_number)
        return set()

    # Accumulate stake for each UID bit
    stake_per_uid: dict[int, int] = {}
    for hotkey, commitment in all_commitments_by_hotkey.items():
        stake = stakes_by_hotkey.get(hotkey, 0)
        if stake == 0:
            continue

        # Extract banned UIDs from bitmap
        banned_uids = _extract_banned_uids_from_bitmap(commitment.bans)
        for uid in banned_uids:
            stake_per_uid[uid] = stake_per_uid.get(uid, 0) + stake

    # Filter UIDs with >50% stake support
    threshold = total_stake // 2  # Integer division for >50% (strictly greater)
    banned_uids = {uid for uid, stake in stake_per_uid.items() if stake > threshold}

    # Map UIDs to hotkeys
    banned_hotkeys = {uid_to_hotkey[uid] for uid in banned_uids if uid in uid_to_hotkey}

    logger.info(
        "Ban consensus computed",
        netuid=netuid,
        block=block_number,
        total_stake=total_stake,
        banned_uids=sorted(banned_uids),
        banned_hotkeys_count=len(banned_hotkeys),
    )

    return banned_hotkeys


def _parse_all_price_commitments(commits: dict[str, bytes | str]) -> dict[str, PriceCommitment]:
    """Parse all price commitments including ban bitmaps.

    Returns:
        Mapping of hotkey -> PriceCommitment
    """
    from .parser import parse_commitment

    out: dict[str, PriceCommitment] = {}
    for hotkey, raw in commits.items():
        # Use generic parser with type filtering
        model = parse_commitment(raw, expected_types=[PriceCommitment])
        if model is not None:
            out[hotkey] = model

    return out


def _parse_ban_commitments(commits: dict[str, bytes | str]) -> dict[str, bytes]:
    """Parse price commitments and extract ban bitmaps.

    Returns:
        Mapping of hotkey -> ban_bitmap (32 bytes)
    """
    from .parser import parse_commitment

    out: dict[str, bytes] = {}
    for hotkey, raw in commits.items():
        # Use generic parser with type filtering
        model = parse_commitment(raw, expected_types=[PriceCommitment])
        if model is None:
            continue

        # Only include if ban bitmap is non-zero (has at least one ban)
        if model.bans and model.bans != b"\x00" * 32:
            out[hotkey] = model.bans

    return out


def _extract_banned_uids_from_bitmap(bitmap: bytes) -> set[int]:
    """Extract set of banned UIDs from 256-bit bitmap."""
    banned = set()
    for uid in range(256):
        byte_idx = uid // 8
        bit_idx = uid % 8
        if byte_idx < len(bitmap):
            if (bitmap[byte_idx] >> bit_idx) & 1:
                banned.add(uid)
    return banned
