import csv
import json
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import pytest
from asgiref.sync import sync_to_async
from django.utils import timezone
from scalecodec.utils import ss58

from infinite_hashes.consensus.bidding import BiddingCommitment
from infinite_hashes.consensus.price import _fp18_to_min_decimal_str
from infinite_hashes.validator import auction_processing as auct
from infinite_hashes.validator import tasks as vtasks
from infinite_hashes.validator.models import AuctionResult


def _load_workers_csv(limit: int | None = None) -> dict[str, list[float]]:
    """Load miners -> list of ASIC capacities (in PH) from CSV.

    - Groups rows by base SS58 before the first '.' as the hotkey.
    - Collects all PH entries per miner (ASICs), capped to keep test fast.
    - Returns dict {hotkey: [ph1, ph2, ...]} for up to `limit` miners.
    """
    csv_path = Path(__file__).resolve().parent / "data" / "pool-workers-2025-08-21.csv"
    by_hotkey: dict[str, list[float]] = {}
    with csv_path.open("r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["Worker Name"].strip()
            base_hk = name.split(".", 1)[0]
            val = row["Hashrate"].strip()
            if not val:
                continue
            try:
                num_str, unit = val.split()
            except ValueError:
                continue
            try:
                x = float(num_str)
            except ValueError:
                continue
            unit = unit.upper()
            if unit == "TH":
                ph = x / 1000.0
            elif unit == "PH":
                ph = x
            else:
                continue
            lst = by_hotkey.setdefault(base_hk, [])
            lst.append(ph)
            # no limiting on number of miners; gather all
    # Sort capacities per miner descending; keep all entries (no cap)
    out: dict[str, list[float]] = {}
    keys = sorted(by_hotkey.keys()) if limit is None else sorted(by_hotkey.keys())[:limit]
    for hk in keys:
        caps = sorted(by_hotkey[hk], reverse=True)
        out[hk] = caps
    return out


def _bt_proxy_factory(sim, start_block: int, end_block: int):
    """Create a Bittensor-like client proxy backed by the simulator."""

    def _block(n: int):
        class _B:
            def __init__(self, number: int):
                self.number = number
                self.hash = f"0x{number:064x}"

            async def get_timestamp(self):
                return timezone.now()

        class _BH:
            def __init__(self, number: int):
                self._b = _B(number)

            async def get(self):
                return self._b

        return _BH(n)

    class _BTProxy:
        def __init__(self, *_args, **_kwargs):
            self._sim = sim
            # Create a wrapper for subtensor with mocked state
            self._subtensor_wrapper = self._create_subtensor_wrapper()

        def _create_subtensor_wrapper(self):
            class _StateWrapper:
                def __init__(self, original_state):
                    self._original_state = original_state

                async def getStorage(self, key, *params, **kwargs):
                    # Mock MechanismEmissionSplit storage
                    if key == "SubtensorModule.MechanismEmissionSplit":
                        # Return a simple 100% split for single mechanism
                        return [0, 65535]  # 65535/65535 = 100% for mechanism 0
                    # Delegate to original state for other calls
                    return await self._original_state.getStorage(key, *params, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._original_state, name)

            class _SubtensorWrapper:
                def __init__(self, original_subtensor):
                    self._original_subtensor = original_subtensor
                    self.state = _StateWrapper(original_subtensor.state)

                def __getattr__(self, name):
                    return getattr(self._original_subtensor, name)

            return _SubtensorWrapper(self._sim.bt.subtensor)

        @property
        def subtensor(self):
            return self._subtensor_wrapper

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        class head:
            @staticmethod
            async def get():
                class _H:
                    number = end_block + 100

                return _H()

        def subnet(self, netuid):  # noqa: ARG002
            sim_bt_subnet = self._sim.bt.subnet(netuid)

            class _Subnet:
                tempo = 600

                async def get(self):
                    return self

                @staticmethod
                def epoch(_n: int):
                    class _E:
                        start = start_block

                    return _E()

                class commitments:
                    @staticmethod
                    async def fetch(*, block_hash: str):  # noqa: ARG002
                        return await sim_bt_subnet.commitments.fetch(block_hash=block_hash)

                class neurons:
                    @staticmethod
                    async def all(*, block_hash: str):  # noqa: ARG002
                        return await sim_bt_subnet.neurons.all(block_hash=block_hash)

            return _Subnet()

        def block(self, n: int):
            return _block(n)

    _BTProxy._block = _block  # type: ignore[attr-defined]
    return _BTProxy


def _build_bids_by_hotkey(workers: dict[str, list[float]]) -> dict[str, list[tuple[str, int]]]:
    """Create compact-encodable bid chunks per hotkey, splitting across clones when needed."""
    bids_by_hotkey: dict[str, list[tuple[str, int]]] = {}
    sorted_hks = sorted(workers.keys())
    for hk in sorted_hks:
        caps = workers[hk]
        idx = sorted_hks.index(hk)
        n_levels = 3 + (idx % 3)
        base = 1.03 + 0.01 * (idx % 3)
        levels = [round(base + 0.01 * j, 3) for j in range(n_levels)]
        items: list[tuple[str, int]] = []
        for j, cap in enumerate(sorted(caps, reverse=True)):
            mult = levels[j % n_levels]
            price_fp = int(mult * (10**18))
            cap_fp = int(round(cap * 10**18))
            cap_s = _fp18_to_min_decimal_str(cap_fp)
            items.append((cap_s, price_fp))

        chunks: list[list[tuple[str, int]]] = []
        for cap_s, price_fp in items:
            placed = False
            for chunk in chunks:
                trial = list(chunk) + [(cap_s, price_fp)]
                try:
                    _ = BiddingCommitment(t="b", bids=trial, v=1).to_compact().encode("utf-8")
                    chunk.append((cap_s, price_fp))
                    placed = True
                    break
                except ValueError:
                    continue
            if not placed:
                chunk = [(cap_s, price_fp)]
                _ = BiddingCommitment(t="b", bids=chunk, v=1).to_compact().encode("utf-8")
                chunks.append(chunk)

        base_pub = _ss58_pubkey_bytes(hk)
        for part_idx, chunk in enumerate(chunks):
            if part_idx == 0:
                out_hk = hk
            else:
                pub = bytes(base_pub[:-1] + bytes([(base_pub[-1] + part_idx) % 256]))
                out_hk = ss58.ss58_encode(pub)
            bids_by_hotkey[out_hk] = list(chunk)
    return bids_by_hotkey


def _apply_commitments(sim, settings, bids_by_hotkey: dict[str, list[tuple[str, int]]]) -> None:
    commits_storage: dict[str, bytes] = {}
    for hk, bids in bids_by_hotkey.items():
        c = BiddingCommitment(t="b", bids=list(bids), v=1)
        comp = c.to_compact().encode("utf-8")
        commits_storage[hk] = comp
    sim.set_commitments(settings.BITTENSOR_NETUID, commits_storage)


def _make_fake_consensus(start_block: int, end_block: int, budget_frac: float, total_csv_ph: float):
    async def _fake_consensus(netuid: int, block_number: int, metrics: list[str], *, bt: Any):  # noqa: ARG002
        FP = 10**18
        miners_share = 0.41
        blocks = end_block - start_block + 1
        target_ph = max(0.000001, budget_frac * total_csv_ph)
        hashp_usdc = (miners_share * blocks) / target_ph
        return {"TAO_USDC": 1 * FP, "ALPHA_TAO": 1 * FP, "HASHP_USDC": int(hashp_usdc * FP)}

    return _fake_consensus


def _make_selector_wrapper(winners_record: dict[str, list[dict[str, Any]]]):
    async def _selector_wrapper(bt, netuid, start_b, end_b, bids_map, **kw):  # noqa: ARG002
        from infinite_hashes.consensus.bidding import select_auction_winners_async as real

        winners = await real(bt, netuid, start_b, end_b, bids_map, **kw)
        winners_record["winners"] = winners
        return winners

    return _selector_wrapper


def _choose_below_threshold(bids_by_hotkey: dict[str, list[tuple[str, int]]]) -> set[str]:
    all_hks_sorted = sorted(bids_by_hotkey.keys())
    return set(all_hks_sorted[::2])


def _make_fake_hashrates(
    bids_by_hotkey: dict[str, list[tuple[str, int]]],
    winners_record: dict[str, list[dict[str, Any]]],
    below_threshold: set[str],
):
    async def _fake_hashrates(subaccount_name, start, end, page_size=100, tick_size=vtasks.TickSize.HOUR):  # noqa: ARG002
        out: dict[str, list[list[int]]] = {}
        won_by_hotkey: dict[str, float] = {}
        for w in winners_record.get("winners", []):
            won_by_hotkey[w["hotkey"]] = won_by_hotkey.get(w["hotkey"], 0.0) + float(w["hashrate"])  # PH
        for hk in bids_by_hotkey.keys():
            requested_ph = won_by_hotkey.get(hk, 0.0)
            delivered_ph = (0.88 if hk in below_threshold else 0.95) * requested_ph
            delivered_hs = int(delivered_ph * 1e15)
            out[hk] = [[delivered_hs]]  # List of samples, each sample is list of worker hashrates
        return out

    return _fake_hashrates


def _aggregate_winners_snapshot(winners: list[dict[str, Any]], budget_frac: float) -> dict[str, Any]:
    getcontext().prec = 40
    agg: dict[tuple[str, int], Decimal] = {}
    total_dec = Decimal(0)
    for w in winners:
        hk = str(w["hotkey"])  # split hotkeys remain separate
        price = int(w["price"])  # FP18 int
        hr_dec = Decimal(str(w["hashrate"]))
        agg[(hk, price)] = agg.get((hk, price), Decimal(0)) + hr_dec
        total_dec += hr_dec

    def _dec_to_min_str(d: Decimal) -> str:
        fp = int((d * Decimal(10**18)).to_integral_value())
        return _fp18_to_min_decimal_str(fp)

    entries = [{"hotkey": hk, "price": price, "hashrate": _dec_to_min_str(val)} for (hk, price), val in agg.items()]
    entries.sort(key=lambda e: (e["hotkey"], e["price"]))
    return {
        "budget_frac": float(budget_frac),
        "total": _dec_to_min_str(total_dec),
        "entries": entries,
    }


def _assert_snapshot_for_budget(budget_frac: float, expected_data: dict[str, Any]) -> None:
    exp_path = Path(__file__).resolve().parent / "data" / f"expected_winners_{budget_frac:.2f}.json"
    if exp_path.exists():
        on_disk = json.loads(exp_path.read_text())
        assert on_disk == expected_data
    else:
        exp_path.write_text(json.dumps(expected_data, indent=2, sort_keys=True))
        pytest.fail(f"Stored expected winners snapshot to {exp_path}. Re-run the test to validate against it.")


@pytest.mark.django_db
@pytest.mark.asyncio
@pytest.mark.parametrize("budget_frac", [0.2, 0.5, 0.9, 1.1])
async def test_process_auctions_creates_results_with_delivery_threshold(monkeypatch, settings, sim, budget_frac):
    # Use a deterministic window set: single 10-min window (50 blocks)
    start_block = 10_000
    end_block = start_block + 49
    # Patch validation_windows_for_subnet_epoch to return our single test window
    from infinite_hashes.auctions import utils as auction_utils

    monkeypatch.setattr(
        auction_utils, "validation_windows_for_subnet_epoch", lambda *_args, **_kw: [(start_block, end_block)]
    )

    # Subtensor simulator context and proxy
    sim.set_block_context(number=end_block + 100)
    _BTProxy = _bt_proxy_factory(sim, start_block, end_block)
    monkeypatch.setattr(auct.turbobt, "Bittensor", _BTProxy)
    monkeypatch.setattr(auct, "DELIVERY_THRESHOLD_FRACTION", 0.95)

    # Prepare commitments from CSV workers
    workers = _load_workers_csv(limit=None)
    total_csv_ph = sum(sum(v) for v in workers.values())
    bids_by_hotkey = _build_bids_by_hotkey(workers)
    _apply_commitments(sim, settings, bids_by_hotkey)

    # Fix price consensus to make budget predictable
    fake_consensus = _make_fake_consensus(start_block, end_block, budget_frac, total_csv_ph)
    monkeypatch.setattr("infinite_hashes.consensus.bidding.compute_price_consensus", fake_consensus)

    # Mock ban consensus to return empty set (no bans)
    async def _no_bans(*args, **kwargs):
        return set()

    monkeypatch.setattr("infinite_hashes.consensus.price.compute_ban_consensus", _no_bans)

    # Mock scraping data gaps check to return False (no gaps = delivery check proceeds)
    async def _no_gaps(*args, **kwargs):
        return False

    monkeypatch.setattr(auct, "has_scraping_data_gaps", _no_gaps)

    # Capture winners to shape delivery exactly to what was requested
    winners_record: dict[str, list[dict[str, Any]]] = {}
    monkeypatch.setattr(auct, "select_auction_winners_async", _make_selector_wrapper(winners_record), raising=True)

    # Define pass/fail miners deterministically and patch hashrates provider
    below_threshold = _choose_below_threshold(bids_by_hotkey)
    monkeypatch.setattr(
        auct,
        "get_hashrates_from_snapshots_async",
        _make_fake_hashrates(bids_by_hotkey, winners_record, below_threshold),
    )

    # Ensure clean slate
    await sync_to_async(lambda: AuctionResult.objects.all().delete())()

    # Run auction processing
    processed = await auct.process_auctions_async()
    assert processed == 1

    # Verify AuctionResult created and winners populated with delivery flags
    ar = await sync_to_async(lambda: AuctionResult.objects.get(end_block=end_block))()
    assert ar.epoch_start == start_block
    assert isinstance(ar.winners, list) and ar.winners
    # All stored winners should be delivered=True (non-delivered are banned and excluded)
    delivered_flags = [w.get("delivered", False) for w in ar.winners]
    assert all(delivered_flags), "All stored winners should have delivered=True"

    # Aggregate winners per (hotkey, price) and assert against stored expectations
    expected_data = _aggregate_winners_snapshot(ar.winners, budget_frac)
    _assert_snapshot_for_budget(budget_frac, expected_data)


def _ss58_pubkey_bytes(hk: str) -> bytearray:
    """Decode SS58 hotkey to 32-byte public key, handling hex or bytes forms."""
    val = ss58.ss58_decode(hk)
    if isinstance(val, bytes | bytearray):
        return bytearray(val)
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("0x"):
            return bytearray(bytes.fromhex(s[2:]))
        try:
            return bytearray(bytes.fromhex(s))
        except ValueError:
            return bytearray(s.encode("utf-8"))
    # Fallback: coerce to bytes
    return bytearray(bytes(val))
