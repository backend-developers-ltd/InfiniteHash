import asyncio
import datetime
from decimal import Decimal, InvalidOperation

import structlog
import turbobt
from celery.utils.log import get_task_logger
from django.conf import settings

from infinite_hashes.auctions import utils as auction_utils
from infinite_hashes.consensus.bidding import select_auction_winners_async
from infinite_hashes.consensus.price import _parse_decimal_to_fp18_int
from infinite_hashes.validator.models import AuctionResult, BannedMiner, ValidatorScrapingEvent

from .hashrates import get_hashrates_from_snapshots_async

logger = structlog.wrap_logger(get_task_logger(__name__))
DELIVERY_THRESHOLD_FRACTION = 0.0  # Minimum delivered share required to consider winner compliant


async def existing_end_blocks(epoch_start: int | None = None) -> set[int]:
    """Get existing end blocks for auction results.

    Args:
        epoch_start: If provided, only return end blocks for this epoch.
                    If None, return all end blocks across all epochs.
    """

    def _fetch():
        qs = AuctionResult.objects.all()
        if epoch_start is not None:
            qs = qs.filter(epoch_start=epoch_start)
        return list(qs.values_list("end_block", flat=True))

    rows = await asyncio.to_thread(_fetch)
    return set(rows)


async def has_scraping_data_gaps(start_block: int, end_block: int) -> bool:
    """Check if validator has data gaps > 7 blocks during the window.

    Scraping should occur every ~2.5 blocks (30s scrape interval / 12s block time).
    A gap of > 7 blocks (~84s) indicates missed scraping runs (2+ consecutive failures).

    Args:
        start_block: Window start block
        end_block: Window end block

    Returns:
        True if there are gaps > 7 blocks (data incomplete, skip underdelivery check)
        False if data is complete (safe to check underdelivery)
    """

    def _fetch():
        return list(
            ValidatorScrapingEvent.objects.filter(
                block_number__gte=start_block,
                block_number__lte=end_block,
            )
            .order_by("block_number")
            .values_list("block_number", flat=True)
        )

    scraping_blocks = await asyncio.to_thread(_fetch)

    if not scraping_blocks:
        # No scraping events at all during window - validator was down/broken
        logger.warning(
            "No scraping events found during window - skipping underdelivery check",
            start_block=start_block,
            end_block=end_block,
        )
        return True

    # Check for gaps > 7 blocks between consecutive scraping events
    MAX_GAP_BLOCKS = 7
    for i in range(1, len(scraping_blocks)):
        gap = scraping_blocks[i] - scraping_blocks[i - 1]
        if gap > MAX_GAP_BLOCKS:
            logger.warning(
                "Large gap in scraping events detected - skipping underdelivery check",
                start_block=start_block,
                end_block=end_block,
                gap_blocks=gap,
                gap_between=(scraping_blocks[i - 1], scraping_blocks[i]),
                max_allowed_gap=MAX_GAP_BLOCKS,
            )
            return True

    # Also check gap from window start to first scraping event
    first_gap = scraping_blocks[0] - start_block
    if first_gap > MAX_GAP_BLOCKS:
        logger.warning(
            "Large gap at window start - skipping underdelivery check",
            start_block=start_block,
            end_block=end_block,
            first_scraping_block=scraping_blocks[0],
            gap_blocks=first_gap,
        )
        return True

    # Check gap from last scraping event to window end
    last_gap = end_block - scraping_blocks[-1]
    if last_gap > MAX_GAP_BLOCKS:
        logger.warning(
            "Large gap at window end - skipping underdelivery check",
            start_block=start_block,
            end_block=end_block,
            last_scraping_block=scraping_blocks[-1],
            gap_blocks=last_gap,
        )
        return True

    logger.debug(
        "Scraping data is complete - underdelivery check will proceed",
        start_block=start_block,
        end_block=end_block,
        scraping_events_count=len(scraping_blocks),
        max_gap=max((scraping_blocks[i] - scraping_blocks[i - 1]) for i in range(1, len(scraping_blocks)))
        if len(scraping_blocks) > 1
        else 0,
    )
    return False


def mark_delivery(winners: list[dict], hashrates: dict[str, list[list[int]]]) -> list[dict]:
    aggregated_enabled = settings.AGGREGATED_DELIVERIES

    # Group winners by hotkey and collect their commitments
    hotkey_commitments: dict[str, list[float]] = {}
    hotkey_prices: dict[str, set] = {}
    for w in winners:
        hk = w["hotkey"]
        req_ph = float(_parse_decimal_to_fp18_int(str(w["hashrate"]))) / 1e18
        if hk not in hotkey_commitments:
            hotkey_commitments[hk] = []
        hotkey_commitments[hk].append(req_ph)
        if aggregated_enabled:
            hotkey_prices.setdefault(hk, set()).add(w.get("price"))

    # Sort commitments by size (descending) for greedy allocation
    for hk in hotkey_commitments:
        hotkey_commitments[hk].sort(reverse=True)

    # Check delivery once per hotkey using per-commitment allocation
    def _delivered_ok(hk: str) -> bool:
        vals = hashrates.get(hk, [])
        commitments = hotkey_commitments[hk]
        total_commitment = sum(commitments)
        prices = hotkey_prices.get(hk, set()) if aggregated_enabled else set()
        use_aggregated = aggregated_enabled and len(prices) <= 1

        if not vals:
            logger.debug("No hashrate values for hotkey", hotkey=hk, commitments=commitments)
            return False

        # For each sample (list of worker hashrates), match workers to commitments
        # Sort both sample values and commitments descending, then pair and cap
        total_delivered = 0.0
        sample_details = []

        for sample in vals:
            # Convert worker hashrates to PH/s and sort descending
            sample_ph_list = sorted([v / 1e15 for v in sample], reverse=True)

            if use_aggregated:
                # Aggregate all workers and cap at the total commitment
                sample_total_ph = sum(sample_ph_list)
                sample_delivered = min(sample_total_ph, total_commitment)
                total_delivered += sample_delivered

                sample_details.append(
                    {
                        "aggregation_mode": "aggregated",
                        "workers_ph": sample_ph_list,
                        "delivered": sample_delivered,
                        "total_workers_ph": sample_total_ph,
                        "capped": max(sample_total_ph - total_commitment, 0.0),
                    }
                )
                continue

            # Match each worker's hashrate to a commitment (both sorted descending)
            allocated = []
            capped_amount = 0.0
            for i, commitment_ph in enumerate(commitments):
                if i < len(sample_ph_list):
                    # Cap worker's delivery at the commitment size
                    alloc = min(sample_ph_list[i], commitment_ph)
                    # Track how much was capped
                    capped_amount += sample_ph_list[i] - alloc
                else:
                    # No worker for this commitment
                    alloc = 0.0
                allocated.append(alloc)

            # Also track workers beyond commitments count (fully ignored)
            workers_beyond = sum(sample_ph_list[len(commitments) :]) if len(sample_ph_list) > len(commitments) else 0.0
            total_capped = capped_amount + workers_beyond

            sample_delivered = sum(allocated)
            total_delivered += sample_delivered

            sample_details.append(
                {
                    "aggregation_mode": "per_commitment",
                    "workers_ph": sample_ph_list,
                    "allocated": allocated,
                    "delivered": sample_delivered,
                    "capped": total_capped,
                    "workers_beyond": workers_beyond,
                }
            )

        # Average delivered across all samples
        avg_delivered = total_delivered / len(vals)
        threshold = DELIVERY_THRESHOLD_FRACTION * total_commitment
        is_delivered = avg_delivered >= threshold

        logger.debug(
            "Delivery check",
            hotkey=hk,
            commitments=commitments,
            total_commitment=total_commitment,
            samples_count=len(vals),
            avg_delivered=avg_delivered,
            threshold=threshold,
            is_delivered=is_delivered,
            aggregated_delivery=use_aggregated,
            price_levels=list(prices) if prices else [],
            sample_details=sample_details[:3] if len(sample_details) > 3 else sample_details,  # Log first 3 samples
        )
        return is_delivered

    # Check each hotkey once and cache the result
    delivery_status: dict[str, bool] = {}
    for hk in hotkey_commitments:
        delivery_status[hk] = _delivered_ok(hk)

    # Apply the delivery status to all wins for each hotkey
    out: list[dict] = []
    for w in winners:
        out.append(
            {
                "hotkey": w["hotkey"],
                "hashrate": w["hashrate"],
                "price": w["price"],
                "delivered": delivery_status[w["hotkey"]],
            }
        )
    return out


async def record_result(
    *,
    epoch_start: int,
    start_block: int,
    end_block: int,
    commitments_count: int,
    winners_with_delivery: list[dict],
    skipped_delivery_check: bool = False,
    underdelivered_hotkeys: list[str] | None = None,
    banned_hotkeys: list[str] | None = None,
    hashp_usdc_fp18: int | None = None,
    alpha_tao_fp18: int | None = None,
    tao_usdc_fp18: int | None = None,
    commitments_ph_by_hotkey: dict[str, float] | None = None,
    wins_ph_by_hotkey: dict[str, float] | None = None,
    total_budget_ph: float | None = None,
) -> None:
    """Record auction result with price consensus for weight calculation.

    Args:
        skipped_delivery_check: True when delivery verification was skipped due to scraping gaps
        underdelivered_hotkeys: Hotkeys that failed delivery validation during the window
        banned_hotkeys: Hotkeys that were banned via ban consensus for the window
        hashp_usdc_fp18: Hashrate price consensus (USDC per PH per day) in FP18 format
        alpha_tao_fp18: ALPHA/TAO price consensus in FP18 format
        tao_usdc_fp18: TAO/USDC price consensus in FP18 format
    """
    from decimal import Decimal

    # Convert FP18 integers to Decimal for storage
    FP18 = 10**18
    hashp_usdc = Decimal(hashp_usdc_fp18) / FP18 if hashp_usdc_fp18 else None
    alpha_tao = Decimal(alpha_tao_fp18) / FP18 if alpha_tao_fp18 else None
    tao_usdc = Decimal(tao_usdc_fp18) / FP18 if tao_usdc_fp18 else None

    # Keep atomicity if caller wraps in a larger transaction
    await AuctionResult.objects.acreate(
        epoch_start=epoch_start,
        start_block=start_block,
        end_block=end_block,
        commitments_count=commitments_count,
        winners=winners_with_delivery,
        skipped_delivery_check=skipped_delivery_check,
        underdelivered_hotkeys=sorted(set(underdelivered_hotkeys or [])),
        banned_hotkeys=sorted(set(banned_hotkeys or [])),
        hashp_usdc=hashp_usdc,
        alpha_tao=alpha_tao,
        tao_usdc=tao_usdc,
        commitments_ph_by_hotkey=commitments_ph_by_hotkey or {},
        wins_ph_by_hotkey=wins_ph_by_hotkey or {},
        total_budget_ph=total_budget_ph,
    )


async def process_auctions_async() -> int:
    """Process finished auction windows based on validation rhythm.

    Validation windows start 60 blocks before subnet epoch:
    - Window 0: Check PREVIOUS validation epoch's window 5 and missed windows
    - Windows 1-5: Check CURRENT validation epoch's finished windows and missed windows

    This function relies on symbols in `validator.tasks` for patchability in tests.
    """
    processed = 0

    async with turbobt.Bittensor(settings.BITTENSOR_NETWORK) as bittensor:
        head, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(settings.BITTENSOR_NETUID).get(),
        )

        current_epoch = subnet.epoch(head.number)
        blocks_per_window = auction_utils.blocks_per_window_default()
        subnet_tempo = subnet.tempo + 1

        # Determine which validation window we're in
        window_number, (window_start, window_end) = auction_utils.current_validation_window(
            head.number, current_epoch.start, blocks_per_window
        )

        # Determine which subnet epoch the returned window belongs to
        # The window_start tells us the validation epoch start
        if window_number == 0:
            # Window 0 starts at a new validation epoch boundary
            # Calculate which subnet epoch this window belongs to
            validation_epoch_start_for_window = window_start
            subnet_epoch_for_window = validation_epoch_start_for_window + blocks_per_window

            # Process the PREVIOUS subnet epoch (relative to the window we're in)
            target_subnet_epoch_start = subnet_epoch_for_window - subnet_tempo
        else:
            # Windows 1-5: we're processing the current subnet epoch
            # Find the validation epoch start that contains this window
            current_validation_start, _ = auction_utils.validation_epoch_for_subnet_epoch(
                current_epoch.start, blocks_per_window
            )
            target_subnet_epoch_start = current_epoch.start

        # Get the validation epoch boundaries for the target
        target_validation_start, target_validation_end = auction_utils.validation_epoch_for_subnet_epoch(
            target_subnet_epoch_start, blocks_per_window
        )

        # Determine which validation epoch to check based on current window
        candidates = []
        validation_epochs_to_check = [(target_subnet_epoch_start, target_validation_start, target_validation_end)]

        logger.debug(
            "Determined target validation epoch",
            window_number=window_number,
            window_range=(window_start, window_end),
            head_number=head.number,
            current_subnet_epoch=current_epoch.start,
            target_subnet_epoch=target_subnet_epoch_start,
            target_validation_epoch=(target_validation_start, target_validation_end),
        )

        # Get existing end blocks to avoid reprocessing
        existing = await existing_end_blocks()

        # Collect candidate windows from the validation epochs to check
        for subnet_epoch_start, val_start, val_end in validation_epochs_to_check:
            # Get validation windows for this subnet epoch
            windows = auction_utils.validation_windows_for_subnet_epoch(subnet_epoch_start, blocks_per_window)

            # Find finished windows that haven't been processed
            for window_num, (start, end) in enumerate(windows, start=0):
                if end <= head.number and end not in existing:
                    candidates.append((subnet_epoch_start, start, end, window_num))

            logger.debug(
                "Checked validation epoch for unprocessed windows",
                subnet_epoch_start=subnet_epoch_start,
                validation_range=(val_start, val_end),
                head_number=head.number,
                windows=windows,
                candidates_found=sum(1 for s, e, _, _ in candidates if val_start <= s <= val_end),
            )

        logger.info(
            "Auction processing",
            head_number=head.number,
            current_epoch_start=current_epoch.start,
            current_window=window_number,
            tempo=subnet.tempo,
            blocks_per_window=blocks_per_window,
            existing_end_blocks_count=len(existing),
            candidates_count=len(candidates),
            candidates=[(epoch, s, e, w) for epoch, s, e, w in candidates],
        )

        for subnet_epoch_start, start_block, end_block, window_num in candidates:
            try:
                start_blk, bids_by_hotkey, consensus_banned_hotkeys = await auction_utils.fetch_bids_for_start_block(
                    bittensor, subnet, start_block, settings.BITTENSOR_NETUID
                )
                cbc_seed = auction_utils.cbc_seed_from_hash(start_blk.hash)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Window skipped: unable to fetch bids/start block (likely pruned/unknown block)",
                    start_block=start_block,
                    end_block=end_block,
                    window_number=window_num,
                    error=str(exc),
                )
                await record_result(
                    epoch_start=subnet_epoch_start,
                    start_block=start_block,
                    end_block=end_block,
                    commitments_count=0,
                    winners_with_delivery=[],
                    skipped_delivery_check=True,
                    underdelivered_hotkeys=[],
                    banned_hotkeys=[],
                    hashp_usdc_fp18=None,
                    alpha_tao_fp18=None,
                    tao_usdc_fp18=None,
                    commitments_ph_by_hotkey={},
                    wins_ph_by_hotkey={},
                    total_budget_ph=None,
                )
                processed += 1
                continue

            # Compute price consensus for weight calculation
            from infinite_hashes.consensus.price import compute_price_consensus

            prices = await compute_price_consensus(
                settings.BITTENSOR_NETUID,
                start_block,
                ["HASHP_USDC", "ALPHA_TAO", "TAO_USDC"],
                bt=bittensor,
            )
            hashp_usdc_fp18 = prices.get("HASHP_USDC") if prices else None
            alpha_tao_fp18 = prices.get("ALPHA_TAO") if prices else None
            tao_usdc_fp18 = prices.get("TAO_USDC") if prices else None

            winners, budget_ph = await select_auction_winners_async(
                bittensor,
                settings.BITTENSOR_NETUID,
                start_block,
                end_block,
                bids_by_hotkey,
                cbc_max_nodes=settings.AUCTION_ILP_CBC_MAX_NODES,
                cbc_seed=cbc_seed,
                max_price_multiplier=settings.MAX_PRICE_MULTIPLIER,
            )

            start_ts, end_ts = await auction_utils.window_timestamps(bittensor, start_block, end_block)
            logger.debug(
                "Window timestamps retrieved",
                start_block=start_block,
                end_block=end_block,
                start_ts=start_ts.isoformat(),
                end_ts=end_ts.isoformat(),
            )
            # Check for scraping data gaps before underdelivery check
            has_gaps = await has_scraping_data_gaps(start_block, end_block)

            skipped_delivery_check = has_gaps

            if has_gaps:
                # Skip underdelivery check - mark all winners as delivered
                # This prevents unfair banning when validator has incomplete data
                winners_with_delivery = [
                    {
                        "hotkey": w["hotkey"],
                        "hashrate": w["hashrate"],
                        "price": w["price"],
                        "delivered": True,  # Assume delivery when data is incomplete
                    }
                    for w in winners
                ]
                logger.info(
                    "Skipped underdelivery check due to data gaps",
                    start_block=start_block,
                    end_block=end_block,
                    winners_count=len(winners),
                )
            else:
                # Use scraped snapshots for delivery verification
                hashrates = await get_hashrates_from_snapshots_async(
                    settings.LUXOR_SUBACCOUNT_NAME_MECHANISM_1,
                    start=start_ts,
                    end=end_ts,
                )
                winners_with_delivery = mark_delivery(winners, hashrates)

            if skipped_delivery_check:
                underdelivered_hotkeys: list[str] = []
            else:
                underdelivered_hotkeys = sorted(
                    {
                        winner.get("hotkey")
                        for winner in winners_with_delivery
                        if winner.get("hotkey") and not winner.get("delivered", False)
                    }
                )

            delivered_counts = {
                True: sum(1 for x in winners_with_delivery if x.get("delivered")),
                False: sum(1 for x in winners_with_delivery if not x.get("delivered")),
            }
            logger.debug(
                "Auction winners decided",
                subnet_epoch_start=subnet_epoch_start,
                window_number=window_num,
                start_block=start_block,
                end_block=end_block,
                commitments_count=len(bids_by_hotkey),
                winners=winners_with_delivery,
                delivered_counts=delivered_counts,
            )

            # Group winners by hotkey and check if ANY bid is underdelivered
            hotkey_bids: dict[str, list[dict]] = {}
            for winner in winners_with_delivery:
                hotkey = winner.get("hotkey")
                if hotkey:
                    if hotkey not in hotkey_bids:
                        hotkey_bids[hotkey] = []
                    hotkey_bids[hotkey].append(winner)

            # Determine which hotkeys to ban (ANY underdelivered bid)
            hotkeys_to_ban: set[str] = set()
            for hotkey, bids in hotkey_bids.items():
                if any(not bid.get("delivered", False) for bid in bids):
                    hotkeys_to_ban.add(hotkey)

            # Ban each hotkey once
            for hotkey in hotkeys_to_ban:

                def _ban_miner():
                    now = datetime.datetime.now(tz=datetime.UTC)
                    ban, created = BannedMiner.objects.get_or_create(
                        hotkey=hotkey,
                        defaults={
                            "banned_at": now,
                            "epoch_start": subnet_epoch_start,
                            "window_number": window_num,
                            "reason": f"Underdelivered in epoch {subnet_epoch_start} window {window_num}",
                        },
                    )
                    if not created:
                        # Miner cheated again - update ban timestamp
                        ban.banned_at = now
                        ban.epoch_start = subnet_epoch_start
                        ban.window_number = window_num
                        ban.reason = (
                            f"Underdelivered in epoch {subnet_epoch_start} window {window_num} (repeat offense)"
                        )
                        ban.save()
                    return ban, created

                ban, created = await asyncio.to_thread(_ban_miner)
                logger.warning(
                    "Banned miner for underdelivery" if created else "Updated ban for repeat underdelivery",
                    hotkey=hotkey,
                    window=window_num,
                    epoch=subnet_epoch_start,
                    created=created,
                    bid_count=len(hotkey_bids[hotkey]),
                )

            # Filter out ALL bids from banned hotkeys (only keep delivered from non-banned)
            delivered_winners = [
                winner
                for winner in winners_with_delivery
                if winner.get("hotkey") not in hotkeys_to_ban and winner.get("delivered", False)
            ]

            # Aggregate commitments and wins in PH
            commitments_ph_by_hotkey: dict[str, float] = {}
            for hk, bids in (bids_by_hotkey or {}).items():
                total = Decimal(0)
                for hr_str, _price in bids or []:
                    try:
                        hr_fp = _parse_decimal_to_fp18_int(str(hr_str))
                    except ValueError:
                        continue
                    total += Decimal(hr_fp) / Decimal(10**18)
                if total > 0:
                    commitments_ph_by_hotkey[hk] = float(total)

            wins_ph_by_hotkey: dict[str, float] = {}
            for winner in winners:
                hk = winner.get("hotkey")
                if not hk:
                    continue
                try:
                    hr_ph = Decimal(str(winner.get("hashrate", 0)))
                except (InvalidOperation, ValueError):
                    continue
                wins_ph_by_hotkey[hk] = float(Decimal(str(wins_ph_by_hotkey.get(hk, 0.0))) + hr_ph)

            # Estimate window budget in PH using price consensus and mechanism share
            total_budget_ph: float | None = None
            total_budget_ph = budget_ph if budget_ph and budget_ph > 0 else None

            # Store only delivered winners (non-delivered are banned and excluded)
            await record_result(
                epoch_start=subnet_epoch_start,
                start_block=start_block,
                end_block=end_block,
                commitments_count=len(bids_by_hotkey),
                winners_with_delivery=delivered_winners,
                skipped_delivery_check=skipped_delivery_check,
                underdelivered_hotkeys=underdelivered_hotkeys,
                banned_hotkeys=sorted(consensus_banned_hotkeys),
                hashp_usdc_fp18=hashp_usdc_fp18,
                alpha_tao_fp18=alpha_tao_fp18,
                tao_usdc_fp18=tao_usdc_fp18,
                commitments_ph_by_hotkey=commitments_ph_by_hotkey,
                wins_ph_by_hotkey=wins_ph_by_hotkey,
                total_budget_ph=total_budget_ph,
            )
            processed += 1

    return processed
