"""Test Luxor scraper task with real API integration."""

import asyncio
import datetime as dt
import os
import unittest.mock

import pytest

from infinite_hashes.validator.models import LuxorSnapshot, ValidatorScrapingEvent
from infinite_hashes.validator.tasks import scrape_luxor_async

POLL_INTERVAL = 30  # 30 seconds (matches production scraper)
NUM_POLLS = 8  # Poll 8 times over 4 minutes to capture 3 Luxor data updates (~1min each)
EXPECTED_DATA_POINTS = 3  # Expect at least 3 distinct data points from Luxor
TEST_DURATION = NUM_POLLS * POLL_INTERVAL  # 4 minutes
LOW_HASHRATE_THRESHOLD_PH = 0.05  # Ignore assertions for workers below this average PH/s


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
@pytest.mark.api_integration_long
async def test_luxor_scraper_long_running_real_api():
    """Test scraper over multiple polls with real Luxor API.

    This test:
    - Connects to the real Luxor API
    - Mocks bittensor to return incrementing block numbers
    - Polls every 30 seconds for 4 minutes (8 polls)
    - Luxor updates data every ~1 minute
    - Verifies we capture at least 3 distinct data points for each worker
    - Verifies ValidatorScrapingEvent is created for each poll with correct block numbers
    - Shows verbose output of what's happening at each poll

    Requires:
    - RUN_API_INTEGRATION_LONG=1
    - LUXOR_API_KEY_FOR_TESTS set in environment
    - LUXOR_API_URL_FOR_TESTS set in environment (optional, defaults to https://app.luxor.tech/api)
    - LUXOR_SUBACCOUNT_NAME_MECHANISM_1_FOR_TESTS set in environment (optional, defaults to infinite)
    """
    from django.conf import settings

    # Get subaccount from settings or env
    subaccount = getattr(
        settings, "LUXOR_SUBACCOUNT_NAME_MECHANISM_1", os.getenv("LUXOR_SUBACCOUNT_NAME_MECHANISM_1", "infinite")
    )

    print("\n" + "=" * 80)
    print("LUXOR SCRAPER LONG-RUNNING INTEGRATION TEST (REAL API)")
    print(f"Duration: {TEST_DURATION}s (~{TEST_DURATION / 60:.1f} minutes)")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Expected polls: {NUM_POLLS}")
    print(f"Expected data points: at least {EXPECTED_DATA_POINTS}")
    print(f"Subaccount: {subaccount}")
    print(f"API URL: {settings.LUXOR_API_URL}")
    print("=" * 80 + "\n")

    # Clear any existing snapshots and scraping events
    deleted_snapshots = await LuxorSnapshot.objects.filter(subaccount_name=subaccount).adelete()
    if isinstance(deleted_snapshots, tuple):
        deleted_snapshots = deleted_snapshots[0]
    deleted_events = await ValidatorScrapingEvent.objects.all().adelete()
    if isinstance(deleted_events, tuple):
        deleted_events = deleted_events[0]
    print(f"[SETUP] Cleared {deleted_snapshots} existing snapshots for {subaccount}")
    print(f"[SETUP] Cleared {deleted_events} existing scraping events\n")

    # Mock bittensor to return incrementing block numbers
    base_block = 10000
    current_block = [base_block]  # Use list to make it mutable
    expected_blocks = []

    async def mock_head_get():
        mock_head = unittest.mock.Mock()
        block_num = current_block[0]
        mock_head.number = block_num
        expected_blocks.append(block_num)
        current_block[0] += 15  # Increment by ~15 blocks per poll (~3 minutes between polls)
        return mock_head

    poll_results = []

    # Patch turbobt.Bittensor to return incrementing block numbers
    with unittest.mock.patch("infinite_hashes.validator.tasks.turbobt.Bittensor") as mock_bittensor_cls:
        # Setup async context manager
        mock_bittensor = unittest.mock.AsyncMock()
        mock_bittensor_cls.return_value.__aenter__.return_value = mock_bittensor
        mock_bittensor_cls.return_value.__aexit__.return_value = None

        # Mock head.get to return incrementing block numbers
        mock_bittensor.head.get = mock_head_get

        for poll_num in range(1, NUM_POLLS + 1):
            print(f"{'─' * 80}")
            print(f"POLL #{poll_num}/{NUM_POLLS} @ {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─' * 80}")

            # Count snapshots before this poll
            total_before = await LuxorSnapshot.objects.filter(subaccount_name=subaccount).acount()
            workers_before = (
                await LuxorSnapshot.objects.filter(subaccount_name=subaccount).values("worker_name").distinct().acount()
            )

            # Run scraper - this hits the real Luxor API
            start_time = dt.datetime.now()
            try:
                workers_recorded = await scrape_luxor_async(subaccount)
                duration_ms = (dt.datetime.now() - start_time).total_seconds() * 1000
                error = None
            except Exception as e:
                duration_ms = (dt.datetime.now() - start_time).total_seconds() * 1000
                error = str(e)
                workers_recorded = 0
                print(f"  ✗ ERROR: {error}")
                print(f"  Duration: {duration_ms:.1f}ms")
                if poll_num < NUM_POLLS:
                    print(f"\n  ⏳ Waiting {POLL_INTERVAL}s until next poll...")
                    await asyncio.sleep(POLL_INTERVAL)
                    print()
                continue

            # Count snapshots after this poll
            total_after = await LuxorSnapshot.objects.filter(subaccount_name=subaccount).acount()
            workers_after = (
                await LuxorSnapshot.objects.filter(subaccount_name=subaccount).values("worker_name").distinct().acount()
            )
            new_snapshots = total_after - total_before

            # Group by worker
            snapshots_by_worker = {}
            async for snapshot in LuxorSnapshot.objects.filter(subaccount_name=subaccount).order_by(
                "worker_name", "-snapshot_time"
            ):
                if snapshot.worker_name not in snapshots_by_worker:
                    snapshots_by_worker[snapshot.worker_name] = []
                snapshots_by_worker[snapshot.worker_name].append(snapshot)

            print(f"  Status: {'✓ RECORDED' if workers_recorded > 0 else '○ NO CHANGE'}")
            print(f"  Duration: {duration_ms:.1f}ms")
            print(f"  Workers recorded: {workers_recorded}")
            print(f"  New snapshots: {new_snapshots}")
            print(f"  Total snapshots: {total_before} → {total_after}")
            print(f"  Unique workers: {workers_before} → {workers_after}")

            if snapshots_by_worker:
                print("\n  Snapshots by worker:")
                for worker_name in sorted(snapshots_by_worker.keys()):
                    snapshots = snapshots_by_worker[worker_name]
                    print(f"    {worker_name[:60]}...")
                    print(f"      Count: {len(snapshots)}")
                    for i, snap in enumerate(snapshots[:5], 1):  # Show first 5
                        hashrate_ph = snap.hashrate / 1e15
                        time_str = snap.snapshot_time.strftime("%H:%M:%S")
                        last_updated_str = snap.last_updated.strftime("%H:%M:%S")
                        print(
                            f"        [{i}] snap={time_str} | {hashrate_ph:.2f} PH/s | "
                            f"eff={snap.efficiency}% | last_share={last_updated_str}"
                        )

            poll_results.append(
                {
                    "poll_num": poll_num,
                    "workers_recorded": workers_recorded,
                    "new_snapshots": new_snapshots,
                    "total_snapshots": total_after,
                    "unique_workers": workers_after,
                    "snapshots_by_worker": {w: len(s) for w, s in snapshots_by_worker.items()},
                    "error": error,
                }
            )

            # Wait for next poll (unless this is the last one)
            if poll_num < NUM_POLLS:
                print(f"\n  ⏳ Waiting {POLL_INTERVAL}s until next poll...")
                await asyncio.sleep(POLL_INTERVAL)
                print()

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    total_snapshots = await LuxorSnapshot.objects.filter(subaccount_name=subaccount).acount()
    unique_workers = (
        await LuxorSnapshot.objects.filter(subaccount_name=subaccount).values("worker_name").distinct().acount()
    )

    print(f"Total polls: {NUM_POLLS}")
    print(f"Total snapshots recorded: {total_snapshots}")
    print(f"Unique workers: {unique_workers}")
    print(f"Expected data points per worker: at least {EXPECTED_DATA_POINTS}")

    # Verify each worker has expected number of snapshots
    snapshots_by_worker = {}
    async for snapshot in LuxorSnapshot.objects.filter(subaccount_name=subaccount).order_by(
        "worker_name", "-snapshot_time"
    ):
        if snapshot.worker_name not in snapshots_by_worker:
            snapshots_by_worker[snapshot.worker_name] = []
        snapshots_by_worker[snapshot.worker_name].append(snapshot)

    print("\nPer-worker snapshot counts:")
    all_workers_ok = True
    low_hashrate_workers = set()
    for worker_name in sorted(snapshots_by_worker.keys()):
        snapshots = snapshots_by_worker[worker_name]
        count = len(snapshots)
        avg_hashrate_hs = sum(s.hashrate for s in snapshots) / count
        avg_hashrate_ph = avg_hashrate_hs / 1e15
        ignored = avg_hashrate_ph < LOW_HASHRATE_THRESHOLD_PH
        status = "○" if ignored else ("✓" if count >= EXPECTED_DATA_POINTS else "✗")
        note = " (ignored: low hashrate)" if ignored else ""
        print(
            f"  {status} {worker_name[:60]}: {count} "
            f"(expected ≥{EXPECTED_DATA_POINTS}, avg={avg_hashrate_ph:.2f} PH/s){note}"
        )
        if ignored:
            low_hashrate_workers.add(worker_name)
            continue
        if count < EXPECTED_DATA_POINTS:
            all_workers_ok = False

    print("\nPoll results summary:")
    for result in poll_results:
        status = "ERROR" if result["error"] else "OK"
        print(
            f"  Poll {result['poll_num']}: {status} | "
            f"{result['workers_recorded']} workers, "
            f"{result['new_snapshots']} new snapshots, "
            f"{result['total_snapshots']} total"
        )

    # Verify ValidatorScrapingEvent tracking
    print(f"\n{'─' * 80}")
    print("SCRAPING EVENT TRACKING")
    print(f"{'─' * 80}")

    all_scraping_events_list = []
    async for event in ValidatorScrapingEvent.objects.all().order_by("block_number"):
        all_scraping_events_list.append(event)

    print(f"\nTotal scraping events: {len(all_scraping_events_list)}")
    print(f"Expected: {NUM_POLLS}")

    print("\nBlock number verification:")
    for i, (event, expected_block) in enumerate(zip(all_scraping_events_list, expected_blocks)):
        status = "✓" if event.block_number == expected_block else "✗"
        print(
            f"  {status} Event {i + 1}: block={event.block_number}, "
            f"expected={expected_block}, workers={event.worker_count}"
        )

    scraping_event_block_numbers = [e.block_number for e in all_scraping_events_list]
    print(f"\nBlock numbers: {scraping_event_block_numbers}")
    print(f"Expected blocks: {expected_blocks}")

    print(f"\n{'=' * 80}\n")

    # Assertions
    errors = [r for r in poll_results if r["error"]]
    assert not errors, f"Some polls failed: {[r['error'] for r in errors]}"

    assert unique_workers > 0, f"Expected at least 1 worker, got {unique_workers}"

    # Verify each worker has at least EXPECTED_DATA_POINTS
    assert all_workers_ok, f"Not all workers have at least {EXPECTED_DATA_POINTS} snapshots"
    for worker_name, snapshots in snapshots_by_worker.items():
        if worker_name in low_hashrate_workers:
            continue
        assert (
            len(snapshots) >= EXPECTED_DATA_POINTS
        ), f"Worker {worker_name} has {len(snapshots)} snapshots, expected at least {EXPECTED_DATA_POINTS}"

    # Verify all snapshots are for correct subaccount
    wrong_subaccount = await LuxorSnapshot.objects.exclude(subaccount_name=subaccount).acount()
    assert wrong_subaccount == 0, f"Found {wrong_subaccount} snapshots with wrong subaccount"

    # Verify timestamps are properly ordered
    for worker_name, snapshots in snapshots_by_worker.items():
        times = [s.snapshot_time for s in snapshots]
        assert times == sorted(times, reverse=True), f"Snapshots for {worker_name} not in descending order"

    # Verify ValidatorScrapingEvent assertions
    assert (
        len(all_scraping_events_list) == NUM_POLLS
    ), f"Expected {NUM_POLLS} scraping events, got {len(all_scraping_events_list)}"

    # Verify block numbers match expected
    for i, (event, expected_block) in enumerate(zip(all_scraping_events_list, expected_blocks)):
        assert (
            event.block_number == expected_block
        ), f"Event {i + 1} has block {event.block_number}, expected {expected_block}"

    # Verify block numbers are monotonically increasing
    assert scraping_event_block_numbers == sorted(
        scraping_event_block_numbers
    ), "Block numbers should be monotonically increasing"
    assert len(set(scraping_event_block_numbers)) == len(
        scraping_event_block_numbers
    ), "All block numbers should be unique"

    # Verify worker_count is non-negative
    for event in all_scraping_events_list:
        assert event.worker_count >= 0, f"Worker count should be non-negative, got {event.worker_count}"

    print("✓ All assertions passed")
