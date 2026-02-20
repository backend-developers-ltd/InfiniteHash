# APScheduler-Based Miner

Standalone miner implementation using APScheduler instead of Django/Celery.

## Features

- **No Database**: All state is retrieved from the blockchain
- **TOML Configuration**: Simple configuration file format
- **Hot Reload**: Configuration reloaded on each task run (no restart needed)
- **APScheduler**: Periodic task execution with hard timeouts
- **Multiprocess Isolation**: Each job runs in a separate process with enforced timeouts
- **Callback System**: Extensible hooks for won/lost bid handling

## Architecture

### Components

- **config.py**: TOML configuration loading and validation
- **models.py**: Dataclasses for workers and auction results
- **tasks.py**: Core miner logic (commitment publishing, auction computation)
- **callbacks.py**: Handlers for auction results (integrate with ASIC routing here)
- **executor.py**: Custom APScheduler executor with hard timeouts
- **scheduler.py**: APScheduler setup and job configuration
- **__main__.py**: Entry point

### How It Works

1. **Configuration**: Load worker hashrates and price from TOML file (hot-reloaded on each run)
2. **Commitment**: Periodically check on-chain commitment and update if workers changed
3. **Auction**: Smart window transition detection
   - Runs every 60 seconds
   - If close to next window (within 45s), actively waits for window transition
   - Polls blockchain every 1s for up to 48s
   - Processes newly completed window if detected, otherwise processes current window
   - Ensures timely auction result computation for just-completed windows
4. **Callbacks**: Call handlers with won/lost bids for routing integration

## Configuration

Create a `config.toml` file (see `config.toml.example`):

```toml
[bittensor]
network = "finney"
netuid = 42

[wallet]
name = "default"
hotkey_name = "default"
directory = "~/.bittensor/wallets"

[workers]
# Single price for all workers
price_multiplier = "1.0"

# Hashrates are expressed in PH.
# v1 (legacy): list of worker hashrates (each entry = 1 worker)
hashrates = ["0.1", "0.2", "0.3"]

# v2 (recommended): hashrate -> worker count (auto-enables v2 commitments)
#
# Inline-table form (must be a single line; no trailing comma):
# worker_sizes = { "0.25" = 4, "0.45" = 2 }
#
# Multi-line form (use this for longer configs):
# [workers.worker_sizes]
# "0.25" = 4
# "0.45" = 2
#
# (Alternative spelling also supported: `hashrates` can be a dict for v2)
# hashrates = { "0.25" = 4, "0.45" = 2 }
# [workers.hashrates]
# "0.25" = 4
# "0.45" = 2

```

**System Constants** (defined in `constants.py`, not user-configurable):
- Block time: 12s
- Commitment interval: 300s (5 minutes)
- Auction interval: 60s (1 minute)
- Job timeout: 120s (2 minutes)
- Window transition threshold: 75% (45s at 60s interval)
- Window transition timeout: 80% (48s at 60s interval)
- ILP CBC max nodes: 100,000

## Usage

```bash
# Run the miner
python -m infinite_hashes.aps_miner config.toml
```

### Hot Reload

Configuration is reloaded from disk on each task execution. This means you can:

1. Edit `config.toml` to change worker hashrates or price
2. Save the file
3. Changes take effect on the next scheduled run (no restart needed)

**Example:**
```bash
# Start miner with 3 workers
python -m infinite_hashes.aps_miner config.toml

# Edit config.toml to add more workers or change price
vim config.toml

# Changes automatically picked up on next commitment/auction cycle
```

This is particularly useful for:
- Adjusting prices based on market conditions
- Adding/removing worker capacity dynamically
- Testing different configurations without downtime

**Note**: System constants (intervals, timeouts, etc.) are hardcoded and cannot be changed without modifying the code.

### Smart Window Transition

The auction computation task implements intelligent timing to process newly completed windows:

**Problem**: Validation windows complete at specific blocks. If we always process the "current" window, we might repeatedly process the same window and miss newly completed ones.

**Solution**: Smart transition detection
1. Calculate time until next window starts (blocks × 12s)
2. If we're close to transition (≤45s away):
   - Actively wait and poll for window change
   - Check blockchain every 1 second
   - If new window detected (within 48s), process the newly completed window
   - If timeout (48s), process current window and let next schedule handle new window
3. If not close to transition (>45s away):
   - Process current window immediately

**Example Timeline**:
```
Block 1000: Window 10 (blocks 900-999)
Block 1005: Window 11 (blocks 1000-1099)

Scenario 1: Job runs at block 995
- 5 blocks until window 11 = ~60s
- Not close to transition (60s > 45s)
- Process window 10 immediately

Scenario 2: Job runs at block 998
- 2 blocks until window 11 = ~24s
- Close to transition (24s < 45s)
- Wait and poll for window change
- At block 1000, detect window 11
- Process newly completed window 10 with fresh data

Scenario 3: Job runs at block 999, but blocks are slow
- 1 block until window 11 = ~12s
- Close to transition (12s < 45s)
- Wait 48s, still at block 999 (slow blocks)
- Timeout, process window 10
- Next scheduled run (60s later) will catch window 11
```

**Benefit**: Maximum latency reduced from ~60s to ~15s (25% of interval)
- **Before**: Could be up to 60s late if window completed right after job ran
- **After**: Maximum 15s late (windows beyond 45s threshold caught by next job in 60-45=15s)

This ensures auction results are computed within 15 seconds of window completion, enabling timely ASIC routing decisions.

## Integration Points

### ASIC Routing

Modify `callbacks.py` to integrate with your ASIC routing system:

```python
def _handle_won_bids(won_bids: list[BidResult], result: AuctionResult) -> None:
    """Route winning workers to mining pool."""
    for bid in won_bids:
        # TODO: Configure ASIC with hashrate=bid.hashrate
        # to mine for the subnet during the window
        pass

def _handle_lost_bids(lost_bids: list[BidResult], result: AuctionResult) -> None:
    """Route losing workers to alternative tasks."""
    for bid in lost_bids:
        # TODO: Configure ASIC with hashrate=bid.hashrate
        # for spot market or idle
        pass
```

## Key Differences from Django Version

| Feature | Django/Celery | APScheduler |
|---------|---------------|-------------|
| Configuration | Django settings | TOML file |
| Database | PostgreSQL | None (blockchain state) |
| Task execution | Celery workers | APScheduler + multiprocessing |
| State persistence | Database models | On-chain commitment |
| Worker management | Database records | Config file list |
| Auction results | Saved to DB | Callbacks |

## Advantages

1. **Simpler deployment**: No database or Celery broker required
2. **Faster iteration**: Edit TOML without restart (hot reload), no migrations
3. **Process isolation**: Each job in separate process with hard timeout
4. **Stateless**: Truth comes from blockchain, not local DB
5. **Easier testing**: No Django app context required

## Testing

Run tasks manually without scheduler:

```python
from infinite_hashes.aps_miner.tasks import ensure_worker_commitment, compute_current_auction

# Test commitment (pass config path)
ensure_worker_commitment("config.toml", force=True)

# Test auction (pass config path)
result = compute_current_auction("config.toml")
print(result)
```

For a full end-to-end manual test with the blockchain simulator, run:

- `python manual_test_miner_compose.py` – builds the miner image (or reuses an override), launches the Docker Compose stack (miner plus IHP proxy stack), and validates automatic `pools.toml` target hashrate updates with proxy reload signaling.

## Logging

Structured JSON logging to stdout using structlog:

```json
{"event": "Miner commitment updated", "worker_count": 3, "timestamp": "2025-01-15T10:30:00Z"}
{"event": "Auction results computed", "won_count": 2, "lost_count": 1}
```

## Future Enhancements

- [ ] Integrate with ASIC routing system
- [ ] Add metrics/monitoring (Prometheus)
- [x] Support dynamic worker configuration (hot reload)
- [ ] Add healthcheck endpoints
- [ ] Support multiple price tiers per worker
