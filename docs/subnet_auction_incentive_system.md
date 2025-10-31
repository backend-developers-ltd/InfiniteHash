# Subnet Auction Incentive System

This document provides a reference description of the validator-managed auction
system introduced in commit `1c1d576`. The implementation replaces the previous
ad-hoc incentive logic with a deterministic mechanism that allocates subnet
emissions through repeated auctions. The goal of this note is to outline the
behaviour of the system, the data it relies on, and the responsibilities of the
supporting services.

## System Components

### APS Miner

`infinite_hashes/aps_miner` contains a stateless miner that submits bids and
collects telemetry without requiring a database or Celery worker pool. The miner
is driven by APScheduler, executes each job in an isolated child process, and
reads configuration from a hot-reloadable TOML file. Callback hooks make it
possible to integrate vendor-specific routing or monitoring.

> **Note:** the current implementation focuses on bid submission and telemetry.
> Handling of auction win/loss notifications and subsequent workload management
> is not yet implemented in the APS miner.

### Validator Service

The validator processes auctions, verifies delivery, manages ban state, and
publishes weights. Long-running tasks remain in the Django project
(`validator.tasks`), but the heavy auction orchestration lives in
`validator.auction_processing`. The validator depends on external data sources:

- Bittensor RPC access (`turbobt`) for chain state, price commitments, and the
  mechanism emission split.
- Luxor scraping snapshots for observed hashrate delivery when aggregated
  deliveries are enabled.

### Consensus Toolkit

`infinite_hashes.consensus` hosts shared parsers and algorithms:

- `price.compute_price_consensus` aggregates TAO/USDC, ALPHA/TAO, and hashed
  price commitments to produce FP18 fixed-point prices.
- `price.compute_ban_consensus` evaluates ban bitmaps carried in commitments and
  produces the set of hotkeys excluded from future auctions.
- `bidding.BiddingCommitment` parses the bid payload published by miners and
  `select_auction_winners_async` runs the winner selection flow.

Test utilities in `infinite_hashes.testutils` provide deterministic simulations,
integration scenarios, and unit fixtures that exercise the full pipeline.

## Auction Lifecycle

### Window Selection

Auctions run inside six validation windows per subnet epoch. Each window spans
60 blocks, except the final window which absorbs any remainder (typically 61
blocks). Validation windows start one window (60 blocks) ahead of the subnet
epoch boundary:

1. Window 0 covers the 60 blocks immediately preceding the new subnet epoch.
2. Windows 1-4 cover the first 240 blocks of the epoch.
3. Window 5 covers the trailing portion, ensuring continuous coverage.

`auction_processing.process_auctions_async` determines which finished windows
have not been processed yet, skipping any windows already stored in
`AuctionResult`.

### Signal Collection

For each candidate window the validator gathers the following inputs:

- **Bids** – commitments posted by miners encode `(hashrate, price)` pairs. The
  parser tolerates binary suffixes and filters out hotkeys that are already
  banned by consensus.
- **Price commitments** – TAO/USDC, ALPHA/TAO, and HASHP/USDC price feeds are
  combined into a consensus value for the window start block. These prices are
  required for budget calculations and for later weight normalisation.
- **Mechanism emission split** – the validator reads the fraction of subnet
  emissions allocated to the auction mechanism from on-chain state. If the split
  for the configured mechanism is zero or missing the window is skipped.

### Budget Computation

The available budget is derived from three values:

1. The miners' share of the subnet emission (base 41% multiplied by the on-chain
   mechanism share).
2. The ALPHA/TAO price, which converts emission units into TAO.
3. The HASHP/USDC price, which converts TAO into purchasable PH/s capacity.

The result is a PH/s budget that expresses how much hashrate can be purchased in
the current conditions. If the budget is zero or no valid bids are present the
window produces no winners.

### Winner Selection

`select_auction_winners_async` builds an optimisation problem that maximises
hashrate purchased under the PH/s budget while respecting per-bid price limits.
The solver:

1. Expands each `(hashrate, price)` pair into a worker item and discards bids
   priced more than `MAX_PRICE_MULTIPLIER` above the consensus price.
2. Solves an integer linear program (ILP) using PuLP's CBC backend to choose the
   combination of bids that fit within the budget.
3. Orders the selected bids deterministically so that the outcome is reproducible
   on every validator.

The resulting winners capture the hotkey, committed hashrate, and accepted price
for each fulfilment.

### Delivery Verification

Delivery is evaluated once per hotkey. For each winner the validator compares
the committed hashrate to observed Luxor snapshots covering the window. When
aggregated deliveries are enabled all workers under a hotkey can be combined as
long as they participate at a single price level; otherwise the allocation
procedure matches workers to commitments greedily. Average delivered PH/s across
the samples must meet at least 95% of the committed amount to pass.

If the validator detects gaps in its own scraping activity (missing data for
more than seven blocks) the underdelivery check is skipped and all winners are
assumed to have delivered. This prevents penalties caused by validator outages
rather than miner behaviour.

### Ban Handling

Hotkeys that fail delivery in any bid during the window are added to the
`BannedMiner` model. Subsequent auctions ignore their commitments until the ban
expires or is manually cleared. Ban consensus data published by other validators
is also honoured when fetching commitments, ensuring network-wide coordination.

### Result Storage and Weight Publication

Only delivered winners are recorded in `AuctionResult` along with the counts of
commitments and the consensus price tuple. When the validator later computes
weights it:

1. Aggregates ALPHA payments implied by the delivered hashrate.
2. Calculates unused budget for each window and redirects it to the subnet owner
   (burn account) if a qualifying neuron exists.
3. Normalises weights using the commit/reveal scheme so that the allocation can
   be posted on-chain for the chosen mechanism.

The commit phase uses `commit_timelocked_mechanism_weights`, and the reveal phase
publishes the actual weights once the reveal round opens.

## Supporting Infrastructure

The repository introduces a comprehensive test environment:

- `testutils.integration` spins up multi-process scenarios that emulate miners,
  validators, and network timing.
- `testutils.simulator` reproduces Subtensor runtime calls for deterministic
  auction windows and price updates.
- Integration tests cover auction settlement, burn accounting, commit/reveal
  sequencing, and ban propagation.

These tools allow developers to iterate on the incentive mechanism without
depending on a live network and serve as executable documentation for the new
behaviour.
