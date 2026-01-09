"""
System constants for the miner.

These values are not user-configurable and are set at the system level.
"""

# Blockchain timing
BLOCK_TIME = 12  # seconds per block

# Auction computation settings
AUCTION_ILP_CBC_MAX_NODES = 100000
MAX_PRICE_MULTIPLIER = 1.05  # Maximum acceptable price multiplier for bids

# Scheduler intervals (in seconds)
COMMITMENT_INTERVAL = 300  # 5 minutes
COMMITMENT_RENEW_AGE_SECONDS = 72 * 60  # 72 minutes
AUCTION_INTERVAL = 60  # 1 minute
JOB_TIMEOUT = 120  # 2 minutes

# Window transition thresholds
WINDOW_TRANSITION_THRESHOLD = 0.75  # Start checking for new window at 75% of schedule time
WINDOW_TRANSITION_TIMEOUT = 0.80  # Give up waiting for new window at 80% of schedule time
