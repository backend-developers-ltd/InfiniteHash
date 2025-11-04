"""Scenario-based event system for integration testing.

Provides a declarative event-driven framework for defining test scenarios:
- TimeAddress: Address specific points in simulation time
- ScenarioEvent: Base class for events (RegisterValidator, RegisterMiner, SetPrices, etc.)
- Scenario: Timeline of events with delivery hooks

# Runtime Registration

Validators and miners can be registered during simulation (not just at initialization):

```python
scenario = Scenario(num_epochs=2)

# Initialization (epoch < 0)
scenario.add_event(
    RegisterValidator(TimeAddress(-1, 5, 0), "validator_0", stake=10000.0)
)

# Runtime registration (epoch >= 0) - new in this version!
scenario.add_event(
    RegisterMiner(
        time=TimeAddress(0, 2, 0),  # During epoch 0, window 2
        name="late_miner",
        workers=[{"identifier": "worker1", "hashrate_ph": "5.0", "price_multiplier": "1.0"}],
    )
)
```

Runtime-registered entities are immediately started and registered on the blockchain.
```

# Delivery Hook System

Control hashrate delivery **per miner or per worker** using flexible hooks with hierarchical lookup.

## Hierarchical Hook Lookup

Hooks are looked up in this order:
1. Worker-specific: `"miner_0.worker1"` (if configured)
2. Miner-level: `"miner_0"` (if configured)
3. Default hook (always available)

## Basic Usage

```python
from infinite_hashes.testutils.integration.scenario import (
    Scenario, TimeAddress, SetDeliveryHook,
    perfect_delivery_hook, no_delivery_hook, DeliveryParams,
)

# 1. Simple: Perfect delivery for all miners (via default)
scenario = Scenario(num_epochs=1, default_delivery_hook=perfect_delivery_hook)

# 2. Per-miner control - different hook for each miner
scenario = Scenario(
    num_epochs=1,
    delivery_hooks={
        "miner_0": perfect_delivery_hook,  # Perfect delivery for all miner_0 workers
        "miner_1": no_delivery_hook,        # No delivery for all miner_1 workers
    },
    default_delivery_hook=default_delivery_hook,  # Fallback for other miners
)

# 3. Per-worker control - mix of miner-level and worker-level hooks
scenario = Scenario(
    num_epochs=1,
    delivery_hooks={
        "miner_0": perfect_delivery_hook,              # Default for all miner_0 workers
        "miner_0.worker1": no_delivery_hook,           # Override for worker1 only
        "miner_1.worker2": lambda m, t: DeliveryParams((0.5, 0.7), 0.1),  # Specific worker
    },
)

# 4. Dynamic changes during simulation - change at miner or worker level
scenario.add_event(
    SetDeliveryHook(
        time=TimeAddress(0, 2, 0),
        target="miner_0",              # Affect all miner_0 workers
        hook=no_delivery_hook,
    )
)
scenario.add_event(
    SetDeliveryHook(
        time=TimeAddress(0, 3, 0),
        target="miner_0.worker1",      # Affect only miner_0.worker1
        hook=perfect_delivery_hook,
    )
)
```

## Advanced: Stateful and Worker-Specific Hooks

```python
# Stateful hook: perfect delivery in epoch 0, degrading after
def create_degrading_hook():
    def hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
        if time.epoch == 0:
            return DeliveryParams(hashrate_multiplier_range=(1.0, 1.0), dropout_rate=0.0)
        dropout = min(0.5, time.epoch * 0.1)
        return DeliveryParams(hashrate_multiplier_range=(0.8, 1.0), dropout_rate=dropout)
    return hook

# Intermittent delivery every 3rd window
def create_intermittent_hook():
    def hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
        if time.window % 3 == 0:
            return DeliveryParams(hashrate_multiplier_range=(0.95, 1.05), dropout_rate=0.01)
        return DeliveryParams(hashrate_multiplier_range=(0.3, 0.5), dropout_rate=0.3)
    return hook

scenario = Scenario(
    num_epochs=2,
    delivery_hooks={
        "miner_0": create_degrading_hook(),       # All miner_0 workers
        "miner_1": perfect_delivery_hook,         # All miner_1 workers
        "miner_1.worker2": create_intermittent_hook(),  # Override for worker2
    }
)

# Dynamic changes during simulation (can target miner or specific worker)
scenario.add_event(
    SetDeliveryHook(TimeAddress(1, 0, 0), target="miner_0", hook=perfect_delivery_hook)
)
scenario.add_event(
    SetDeliveryHook(TimeAddress(1, 2, 0), target="miner_1.worker2", hook=no_delivery_hook)
)
```
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class TimeAddress:
    """Address a specific point in simulation time.

    - epoch: validation epoch index (can be negative for initialization, e.g., -1)
    - window: window index within epoch (0-5)
    - block: block offset within window (0-59 for windows 0-4, 0-60 for window 5)
    """

    epoch: int
    window: int
    block: int

    # Window configuration (class variables)
    BLOCKS_PER_WINDOW = 60  # Windows 0-4
    BLOCKS_IN_LAST_WINDOW = 61  # Window 5 has 61 blocks
    WINDOWS_PER_EPOCH = 6

    def __post_init__(self):
        """Validate TimeAddress after initialization."""
        # Validate window
        if self.window < 0 or self.window >= TimeAddress.WINDOWS_PER_EPOCH:
            raise ValueError(f"Window must be 0-5, got {self.window}")

        # Validate block based on window
        max_block = TimeAddress.BLOCKS_IN_LAST_WINDOW if self.window == 5 else TimeAddress.BLOCKS_PER_WINDOW
        if self.block < 0 or self.block >= max_block:
            raise ValueError(f"Block must be 0-{max_block - 1} for window {self.window}, got {self.block}")

    def __str__(self) -> str:
        return f"({self.epoch}, {self.window}, {self.block})"

    def dt(self, epoch: int = 0, window: int = 0, block: int = 0) -> "TimeAddress":
        """Create relative time address with delta, handling overflows.

        Automatically handles block/window/epoch overflow:
        - Block overflow -> rolls to next window
        - Window overflow -> rolls to next epoch
        - Negative values roll backwards

        Args:
            epoch: Epoch delta
            window: Window delta
            block: Block delta

        Returns:
            New TimeAddress with normalized values

        Example:
            >>> TimeAddress(1, 5, 60).dt(block=3)
            TimeAddress(epoch=2, window=0, block=2)  # Overflowed to next epoch
        """
        new_epoch = self.epoch + epoch
        new_window = self.window + window
        new_block = self.block + block

        # Handle block overflow/underflow
        while new_block < 0 or new_block >= self._blocks_in_window(new_window):
            if new_block < 0:
                # Underflow - go to previous window
                new_window -= 1
                if new_window < 0:
                    new_window = self.WINDOWS_PER_EPOCH - 1
                    new_epoch -= 1
                new_block += self._blocks_in_window(new_window)
            else:
                # Overflow - go to next window
                new_block -= self._blocks_in_window(new_window)
                new_window += 1
                if new_window >= self.WINDOWS_PER_EPOCH:
                    new_window = 0
                    new_epoch += 1

        # Handle window overflow/underflow
        while new_window < 0 or new_window >= self.WINDOWS_PER_EPOCH:
            if new_window < 0:
                new_window += self.WINDOWS_PER_EPOCH
                new_epoch -= 1
            else:
                new_window -= self.WINDOWS_PER_EPOCH
                new_epoch += 1

        # Validation happens automatically in __new__
        return TimeAddress(new_epoch, new_window, new_block)

    @staticmethod
    def _blocks_in_window(window: int) -> int:
        """Get number of blocks in a window."""
        # Handle negative windows (for underflow calculations)
        normalized_window = window % TimeAddress.WINDOWS_PER_EPOCH
        return TimeAddress.BLOCKS_IN_LAST_WINDOW if normalized_window == 5 else TimeAddress.BLOCKS_PER_WINDOW

    def b_dt(self, blocks: int) -> "TimeAddress":
        """Create relative time address with block delta, handling overflow."""
        return self.dt(block=blocks)

    def w_dt(self, windows: int, blocks: int = 0) -> "TimeAddress":
        """Create relative time address with window delta, handling overflow."""
        return self.dt(window=windows, block=blocks)

    def e_dt(self, epochs: int, windows: int = 0, blocks: int = 0) -> "TimeAddress":
        """Create relative time address with epoch delta."""
        return self.dt(epoch=epochs, window=windows, block=blocks)


@dataclass
class DeliveryParams:
    """Parameters controlling delivery simulation for a miner at a specific time."""

    hashrate_multiplier_range: tuple[float, float] = (0.93, 1.15)  # Random range
    dropout_rate: float = 0.01  # Probability of 0 hashrate

    def sample_multiplier(self) -> float:
        """Sample a random multiplier from the configured range."""
        return random.uniform(*self.hashrate_multiplier_range)


@dataclass
class ScenarioEvent:
    """Base class for scenario events."""

    time: TimeAddress

    def __lt__(self, other: "ScenarioEvent") -> bool:
        """Events are sorted by time address."""
        return self.time < other.time


@dataclass
class RegisterValidator(ScenarioEvent):
    """Register a validator with given name and stake."""

    name: str
    stake: float
    wallet_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisterMiner(ScenarioEvent):
    """Register a miner with given name and workers.

    Args:
        name: Unique name for the miner
        workers: List of worker configurations
        wallet_config: Optional wallet configuration overrides
        replace_miner: Optional miner name to replace (takes over that miner's UID)
    """

    name: str
    workers: list[dict[str, Any]]
    wallet_config: dict[str, Any] = field(default_factory=dict)
    replace_miner: str | None = None


@dataclass
class SetPrices(ScenarioEvent):
    """Validator scrapes and publishes price commitment.

    Args:
        validator_name: Name of the validator
        ph_budget: Optional PH budget to set (adjusts ALPHA_TAO price accordingly)
                   If None, uses default simulated prices
    """

    validator_name: str
    ph_budget: float | None = None


@dataclass
class SetCommitment(ScenarioEvent):
    """Miner publishes bidding commitment."""

    miner_name: str
    commitment_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeWorkers(ScenarioEvent):
    """Update a miner's worker set without restarting the process.

    Args:
        miner_name: Name of the registered miner to update.
        workers: New list of worker configurations (same format as RegisterMiner.workers).
    """

    miner_name: str
    workers: list[dict[str, Any]]


@dataclass
class SetDeliveryHook(ScenarioEvent):
    """Change the delivery hook for a miner or specific worker.

    Target can be:
    - "miner_0" - applies to all workers of miner_0
    - "miner_0.worker1" - applies only to worker1 of miner_0
    """

    target: str  # "miner_name" or "miner_name.worker_identifier"
    hook: Callable[[str, TimeAddress], DeliveryParams]


@dataclass
class AssertWeightsEvent(ScenarioEvent):
    """Assert that validators have committed expected weights for a specific epoch.

    Weights are provided in normalized form (sum=1.0) and converted to
    max-based format (0-65535) for comparison with simulator values.

    The Director tracks when validators commit weights for each epoch, and uses
    this cached information to calculate the correct reveal block for verification.

    Special Keys:
        "__owner__": Use this key to assert burn weight (unused auction budget allocated to subnet owner).
                    The subnet owner is automatically registered at UID 0 during scenario setup.

    Args:
        for_epoch: The epoch number whose weights should be verified (0, 1, 2, ...)
        expected_weights: Dict[validator_name, Dict[miner_name | "__owner__", normalized_weight]]
                         where weights sum to 1.0 and are converted to max-based (0-65535)

    Example:
        scenario.add_event(
            AssertWeightsEvent(
                time=TimeAddress(1, 0, 1),  # Check at epoch 1 start
                for_epoch=0,  # Verify epoch 0 weights
                expected_weights={
                    "validator_0": {
                        "miner_0": 0.4,      # 40% of weight
                        "miner_1": 0.4,      # 40% of weight
                        "__owner__": 0.2,    # 20% burn (unused budget)
                    },  # Sum = 1.0
                    "validator_1": {
                        "miner_0": 0.4,
                        "miner_1": 0.4,
                        "__owner__": 0.2,
                    },  # Sum = 1.0
                },
            )
        )
    """

    for_epoch: int
    expected_weights: dict[str, dict[str, float]]  # validator -> {miner -> normalized weight (sum=1.0)}


@dataclass
class AssertFalseEvent(ScenarioEvent):
    """Stop simulation at a specific time by raising an assertion failure.

    Useful for debugging - allows you to stop the simulation at a specific
    point to inspect state, logs, or verify intermediate conditions.

    Args:
        message: Optional custom message to display when stopping

    Example:
        scenario.add_event(
            AssertFalseEvent(
                time=TimeAddress(1, 3, 30),  # Stop at epoch 1, window 3, block 30
                message="Stopping to inspect state after miner registration"
            )
        )
    """

    message: str = "Simulation stopped by AssertFalseEvent"


# Type alias for delivery hook
DeliveryHook = Callable[[str, TimeAddress], DeliveryParams]


def default_delivery_hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
    """Default delivery simulation: random variance independent of time."""
    return DeliveryParams()


# Helper functions for common delivery patterns


def perfect_delivery_hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
    """Perfect delivery: exactly 100% of committed hashrate."""
    return DeliveryParams(hashrate_multiplier_range=(1.0, 1.0), dropout_rate=0.0)


def no_delivery_hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
    """No delivery: always 0 hashrate."""
    return DeliveryParams(hashrate_multiplier_range=(0.0, 0.0), dropout_rate=1.0)


def delivery_for_miners(**miner_params: DeliveryParams) -> DeliveryHook:
    """Create a hook with specific delivery params per miner.

    Example:
        hook = delivery_for_miners(
            miner_0=DeliveryParams(hashrate_multiplier_range=(1.0, 1.0), dropout_rate=0.0),
            miner_1=DeliveryParams(hashrate_multiplier_range=(0.5, 0.7), dropout_rate=0.1),
        )
    """
    default_params = DeliveryParams()

    def hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
        return miner_params.get(miner_name, default_params)

    return hook


def conditional_delivery(
    condition: Callable[[str, TimeAddress], bool],
    if_true: DeliveryParams,
    if_false: DeliveryParams | None = None,
) -> DeliveryHook:
    """Create a hook that switches delivery based on a condition.

    Example:
        # Perfect delivery only in epoch 0
        hook = conditional_delivery(
            condition=lambda miner, time: time.epoch == 0,
            if_true=DeliveryParams(hashrate_multiplier_range=(1.0, 1.0), dropout_rate=0.0),
            if_false=DeliveryParams(hashrate_multiplier_range=(0.5, 0.7), dropout_rate=0.2),
        )
    """
    default_params = if_false or DeliveryParams()

    def hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
        return if_true if condition(miner_name, time) else default_params

    return hook


@dataclass
class Scenario:
    """Complete scenario definition with events and hooks.

    Attributes:
        num_epochs: Number of validation epochs to simulate
        events: User-defined timeline of events
        delivery_hooks: Hierarchical hooks (target -> hook)
                       Target can be "miner_name" or "miner_name.worker_identifier"
        default_delivery_hook: Fallback hook when no specific hook is configured
        wallet_dir: Base directory for test wallets
    """

    num_epochs: int
    events: list[ScenarioEvent] = field(default_factory=list)
    delivery_hooks: dict[str, DeliveryHook] = field(default_factory=dict)
    default_delivery_hook: DeliveryHook = default_delivery_hook
    wallet_dir: str = ""

    def get_delivery_hook(self, miner_name: str, worker_identifier: str | None = None) -> DeliveryHook:
        """Get the delivery hook with hierarchical lookup.

        Lookup order:
        1. "miner_name.worker_identifier" (if worker_identifier provided)
        2. "miner_name"
        3. default_delivery_hook
        """
        # Try worker-specific hook first
        if worker_identifier:
            worker_key = f"{miner_name}.{worker_identifier}"
            if worker_key in self.delivery_hooks:
                return self.delivery_hooks[worker_key]

        # Try miner-level hook
        if miner_name in self.delivery_hooks:
            return self.delivery_hooks[miner_name]

        # Fall back to default
        return self.default_delivery_hook

    def set_delivery_hook(self, target: str, hook: DeliveryHook) -> None:
        """Set the delivery hook for a target (miner or worker).

        Target can be:
        - "miner_0" - applies to all workers of miner_0
        - "miner_0.worker1" - applies only to worker1
        """
        self.delivery_hooks[target] = hook

    def add_event(self, event: ScenarioEvent) -> None:
        """Add an event to the timeline."""
        self.events.append(event)

    def add_events(self, *events: ScenarioEvent) -> None:
        """Add multiple events to the timeline (more declarative)."""
        self.events.extend(events)

    def get_events_at(self, time: TimeAddress) -> list[ScenarioEvent]:
        """Get all events scheduled at a specific time."""
        return [e for e in self.events if e.time == time]

    def get_events_before(self, time: TimeAddress) -> list[ScenarioEvent]:
        """Get all events scheduled before a specific time."""
        return [e for e in self.events if e.time < time]

    def sorted_events(self) -> list[ScenarioEvent]:
        """Return events sorted by time."""
        return sorted(self.events)
