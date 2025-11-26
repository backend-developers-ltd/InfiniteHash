from django.db import models  # noqa


class WeightsBatch(models.Model):
    block = models.BigIntegerField(
        help_text="Block number for which this batch is scheduled",
    )
    epoch_start = models.BigIntegerField(
        help_text="Epoch's start Block number",
    )
    mechanism_id = models.IntegerField(
        default=0,
        help_text="Mechanism ID (0=legacy, 1=auctions)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    scored = models.BooleanField(default=False)
    should_be_scored = models.BooleanField(default=True)
    weights = models.JSONField(default=dict)

    class Meta:
        verbose_name_plural = "Weights Batches"
        unique_together = [["block", "mechanism_id"]]

    def __str__(self):
        return f"Weights Batch #{self.pk} at block #{self.block} (mech {self.mechanism_id})"


class AuctionResult(models.Model):
    epoch_start = models.BigIntegerField(help_text="Epoch start block")
    start_block = models.BigIntegerField(help_text="Auction window start block")
    end_block = models.BigIntegerField(help_text="Auction window end block", unique=True)
    commitments_count = models.IntegerField(default=0)
    skipped_delivery_check = models.BooleanField(
        default=False,
        help_text="True when delivery validation was skipped due to scraping gaps",
    )
    winners = models.JSONField(default=list, help_text="Delivered winners with hashrate and price multiplier")
    commitments_ph_by_hotkey = models.JSONField(
        default=dict,
        help_text="Total committed hashrate per hotkey (PH) for this window",
    )
    wins_ph_by_hotkey = models.JSONField(
        default=dict,
        help_text="Total winning hashrate per hotkey (PH) for this window",
    )
    total_budget_ph = models.DecimalField(
        max_digits=30,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Estimated auction budget for this window in PH",
    )
    underdelivered_hotkeys = models.JSONField(
        default=list,
        help_text="Unique hotkeys that failed delivery checks during this window",
    )
    banned_hotkeys = models.JSONField(
        default=list,
        help_text="Hotkeys banned from this window after ban consensus",
    )
    hashp_usdc = models.DecimalField(
        max_digits=30,
        decimal_places=18,
        null=True,
        blank=True,
        help_text="Hashrate price consensus (USDC per PH per day)",
    )
    alpha_tao = models.DecimalField(
        max_digits=30, decimal_places=18, null=True, blank=True, help_text="ALPHA/TAO price consensus"
    )
    tao_usdc = models.DecimalField(
        max_digits=30, decimal_places=18, null=True, blank=True, help_text="TAO/USDC price consensus"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-end_block"]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Auction {self.start_block}-{self.end_block} (epoch {self.epoch_start})"


class LuxorSnapshot(models.Model):
    """Scraped Luxor hashrate data for high-resolution historical tracking."""

    snapshot_time = models.DateTimeField(
        db_index=True,
        help_text="When this snapshot was taken by the scraper",
    )
    subaccount_name = models.CharField(
        max_length=255,
        help_text="Luxor subaccount name",
    )
    worker_name = models.CharField(
        max_length=255,
        help_text="Worker identifier (e.g., hotkey_worker1)",
    )
    hashrate = models.BigIntegerField(help_text="Hashrate in H/s")
    efficiency = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        help_text="Efficiency percentage",
    )
    revenue = models.DecimalField(
        max_digits=20,
        decimal_places=10,
        help_text="Revenue in BTC",
    )
    last_updated = models.DateTimeField(
        help_text="Timestamp when Luxor last updated this data",
    )

    class Meta:
        ordering = ["-snapshot_time"]
        indexes = [
            models.Index(fields=["subaccount_name", "worker_name", "snapshot_time"]),
            models.Index(fields=["snapshot_time", "worker_name"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Luxor {self.worker_name} @ {self.snapshot_time.isoformat()}"


class BannedMiner(models.Model):
    """Tracks miners banned for underdelivering hashrate."""

    hotkey = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text="Miner hotkey that is banned",
    )
    banned_at = models.DateTimeField(
        db_index=True,
        help_text="When the ban was applied or last updated (for repeat offenses)",
    )
    epoch_start = models.BigIntegerField(
        help_text="Epoch start block where underdelivery was detected",
    )
    window_number = models.IntegerField(
        help_text="Window number (0-5) where underdelivery occurred",
    )
    reason = models.CharField(
        max_length=500,
        blank=True,
        help_text="Optional reason for ban (e.g., 'underdelivered 90%')",
    )

    class Meta:
        ordering = ["-banned_at"]
        indexes = [
            models.Index(fields=["hotkey", "banned_at"]),
            models.Index(fields=["epoch_start", "window_number"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Banned {self.hotkey[:8]}... @ epoch {self.epoch_start}"


class ValidatorScrapingEvent(models.Model):
    """Tracks when validator successfully scraped Luxor data (heartbeat for data completeness).

    Used to detect data gaps before penalizing miners for underdelivery.
    If there's a gap > 7 blocks (~84s) in scraping events, underdelivery check is skipped
    for that window since the validator may have missed data collection.
    """

    scraped_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="When the scraping occurred",
    )
    block_number = models.BigIntegerField(
        db_index=True,
        help_text="Block number when scraping occurred",
    )
    worker_count = models.IntegerField(
        default=0,
        help_text="Number of workers scraped in this event",
    )

    class Meta:
        ordering = ["-scraped_at"]
        indexes = [
            models.Index(fields=["block_number"]),
            models.Index(fields=["scraped_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Scraping @ block {self.block_number}"


class PriceObservation(models.Model):
    """Tracks scraped price data for consensus mechanisms."""

    metric = models.CharField(max_length=32)
    source = models.CharField(max_length=32)
    observed_at = models.DateTimeField()
    fetched_at = models.DateTimeField(auto_now_add=True)
    # Store integer Q.18 value
    price_fp18 = models.DecimalField(max_digits=40, decimal_places=0)

    class Meta:
        indexes = [
            models.Index(fields=["metric", "observed_at"]),
            models.Index(fields=["metric", "source", "observed_at"]),
        ]
        ordering = ["-observed_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.metric} {self.source} {self.observed_at} {self.price_fp18}"
