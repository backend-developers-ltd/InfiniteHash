from django.db import models  # noqa


class WeightsBatch(models.Model):
    block = models.BigIntegerField(
        help_text="Block number for which this batch is scheduled",
        unique=True,
    )
    epoch_start = models.BigIntegerField(
        help_text="Epoch's start Block number",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    scored = models.BooleanField(default=False)
    should_be_scored = models.BooleanField(default=True)
    weights = models.JSONField(default=dict)

    class Meta:
        verbose_name_plural = "Weights Batches"

    def __str__(self):
        return f"Weights Batch #{self.pk} at block #{self.block}"
