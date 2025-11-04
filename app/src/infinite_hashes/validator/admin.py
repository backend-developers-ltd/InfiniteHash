from django.contrib import admin  # noqa
from django.contrib.admin import register  # noqa
from .models import AuctionResult, PriceObservation


admin.site.site_header = "infinite_hashes Administration"
admin.site.site_title = "infinite_hashes"
admin.site.index_title = "Welcome to infinite_hashes Administration"


@register(AuctionResult)
class AuctionResultAdmin(admin.ModelAdmin):
    list_display = ("epoch_start", "start_block", "end_block", "commitments_count")
    search_fields = ("start_block", "end_block")
    ordering = ("-end_block",)


@register(PriceObservation)
class PriceObservationAdmin(admin.ModelAdmin):
    list_display = ("metric", "source", "observed_at", "price_fp18")
    list_filter = ("metric", "source")
    search_fields = ("metric", "source")
    ordering = ("-observed_at",)
