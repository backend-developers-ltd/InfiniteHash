from django.core.management.base import BaseCommand

from infinite_hashes.validator.tasks import calculate_weights


class Command(BaseCommand):
    help = "Calculate them"

    def handle(self, *args, **kwargs):
        calculate_weights()
