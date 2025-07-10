from django.core.management.base import BaseCommand

from infinite_hashes.validator.tasks import set_weights


class Command(BaseCommand):
    help = "Set them"

    def handle(self, *args, **kwargs):
        set_weights()
