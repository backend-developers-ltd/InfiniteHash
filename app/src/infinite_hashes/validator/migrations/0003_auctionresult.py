from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("validator", "0002_alter_weightsbatch_options"),
    ]

    operations = [
        migrations.CreateModel(
            name="AuctionResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("epoch_start", models.BigIntegerField(help_text="Epoch start block")),
                ("start_block", models.BigIntegerField(help_text="Auction window start block")),
                ("end_block", models.BigIntegerField(help_text="Auction window end block", unique=True)),
                ("commitments_count", models.IntegerField(default=0)),
                ("winners", models.JSONField(default=list, help_text="List of winning hotkeys")),
                ("delivered", models.JSONField(default=dict, help_text="Map hotkey -> delivered bool")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-end_block"],
            },
        ),
    ]
