# Generated by Django 4.2.23 on 2025-07-07 17:05

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="WeightsBatch",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "block",
                    models.BigIntegerField(help_text="Block number for which this batch is scheduled", unique=True),
                ),
                ("epoch_start", models.BigIntegerField(help_text="Epoch's start Block number")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("scored", models.BooleanField(default=False)),
                ("should_be_scored", models.BooleanField(default=True)),
                ("weights", models.JSONField(default=dict)),
            ],
        ),
    ]
