from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("validator", "0003_auctionresult"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="auctionresult",
            name="delivered",
        ),
    ]
