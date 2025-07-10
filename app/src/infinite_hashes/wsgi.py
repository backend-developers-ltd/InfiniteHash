import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")

application = get_wsgi_application()
