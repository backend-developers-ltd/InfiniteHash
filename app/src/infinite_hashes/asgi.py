import os

from django.core.asgi import get_asgi_application

# init django before importing urls
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")
http_app = get_asgi_application()


application = http_app
