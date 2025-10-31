from django.conf import settings
from django.contrib.admin.sites import site
from django.http import HttpResponse
from django.urls import include, path

from infinite_hashes.validator.business_metrics import metrics_manager
from infinite_hashes.validator.metrics import metrics_view

urlpatterns = [
    path("alive/", lambda _: HttpResponse(b"ok")),
    path("admin/", site.urls),
    path("metrics", metrics_view, name="prometheus-django-metrics"),
    path("business-metrics", metrics_manager.view, name="prometheus-business-metrics"),
    path("", include("django.contrib.auth.urls")),
]


if settings.DEBUG_TOOLBAR:
    urlpatterns += [
        path("__debug__/", include("debug_toolbar.urls")),
    ]
