import multiprocessing

import environ
from prometheus_client import multiprocess

env = environ.Env()

workers = env.int("GUNICORN_WORKERS", 2 * multiprocessing.cpu_count() + 1)
max_workers = env.int("GUNICORN_MAX_WORKERS", 0)
if max_workers > 0:
    workers = min(max_workers, workers)
threads = env.int("GUNICORN_THREADS", 1)
preload_app = env.bool("GUNICORN_PRELOAD_APP", True)
bind = "unix:/var/run/gunicorn/gunicorn.sock"
wsgi_app = "infinite_hashes.wsgi:application"
access_logfile = "-"


def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
