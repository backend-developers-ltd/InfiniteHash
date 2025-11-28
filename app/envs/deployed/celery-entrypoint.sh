#!/bin/sh
set -eu

./prometheus-cleanup.sh

# Dedicated queues with one worker each (prefork concurrency controlled via CELERY_QUEUE_CONCURRENCY, default 2)
QUEUES="default luxor auctions weights prices"
CONCURRENCY="${CELERY_QUEUE_CONCURRENCY:-2}"
OPTIONS="-A infinite_hashes -E -l DEBUG --pidfile=/var/run/celery-%n.pid --logfile=/var/log/celery-%n.log"

# shellcheck disable=2086
C_FORCE_ROOT=1 nice celery multi start $QUEUES $OPTIONS \
    -Q:default default --concurrency:default=$CONCURRENCY --max-tasks-per-child:default=1 \
    -Q:luxor luxor --concurrency:luxor=$CONCURRENCY --max-tasks-per-child:luxor=1 \
    -Q:auctions auctions --concurrency:auctions=$CONCURRENCY --max-tasks-per-child:auctions=1 \
    -Q:weights weights --concurrency:weights=$CONCURRENCY --max-tasks-per-child:weights=1 \
    -Q:prices prices --concurrency:prices=$CONCURRENCY --max-tasks-per-child:prices=1

# shellcheck disable=2064
trap "celery multi stop $WORKERS $OPTIONS; exit 0" INT TERM

tail -f /var/log/celery-*.log &

# check celery status periodically to exit if it crashed
while true; do
    sleep 120
    echo "Checking celery status"
    celery -A infinite_hashes status -t 30 > /dev/null 2>&1 || exit 1
    echo "Celery status OK"
done
