ARG BASE_IMAGE=python:3.11-slim

FROM $BASE_IMAGE AS base-image
LABEL builder=true

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /root/src/

COPY pyproject.toml uv.lock ./

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy


RUN apt-get update && apt-get install -y git

RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --frozen --no-install-project --no-dev --no-editable

COPY ./app/src/ ./app/envs/deployed/entrypoint.sh ./app/envs/deployed/gunicorn.conf.py ./app/envs/deployed/celery-entrypoint.sh ./app/envs/deployed/prometheus-cleanup.sh /root/src/

RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --frozen --no-dev --no-editable

RUN ENV=prod ENV_FILL_MISSING_VALUES=1 SECRET_KEY=dummy uv run python3 manage.py collectstatic --no-input --clear


FROM $BASE_IMAGE AS secondary-image
LABEL builder=false

RUN apt-get update \
	&& apt-get install -y wget \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/src/
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/src/.venv/bin:$PATH"

COPY --from=base-image /root/src/ /root/src/


CMD ["./entrypoint.sh"]
