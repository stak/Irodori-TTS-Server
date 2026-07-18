# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}
ARG IRODORI_TTS_BACKEND=cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    # Pin the torch.compile caches to stable paths so compose can mount
    # volumes there; with IRODORI_COMPILE=1 only the first container run
    # then pays the cold-compile cost.
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    TRITON_CACHE_DIR=/root/.cache/triton

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --no-install-project --extra "${IRODORI_TTS_BACKEND}"

COPY README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --no-editable --extra "${IRODORI_TTS_BACKEND}"

EXPOSE 8088

CMD ["/app/.venv/bin/python", "-m", "irodori_openai_tts"]
