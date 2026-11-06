# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    PORT=8090

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root runtime user (uid 1000) and persistent data mount point
RUN groupadd -g 1000 muse && useradd -u 1000 -g 1000 -m -s /bin/bash muse \
    && mkdir -p /data/models /data/calls /data/fillers /app \
    && chown -R muse:muse /data /app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY --chown=muse:muse app /app/app

USER 1000:1000

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8090/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
