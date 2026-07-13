FROM python:3.13-slim

LABEL org.opencontainers.image.source=https://github.com/hxmatt/smoke-fire-detection-service

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv

# OpenCV shared libs (via ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# service deps + CPU torch from the lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group service --extra cpu

COPY src ./src

# Trained weights bundled in so the image runs with no mount or download.
COPY weights /weights

ENV WEIGHTS_DIR=/weights
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
